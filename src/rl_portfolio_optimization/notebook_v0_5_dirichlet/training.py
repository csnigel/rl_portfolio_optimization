from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirichletActorCriticV0_5(nn.Module):
    def __init__(self, state_dim, n_assets, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden, n_assets)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.shared(x)
        logits = self.actor_head(z)
        value = self.critic_head(z).squeeze(-1)
        weights = F.softmax(logits, dim=-1)
        return weights, logits, value


def dirichlet_from_logits(logits, min_conc=1e-3):
    alpha = F.softplus(logits) + min_conc
    return torch.distributions.Dirichlet(alpha), alpha


@dataclass
class DirichletTrainingBundle:
    model: DirichletActorCriticV0_5
    optimizer: torch.optim.Optimizer
    device: torch.device
    train_ep_rewards: list[float]
    val_ep_rewards: list[float]


@torch.no_grad()
def collect_rollout(env, model, device, max_steps=None):
    states, actions, logps, rewards, dones, values, entropies = [], [], [], [], [], [], []
    infos = []

    s = env.reset()
    done = False
    steps = 0

    while not done:
        s_t = torch.tensor(s, dtype=torch.float32, device=device)
        weights, logits, v = model(s_t)
        dist, alpha = dirichlet_from_logits(logits)

        a = dist.sample()
        logp = dist.log_prob(a)
        ent = dist.entropy()

        a_np = a.detach().cpu().numpy().astype(np.float32)
        s2, r, done, info = env.step(a_np)

        states.append(s)
        actions.append(a_np)
        logps.append(float(logp.cpu().item()))
        rewards.append(float(r))
        dones.append(done)
        values.append(float(v.cpu().item()))
        entropies.append(float(ent.cpu().item()))
        infos.append(info)

        s = s2 if s2 is not None else s
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "logps": np.array(logps, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=np.bool_),
        "values": np.array(values, dtype=np.float32),
        "entropies": np.array(entropies, dtype=np.float32),
        "infos": infos,
    }


def compute_returns_advantages(rewards, values, gamma):
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    adv = returns - values
    return returns, adv


def ppo_update(rollout, model, optimizer, device, gamma=0.99, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, ppo_epochs=10, batch_size=64):
    states = torch.tensor(rollout["states"], dtype=torch.float32, device=device)
    actions = torch.tensor(rollout["actions"], dtype=torch.float32, device=device)
    old_logps = torch.tensor(rollout["logps"], dtype=torch.float32, device=device)

    returns_np, adv_np = compute_returns_advantages(rollout["rewards"], rollout["values"], gamma)
    returns = torch.tensor(returns_np, dtype=torch.float32, device=device)
    adv = torch.tensor(adv_np, dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = states.shape[0]
    idx = np.arange(N)

    for _ in range(ppo_epochs):
        np.random.shuffle(idx)
        for start in range(0, N, batch_size):
            mb = idx[start:start + batch_size]

            s_mb = states[mb]
            a_mb = actions[mb]
            old_logp_mb = old_logps[mb]
            ret_mb = returns[mb]
            adv_mb = adv[mb]

            w_mean, logits, v = model(s_mb)
            dist, alpha = dirichlet_from_logits(logits)

            logp = dist.log_prob(a_mb)
            entropy = dist.entropy().mean()
            ratio = torch.exp(logp - old_logp_mb)

            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(v, ret_mb)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def train_dirichlet_v0_5(train_env, val_env):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = train_env.reset().shape[0]
    n_assets = len(train_env.tickers)

    model = DirichletActorCriticV0_5(state_dim, n_assets, hidden=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    train_ep_rewards = []
    val_ep_rewards = []
    n_iters = 300

    for it in range(1, n_iters + 1):
        roll = collect_rollout(train_env, model, device)
        train_ep_rewards.append(float(roll["rewards"].sum()))
        ppo_update(
            roll,
            model,
            optimizer,
            device,
            gamma=0.99,
            clip_eps=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            ppo_epochs=10,
            batch_size=64,
        )

        vroll = collect_rollout(val_env, model, device)
        val_ep_rewards.append(float(vroll["rewards"].sum()))

        if it % 20 == 0:
            print(
                f"iter {it:4d} | "
                f"train_ep_reward {train_ep_rewards[-1]:.4f} | "
                f"val_ep_reward {val_ep_rewards[-1]:.4f}"
            )

    return DirichletTrainingBundle(
        model=model,
        optimizer=optimizer,
        device=device,
        train_ep_rewards=train_ep_rewards,
        val_ep_rewards=val_ep_rewards,
    )
