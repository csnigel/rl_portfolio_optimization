from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_logprob(a, mu, log_std):
    std = torch.exp(log_std)
    var = std ** 2
    log2pi = np.log(2.0 * np.pi)
    logp = -0.5 * (((a - mu) ** 2) / var + 2.0 * log_std + log2pi)
    return logp.sum(dim=-1)


def gaussian_entropy(log_std):
    return (0.5 * (1.0 + np.log(2.0 * np.pi)) + log_std).sum()


class GaussianActorCriticV1(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.v_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        z = self.shared(obs)
        mu = self.mu_head(z)
        v = self.v_head(z).squeeze(-1)
        return mu, self.log_std, v

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        mu, log_std, v = self(obs.unsqueeze(0))
        mu = mu.squeeze(0)
        v = v.squeeze(0)

        if deterministic:
            a = mu
        else:
            std = torch.exp(log_std)
            a = mu + std * torch.randn_like(mu)

        logp = gaussian_logprob(a.unsqueeze(0), mu.unsqueeze(0), log_std).squeeze(0)
        return a, logp, v


@dataclass
class GaussianTrainingBundle:
    model: GaussianActorCriticV1
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
        a_t, logp_t, v_t = model.act(s_t, deterministic=False)
        ent_t = gaussian_entropy(model.log_std)
        a_np = a_t.detach().cpu().numpy().astype(np.float32)
        s2, r, done, info = env.step(a_np)

        states.append(s)
        actions.append(a_np)
        logps.append(float(logp_t.cpu().item()))
        rewards.append(float(r))
        dones.append(bool(done))
        values.append(float(v_t.cpu().item()))
        entropies.append(float(ent_t.cpu().item()))
        infos.append(info)

        s = s2 if s2 is not None else s
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "logps": np.asarray(logps, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "values": np.asarray(values, dtype=np.float32),
        "entropies": np.asarray(entropies, dtype=np.float32),
        "infos": infos,
    }


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)

    lastgaelam = 0.0
    for t in reversed(range(T)):
        not_done = 0.0 if dones[t] else 1.0
        v_next = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * not_done * v_next - values[t]
        lastgaelam = delta + gamma * lam * not_done * lastgaelam
        adv[t] = lastgaelam

    returns = adv + values
    return returns.astype(np.float32), adv.astype(np.float32)


def ppo_update(rollout, model, optimizer, device, gamma=0.99, lam=0.95, clip_eps=0.2, value_coef=0.5, entropy_coef=0.0, ppo_epochs=5, batch_size=64):
    states = torch.tensor(rollout["states"], dtype=torch.float32, device=device)
    actions = torch.tensor(rollout["actions"], dtype=torch.float32, device=device)
    old_logps = torch.tensor(rollout["logps"], dtype=torch.float32, device=device)

    returns_np, adv_np = compute_gae(rollout["rewards"], rollout["values"], rollout["dones"], gamma=gamma, lam=lam)
    returns = torch.tensor(returns_np, dtype=torch.float32, device=device)
    adv = torch.tensor(adv_np, dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = states.shape[0]
    idx = np.arange(N)
    mb = min(batch_size, N)

    for _ in range(ppo_epochs):
        np.random.shuffle(idx)

        for start in range(0, N, mb):
            mb_idx = idx[start:start + mb]
            s_mb = states[mb_idx]
            a_mb = actions[mb_idx]
            old_logp_mb = old_logps[mb_idx]
            ret_mb = returns[mb_idx]
            adv_mb = adv[mb_idx]

            mu_mb, log_std, v_mb = model(s_mb)

            logp_mb = gaussian_logprob(a_mb, mu_mb, log_std)
            ratio = torch.exp(logp_mb - old_logp_mb)

            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(v_mb, ret_mb)
            entropy = gaussian_entropy(model.log_std)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def train_gaussian_v1(train_env, val_env):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = train_env.reset().shape[0]
    n_assets = len(train_env.tickers)

    model = GaussianActorCriticV1(state_dim, n_assets, hidden=64).to(device)
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
            lam=0.95,
            clip_eps=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            ppo_epochs=5,
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

    return GaussianTrainingBundle(
        model=model,
        optimizer=optimizer,
        device=device,
        train_ep_rewards=train_ep_rewards,
        val_ep_rewards=val_ep_rewards,
    )
