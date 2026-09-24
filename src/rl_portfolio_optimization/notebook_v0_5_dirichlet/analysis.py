from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from ..common.data_loading import build_date_level_regime_frame
from ..common.performance import get_performance_metrics
from .training import dirichlet_from_logits

BENCHMARK = {
    "end_wealth": 1.2054414169691656,
    "cagr": 0.09792596151524058,
    "max_drawdown": -0.23166865769509815,
    "annualized_vol": 0.27278683845913476,
    "max_vol_rolling_6m": 0.5389250237567842,
    "sharpe": 0.47729462181507226,
}


@torch.no_grad()
def run_policy_through_env_dirichlet(env, model, device, deterministic=True):
    s = env.reset()
    done = False

    dates, net_rets, rewards, values, entropies = [], [], [], [], []
    W = []
    Z = []

    while not done:
        s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        w_mean, logits, v = model(s_t)
        logits = logits.squeeze(0)
        v = v.squeeze(0)

        z = model.shared(s_t).squeeze(0).detach().cpu().numpy()
        Z.append(z)

        dist, alpha = dirichlet_from_logits(logits)

        if deterministic:
            a = w_mean.squeeze(0)
            ent = dist.entropy()
        else:
            a = dist.sample()
            ent = dist.entropy()

        a_np = a.detach().cpu().numpy().astype(np.float32)
        s2, r, done, info = env.step(a_np)

        dates.append(pd.to_datetime(info["date"]))
        net_rets.append(float(info["net_ret"]))
        rewards.append(float(r))
        values.append(float(v.cpu().item()))
        entropies.append(float(ent.cpu().item()))
        W.append(a_np.astype(float))

        if s2 is None:
            break
        s = s2

    W = np.vstack(W) if len(W) else np.zeros((0, len(env.tickers)))
    out = pd.DataFrame(W, columns=[f"w_{tkr}" for tkr in env.tickers])
    out["net_ret"] = np.array(net_rets, dtype=float)
    out["reward"] = np.array(rewards, dtype=float)
    out["value"] = np.array(values, dtype=float)
    out["entropy"] = np.array(entropies, dtype=float)
    out["wealth"] = np.cumprod(1.0 + out["net_ret"].to_numpy())
    out["date"] = dates

    if len(Z) > 0:
        Zm = np.vstack(Z)
        z_df = pd.DataFrame(Zm, columns=[f"z{j}" for j in range(Zm.shape[1])])
        out = pd.concat([out.reset_index(drop=True), z_df], axis=1)

    return out


def mc_test_policy_dirichlet(env, model, device, num_sims=200, seed=123):
    runs = []
    for i in range(num_sims):
        seed = seed + i
        np.random.seed(seed)
        torch.manual_seed(seed)

        out = run_policy_through_env_dirichlet(env, model, device=device, deterministic=False)
        returns = out["net_ret"].to_numpy()
        end_wealth, cagr, mdd, ann_vol, rol_vol, sharpe = get_performance_metrics(returns, periods=12, rf=0)

        runs.append(
            {
                "seed": seed,
                "simulation": i + 1,
                "end_wealth": float(end_wealth),
                "cagr": float(cagr),
                "max_drawdown": float(mdd),
                "annualized_vol": float(ann_vol),
                "max_vol_rolling_6m": float(rol_vol),
                "sharpe": float(sharpe),
            }
        )
    return pd.DataFrame(runs)


def build_market_regime_outputs(panel):
    df = build_date_level_regime_frame(panel)
    train_mask = df.index <= "2022-12-31"

    scaler = StandardScaler().fit(df.loc[train_mask].values)
    df_scaled = scaler.transform(df)

    K = 4
    hmm = GaussianHMM(n_components=K, covariance_type="diag", n_iter=500, random_state=0)
    hmm.fit(df_scaled[train_mask])

    mkt_regime = pd.Series(hmm.predict(df_scaled), index=df.index, name="mkt_regime")
    return df, mkt_regime


def build_agent_regime_outputs(out):
    z_cols = [c for c in out.columns if c.startswith("z")]
    Z = out[z_cols].values
    Zs = StandardScaler().fit_transform(Z)

    K_agent = 4
    hmm_agent = GaussianHMM(n_components=K_agent, covariance_type="diag", n_iter=500, random_state=0)
    hmm_agent.fit(Zs)

    agent_regime = pd.Series(hmm_agent.predict(Zs), index=out.index, name="agent_regime")
    return agent_regime
