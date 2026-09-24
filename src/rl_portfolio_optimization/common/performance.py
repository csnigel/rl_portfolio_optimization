from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


def get_performance_metrics(returns, periods=12, rf=0):
    returns = np.asarray(returns)
    wealth = np.cumprod(1 + returns)

    cagr = np.nan
    years = len(returns) / periods
    if years > 0:
        cagr = (wealth[-1] ** (1 / years)) - 1

    peak = np.maximum.accumulate(wealth)
    max_drawdown = float(((wealth / peak) - 1).min())
    annualized_vol = np.std(returns, ddof=1) * np.sqrt(periods)
    vol = pd.Series(returns).rolling(6).std() * np.sqrt(12)
    max_rolling_vol = vol.max()
    sharpe = np.mean(returns) * periods / annualized_vol

    return wealth[-1], cagr, max_drawdown, annualized_vol, max_rolling_vol, sharpe


def perf_stats(x):
    ann_ret = np.mean(x) * 12
    ann_vol = np.std(x, ddof=1) * np.sqrt(12)
    sharpe = ann_ret / (ann_vol + 1e-12)
    return pd.Series({"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe})


def plot_metrics_dashboard(mc_df, benchmark, title):
    fig = plt.figure(figsize=(16, 12))
    cols = mc_df.drop(columns=["simulation", "seed"]).columns.to_list()

    for i, col in enumerate(cols):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.hist(mc_df[col].dropna(), bins=40)
        ax.set_title(f"{col} distribution")
        ax.axvline(benchmark[col], color="red", linestyle="--", linewidth=2, label="Benchmark")
        ax.legend()

    ax6 = fig.add_subplot(3, 3, 7)
    ax6.scatter(mc_df["annualized_vol"], mc_df["cagr"], alpha=0.5)
    ax6.set_title("Vol vs CAGR")
    ax6.set_xlabel("Annualized Volatility")
    ax6.set_ylabel("CAGR")
    ax6.scatter(benchmark["annualized_vol"], benchmark["cagr"], color="red", s=120, zorder=5, label="Benchmark")
    ax6.legend()

    ax7 = fig.add_subplot(3, 3, 8)
    ax7.scatter(mc_df["max_drawdown"], mc_df["cagr"], alpha=0.5)
    ax7.set_title("Max Drawdown vs CAGR")
    ax7.set_xlabel("Max Drawdown")
    ax7.set_ylabel("CAGR")
    ax7.scatter(benchmark["max_drawdown"], benchmark["cagr"], color="red", s=120, zorder=5, label="Benchmark")
    ax7.legend()

    ax8 = fig.add_subplot(3, 3, 9)
    ax8.scatter(mc_df["max_vol_rolling_6m"], mc_df["cagr"], alpha=0.5)
    ax8.set_title("Max Vol (6m Rolling) vs CAGR")
    ax8.set_xlabel("Max Vol Rolling 6m")
    ax8.set_ylabel("CAGR")
    ax8.scatter(
        benchmark["max_vol_rolling_6m"],
        benchmark["cagr"],
        color="red",
        s=120,
        zorder=5,
        label="Benchmark",
    )
    ax8.legend()

    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    return fig


def plot_representative_paths(env, model, device, mc_df, runner):
    best_i = int(mc_df["cagr"].idxmax())
    worst_i = int(mc_df["cagr"].idxmin())
    med_target = mc_df["cagr"].median()
    median_i = int((mc_df["cagr"] - med_target).abs().idxmin())

    picks = [("Best", best_i), ("Median", median_i), ("Worst", worst_i)]

    plt.figure(figsize=(12, 5))
    for label, idx in picks:
        seed = int(mc_df.loc[idx, "seed"])
        np.random.seed(seed)
        torch.manual_seed(seed)
        out = runner(env, model, device=device, deterministic=False)
        plt.plot(out["date"], out["wealth"], label=f"{label} (seed={seed})")

    plt.title("Representative MC Wealth Paths")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()
