from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from rl_portfolio_optimization.common.runtime import configure_runtime

configure_runtime(REPO_ROOT)

import matplotlib.pyplot as plt
import pandas as pd
import torch

from rl_portfolio_optimization.common import (
    DEFAULT_ASSET_COLUMNS,
    DEFAULT_MACRO_COLUMNS,
    DEFAULT_SHARED_COLUMNS,
    DEFAULT_TICKERS,
    RunReport,
    perf_stats,
    plot_metrics_dashboard,
    plot_representative_paths,
)
from rl_portfolio_optimization.common.data_loading import load_notebook_source_data
from rl_portfolio_optimization.notebook_v0_5_dirichlet import (
    BENCHMARK,
    PortfolioEnvV0_5,
    build_agent_regime_outputs,
    build_market_regime_outputs,
    mc_test_policy_dirichlet,
    run_policy_through_env_dirichlet,
    train_dirichlet_v0_5,
)


def main() -> None:
    repo_root = REPO_ROOT
    report = RunReport.create(repo_root / "reports", "notebook_v0_5_dirichlet")
    data = load_notebook_source_data(repo_root / "data" / "raw")
    panel = data["panel"]
    report.add_text(
        "Data Summary",
        f"Panel shape: `{panel.shape}`\n\nMacro shape: `{data['macro_df'].shape}`\n\nMissing macro dates after alignment: `{list(data['missing_macro_dates'])}`",
    )

    train_env = PortfolioEnvV0_5(panel, DEFAULT_TICKERS, DEFAULT_ASSET_COLUMNS, DEFAULT_SHARED_COLUMNS, DEFAULT_MACRO_COLUMNS, tc_bps=20.0, l2_gamma=0.01, start_date="2016-12-31", end_date="2022-12-31")
    val_env = PortfolioEnvV0_5(panel, DEFAULT_TICKERS, DEFAULT_ASSET_COLUMNS, DEFAULT_SHARED_COLUMNS, DEFAULT_MACRO_COLUMNS, tc_bps=20.0, l2_gamma=0.01, start_date="2023-01-31", end_date="2023-12-31")
    test_env = PortfolioEnvV0_5(panel, DEFAULT_TICKERS, DEFAULT_ASSET_COLUMNS, DEFAULT_SHARED_COLUMNS, DEFAULT_MACRO_COLUMNS, tc_bps=20.0, l2_gamma=0.01, start_date="2024-01-31", end_date="2025-12-31")
    complete_env = PortfolioEnvV0_5(panel, DEFAULT_TICKERS, DEFAULT_ASSET_COLUMNS, DEFAULT_SHARED_COLUMNS, DEFAULT_MACRO_COLUMNS, tc_bps=0, l2_gamma=0, start_date="2016-12-31", end_date="2025-12-31")

    bundle = train_dirichlet_v0_5(train_env, val_env)
    model = bundle.model
    device = bundle.device

    fig = plt.figure()
    plt.plot(bundle.train_ep_rewards, label="Train")
    plt.plot(bundle.val_ep_rewards, label="Validation")
    plt.legend()
    plt.title("Training vs Validation Reward")
    plt.tight_layout()
    fig.savefig(report.root_dir / "training_validation_reward.png", dpi=200, bbox_inches="tight")
    report.add_artifact("Training vs Validation Reward", "training_validation_reward.png")
    plt.show()

    out_test = run_policy_through_env_dirichlet(test_env, model, device=device, deterministic=True)
    fig = plt.figure()
    plt.plot(out_test["date"], out_test["wealth"])
    plt.title("PPO Portfolio Cumulative Return (Test, net)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(report.root_dir / "test_cumulative_return.png", dpi=200, bbox_inches="tight")
    report.add_artifact("Test Cumulative Return", "test_cumulative_return.png")
    plt.show()

    mc_df = mc_test_policy_dirichlet(test_env, model, device=device, num_sims=1000)
    report.save_dataframe(mc_df, "mc_metrics.csv")
    fig = plot_metrics_dashboard(mc_df, BENCHMARK, "V0 Agent Perfomance Distribution v/s Benchmark Dashboard")
    fig.savefig(report.root_dir / "mc_dashboard.png", dpi=200, bbox_inches="tight")
    report.add_artifact("Monte Carlo Metrics Dashboard", "mc_dashboard.png")
    plt.show()
    fig = plot_representative_paths(test_env, model, device, mc_df, run_policy_through_env_dirichlet)
    fig.savefig(report.root_dir / "representative_paths.png", dpi=200, bbox_inches="tight")
    report.add_artifact("Representative Paths", "representative_paths.png")
    plt.show()

    df, mkt_regime = build_market_regime_outputs(panel)

    runs = []
    for i in range(1000):
        seed = 123 + i
        torch.manual_seed(seed)
        out = run_policy_through_env_dirichlet(complete_env, model, device=device, deterministic=False)
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")
        runs.append(out)

    out = pd.concat(runs).groupby(level=0).mean()
    report.save_dataframe(out, "complete_env_mc_average.csv")
    agent_regime = build_agent_regime_outputs(out)
    report.save_dataframe(agent_regime.to_frame(), "agent_regime.csv")

    regime_cols = df[["ret_1m_mean", "curve_inv", "VIX_m_z", "HY_OAS_m_z", "credit_spread", "CPI_q_z"]]
    regime_perf_df = out.join(mkt_regime, how="left").join(regime_cols, how="left")
    cols = ["net_ret", "reward", "wealth", "mkt_regime", "ret_1m_mean", "curve_inv", "VIX_m_z", "HY_OAS_m_z", "credit_spread", "CPI_q_z"]
    regime_perf_df = regime_perf_df[cols]
    perf_by_regime = regime_perf_df.groupby("mkt_regime")["net_ret"].apply(perf_stats)
    perf_plot = perf_by_regime.unstack()
    fig = perf_plot[["ann_ret", "sharpe"]].plot(kind="bar", figsize=(10, 5)).get_figure()
    plt.title("Agent Performance by Market Regime")
    plt.ylabel("Value")
    plt.tight_layout()
    fig.savefig(report.root_dir / "performance_by_market_regime.png", dpi=200, bbox_inches="tight")
    report.add_artifact("Performance by Market Regime", "performance_by_market_regime.png")
    plt.show()

    agent_regime_perf_df = out.join(agent_regime, how="left").join(regime_cols, how="left")
    cols = ["net_ret", "reward", "wealth", "agent_regime", "ret_1m_mean", "curve_inv", "VIX_m_z", "HY_OAS_m_z", "credit_spread", "CPI_q_z"]
    agent_regime_perf_df = agent_regime_perf_df[cols]
    perf_by_agent_regime = agent_regime_perf_df.groupby("agent_regime")["net_ret"].apply(perf_stats)
    perf_plot = perf_by_agent_regime.unstack()
    fig = perf_plot[["ann_ret", "sharpe"]].plot(kind="bar", figsize=(10, 5)).get_figure()
    plt.title("Agent Performance by Policy Regime")
    plt.ylabel("Value")
    plt.tight_layout()
    fig.savefig(report.root_dir / "performance_by_policy_regime.png", dpi=200, bbox_inches="tight")
    report.add_artifact("Performance by Policy Regime", "performance_by_policy_regime.png")
    plt.show()

    report.add_text(
        "Key Outputs",
        f"Test wealth last value: `{float(out_test['wealth'].iloc[-1]):.6f}`\n\nMonte Carlo simulations: `{len(mc_df)}`",
    )
    report.write_markdown()
    print(f"Report written to {report.root_dir / 'report.md'}")


if __name__ == "__main__":
    main()
