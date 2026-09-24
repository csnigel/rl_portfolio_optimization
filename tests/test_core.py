from __future__ import annotations

import numpy as np
import pandas as pd

from rl_portfolio_optimization.common.performance import perf_stats
from rl_portfolio_optimization.notebook_v0_5_dirichlet.env import PortfolioEnvV0_5
from rl_portfolio_optimization.notebook_v1_gaussian.env import PortfolioEnvV1


TICKERS = ["BIL", "RISKY"]
ASSET_COLUMNS = ["ret_1m", "vol_1m", "amihud_1m"]
SHARED_COLUMNS = ["avg_corr_1m", "xsec_disp_1m"]
MACRO_COLUMNS = ["macro"]


def sample_panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    index = pd.MultiIndex.from_product([dates, TICKERS], names=["date", "ticker"])
    panel = pd.DataFrame(index=index)
    panel["ret_1m"] = [0.001, 0.02, 0.001, -0.01, 0.001, 0.03]
    panel["vol_1m"] = 0.10
    panel["amihud_1m"] = 0.0
    panel["avg_corr_1m"] = 0.25
    panel["xsec_disp_1m"] = 0.05
    panel["macro"] = 0.0
    return panel


def test_dirichlet_environment_produces_long_only_fully_invested_weights() -> None:
    env = PortfolioEnvV0_5(
        sample_panel(),
        TICKERS,
        ASSET_COLUMNS,
        SHARED_COLUMNS,
        MACRO_COLUMNS,
    )
    env.step(np.array([-2.0, 1.0], dtype=np.float32))
    weights = env.w_prev

    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)


def test_gaussian_environment_assigns_residual_to_bil() -> None:
    env = PortfolioEnvV1(
        sample_panel(),
        TICKERS,
        ASSET_COLUMNS,
        SHARED_COLUMNS,
        MACRO_COLUMNS,
    )
    weights = env._action_to_weights_with_BIL_residual(
        np.array([0.0, 0.40], dtype=np.float32),
        bil_idx=env.bil_idx,
    )

    assert np.allclose(weights, [0.60, 0.40])
    assert np.isclose(weights.sum(), 1.0)


def test_environment_step_applies_turnover_cost() -> None:
    env = PortfolioEnvV0_5(
        sample_panel(),
        TICKERS,
        ASSET_COLUMNS,
        SHARED_COLUMNS,
        MACRO_COLUMNS,
        tc_bps=10.0,
    )
    env.reset()
    _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))

    assert info["turnover"] > 0.0
    assert info["cost"] == info["turnover"] * 10.0 / 10_000.0
    assert info["net_ret"] == info["gross_ret"] - info["cost"]


def test_performance_summary_has_expected_fields() -> None:
    result = perf_stats(pd.Series([0.01, -0.005, 0.02, 0.0]))

    assert list(result.index) == ["ann_ret", "ann_vol", "sharpe"]
    assert np.isfinite(result.to_numpy()).all()
