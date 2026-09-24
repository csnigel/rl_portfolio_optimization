from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_TICKERS = ["BIL", "VOX", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
DEFAULT_ASSET_COLUMNS = ["ret_1m", "vol_1m", "amihud_1m"]
DEFAULT_SHARED_COLUMNS = ["avg_corr_1m", "xsec_disp_1m"]
DEFAULT_MACRO_COLUMNS = [
    "CPI_q_z",
    "INDPRO_q_z",
    "TERM_SPREAD_m_z",
    "FED_RATE_m_z",
    "HY_OAS_m_z",
    "VIX_m_z",
    "curve_inv",
    "high_vol",
    "credit_spread",
]


@dataclass(frozen=True)
class DataPaths:
    root: Path = Path("data")

    @property
    def macro_data(self) -> Path:
        return self.root / "raw" / "macro_data.csv"

    @property
    def sector_panel(self) -> Path:
        return self.root / "raw" / "sector_panel.parquet"

    @property
    def market_regime(self) -> Path:
        return self.root / "derived" / "market_regime.csv"


@dataclass(frozen=True)
class EnvConfig:
    tc_bps: float = 0.0
    l2_gamma: float = 0.0


@dataclass(frozen=True)
class SplitConfig:
    train_start: str = "2016-12-31"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-31"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-31"
    test_end: str = "2025-12-31"
    complete_start: str = "2016-12-31"
    complete_end: str = "2025-12-31"


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    ppo_epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 3e-4
    hidden_size: int = 64
    n_iters: int = 300
