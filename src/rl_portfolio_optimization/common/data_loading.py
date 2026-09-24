from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def load_notebook_source_data(data_dir: str | Path):
    data_dir = Path(data_dir)

    macro_df = pd.read_csv(data_dir / "macro_data.csv")
    macro_df["Unnamed: 0"] = pd.to_datetime(macro_df["Unnamed: 0"])
    macro_df = macro_df.set_index("Unnamed: 0").sort_index()
    macro_df.index.name = "date"

    etf_df = pd.read_parquet(data_dir / "sector_panel.parquet")

    panel = etf_df.join(macro_df, how="left")

    missing = panel[macro_df.columns].isna().any(axis=1).groupby(level=0).any()
    common_dates = etf_df.index.get_level_values(0).unique().intersection(macro_df.index)
    panel = etf_df.loc[common_dates].join(macro_df, how="left")
    missing = panel[macro_df.columns].isna().any(axis=1).groupby(level=0).any()
    dates = sorted(panel.index.get_level_values(0).unique())

    return {
        "macro_df": macro_df,
        "etf_df": etf_df,
        "panel": panel,
        "dates": dates,
        "missing_macro_dates": missing[missing].index,
    }


def build_date_level_regime_frame(panel: pd.DataFrame) -> pd.DataFrame:
    asset_cols = ["ret_1m", "vol_1m"]
    date_cols = [
        "avg_corr_1m",
        "xsec_disp_1m",
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

    asset_mean = panel[asset_cols].groupby(level=0).mean().add_suffix("_mean")
    asset_std = panel[asset_cols].groupby(level=0).std().add_suffix("_std")
    date_level = panel[date_cols].groupby(level=0).first()
    df = pd.concat([asset_mean, asset_std, date_level], axis=1).dropna()
    return df
