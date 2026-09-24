from __future__ import annotations

import numpy as np
import pandas as pd


class PortfolioEnvV0_5:
    def __init__(self, panel, tickers, asset_cols, shared_cols, macro_cols, tc_bps=10.0, l2_gamma=0.01, start_date=None, end_date=None):
        self.panel = panel
        self.tickers = list(tickers)
        self.asset_cols = list(asset_cols)
        self.shared_cols = list(shared_cols)
        self.macro_cols = list(macro_cols)
        self.l2_gamma = float(l2_gamma)
        self.tc_bps = float(tc_bps)
        self.tc = self.tc_bps / 10000.0

        all_dates = sorted(panel.index.get_level_values(0).unique())
        if start_date is not None:
            all_dates = [d for d in all_dates if d >= pd.to_datetime(start_date)]
        if end_date is not None:
            all_dates = [d for d in all_dates if d <= pd.to_datetime(end_date)]

        if len(all_dates) < 2:
            raise ValueError("Need at least 2 dates (t and t+1) in the env range.")

        self.dates = all_dates
        self.n_assets = len(self.tickers)
        self.reset()

    def _slice(self, date):
        return self.panel.loc[date].reindex(self.tickers)

    def _get_state(self, date):
        df = self._slice(date)
        asset_block = df[self.asset_cols].to_numpy(np.float32).reshape(-1)
        shared = df[self.shared_cols].iloc[0].to_numpy(np.float32)
        macro = df[self.macro_cols].iloc[0].to_numpy(np.float32)
        state = np.concatenate([asset_block, shared, macro, self.w_prev], axis=0).astype(np.float32)
        return state

    def reset(self):
        self.t = 0
        self.done = False
        self.w_prev = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        return self._get_state(self.dates[self.t])

    def step(self, w):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        w = np.asarray(w, dtype=np.float32)
        w = np.clip(w, 0, None)
        w = w / (w.sum() + 1e-12)

        date_tp1 = self.dates[self.t + 1]
        ret_tp1 = self._slice(date_tp1)["ret_1m"].to_numpy(np.float32)

        gross_ret = float(np.dot(w, ret_tp1))
        turnover = float(np.sum(np.abs(w - self.w_prev)))
        cost = float(self.tc * turnover)
        net_ret = gross_ret - cost
        net_ret = max(net_ret, -0.999999)
        log_ret = float(np.log1p(net_ret))
        l2_pen = float(np.sum(w * w))
        reward = log_ret - self.l2_gamma * l2_pen

        self.w_prev = w
        self.t += 1

        info = {
            "date": date_tp1,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "turnover": turnover,
            "cost": cost,
        }

        if self.t >= len(self.dates) - 1:
            self.done = True
            return None, reward, True, info

        next_state = self._get_state(self.dates[self.t])
        return next_state, reward, False, info
