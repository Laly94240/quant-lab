"""Baselines: SMA trend following (long/flat) and an equal-weight buy-and-hold benchmark.

Hypothesis (trend): asset returns show time-series momentum at multi-month horizons, so being
invested only when price is above its long moving average cuts exposure to prolonged
drawdowns. It is a hypothesis, not a fact: the pipeline is there to try to falsify it.
"""
from __future__ import annotations

import pandas as pd

from .base import Strategy, hold_between_rebalances


class TrendFollowing(Strategy):
    name = "trend_sma"

    def __init__(self, lookback: int = 200, rebalance: str = "M", weighting: str = "equal",
                 vol_lookback: int = 60):
        super().__init__(lookback=lookback, rebalance=rebalance, weighting=weighting,
                         vol_lookback=vol_lookback)
        self.lookback, self.rebalance = lookback, rebalance
        self.weighting, self.vol_lookback = weighting, vol_lookback

    def target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        sma = prices.rolling(self.lookback, min_periods=self.lookback).mean()
        signal = (prices > sma).astype(float)            # 1 = invested, 0 = cash
        n = prices.shape[1]
        if self.weighting == "equal":
            w = signal / n                               # each asset owns a 1/n slot; off => cash
        elif self.weighting == "inv_vol":
            vol = prices.pct_change().rolling(self.vol_lookback).std()
            inv = 1.0 / vol
            w = signal * inv.div(inv.sum(axis=1), axis=0)   # normalised over the whole universe
        else:
            raise ValueError(f"unknown weighting {self.weighting!r}")
        return hold_between_rebalances(w.fillna(0.0), self.rebalance)


class BuyAndHold(Strategy):
    """Equal-weight, periodically rebalanced, always invested. Used as the benchmark."""
    name = "equal_weight_bh"

    def __init__(self, rebalance: str = "M"):
        super().__init__(rebalance=rebalance)
        self.rebalance = rebalance

    def target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
        return hold_between_rebalances(w, self.rebalance)
