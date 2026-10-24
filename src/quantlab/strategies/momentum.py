"""Cross-sectional momentum with an absolute-momentum filter (long-only, 'dual momentum' style).

Hypothesis: assets that outperformed over the past 12 months (skipping the latest month to
avoid short-term reversal) tend to keep outperforming over the next month(s).
"""
from __future__ import annotations

import pandas as pd

from .base import Strategy, hold_between_rebalances


class CrossSectionalMomentum(Strategy):
    name = "xs_momentum"

    def __init__(self, lookback: int = 252, skip: int = 21, top_k: int = 3,
                 rebalance: str = "M", abs_filter: bool = True):
        super().__init__(lookback=lookback, skip=skip, top_k=top_k, rebalance=rebalance,
                         abs_filter=abs_filter)
        self.lookback, self.skip, self.top_k = lookback, skip, top_k
        self.rebalance, self.abs_filter = rebalance, abs_filter

    def target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        score = prices.shift(self.skip) / prices.shift(self.lookback) - 1.0
        rank = score.rank(axis=1, ascending=False, method="first")
        selected = rank <= self.top_k
        if self.abs_filter:
            selected &= score > 0
        w = selected.astype(float) / self.top_k           # fixed slots: unselected slots stay cash
        return hold_between_rebalances(w, self.rebalance)
