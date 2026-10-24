"""Strategy interface and the anti-look-ahead contract.

Contract: row t of `target_weights(prices)` may use ONLY prices[: t + 1] (information known at
the close of bar t). `check_no_lookahead` enforces it empirically by truncation: recomputing
the weights on a truncated history must reproduce the same rows. Any dependence on the future
(shift(-1), centered windows, full-sample normalisation, ...) breaks that equality.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class LookaheadError(AssertionError):
    pass


class Strategy(ABC):
    name: str = "strategy"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Target portfolio weights decided at each close (fraction of equity, same shape)."""

    def describe(self) -> dict:
        return {"name": self.name, "params": dict(self.params)}


def rebalance_mask(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """True on the first bar of each new period. Depends only on past bars (no look-ahead)."""
    if freq == "D":
        return np.ones(len(index), dtype=bool)
    code = {"W": "W", "M": "M", "Q": "Q"}.get(freq)
    if code is None:
        raise ValueError(f"unsupported rebalance frequency {freq!r}")
    key = index.to_period(code)
    mask = np.ones(len(index), dtype=bool)
    mask[1:] = np.asarray(key[1:] != key[:-1])
    return mask


def hold_between_rebalances(weights: pd.DataFrame, freq: str) -> pd.DataFrame:
    mask = rebalance_mask(weights.index, freq)
    out = weights.copy()
    out.loc[~mask, :] = np.nan
    return out.ffill().fillna(0.0)


def check_no_lookahead(
    strategy: Strategy, prices: pd.DataFrame, n_checks: int = 6, seed: int = 0, min_history: int = 30
) -> None:
    full = strategy.target_weights(prices)
    rng = np.random.default_rng(seed)
    cuts = rng.choice(np.arange(min_history, len(prices) - 1), size=n_checks, replace=False)
    for k in sorted(cuts):
        partial = strategy.target_weights(prices.iloc[: k + 1])
        if not np.allclose(partial.to_numpy(), full.iloc[: k + 1].to_numpy(), atol=1e-10, equal_nan=True):
            raise LookaheadError(
                f"{strategy.name}: weights up to {prices.index[k].date()} change when future "
                "data is removed -> the strategy uses information from the future."
            )
