"""Synthetic price generator.

!!! Synthetic data is for TESTING THE PIPELINE ONLY. Any performance number obtained on it
!!! has no financial meaning, and runs on it are flagged `is_synthetic` in the experiment store.

Two scenarios make the harness falsifiable:
  * random_walk: iid returns, no predictability. A sound pipeline must REJECT strategies here.
  * regime: a persistent common drift regime (Markov chain). Trend following has a real edge
    here; a sound pipeline must be able to DETECT it.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .base import DataProvider
from .csv_provider import _slice


def generate_synthetic_prices(
    tickers: Sequence[str],
    n_days: int = 252 * 15,
    seed: int = 0,
    scenario: str = "random_walk",
    start: str = "2007-01-03",
    annual_vol: float = 0.16,
    annual_drift: float = 0.05,
    correlation: float = 0.4,
    regime_drift: float = 0.40,
    regime_stay_prob: float = 0.995,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(tickers)
    idx = pd.bdate_range(start=start, periods=n_days)
    corr = np.full((n, n), correlation) + np.eye(n) * (1 - correlation)
    chol = np.linalg.cholesky(corr)
    daily_vol = annual_vol / np.sqrt(252)
    shocks = rng.standard_normal((n_days, n)) @ chol.T * daily_vol

    if scenario == "random_walk":
        drift = np.full(n_days, annual_drift / 252)
    elif scenario == "regime":
        state = np.empty(n_days)
        s = 1.0
        for t in range(n_days):
            if rng.random() > regime_stay_prob:
                s = -s
            state[t] = s
        drift = state * regime_drift / 252
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    rets = shocks + drift[:, None]
    prices = 100.0 * np.exp(np.cumsum(rets - 0.5 * daily_vol**2, axis=0))
    return pd.DataFrame(prices, index=idx, columns=list(tickers))


class SyntheticProvider(DataProvider):
    name = "synthetic"
    is_synthetic = True

    def __init__(self, seed: int = 0, scenario: str = "random_walk", n_days: int = 252 * 15):
        self.seed, self.scenario, self.n_days = seed, scenario, n_days

    def get_prices(self, tickers, start=None, end=None) -> pd.DataFrame:
        prices = generate_synthetic_prices(
            tickers, n_days=self.n_days, seed=self.seed, scenario=self.scenario
        )
        return _slice(prices, start, end)
