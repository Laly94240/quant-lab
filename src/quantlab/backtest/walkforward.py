"""Walk-forward validation with parameter selection on the TRAIN window only.

For each fold, the parameter set with the best in-sample Sharpe is selected on data strictly
before the test window; its target weights for the test window are then stitched together and
simulated once, so that switching costs between parameter sets are charged. Because every
strategy only uses past data (enforced by `check_no_lookahead`), full-sample weights restricted
to a window equal what would have been computed live at the time.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..risk import apply_weight_limits
from .engine import BacktestConfig, BacktestResult, simulate
from .metrics import sharpe_per_period


@dataclass
class Fold:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    params: dict
    train_sharpe_ann: float | None
    test_sharpe_ann: float | None


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    oos_start: pd.Timestamp
    stitched_target: pd.DataFrame
    oos: BacktestResult
    trial_sharpes: list[float]    # per-period full-sample Sharpe of every parameter set tried
    n_trials: int


def param_combinations(grid: dict, fixed: dict | None = None) -> list[dict]:
    fixed = fixed or {}
    if not grid:
        return [dict(fixed)]
    keys = list(grid)
    return [{**fixed, **dict(zip(keys, vals, strict=True))} for vals in itertools.product(*(grid[k] for k in keys))]


def fold_windows(index: pd.DatetimeIndex, train_years: int, test_years: int, anchored: bool = True):
    start, last = index[0], index[-1]
    test_start = start + pd.DateOffset(years=train_years)
    while test_start <= last:
        test_end = test_start + pd.DateOffset(years=test_years)
        train_start = start if anchored else test_start - pd.DateOffset(years=train_years)
        yield train_start, test_start, test_end
        test_start = test_end


def walk_forward(
    prices: pd.DataFrame,
    strategy_cls,
    combos: list[dict],
    cfg: BacktestConfig,
    train_years: int = 5,
    test_years: int = 1,
    anchored: bool = True,
    min_test_obs: int = 60,
) -> WalkForwardResult:
    idx = prices.index
    ppy = cfg.periods_per_year
    targets, full_runs = [], []
    for p in combos:
        tgt = apply_weight_limits(strategy_cls(**p).target_weights(prices), cfg.risk)
        targets.append(tgt)
        full_runs.append(simulate(prices, tgt, cfg))
    trial_sharpes = [sharpe_per_period(r.returns) for r in full_runs]

    stitched = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    folds: list[Fold] = []
    oos_start = None
    for tr_s, te_s, te_e in fold_windows(idx, train_years, test_years, anchored):
        train_m = (idx >= tr_s) & (idx < te_s)
        test_m = (idx >= te_s) & (idx < te_e)
        if test_m.sum() < min_test_obs or train_m.sum() < ppy:
            continue
        train_sh = [sharpe_per_period(r.returns[train_m]) for r in full_runs]
        scores = np.array([-np.inf if not np.isfinite(s) else s for s in train_sh])
        best = int(np.argmax(scores))
        stitched.loc[test_m] = targets[best].loc[test_m].to_numpy()
        oos_start = oos_start or idx[test_m][0]
        test_sh = sharpe_per_period(full_runs[best].returns[test_m])
        folds.append(Fold(
            str(idx[train_m][0].date()), str(idx[train_m][-1].date()),
            str(idx[test_m][0].date()), str(idx[test_m][-1].date()),
            combos[best],
            float(train_sh[best] * np.sqrt(ppy)) if np.isfinite(train_sh[best]) else None,
            float(test_sh * np.sqrt(ppy)) if np.isfinite(test_sh) else None,
        ))
    if not folds:
        raise ValueError("no valid walk-forward fold: history too short for train/test settings")

    oos = simulate(prices, stitched, cfg).slice_from(oos_start)
    return WalkForwardResult(folds, oos_start, stitched, oos, trial_sharpes, len(combos))
