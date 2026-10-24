"""Multiple-testing aware statistics (Bailey & Lopez de Prado).

All Sharpe ratios here are PER-PERIOD (not annualised), consistent with the formulas.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


def probabilistic_sharpe_ratio(
    sr_hat: float, sr_benchmark: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """P(true SR > sr_benchmark) given an observed SR, accounting for skew and fat tails."""
    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    if n_obs < 3 or denom <= 0:
        return float("nan")
    z = (sr_hat - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(sps.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum of n_trials Sharpe estimates under the null of zero true skill."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe_ratio(
    sr_hat: float, n_obs: int, n_trials: int, var_sr_trials: float,
    skew: float = 0.0, kurt: float = 3.0,
) -> float:
    """PSR against the Sharpe one would expect from the best of `n_trials` skill-less trials."""
    sr0 = expected_max_sharpe(n_trials, var_sr_trials)
    return probabilistic_sharpe_ratio(sr_hat, sr0, n_obs, skew, kurt)


def sample_var(values) -> float:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    return float(np.var(v, ddof=1)) if len(v) > 1 else 0.0
