"""Performance metrics computed from a net daily return series."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps


def _nan_to_none(x):
    return None if (x is None or (isinstance(x, float) and not math.isfinite(x))) else x


def max_drawdown(returns: pd.Series) -> float:
    eq = np.concatenate([[1.0], (1.0 + returns.to_numpy()).cumprod()])
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def sharpe_per_period(returns: pd.Series, rf_per_period: float = 0.0) -> float:
    ex = returns - rf_per_period
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd) if sd > 0 else float("nan")


def compute_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
    exposure: pd.Series | None = None,
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    ppy = periods_per_year
    n = len(returns)
    out: dict = {"n_obs": n}
    if n < 2:
        return out
    rf = rf_annual / ppy
    ex = returns - rf
    eq_end = float((1.0 + returns).prod())
    sd = float(ex.std(ddof=1))
    downside = float(np.sqrt(np.mean(np.minimum(ex.to_numpy(), 0.0) ** 2)))
    mdd = max_drawdown(returns)
    cagr = eq_end ** (ppy / n) - 1.0 if eq_end > 0 else -1.0

    active = returns[exposure > 1e-12] if exposure is not None else returns[returns != 0]
    wins, losses = active[active > 0], active[active < 0]

    out.update(
        total_return=eq_end - 1.0,
        cagr=cagr,
        ann_vol=sd * math.sqrt(ppy),
        sharpe=float(ex.mean() / sd * math.sqrt(ppy)) if sd > 0 else float("nan"),
        sortino=float(ex.mean() / downside * math.sqrt(ppy)) if downside > 0 else float("nan"),
        max_drawdown=mdd,
        calmar=cagr / abs(mdd) if mdd < 0 else float("nan"),
        daily_win_rate=float((active > 0).mean()) if len(active) else float("nan"),
        avg_win_over_avg_loss=(
            float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else float("nan")
        ),
        skew=float(sps.skew(returns)),
        kurtosis=float(sps.kurtosis(returns, fisher=False)),
    )
    if turnover is not None:
        out["annual_turnover"] = float(turnover.mean() * ppy)
    if costs is not None:
        out["annual_cost_drag"] = float(costs.mean() * ppy)
    if exposure is not None:
        out["avg_gross_exposure"] = float(exposure.mean())
    return {k: _nan_to_none(v) if isinstance(v, float) else v for k, v in out.items()}
