"""Deterministic risk controls. Plain, tested code: the AI layer never bypasses it.

The same `apply_weight_limits` is used in the backtest and in the live/paper path, so the
research results already include the constraints that will apply in production.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskLimits:
    max_position_weight: float = 0.35     # per-asset cap, as a fraction of equity
    max_gross_exposure: float = 1.0       # sum |w|; 1.0 = no leverage
    allow_short: bool = False
    max_daily_loss: float = 0.03          # halt if one-day loss exceeds this
    max_drawdown: float = 0.25            # halt if drawdown from peak exceeds this
    max_orders_per_day: int = 20
    max_data_age_bdays: int = 3           # stale-data guard (business days)


def apply_weight_limits(weights: pd.DataFrame | pd.Series, limits: RiskLimits):
    """Clip per-asset weights, then scale rows down so gross exposure <= max_gross_exposure."""
    lower = -limits.max_position_weight if limits.allow_short else 0.0
    w = weights.clip(lower=lower, upper=limits.max_position_weight)
    if isinstance(w, pd.Series):
        gross = float(w.abs().sum())
        return w * (limits.max_gross_exposure / gross) if gross > limits.max_gross_exposure else w
    gross = w.abs().sum(axis=1)
    scale = np.where(gross > limits.max_gross_exposure, limits.max_gross_exposure / gross, 1.0)
    return w.mul(scale, axis=0)


def check_halt(equity: pd.Series, limits: RiskLimits) -> str | None:
    """Return a halt reason (kill-switch trigger) or None. `equity` is the account value series."""
    if len(equity) < 2:
        return None
    day = equity.iloc[-1] / equity.iloc[-2] - 1.0
    if day < -limits.max_daily_loss:
        return f"daily loss {day:.2%} exceeds limit {limits.max_daily_loss:.2%}"
    dd = equity.iloc[-1] / equity.cummax().iloc[-1] - 1.0
    if dd < -limits.max_drawdown:
        return f"drawdown {dd:.2%} exceeds limit {limits.max_drawdown:.2%}"
    return None


def check_data_freshness(last_bar: pd.Timestamp, today: pd.Timestamp, limits: RiskLimits) -> str | None:
    """Return an error if the latest market-data bar is older than the allowed business days."""
    age = int(np.busday_count(last_bar.date(), today.date()))
    if age > limits.max_data_age_bdays:
        return f"stale market data: last bar {last_bar.date()} is {age} business days old"
    return None


def check_order_count(n_orders_today: int, limits: RiskLimits) -> str | None:
    if n_orders_today > limits.max_orders_per_day:
        return f"{n_orders_today} orders exceed daily limit {limits.max_orders_per_day}"
    return None
