"""Optional, REDUCE-ONLY exposure overlay driven by a validated news briefing.

Guarantees (enforced by code and by tests):
  * it can only shrink positions: |output| <= |input| for every asset, never add or flip;
  * the shrink is bounded (`max_reduction`, default 50 %), per flagged asset only;
  * only assets present in the portfolio are touched; a stale briefing (older than `ttl_hours`) is ignored;
  * disabled by default. It has NO historical validation (no archive of past news is available to
    backtest it), so it must be judged in paper trading before anyone relies on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from .models import Briefing

_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class OverlayConfig:
    enabled: bool = False
    max_reduction: float = 0.5
    min_severity: str = "high"
    ttl_hours: float = 24.0


def apply_event_overlay(weights: pd.Series, briefing: Briefing | None, cfg: OverlayConfig,
                        now: datetime | None = None) -> pd.Series:
    if not cfg.enabled or briefing is None:
        return weights
    max_red = min(max(cfg.max_reduction, 0.0), 1.0)
    now = now or datetime.now(timezone.utc)
    try:
        created = datetime.fromisoformat(briefing.created_utc)
    except ValueError:
        return weights                                      # unknown age => ignore
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    if now - created > timedelta(hours=cfg.ttl_hours):
        return weights

    factor = pd.Series(1.0, index=weights.index)
    floor = _RANK[cfg.min_severity]
    for flag in briefing.risk_flags:
        rank = _RANK.get(flag.severity, -1)
        if rank < floor:
            continue
        reduction = max_red if rank >= 2 else max_red / 2.0
        for sym in flag.affected_assets:
            if sym in factor.index:
                factor[sym] = min(factor[sym], 1.0 - reduction)
    return weights * factor
