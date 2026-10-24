"""Data-quality checks. Nothing is silently repaired: every fix is counted and reported, and
anything we cannot fix safely is an error that stops the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class DataQualityError(ValueError):
    pass


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    tickers: list[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    issues: list[Issue] = field(default_factory=list)
    n_rows: int = 0
    start: str | None = None
    end: str | None = None
    n_forward_filled: dict[str, int] = field(default_factory=dict)
    n_rows_dropped_leading: int = 0

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise DataQualityError("; ".join(f"[{i.code}] {i.message}" for i in self.errors))

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_rows": self.n_rows,
            "start": self.start,
            "end": self.end,
            "n_forward_filled": self.n_forward_filled,
            "n_rows_dropped_leading": self.n_rows_dropped_leading,
            "issues": [i.__dict__ for i in self.issues],
        }


def validate_prices(
    prices: pd.DataFrame,
    max_ffill: int = 5,
    max_abs_return: float = 0.40,
    stale_run: int = 10,
    max_calendar_gap_days: int = 7,
    min_rows: int = 252,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Return (clean_prices, report). Clean prices are aligned on the common history."""
    rep = DataQualityReport()
    if not isinstance(prices.index, pd.DatetimeIndex):
        rep.issues.append(Issue("error", "index", "index is not a DatetimeIndex"))
        return prices, rep
    if prices.index.has_duplicates:
        rep.issues.append(Issue("error", "duplicates", "duplicate timestamps in index"))
        return prices, rep
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()
        rep.issues.append(Issue("warning", "unsorted", "index was not sorted; sorted ascending"))

    bad = prices.columns[((prices <= 0) | np.isinf(prices)).any()].tolist()
    if bad:
        rep.issues.append(Issue("error", "non_positive", "non-positive or infinite prices", bad))
        return prices, rep

    # Align on the common start (assets listed later cut the history) and report the cut.
    first_valid = prices.apply(pd.Series.first_valid_index)
    if first_valid.isna().any():
        empty = first_valid[first_valid.isna()].index.tolist()
        rep.issues.append(Issue("error", "empty_series", "ticker has no data", empty))
        return prices, rep
    common_start = first_valid.max()
    before = len(prices)
    prices = prices.loc[common_start:]
    rep.n_rows_dropped_leading = before - len(prices)
    if first_valid.nunique() > 1:
        late = first_valid[first_valid > first_valid.min()].index.tolist()
        rep.issues.append(
            Issue("warning", "late_start", f"history cut to common start {common_start.date()}", late)
        )

    # Limited forward-fill of short gaps (e.g. local holidays); longer gaps are errors.
    n_nan = prices.isna().sum()
    filled = prices.ffill(limit=max_ffill)
    rep.n_forward_filled = {c: int(n_nan[c]) for c in prices.columns if n_nan[c] > 0}
    remaining = filled.columns[filled.isna().any()].tolist()
    if remaining:
        rep.issues.append(
            Issue("error", "gaps", f"NaN gaps longer than {max_ffill} bars remain", remaining)
        )
        return filled, rep
    prices = filled

    rets = prices.pct_change().iloc[1:]
    extreme = rets.columns[(rets.abs() > max_abs_return).any()].tolist()
    if extreme:
        msg = f"|daily return| > {max_abs_return:.0%}: check splits / bad ticks"
        rep.issues.append(Issue("warning", "extreme_return", msg, extreme))

    unchanged = prices.diff() == 0
    stale = []
    for c in prices.columns:
        run = unchanged[c].astype(int).groupby((~unchanged[c]).cumsum()).sum()
        if run.max() >= stale_run:
            stale.append(c)
    if stale:
        msg = f"price unchanged for >= {stale_run} consecutive bars"
        rep.issues.append(Issue("warning", "stale", msg, stale))

    gaps = prices.index.to_series().diff().dt.days.dropna()
    if (gaps > max_calendar_gap_days).any():
        msg = f"calendar gap > {max_calendar_gap_days} days in index"
        rep.issues.append(Issue("warning", "calendar_gap", msg))

    if len(prices) < min_rows:
        rep.issues.append(Issue("error", "too_short", f"only {len(prices)} rows (< {min_rows})"))

    rep.n_rows = len(prices)
    rep.start = str(prices.index[0].date()) if len(prices) else None
    rep.end = str(prices.index[-1].date()) if len(prices) else None
    return prices, rep
