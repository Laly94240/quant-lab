"""Local CSV provider: one file per ticker, `<root>/<TICKER>.csv`, with a Date column and an
'Adj Close' (preferred) or 'Close' column. This is the reproducible, offline path: fetch once,
record the data hash, rerun forever."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import DataProvider


class CSVProvider(DataProvider):
    name = "csv"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def get_prices(self, tickers, start=None, end=None) -> pd.DataFrame:
        missing = [t for t in tickers if not (self.root / f"{t}.csv").exists()]
        if missing:
            raise FileNotFoundError(f"No CSV in {self.root} for: {missing}")
        cols = {}
        for t in tickers:
            df = pd.read_csv(self.root / f"{t}.csv")
            lower = {c.lower().strip(): c for c in df.columns}
            date_col = lower.get("date") or df.columns[0]
            price_col = lower.get("adj close") or lower.get("adj_close") or lower.get("close")
            if price_col is None:
                raise ValueError(f"{t}.csv has no 'Adj Close'/'Close' column: {list(df.columns)}")
            cols[t] = pd.Series(
                df[price_col].to_numpy(dtype=float),
                index=pd.to_datetime(df[date_col]),
                name=t,
            )
        prices = pd.DataFrame(cols).sort_index()
        prices.index = pd.DatetimeIndex(prices.index).rename(None)
        return _slice(prices, start, end)


def _slice(prices: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start is not None:
        prices = prices.loc[pd.Timestamp(start):]
    if end is not None:
        prices = prices.loc[: pd.Timestamp(end)]
    return prices
