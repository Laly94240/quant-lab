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

    def get_ohlc(self, tickers, start=None, end=None) -> dict[str, pd.DataFrame]:
        """{'open','high','low','close'} -> DataFrame (one column per ticker). Prices are the adjusted
        series written by YahooProvider (open/high/low/close adjusted with the same factor)."""
        missing = [t for t in tickers if not (self.root / f"{t}.csv").exists()]
        if missing:
            raise FileNotFoundError(f"No CSV in {self.root} for: {missing}")
        frames = {t: _read_ohlc_file(self.root / f"{t}.csv", t) for t in tickers}
        return {f: _slice(pd.DataFrame({t: frames[t][f] for t in tickers}).sort_index(), start, end)
                for f in ("open", "high", "low", "close")}


def _read_ohlc_file(path: Path, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}
    close_col = lower.get("close") or lower.get("adj close")
    missing = [f for f in ("open", "high", "low") if f not in lower] + ([] if close_col else ["close"])
    if missing:
        raise ValueError(
            f"{ticker}.csv lacks {missing}: re-run `quantlab fetch` (older snapshots only hold 'Adj Close')")
    date_col = lower.get("date") or df.columns[0]
    out = pd.DataFrame({
        "open": df[lower["open"]].to_numpy(dtype=float), "high": df[lower["high"]].to_numpy(dtype=float),
        "low": df[lower["low"]].to_numpy(dtype=float), "close": df[close_col].to_numpy(dtype=float)},
        index=pd.DatetimeIndex(pd.to_datetime(df[date_col])).rename(None))
    return out.sort_index()


def _slice(prices: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start is not None:
        prices = prices.loc[pd.Timestamp(start):]
    if end is not None:
        prices = prices.loc[: pd.Timestamp(end)]
    return prices
