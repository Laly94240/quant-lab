"""Yahoo Finance provider (via the unofficial `yfinance` wrapper).

Good enough for research prototyping on daily ETF data. NOT a production data source: the
endpoints are undocumented, can change or throttle without notice, and Yahoo's terms restrict
use to personal purposes. Always snapshot the download to CSV (see `quantlab fetch`) so that
experiments stay reproducible even if the upstream data later changes.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .base import DataProvider


class YahooProvider(DataProvider):
    name = "yahoo"

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def get_prices(self, tickers: Sequence[str], start=None, end=None) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("Install the optional extra: pip install 'quantlab[yahoo]'") from exc

        tickers = list(tickers)
        raw = yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column"
        )
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
        close = close[tickers].astype(float)
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = close.sort_index()
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for t in tickers:
                out = close[t].dropna().rename("Adj Close").rename_axis("Date")
                out.to_csv(self.cache_dir / f"{t}.csv")
        return close
