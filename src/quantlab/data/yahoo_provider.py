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

    def _download(self, tickers: Sequence[str], start, end) -> dict[str, pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("Install the optional extra: pip install 'quantlab[yahoo]'") from exc

        tickers = list(tickers)
        raw = yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column"
        )
        fields: dict[str, pd.DataFrame] = {}
        for name in ("Open", "High", "Low", "Close", "Volume"):
            try:
                frame = raw[name]
            except KeyError:
                continue
            if isinstance(frame, pd.Series):
                frame = frame.to_frame(tickers[0])
            frame = frame[tickers].astype(float)
            frame.index = pd.to_datetime(frame.index).tz_localize(None)
            fields[name.lower()] = frame.sort_index()
        if "close" not in fields:
            raise ValueError("Yahoo response has no Close column")
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for t in tickers:
                snap = pd.DataFrame({f.capitalize(): fields[f][t] for f in fields})
                snap["Adj Close"] = snap["Close"]            # auto_adjust=True: Close is already adjusted
                snap = snap.dropna(subset=["Close"]).rename_axis("Date")
                snap.to_csv(self.cache_dir / f"{t}.csv")
        return fields

    def get_prices(self, tickers: Sequence[str], start=None, end=None) -> pd.DataFrame:
        return self._download(tickers, start, end)["close"]

    def get_ohlc(self, tickers: Sequence[str], start=None, end=None) -> dict[str, pd.DataFrame]:
        fields = self._download(tickers, start, end)
        missing = [f for f in ("open", "high", "low", "close") if f not in fields]
        if missing:
            raise ValueError(f"Yahoo response lacks {missing}")
        return {f: fields[f] for f in ("open", "high", "low", "close")}
