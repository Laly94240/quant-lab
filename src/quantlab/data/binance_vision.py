"""Binance USD-M futures public data (https://data.binance.vision), no API key, no account.

Static files only: monthly archives for complete months, daily files for the current month.
Binance's live API is NOT used (it is geo-blocked from some countries and rate limited).
The host is fixed: no URL coming from user input is ever requested.

Not tested against the live site from the development sandbox (no network); parsing is tested on
files shaped like Binance's documented layout, including both known variants (with / without a
header row, millisecond / microsecond timestamps).
"""
from __future__ import annotations

import io
import logging
import re
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .base import DataProvider

log = logging.getLogger(__name__)
BASE = "https://data.binance.vision/data/futures/um"
SYMBOL_RE = re.compile(r"^[A-Z0-9]{3,20}$")
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def check_symbol(symbol: str) -> str:
    if not SYMBOL_RE.match(symbol):
        raise ValueError(f"invalid symbol {symbol!r}")
    return symbol


def months_between(start: str, end: date) -> list[str]:
    y, m = map(int, start.split("-"))
    out = []
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def kline_url(symbol: str, interval: str, period: str, daily: bool = False) -> str:
    kind = "daily" if daily else "monthly"
    return f"{BASE}/{kind}/klines/{symbol}/{interval}/{symbol}-{interval}-{period}.zip"


def funding_url(symbol: str, month: str) -> str:
    return f"{BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"


def _read_zip_text(blob: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return zf.read(zf.namelist()[0]).decode("utf-8")


def _to_datetime(ts: pd.Series) -> pd.DatetimeIndex:
    ts = ts.astype("int64")
    unit = "us" if ts.iloc[0] > 10**14 else "ms"          # some archives switched to microseconds
    return pd.DatetimeIndex(pd.to_datetime(ts, unit=unit))


def parse_klines_csv(text: str) -> pd.DataFrame:
    header = 0 if not text.lstrip()[:1].isdigit() else None
    df = pd.read_csv(io.StringIO(text), header=header, names=None if header == 0 else KLINE_COLS)
    df.index = _to_datetime(df["open_time"])
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def parse_funding_csv(text: str) -> pd.Series:
    header = 0 if not text.lstrip()[:1].isdigit() else None
    names = None if header == 0 else ["calc_time", "funding_interval_hours", "last_funding_rate"]
    df = pd.read_csv(io.StringIO(text), header=header, names=names)
    s = pd.Series(df["last_funding_rate"].astype(float).to_numpy(), index=_to_datetime(df["calc_time"]))
    return s


def _default_fetch(url: str) -> bytes | None:
    if not url.startswith(BASE + "/"):
        raise ValueError("refusing to fetch a URL outside data.binance.vision")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "quantlab/0.1"}),
                                    timeout=60) as r:                      # noqa: S310 - fixed https host
            return r.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None                                                    # file not published (yet)
        raise


class BinanceVisionProvider(DataProvider):
    name = "binance_futures"

    def __init__(self, interval: str = "1d", cache_dir: str | Path | None = None,
                 fetch: Callable[[str], bytes | None] | None = None, today: date | None = None):
        self.interval = interval
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._fetch = fetch or _default_fetch
        self.today = today or date.today()
        self.warnings: list[str] = []

    def _get(self, url: str, symbol: str) -> bytes | None:
        cache = self.cache_dir / symbol / url.rsplit("/", 1)[-1] if self.cache_dir else None
        if cache is not None and cache.exists():
            return cache.read_bytes()
        blob = self._fetch(url)
        if blob is not None and cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(blob)
        return blob

    def _periods(self, start: str, end: str | None):
        """Monthly files for complete months, daily files for the current month."""
        last = date.fromisoformat(end + "-01") if end and len(end) == 7 else self.today
        for ym in months_between(start[:7], last):
            if (int(ym[:4]), int(ym[5:])) < (self.today.year, self.today.month):
                yield ym, False
            else:
                d = date(self.today.year, self.today.month, 1)
                while d < self.today:
                    yield d.isoformat(), True
                    d += timedelta(days=1)
                return

    def get_ohlcv(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        check_symbol(symbol)
        frames = []
        for period, daily in self._periods(start, end):
            blob = self._get(kline_url(symbol, self.interval, period, daily), symbol)
            if blob is None:
                self.warnings.append(f"{symbol}: no {'daily' if daily else 'monthly'} file for {period}")
                continue
            frames.append(parse_klines_csv(_read_zip_text(blob)))
        if not frames:
            raise FileNotFoundError(f"no data downloaded for {symbol} from {start}")
        df = pd.concat(frames).sort_index()
        return df[~df.index.duplicated(keep="last")]

    def get_prices(self, tickers: Sequence[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
        start = (start or "2020-01")[:7]
        cols = {t: self.get_ohlcv(t, start, end)["close"] for t in tickers}
        prices = pd.DataFrame(cols).sort_index()
        if self.interval == "1d":
            prices.index = prices.index.normalize()
        return prices

    def get_funding(self, tickers: Sequence[str], start: str, end: str | None = None) -> pd.DataFrame | None:
        """Daily sum of funding rates per symbol (positive = longs pay shorts). None if unavailable."""
        out = {}
        for t in tickers:
            check_symbol(t)
            parts = []
            for ym in months_between(start[:7], self.today):
                if (int(ym[:4]), int(ym[5:])) >= (self.today.year, self.today.month):
                    break
                blob = self._get(funding_url(t, ym), t)
                if blob is not None:
                    parts.append(parse_funding_csv(_read_zip_text(blob)))
            if not parts:
                self.warnings.append(f"{t}: no funding-rate files found, funding cost NOT included")
                return None
            s = pd.concat(parts).sort_index()
            out[t] = s.groupby(s.index.normalize()).sum()
        return pd.DataFrame(out)
