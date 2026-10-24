"""Data provider interface. Providers return adjusted-close (total return) prices."""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd


class DataProvider(ABC):
    name: str = "base"
    is_synthetic: bool = False

    @abstractmethod
    def get_prices(
        self, tickers: Sequence[str], start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Return adjusted closes: ascending tz-naive DatetimeIndex, one column per ticker."""


def dataframe_hash(df: pd.DataFrame) -> str:
    """Stable short fingerprint of a price table (values, index, columns) for reproducibility."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    h.update(",".join(map(str, df.columns)).encode())
    return h.hexdigest()[:16]
