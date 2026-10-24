import sys
import types

import numpy as np
import pandas as pd
import pytest
from conftest import make_prices

from quantlab.data import (
    CSVProvider,
    SyntheticProvider,
    YahooProvider,
    dataframe_hash,
    generate_synthetic_prices,
    validate_prices,
)


def _long(n=400, **kw):
    rng = np.random.default_rng(0)
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return make_prices({"A": base, "B": base * 1.1, **kw})


def test_clean_data_passes():
    clean, rep = validate_prices(_long())
    assert rep.ok and rep.n_rows == 400 and not rep.errors


def test_duplicate_index_is_an_error():
    p = _long()
    p = pd.concat([p, p.iloc[[5]]]).sort_index()
    _, rep = validate_prices(p)
    assert not rep.ok and rep.errors[0].code == "duplicates"


def test_non_positive_price_is_an_error():
    p = _long()
    p.iloc[10, 0] = 0.0
    _, rep = validate_prices(p)
    assert [i.code for i in rep.errors] == ["non_positive"]


def test_short_gap_is_filled_and_counted_long_gap_is_error():
    p = _long()
    p.iloc[50:53, 0] = np.nan
    clean, rep = validate_prices(p, max_ffill=5)
    assert rep.ok and rep.n_forward_filled == {"A": 3}
    assert clean["A"].iloc[51] == p["A"].iloc[49]          # carried forward, not invented

    p.iloc[100:120, 1] = np.nan
    _, rep = validate_prices(p, max_ffill=5)
    assert [i.code for i in rep.errors] == ["gaps"]


def test_leading_nans_align_on_common_start_and_are_reported():
    p = _long()
    p.iloc[:30, 1] = np.nan
    clean, rep = validate_prices(p)
    assert len(clean) == 370 and rep.n_rows_dropped_leading == 30
    assert any(i.code == "late_start" for i in rep.warnings)


def test_extreme_return_and_stale_series_warn():
    p = _long()
    p.iloc[200:, 0] *= 0.4                                   # e.g. an unadjusted split
    p.iloc[300:315, 1] = p.iloc[300, 1]
    _, rep = validate_prices(p)
    codes = {i.code for i in rep.warnings}
    assert {"extreme_return", "stale"} <= codes


def test_too_short_history_is_an_error():
    _, rep = validate_prices(_long(n=100))
    assert any(i.code == "too_short" for i in rep.errors)


def test_synthetic_is_deterministic_and_flagged():
    a = SyntheticProvider(seed=1).get_prices(["X", "Y"])
    b = SyntheticProvider(seed=1).get_prices(["X", "Y"])
    c = SyntheticProvider(seed=2).get_prices(["X", "Y"])
    assert a.equals(b) and not a.equals(c)
    assert SyntheticProvider.is_synthetic is True
    assert dataframe_hash(a) == dataframe_hash(b) != dataframe_hash(c)


def test_csv_roundtrip_and_missing_ticker(tmp_path):
    p = generate_synthetic_prices(["X", "Y"], n_days=300, seed=4)
    for t in p:
        p[t].rename("Adj Close").rename_axis("Date").to_csv(tmp_path / f"{t}.csv")
    got = CSVProvider(tmp_path).get_prices(["X", "Y"])
    pd.testing.assert_frame_equal(got, p, check_freq=False, check_exact=False, rtol=1e-9)
    with pytest.raises(FileNotFoundError):
        CSVProvider(tmp_path).get_prices(["X", "NOPE"])


def test_yahoo_provider_parses_multiindex_frame_and_snapshots_csv(tmp_path, monkeypatch):
    """Uses a FAKE yfinance module shaped like the documented output; it checks OUR parsing
    and caching, not Yahoo's real behaviour (the sandbox has no access to it)."""
    idx = pd.date_range("2024-01-01", periods=5, tz="UTC")
    cols = pd.MultiIndex.from_product([["Close", "Volume"], ["SPY", "TLT"]])
    raw = pd.DataFrame(np.arange(20, dtype=float).reshape(5, 4) + 1, index=idx, columns=cols)
    fake = types.SimpleNamespace(download=lambda *a, **k: raw)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    out = YahooProvider(cache_dir=tmp_path).get_prices(["SPY", "TLT"], "2024-01-01", None)
    assert list(out.columns) == ["SPY", "TLT"] and out.index.tz is None
    assert (tmp_path / "SPY.csv").exists()
    again = CSVProvider(tmp_path).get_prices(["SPY", "TLT"])
    assert np.allclose(again.to_numpy(), out.to_numpy())
