import io
import json
import zipfile
from datetime import date

import pandas as pd
import pytest

from quantlab.backtest import CostModel
from quantlab.cli import main
from quantlab.crypto_report import build_report, extract_trades, render_html
from quantlab.data import BinanceVisionProvider, generate_synthetic_prices
from quantlab.data.binance_vision import (
    BASE,
    _default_fetch,
    check_symbol,
    funding_url,
    kline_url,
    months_between,
    parse_funding_csv,
    parse_klines_csv,
)

DAY = 86_400_000


def kline_csv(start_ms, n, price0=100.0, header=False, micro=False):
    rows = ["open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"] if header else []
    for i in range(n):
        t = start_ms + i * DAY
        t = t * 1000 if micro else t
        p = price0 + i
        rows.append(f"{t},{p},{p + 1},{p - 1},{p + 0.5},10,{t + DAY - 1},1000,5,5,500,0")
    return "\n".join(rows) + "\n"


def zip_of(text, name="x.csv"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def ms(d):
    return int(pd.Timestamp(d).timestamp() * 1000)


def test_urls_symbols_and_months():
    assert kline_url("BTCUSDT", "1d", "2024-01") == f"{BASE}/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
    assert kline_url("BTCUSDT", "1d", "2024-01-05", daily=True).startswith(f"{BASE}/daily/klines/BTCUSDT/1d/")
    assert funding_url("ETHUSDT", "2024-01").endswith("/fundingRate/ETHUSDT/ETHUSDT-fundingRate-2024-01.zip")
    assert months_between("2024-11", date(2025, 2, 10)) == ["2024-11", "2024-12", "2025-01", "2025-02"]
    for bad in ("btcusdt", "BTC/USDT", "BTCUSDT; rm -rf", "../x", ""):
        with pytest.raises(ValueError):
            check_symbol(bad)
    with pytest.raises(ValueError):
        _default_fetch("https://evil.example/x.zip")                      # fixed host only


@pytest.mark.parametrize("header,micro", [(False, False), (True, False), (False, True), (True, True)])
def test_kline_parsing_variants(header, micro):
    df = parse_klines_csv(kline_csv(ms("2024-01-01"), 3, header=header, micro=micro))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index[0] == pd.Timestamp("2024-01-01") and df["close"].iloc[1] == 101.5


def test_funding_parsing_and_daily_sum():
    txt = "calc_time,funding_interval_hours,last_funding_rate\n" + "\n".join(
        f"{ms('2024-01-01') + h * 3_600_000 * 8},8,0.0001" for h in range(6))
    s = parse_funding_csv(txt)
    assert len(s) == 6 and s.iloc[0] == pytest.approx(0.0001)
    assert len(parse_funding_csv(txt.split("\n", 1)[1])) == 6              # headerless variant


def make_provider(files, today=date(2024, 3, 10), cache=None):
    calls = []
    def fetch(url):
        calls.append(url)
        return files.get(url)
    return BinanceVisionProvider("1d", cache_dir=cache, fetch=fetch, today=today), calls


def test_provider_uses_monthly_files_then_daily_files_for_current_month(tmp_path):
    files = {
        kline_url("BTCUSDT", "1d", "2024-01"): zip_of(kline_csv(ms("2024-01-01"), 31)),
        kline_url("BTCUSDT", "1d", "2024-02"): zip_of(kline_csv(ms("2024-02-01"), 29, 200.0)),
    }
    for d in range(1, 10):                                                # March 1..9 as daily files
        files[kline_url("BTCUSDT", "1d", f"2024-03-{d:02d}", daily=True)] = zip_of(kline_csv(ms(f"2024-03-{d:02d}"), 1, 300.0 + d))
    prov, calls = make_provider(files, cache=tmp_path)
    prices = prov.get_prices(["BTCUSDT"], "2024-01")
    assert len(prices) == 31 + 29 + 9 and prices.index.is_monotonic_increasing and not prov.warnings
    assert sum("daily" in c for c in calls) == 9 and sum("monthly" in c for c in calls) == 2
    prov2, calls2 = make_provider({}, cache=tmp_path)                       # second run: served from the cache
    assert len(prov2.get_prices(["BTCUSDT"], "2024-01")) == 69 and calls2 == [] or all("daily" not in c for c in calls2)


def test_missing_files_are_reported_not_hidden():
    files = {kline_url("BTCUSDT", "1d", "2024-01"): zip_of(kline_csv(ms("2024-01-01"), 31))}
    prov, _ = make_provider(files, today=date(2024, 3, 1))
    prices = prov.get_prices(["BTCUSDT"], "2024-01")
    assert len(prices) == 31 and any("2024-02" in w for w in prov.warnings)
    empty, _ = make_provider({}, today=date(2024, 3, 1))
    with pytest.raises(FileNotFoundError):
        empty.get_prices(["BTCUSDT"], "2024-01")
    with pytest.raises(ValueError):
        prov.get_prices(["BTCUSDT; evil"], "2024-01")


def test_funding_provider_returns_daily_sums_or_none():
    txt = "calc_time,funding_interval_hours,last_funding_rate\n" + "\n".join(
        f"{ms('2024-01-01') + h * 3_600_000 * 8},8,0.0001" for h in range(3 * 31))
    prov, _ = make_provider({funding_url("BTCUSDT", "2024-01"): zip_of(txt)}, today=date(2024, 2, 15))
    f = prov.get_funding(["BTCUSDT"], "2024-01")
    assert f["BTCUSDT"].iloc[0] == pytest.approx(0.0003) and len(f) == 31
    none, _ = make_provider({}, today=date(2024, 2, 15))
    assert none.get_funding(["BTCUSDT"], "2024-01") is None and none.warnings


def test_extract_trades_toy():
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    prices = pd.DataFrame({"A": [100, 100, 110, 120, 90, 90, 95, 100.0]}, index=idx)
    held = pd.DataFrame({"A": [0, 0, 1, 1, 0, 0, 1, 1.0]}, index=idx)      # earns r[2], r[3] then r[6], r[7]
    t = extract_trades(prices, held)
    assert len(t) == 2
    assert (t[0].entry_date, t[0].exit_date, t[0].entry_px, t[0].exit_px) == ("2024-01-02", "2024-01-04", 100.0, 120.0)
    assert t[0].ret == pytest.approx(0.2) and t[0].days == 2 and not t[0].open
    assert t[1].open and t[1].entry_px == 90.0 and t[1].ret == pytest.approx(100 / 90 - 1)


@pytest.fixture(scope="module")
def crypto_prices():
    return generate_synthetic_prices(["BTCUSDT", "ETHUSDT", "BNBUSDT"], n_days=365 * 4, seed=2, scenario="regime",
                                     start="2021-01-01", annual_vol=0.7, annual_drift=0.3, regime_drift=0.8,
                                     freq="D", periods_per_year=365)


def test_report_content_and_safety(crypto_prices):
    rep = build_report(crypto_prices, source="<b>x</b>", is_synthetic=True, warnings=["<script>alert(1)</script>"])
    html = render_html(rep)
    for needle in ("Comment la stratégie de tendance aurait été appliquée", "Données synthétiques", "Sensibilité aux coûts",
                   "Rendements mensuels", "<polyline", "<polygon", "Limites"):
        assert needle in html
    assert "<script" not in html and "<b>x</b>" not in html and "&lt;script&gt;" in html
    assert rep["tstats"]["n_trades"] == len(rep["trades"]) > 0
    assert set(rep["metrics"]) == {"Stratégie tendance", "Panier équipondéré (acheter et garder)", "BTCUSDT (acheter et garder)"}


def test_cost_sensitivity_is_monotone_and_zero_cost_is_best(crypto_prices):
    rep = build_report(crypto_prices, source="t", is_synthetic=True)
    f = rep["cost_finals"]
    assert f == sorted(f, reverse=True) and rep["cost_mult"][0] == 0


def test_funding_only_ever_reduces_long_returns(crypto_prices):
    base = build_report(crypto_prices, source="t", is_synthetic=True)
    funding = pd.DataFrame(0.0003, index=crypto_prices.index, columns=crypto_prices.columns)   # longs pay
    paid = build_report(crypto_prices, source="t", is_synthetic=True, funding=funding)
    assert paid["metrics"]["Stratégie tendance"]["total_return"] < base["metrics"]["Stratégie tendance"]["total_return"]
    assert paid["metrics"]["Stratégie tendance"]["annual_funding_drag"] > 0
    assert paid["params"]["funding_included"] is True


def test_strategy_is_long_flat_and_unlevered(crypto_prices):
    rep = build_report(crypto_prices, source="t", is_synthetic=True, cost=CostModel(0, 0, 0))
    w = rep["res"].weights
    assert (w >= 0).all().all() and w.sum(axis=1).max() <= 1.0 + 1e-9


def test_cli_synthetic_preview(tmp_path, capsys):
    out, js = tmp_path / "r.html", tmp_path / "r.json"
    assert main(["binance-report", "--synthetic", "--out", str(out), "--json", str(js)]) == 0
    assert "report written" in capsys.readouterr().out
    assert "Données synthétiques" in out.read_text(encoding="utf-8")
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["is_synthetic"] is True and "metrics" in data and data["params"]["lookback"] == 200


def test_cli_rejects_dangerous_symbols(tmp_path):
    with pytest.raises(ValueError):
        main(["binance-report", "--synthetic", "--symbols", "BTCUSDT;rm", "--out", str(tmp_path / "x.html")])
