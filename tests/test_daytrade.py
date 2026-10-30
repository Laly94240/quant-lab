import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quantlab.backtest import BacktestConfig, CostModel
from quantlab.cli import main
from quantlab.data import CSVProvider, YahooProvider, generate_synthetic_ohlc
from quantlab.daytrade import (
    OHLC,
    DayStrategy,
    GapFade,
    IBSReversal,
    IntradayTrend,
    LookaheadError,
    check_no_lookahead_day,
    clean_ohlc,
    day_weights,
    decompose,
    run_daytrade_research,
    simulate_day,
    trade_stats,
    walk_forward_day,
)
from quantlab.daytrade_report import render_daytrade_html
from quantlab.experiments import ExperimentStore
from quantlab.risk import RiskLimits

CFG_PATH = Path(__file__).parents[1] / "configs" / "etf_daytrade.yaml"
CFG = yaml.safe_load(CFG_PATH.read_text())
TICKERS = CFG["universe"]
ZERO = BacktestConfig(cost=CostModel(0, 0, 0), execution_delay=0, risk=RiskLimits(max_position_weight=1.0))


def toy(o, h, lo, c):
    idx = pd.bdate_range("2024-01-01", periods=len(o))
    mk = lambda v: pd.DataFrame({"A": np.asarray(v, dtype=float)}, index=idx)   # noqa: E731
    return OHLC(mk(o), mk(h), mk(lo), mk(c))


@pytest.fixture(scope="module")
def ohlc_rw():
    return clean_ohlc(generate_synthetic_ohlc(TICKERS, seed=7, scenario="random_walk"))[0]


def test_session_and_night_returns_compound_exactly_to_close_to_close(ohlc_rw):
    o, c = ohlc_rw.open, ohlc_rw.close
    intra, night = c / o - 1, o / c.shift(1) - 1
    total = c / c.shift(1) - 1
    assert np.allclose(((1 + intra) * (1 + night) - 1).iloc[1:], total.iloc[1:])
    d = decompose(ohlc_rw)
    assert set(d["per_asset"]) == set(TICKERS) and len(d["portfolio"]) == 3


def test_simulate_day_hand_computed_with_round_trip_costs():
    ohlc = toy([100, 100, 100], [102, 102, 102], [99, 99, 99], [101, 99, 100])
    w = pd.DataFrame({"A": [1.0, 1.0, 0.0]}, index=ohlc.close.index)
    free = simulate_day(ohlc, w, ZERO)
    assert free.returns.round(6).tolist() == [0.01, -0.01, 0.0] and free.turnover.tolist() == [2.0, 2.0, 0.0]
    cfg = BacktestConfig(cost=CostModel(commission_bps=2, spread_bps=0, slippage_bps=1), execution_delay=0,
                         risk=RiskLimits(max_position_weight=1.0))
    paid = simulate_day(ohlc, w, cfg, extra_bps=2.0)             # round trip = 2 x (3 + 2) = 10 bps per full position
    assert paid.costs.round(6).tolist() == [0.001, 0.001, 0.0]
    assert paid.returns.iloc[0] == pytest.approx(0.01 - 0.001) and paid.returns.iloc[2] == 0.0


def test_no_trade_when_open_is_missing():
    ohlc = toy([100, np.nan, 100], [101] * 3, [99] * 3, [101, 101, 101])
    w = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=ohlc.close.index)
    res = simulate_day(ohlc, w, ZERO)
    assert res.returns.iloc[1] == 0.0 and res.weights["A"].iloc[1] == 0.0


def test_strategy_signal_logic_on_toy_series():
    n = 40
    close = np.linspace(100, 140, n)                           # steady uptrend
    openp = np.r_[close[0], close[:-1]] * 1.0
    o, h, lo, c = openp.copy(), close * 1.01, close * 0.99, close.copy()
    o[30] = c[29] * 0.97                                       # a 3% gap down on day 30
    ohlc = toy(o, h, lo, c)
    g = GapFade(k=1.0, vol_window=5, trend_lookback=10).signals(ohlc)
    assert g["A"].iloc[30] == 1.0 and g["A"].iloc[29] == 0.0 and g["A"].iloc[31] == 0.0
    lo2, h2, c2 = lo.copy(), h.copy(), c.copy()                # day 20 closes near its low, still above its SMA
    h2[20], lo2[20], c2[20] = c[20], c[20] * 0.99, c[20] * 0.9905
    ibs = IBSReversal(theta=0.2, trend_lookback=10).signals(toy(o, h2, lo2, c2))
    assert ibs["A"].iloc[21] == 1.0 and ibs["A"].iloc[20] == 0.0
    tr = IntradayTrend(lookback=10).signals(ohlc)
    assert tr["A"].iloc[:10].sum() == 0 and tr["A"].iloc[-1] == 1.0


@pytest.mark.parametrize("strat", [IntradayTrend(lookback=100), GapFade(k=0.5), IBSReversal(theta=0.3)],
                         ids=lambda s: s.name)
def test_registered_strategies_do_not_use_the_sessions_high_low_close(strat, ohlc_rw):
    check_no_lookahead_day(strat, ohlc_rw)


def test_lookahead_detector_catches_same_day_information(ohlc_rw):
    class PeekClose(DayStrategy):
        name = "peek_close"
        def signals(self, ohlc):
            return (ohlc.close > ohlc.open).astype(float)         # today's close: unknowable at the open

    class PeekHigh(DayStrategy):
        name = "peek_high"
        def signals(self, ohlc):
            return (ohlc.high > ohlc.open * 1.002).astype(float)

    class SameDayOpenOnly(DayStrategy):
        name = "ok_open_only"
        def signals(self, ohlc):
            return (ohlc.open > ohlc.close.shift(1)).astype(float)   # the open is known: allowed

    for bad in (PeekClose(), PeekHigh()):
        with pytest.raises(LookaheadError):
            check_no_lookahead_day(bad, ohlc_rw)
    check_no_lookahead_day(SameDayOpenOnly(), ohlc_rw)


def test_a_rule_that_peeks_at_the_close_would_look_great_without_the_check(ohlc_rw):
    """Why the check matters: peeking at the close turns a no-edge market into a spectacular backtest."""
    peek = (ohlc_rw.close > ohlc_rw.open).astype(float) / len(TICKERS)
    res = simulate_day(ohlc_rw, peek, ZERO)
    assert res.returns.mean() * 252 > 0.5


def test_clean_ohlc_repairs_and_reports_inconsistent_bars_but_rejects_garbage():
    raw = generate_synthetic_ohlc(["A", "B"], n_days=600, seed=1)
    raw["high"].iloc[10, 0] = raw["close"].iloc[10, 0] * 0.9    # high below the close
    ohlc, warnings, _ = clean_ohlc(raw)
    assert any("repaired" in w for w in warnings)
    assert (ohlc.high >= np.maximum(ohlc.open, ohlc.close) - 1e-9).all().all()
    bad = generate_synthetic_ohlc(["A"], n_days=600, seed=1)
    bad["high"] = bad["low"] * 0.5                              # systematically inconsistent
    with pytest.raises(ValueError):
        clean_ohlc(bad)
    neg = generate_synthetic_ohlc(["A"], n_days=600, seed=1)
    neg["open"].iloc[5, 0] = -1.0
    with pytest.raises(ValueError):
        clean_ohlc(neg)


def test_trade_stats_edge_and_cost(ohlc_rw):
    cfg = BacktestConfig(cost=CostModel(), execution_delay=0)
    res = simulate_day(ohlc_rw, day_weights(IntradayTrend(lookback=100), ohlc_rw, cfg), cfg, 2.0)
    t = trade_stats(res, ohlc_rw, cfg, 2.0)
    assert t["round_trip_cost_bps"] == pytest.approx(2 * (CostModel().one_way_bps + 2.0))
    assert t["net_edge_bps"] == pytest.approx(t["gross_edge_bps"] - t["round_trip_cost_bps"])
    assert 0.2 < t["pct_days_in_market"] < 1.0 and t["trades_per_year"] > 100


def test_walk_forward_day_ignores_future_data(ohlc_rw):
    combos = [{"lookback": 100}, {"lookback": 200}]
    cfg = BacktestConfig(cost=CostModel(), execution_delay=0)
    full = walk_forward_day(ohlc_rw, IntradayTrend, combos, cfg, 2.0)
    cut_ts = pd.Timestamp(full.folds[3].test_end) + pd.Timedelta(days=1)
    part_ohlc = OHLC(*(getattr(ohlc_rw, f).loc[:cut_ts] for f in ("open", "high", "low", "close")))
    part = walk_forward_day(part_ohlc, IntradayTrend, combos, cfg, 2.0)
    for a, b in zip(full.folds[:4], part.folds[:4], strict=True):
        assert a.params == b.params
    end = pd.Timestamp(full.folds[3].test_end)
    pd.testing.assert_series_equal(full.oos.returns.loc[:end], part.oos.returns.loc[:end], check_exact=False, rtol=1e-12)


def _run(scenario, seed, store=None, cfg=CFG):
    ohlc, w, dq = clean_ohlc(generate_synthetic_ohlc(TICKERS, seed=seed, scenario=scenario))
    return run_daytrade_research(cfg, ohlc, is_synthetic=True, data_source="synthetic", store=store,
                                 data_warnings=w, dq=dq)


def test_false_positive_control_on_random_walk_markets():
    passes = sum(r["verdict"] == "PASS_TO_PAPER" for s in range(8) for r in _run("random_walk", s)["reports"])
    assert passes == 0


def test_detects_a_planted_gap_reversal_edge_and_only_the_right_rule():
    results = [{r["strategy"]: r["verdict"] for r in _run("gap_reversal", s)["reports"]} for s in range(6)]
    assert sum(v["gap_fade"] == "PASS_TO_PAPER" for v in results) >= 4
    assert all(v["intraday_trend"] == "REJECT" and v["ibs_reversal"] == "REJECT" for v in results)


def test_trading_every_day_is_destroyed_by_costs_even_without_any_edge():
    r = {x["strategy"]: x for x in _run("random_walk", 3)["reports"]}["intraday_trend"]
    t = r["trade_stats"]
    assert t["trades_per_year"] > 500 and abs(t["gross_edge_bps"]) < t["round_trip_cost_bps"]
    assert r["oos_metrics"]["cagr"] < 0 and "cost_stress_sharpe" in r["failed_gates"]


def test_store_records_runs_counts_trials_and_skips_identical_experiments(tmp_path):
    store = ExperimentStore(tmp_path)
    r1 = _run("random_walk", 2, store)
    assert len(r1["reports"]) == 3 and r1["n_trials_total"] == 10 == store.cumulative_trials(CFG["research_id"])
    run = tmp_path / r1["reports"][0]["run_id"]
    assert (run / "report.json").exists() and (run / "benchmark_returns.csv").exists()
    r2 = _run("random_walk", 2, store)
    assert len(r2["skipped"]) == 3 and not r2["reports"]
    cfg2 = deepcopy(CFG)
    cfg2["strategies"] = [{"name": "gap_fade", "grid": {"k": [0.75, 1.25]}}]
    assert _run("random_walk", 2, store, cfg2)["n_trials_total"] == 12


def test_report_is_self_contained_and_escapes_everything():
    res = _run("random_walk", 1)
    res["data"]["warnings"] = ["<script>alert(1)</script>"]
    res["data"]["source"] = "<img src=x onerror=alert(1)>"
    html = render_daytrade_html(res)
    for needle in ("Day trading sur données journalières", "Données synthétiques", "D'où vient le rendement",
                   "Long chaque jour", "Critères de validation", "<polyline"):
        assert needle in html
    assert "<script" not in html and "<img src=x" not in html and "http://" not in html


def test_cli_end_to_end_synthetic(tmp_path, capsys):
    out, js = tmp_path / "d.html", tmp_path / "d.json"
    assert main(["daytrade", "--config", str(CFG_PATH), "--provider", "synthetic", "--scenario", "gap_reversal",
                 "--seed", "2", "--store-dir", str(tmp_path / "runs"), "--out", str(out), "--json", str(js)]) == 0
    text = capsys.readouterr().out
    assert "SYNTHETIC DATA" in text and "gap_fade" in text and "edge bp" in text
    data = json.loads(js.read_text(encoding="utf-8"))
    assert {r["strategy"] for r in data["reports"]} == {"intraday_trend", "gap_fade", "ibs_reversal"}
    assert "Day trading" in out.read_text(encoding="utf-8")


def test_yahoo_snapshot_keeps_ohlc_and_csv_provider_reads_it_back(tmp_path, monkeypatch):
    """Uses a FAKE yfinance module shaped like the documented output (no network in the sandbox)."""
    import sys
    import types
    idx = pd.date_range("2024-01-01", periods=6, tz="UTC")
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["SPY", "TLT"]])
    base = np.arange(1, 13, dtype=float)
    raw = pd.DataFrame(np.column_stack([np.tile(base[:6] + k, (2, 1)).T.reshape(6, 2) for k in range(5)]),
                       index=idx, columns=cols)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=lambda *a, **k: raw))
    fields = YahooProvider(cache_dir=tmp_path).get_ohlc(["SPY", "TLT"])
    assert set(fields) == {"open", "high", "low", "close"} and fields["open"].index.tz is None
    back = CSVProvider(tmp_path).get_ohlc(["SPY", "TLT"])
    for f in fields:
        assert np.allclose(back[f].to_numpy(), fields[f].to_numpy())
    assert np.allclose(CSVProvider(tmp_path).get_prices(["SPY", "TLT"]).to_numpy(), fields["close"].to_numpy())
    old = tmp_path / "old"
    old.mkdir()
    pd.DataFrame({"Date": ["2024-01-02"], "Adj Close": [1.0]}).to_csv(old / "SPY.csv", index=False)
    with pytest.raises(ValueError, match="re-run"):
        CSVProvider(old).get_ohlc(["SPY"])
