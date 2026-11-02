import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quantlab.cli import main
from quantlab.crypto_intraday import (
    Gates,
    HourReversal,
    IntradayStrategy,
    LockboxBurned,
    LookaheadError,
    OpenMomentum,
    RangeBreakout,
    TrialRegistry,
    check_no_lookahead_bars,
    combos_of,
    evaluate_gates,
    generate_synthetic_bars,
    run_protocol,
    run_trades,
    to_bars,
    trade_stats,
)
from quantlab.crypto_intraday_report import render_crypto_intraday_html

CFG_PATH = Path(__file__).parents[1] / "configs" / "crypto_intraday.yaml"
CFG = yaml.safe_load(CFG_PATH.read_text())
CFG_T = deepcopy(CFG)
CFG_T["blocks"] = {"dev_end": "2021-12-31", "validation_end": "2022-12-31", "lockbox_end": None}


def toy_bars(opens, closes, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(opens), freq="15min")
    o, c = np.asarray(opens, dtype=float), np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999, "close": c, "volume": 1.0}, index=idx)


def sig(bars, entries):
    s = pd.DataFrame({"side": 0, "hold": 0}, index=bars.index)
    for t, side, hold in entries:
        s.iloc[t] = (side, hold)
    return s


def test_resampling_to_bars():
    idx = pd.date_range("2024-01-01", periods=30, freq="1min")
    m = pd.DataFrame({"open": np.arange(30.0), "high": np.arange(30.0) + 1, "low": np.arange(30.0) - 1,
                      "close": np.arange(30.0) + 0.5, "volume": 1.0}, index=idx)
    b = to_bars(m, 15)
    assert len(b) == 2 and b["open"].iloc[1] == 15.0 and b["close"].iloc[0] == 14.5
    assert b["high"].iloc[0] == 15.0 and b["low"].iloc[1] == 14.0 and b["volume"].iloc[0] == 15.0


def test_trade_is_entered_at_next_open_exited_at_hold_and_charged_costs():
    b = toy_bars([100, 100, 101, 102, 103, 104, 105, 106], [100, 101, 102, 103, 104, 105, 106, 107])
    tr = run_trades(b, sig(b, [(1, 1, 2)]), 15, fee_bps=5, slip_bps=1)
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t["entry_time"] == b.index[2] and t["exit_time"] == b.index[3] and t["bars"] == 2
    assert t["gross"] == pytest.approx(103 / 101 - 1)                      # open of bar 2, close of bar 3
    assert t["net"] == pytest.approx(103 / 101 - 1 - 0.0012)               # round trip = 2 x 6 bps


def test_short_gains_when_price_falls_and_signals_inside_a_trade_are_ignored():
    b = toy_bars([100, 100, 99, 98, 97, 96, 95, 94], [100, 99, 98, 97, 96, 95, 94, 93])
    tr = run_trades(b, sig(b, [(0, -1, 3), (1, 1, 1), (4, -1, 1)]), 15, 0, 0)
    assert len(tr) == 2                                                    # the signal at bar 1 falls inside the first trade
    assert tr.iloc[0]["side"] == -1 and tr.iloc[0]["gross"] == pytest.approx(-(97 / 100 - 1))
    assert tr.iloc[1]["entry_time"] == b.index[5]


def test_end_of_day_exit_and_edge_cases():
    n = 96 + 4
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    px = np.linspace(100, 110, n)
    b = pd.DataFrame({"open": px, "high": px * 1.001, "low": px * 0.999, "close": px, "volume": 1.0}, index=idx)
    tr = run_trades(b, sig(b, [(10, 1, 0)]), 15, 0, 0)
    assert tr.iloc[0]["exit_time"] == pd.Timestamp("2024-01-01 23:45")      # last bar of the UTC day
    assert run_trades(b, sig(b, [(95, 1, 0)]), 15, 0, 0).empty              # signal on the day's last bar: skipped
    gap = b.drop(b.index[20:23])
    assert run_trades(gap, sig(gap, [(18, 1, 5)]), 15, 0, 0).empty          # a hole in the data: no trade


def test_stats_known_values():
    tr = pd.DataFrame({"entry_time": pd.date_range("2024-01-01", periods=4, freq="D"),
                       "exit_time": pd.date_range("2024-01-01 06:00", periods=4, freq="D"),
                       "side": 1, "gross": [0.02, 0.01, -0.01, -0.02], "net": [0.02, 0.01, -0.01, -0.02], "bars": 4})
    st = trade_stats(tr, "2024-01-01", "2024-01-04", 15)
    assert st["n_trades"] == 4 and st["win_rate"] == 0.5 and st["expectancy_bps"] == pytest.approx(0.0)
    assert st["profit_factor"] == pytest.approx(1.0) and st["time_in_market"] == pytest.approx(4 * 15 * 4 / (4 * 1440))


def test_a_high_win_rate_is_not_enough():
    """60% winners but losses twice the gains: the win-rate gate passes, the expectancy gate must reject."""
    rng = np.random.default_rng(0)
    n = 600
    net = np.where(rng.random(n) < 0.6, 0.0005, -0.0020)
    rt = 0.0012
    tr = pd.DataFrame({"entry_time": pd.date_range("2024-01-01", periods=n, freq="8h"),
                       "exit_time": pd.date_range("2024-01-01 02:00", periods=n, freq="8h"),
                       "side": 1, "gross": net + rt, "net": net, "bars": 4})
    ev = evaluate_gates(tr, "2024-01-01", "2024-09-30", 15, Gates(), rt, 20)
    g = {x["name"]: x for x in ev["gates"]}
    assert g["win_rate"]["passed"] and g["win_rate"]["value"] > 0.55
    assert not g["expectancy_bps_net"]["passed"] and not ev["passed"]


@pytest.fixture(scope="module")
def bars_rw():
    return generate_synthetic_bars(365 * 3, seed=5, scenario="random_walk")


@pytest.mark.parametrize("strat", [OpenMomentum(4, 0.5), OpenMomentum(8, 0.0), HourReversal(2.0, 2), RangeBreakout(12, 2)],
                         ids=lambda s: f"{s.name}-{s.params}")
def test_registered_strategies_use_only_the_past(strat, bars_rw):
    check_no_lookahead_bars(strat, bars_rw, 15, n_checks=3)


def test_lookahead_detector_catches_future_information(bars_rw):
    class Peek(IntradayStrategy):
        name = "peek"

        def signals(self, bars, minutes):
            nxt = bars["close"].shift(-1) / bars["close"] - 1                 # the next bar's return
            return pd.DataFrame({"side": np.sign(nxt).fillna(0).astype(int), "hold": 1}, index=bars.index)

    with pytest.raises(LookaheadError):
        check_no_lookahead_bars(Peek(), bars_rw, 15, n_checks=3)
    peek = Peek().signals(bars_rw, 15)
    tr = run_trades(bars_rw, peek, 15, 5, 1)
    assert tr["gross"].mean() * 1e4 > 5                                       # why the check matters: a peeking rule "works"


def test_planted_effect_is_visible_to_the_momentum_rule_only(bars_rw):
    planted = generate_synthetic_bars(365 * 3, seed=5, scenario="open_momentum")
    span = ("2020-01-01", "2022-12-30")
    m1 = trade_stats(run_trades(planted, OpenMomentum(4, 0.5).signals(planted, 15), 15, 5, 1), *span, 15)
    base = trade_stats(run_trades(bars_rw, OpenMomentum(4, 0.5).signals(bars_rw, 15), 15, 5, 1), *span, 15)
    assert m1["gross_expectancy_bps"] > 50 and m1["win_rate"] > 0.55
    assert abs(base["gross_expectancy_bps"]) < 40 and base["win_rate"] < 0.52


def test_trial_registry_counts_unique_parameter_sets_forever(tmp_path):
    reg = TrialRegistry(tmp_path / "t.jsonl")
    assert reg.register("r", ["a", "b", "c"]) == 3
    assert reg.register("r", ["b", "c", "d"]) == 4                            # only 'd' is new
    assert reg.register("other", ["a"]) == 1
    assert TrialRegistry(tmp_path / "t.jsonl").register("r", []) == 4         # survives a restart
    assert len(combos_of(CFG)) == 23


def _protocol(seed, scenario, state, through="lockbox", **kw):
    bars = generate_synthetic_bars(365 * 4, seed=seed, scenario=scenario)
    return run_protocol(bars, CFG_T, "BTCUSDT", through=through, state_dir=state, check_lookahead=False, **kw)


def test_no_winner_survives_on_random_markets(tmp_path):
    t0 = time.time()
    outcomes = [_protocol(s, "random_walk", tmp_path / f"s{s}")["status"] for s in range(5)]
    assert not any(o.startswith("lockbox_passed") for o in outcomes), outcomes
    assert time.time() - t0 < 120


def test_planted_edge_goes_through_all_three_blocks(tmp_path):
    results = [_protocol(s, "open_momentum", tmp_path / f"s{s}") for s in range(4)]
    passed = [r for r in results if r["status"] == "lockbox_passed"]
    assert len(passed) >= 3, [r["status"] for r in results]
    assert all(r["winner"]["strategy"] == "open_momentum" for r in passed)
    r = passed[0]
    assert r["blocks"]["dev"][1] == "2021-12-31" and r["blocks"]["lockbox"][0] == "2023-01-01"
    assert r["lockbox"]["stats"]["win_rate"] >= 0.51 and r["lockbox"]["stats"]["expectancy_bps"] > 0


def test_the_lockbox_can_be_opened_once_only(tmp_path):
    first = _protocol(1, "open_momentum", tmp_path)
    assert first["status"] == "lockbox_passed" and (tmp_path / "lockbox_used.json").exists()
    with pytest.raises(LockboxBurned):
        _protocol(1, "open_momentum", tmp_path)
    again = _protocol(1, "open_momentum", tmp_path, allow_burned=True)
    assert again["status"].endswith("EXPLORATORY_burned") and again["lockbox"]["burned"] is True


def test_the_lockbox_stays_closed_when_nothing_passes_validation(tmp_path):
    res = _protocol(2, "random_walk", tmp_path)
    assert res["status"] in {"no_candidate_in_development", "no_winner_in_validation"}
    assert res["lockbox"] is None and not (tmp_path / "lockbox_used.json").exists()


def test_stopping_before_the_lockbox_is_the_default(tmp_path):
    res = _protocol(1, "open_momentum", tmp_path, through="validate")
    assert res["status"] == "winner_found_lockbox_not_used" and res["lockbox"] is None
    assert not (tmp_path / "lockbox_used.json").exists()


def test_more_trials_lower_the_deflated_sharpe(tmp_path):
    bars = generate_synthetic_bars(365 * 4, seed=1, scenario="open_momentum")
    few = run_protocol(bars, CFG_T, "BTCUSDT", through="dev", state_dir=tmp_path / "a", check_lookahead=False)
    reg = TrialRegistry(tmp_path / "b" / "trial_registry.jsonl")
    reg.register(CFG_T["research_id"], [f"old|{i}" for i in range(500)])      # earlier, abandoned attempts
    many = run_protocol(bars, CFG_T, "BTCUSDT", through="dev", state_dir=tmp_path / "b", check_lookahead=False)

    def dsr(res):
        row = next(r for r in res["dev"] if r["strategy"] == "open_momentum" and r["params"] == {"x_hours": 4, "min_z": 0.5})
        return next(g["value"] for g in row["gates"] if g["name"] == "deflated_sharpe_ratio")

    assert many["n_trials"] > few["n_trials"] + 400 and dsr(many) < dsr(few)


def test_report_renders_and_escapes(tmp_path):
    res = _protocol(1, "open_momentum", tmp_path)
    res["symbol"] = "<script>alert(1)</script>"
    html = render_crypto_intraday_html([res], CFG_T, True)
    for needle in ("Recherche intraday sur futures crypto", "Données synthétiques", "Le protocole", "lockbox", "<polyline"):
        assert needle in html
    assert "<script>alert" not in html and "http://" not in html


def test_cli_end_to_end_synthetic(tmp_path, capsys):
    out, js = tmp_path / "c.html", tmp_path / "c.json"
    assert main(["crypto-intraday", "--config", str(CFG_PATH), "--synthetic", "--scenario", "open_momentum", "--through", "dev",
                 "--symbols", "BTCUSDT", "--state-dir", str(tmp_path / "st"), "--out", str(out), "--json", str(js)]) == 0
    assert "BTCUSDT:" in capsys.readouterr().out
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data[0]["symbol"] == "BTCUSDT" and len(data[0]["dev"]) == 23
    assert (tmp_path / "st" / "trial_registry.jsonl").exists() and "Recherche intraday" in out.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        main(["crypto-intraday", "--config", str(CFG_PATH), "--synthetic", "--symbols", "BTC;rm", "--out", str(out)])
