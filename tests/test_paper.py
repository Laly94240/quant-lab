import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from quantlab.data import generate_synthetic_prices
from quantlab.news import Briefing, RiskFlag
from quantlab.paper import run_paper_day

CFG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "etf_trend.yaml").read_text())


class Rec:
    def __init__(self):
        self.calls = []

    def send(self, title, body="", *, fields=None, level="info"):
        self.calls.append((title, level))
        return True


@pytest.fixture
def setup(tmp_path):
    cfg = deepcopy(CFG)
    cfg["paper"]["state_file"] = str(tmp_path / "state.json")
    cfg["paper"]["strategy"]["params"] = {"lookback": 50, "rebalance": "D", "weighting": "equal"}
    prices = generate_synthetic_prices(CFG["universe"], n_days=600, seed=2, scenario="regime")
    return cfg, prices, tmp_path


def _now(prices, k=0):
    return prices.index[-1] + pd.Timedelta(days=1 + k)


def test_first_run_submits_orders_and_announces_them(setup):
    cfg, prices, tmp = setup
    rec = Rec()
    s = run_paper_day(cfg, prices, rec, now=_now(prices))
    assert s["status"] == "ok" and s["n_orders_submitted"] > 0
    assert sum("Ordre soumis" in t for t, _ in rec.calls) == s["n_orders_submitted"]
    assert all(t.startswith("[PAPER]") for t, _ in rec.calls)
    state = json.loads((tmp / "state.json").read_text())
    assert state["last_bar"] == str(prices.index[-1].date()) and state["halted"] is None


def test_same_day_rerun_is_a_noop(setup):
    cfg, prices, _ = setup
    rec = Rec()
    run_paper_day(cfg, prices, rec, now=_now(prices))
    n = len(rec.calls)
    assert run_paper_day(cfg, prices, rec, now=_now(prices))["status"] == "already_ran"
    assert len(rec.calls) == n


def test_orders_fill_at_the_next_run_and_fills_are_announced(setup):
    cfg, prices, _ = setup
    rec = Rec()
    run_paper_day(cfg, prices.iloc[:-1], rec, now=prices.index[-2] + pd.Timedelta(days=1))
    submitted = sum("Ordre soumis" in t for t, _ in rec.calls)
    s2 = run_paper_day(cfg, prices, rec, now=_now(prices))
    assert submitted > 0 and sum("Ordre exécuté" in t for t, _ in rec.calls) >= 1
    assert s2["positions"]                                           # positions exist only after the fills
    assert s2["equity"] > 0.9 * cfg["paper"]["initial_cash"]


def test_stale_data_blocks_trading_and_alerts(setup):
    cfg, prices, tmp = setup
    rec = Rec()
    s = run_paper_day(cfg, prices, rec, now=prices.index[-1] + pd.Timedelta(days=30))
    assert s["status"] == "aborted" and "stale" in s["reason"]
    assert rec.calls == [("[PAPER] Données périmées : aucun ordre", "error")]
    assert not (tmp / "state.json").exists()                          # nothing was traded or persisted


def test_bad_data_blocks_trading(setup):
    cfg, prices, _ = setup
    bad = prices.copy()
    bad.iloc[-5, 0] = -1.0
    rec = Rec()
    assert run_paper_day(cfg, bad, rec, now=_now(prices))["status"] == "aborted"


def test_halt_on_loss_stops_new_orders_until_a_human_clears_it(setup):
    cfg, prices, tmp = setup
    run_paper_day(cfg, prices.iloc[:-1], Rec(), now=prices.index[-2] + pd.Timedelta(days=1))
    state_path = tmp / "state.json"
    state = json.loads(state_path.read_text())
    state["equity_history"] = [{"date": "2020-01-01", "equity": 20_000.0}]    # simulate a 50% drawdown
    state_path.write_text(json.dumps(state))
    rec = Rec()
    s = run_paper_day(cfg, prices, rec, now=_now(prices))
    assert s["halted"] and s["n_orders_submitted"] == 0
    assert any("ARRÊT DE SÉCURITÉ" in t for t, _ in rec.calls)
    later = generate_synthetic_prices(CFG["universe"], n_days=601, seed=2, scenario="regime")
    s2 = run_paper_day(cfg, later, Rec(), now=_now(later))
    assert s2["halted"] and s2["n_orders_submitted"] == 0           # still halted: needs a human


def test_news_overlay_only_reduces_what_the_paper_runner_buys(setup):
    cfg, prices, _ = setup
    flag = Briefing("risque", [RiskFlag("tarifs", "high", list(CFG["universe"]), [1])], [], created_utc=pd.Timestamp.now(tz="UTC").isoformat())
    cfg_on = deepcopy(cfg)
    cfg_on["news"]["overlay"]["enabled"] = True
    base = run_paper_day(deepcopy(cfg), prices, Rec(), now=_now(prices))
    cfg_on["paper"]["state_file"] = str(setup[2] / "state_on.json")
    cut = run_paper_day(cfg_on, prices, Rec(), now=_now(prices), briefing=flag)
    assert cut["n_orders_submitted"] <= base["n_orders_submitted"]
