"""Daily paper-trading runner (Stage 2). Simulated money only; no real broker is involved.

Timing mirrors the backtest (execution_delay = 1): orders decided at the close of day D are
filled at the close of the NEXT trading day, i.e. at the start of the following run. A run is
idempotent: re-running on the same last bar does nothing (and order ids are deterministic).

Every order event is announced through the notifier (Discord). Safety stops (stale data, bad
data, daily-loss / drawdown halt) block new orders and are announced too; a halt stays on until
a human clears it.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backtest import CostModel
from .data import validate_prices
from .execution import NotifyingBroker, PaperBroker, generate_orders
from .news.models import Briefing
from .news.overlay import OverlayConfig, apply_event_overlay
from .risk import RiskLimits, check_data_freshness, check_halt, check_order_count
from .strategies import REGISTRY


def load_state(path: Path, initial_cash: float) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"broker": PaperBroker(initial_cash).to_dict(), "equity_history": [], "last_bar": None,
            "halted": None, "initial_cash": initial_cash,
            "created_utc": datetime.now(timezone.utc).isoformat()}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, path)                                   # atomic: never a half-written state


def mark_to_market(broker: PaperBroker, prices: pd.Series) -> float:
    return broker.get_cash() + sum(q * float(prices[s]) for s, q in broker.get_positions().items())


def run_paper_day(cfg: dict, prices: pd.DataFrame, notifier, *, now: pd.Timestamp | None = None,
                  briefing: Briefing | None = None, clear_halt: bool = False) -> dict:
    pcfg = cfg["paper"]
    now = now or pd.Timestamp.now()
    limits = RiskLimits(**cfg.get("risk", {}))
    cost = CostModel(**cfg.get("costs", {}))
    state_path = Path(pcfg.get("state_file", "paper/state.json"))
    state = load_state(state_path, pcfg.get("initial_cash", 10_000.0))
    if clear_halt:
        state["halted"] = None

    prices, dq = validate_prices(prices)
    if not dq.ok:
        msg = "; ".join(f"[{i.code}] {i.message}" for i in dq.errors)
        notifier.send("[PAPER] Données invalides : aucun ordre", msg, level="error")
        return {"status": "aborted", "reason": f"data quality: {msg}"}
    stale = check_data_freshness(prices.index[-1], now, limits)
    if stale:
        notifier.send("[PAPER] Données périmées : aucun ordre", stale, level="error")
        return {"status": "aborted", "reason": stale}

    last_bar = str(prices.index[-1].date())
    if state["last_bar"] == last_bar:
        return {"status": "already_ran", "last_bar": last_bar}

    broker = PaperBroker.from_dict(state["broker"], cost)
    nb = NotifyingBroker(broker, notifier, "PAPER")
    last_px = prices.iloc[-1]

    # 1) fill what was submitted at the previous run, at today's close (delay = 1 bar)
    broker.process_fills({s: float(p) for s, p in last_px.items()})
    n_events = nb.poll()

    # 2) mark to market, update history, check the kill switch
    equity = mark_to_market(broker, last_px)
    state["equity_history"].append({"date": last_bar, "equity": equity})
    series = pd.Series([e["equity"] for e in state["equity_history"]])
    halt = check_halt(series, limits)
    if halt and not state["halted"]:
        state["halted"] = halt
        notifier.send("[PAPER] ARRÊT DE SÉCURITÉ : plus aucun ordre", halt, level="error")

    submitted = []
    if not state["halted"]:
        # 3) decide: target weights at today's close (+ optional, reduce-only news overlay)
        spec = pcfg["strategy"]
        target = REGISTRY[spec["name"]](**spec.get("params", {})).target_weights(prices).iloc[-1]
        overlay = OverlayConfig(**cfg.get("news", {}).get("overlay", {}))
        target = apply_event_overlay(target, briefing, overlay)

        positions = pd.Series(broker.get_positions(), dtype=float).reindex(prices.columns).fillna(0)
        sizing_equity = equity * (1.0 - pcfg.get("cash_buffer", 0.01))
        orders = generate_orders(target, positions, last_px, sizing_equity, last_bar, spec["name"],
                                 limits, pcfg.get("min_trade_value", 0.0))
        too_many = check_order_count(len(orders), limits)
        if too_many:
            notifier.send("[PAPER] Trop d'ordres : aucun envoyé", too_many, level="error")
            orders = []
        for o in orders:
            nb.submit_order(o)
            submitted.append(o.client_order_id)

    summary = {"status": "ok", "last_bar": last_bar, "equity": equity, "n_fills_or_events": n_events,
               "n_orders_submitted": len(submitted), "positions": broker.get_positions(),
               "halted": state["halted"]}
    if pcfg.get("notify_summary", True) and (submitted or n_events):
        pos = ", ".join(f"{s}:{q}" for s, q in sorted(broker.get_positions().items())) or "aucune"
        notifier.send(f"[PAPER] Bilan du {last_bar}", "Simulation, argent fictif.", level="info",
                      fields=[("Valeur", f"{equity:,.2f}"), ("Cash", f"{broker.get_cash():,.2f}"),
                              ("Positions", pos), ("Ordres soumis", str(len(submitted)))])

    state["broker"] = broker.to_dict()
    state["last_bar"] = last_bar
    save_state(state_path, state)
    return summary
