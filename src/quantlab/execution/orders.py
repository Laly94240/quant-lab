"""Turn target weights into proposed orders. Pure functions, no broker, no network."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..risk import RiskLimits, apply_weight_limits


@dataclass(frozen=True)
class Order:
    client_order_id: str
    symbol: str
    side: str            # "BUY" | "SELL"
    quantity: int
    order_type: str = "MKT"
    limit_price: float | None = None


def make_client_order_id(strategy: str, asof, symbol: str, side: str, quantity: int) -> str:
    """Deterministic id: re-generating the same decision yields the same id, so a retry or a
    double run cannot create a second order (idempotency / duplicate-order prevention)."""
    raw = f"{strategy}|{asof}|{symbol}|{side}|{quantity}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def generate_orders(
    target_weights: pd.Series,
    positions: pd.Series,
    prices: pd.Series,
    equity: float,
    asof,
    strategy: str,
    limits: RiskLimits | None = None,
    min_trade_value: float = 0.0,
) -> list[Order]:
    limits = limits or RiskLimits()
    w = apply_weight_limits(target_weights, limits)
    pos = positions.reindex(w.index).fillna(0).astype(int)
    px = prices.reindex(w.index)
    target_shares = np.fix(w * equity / px).astype(int)       # round toward zero: never overspend
    delta = target_shares - pos
    orders = []
    for sym, d in delta.items():
        if d == 0 or abs(d) * px[sym] < min_trade_value:
            continue
        side = "BUY" if d > 0 else "SELL"
        qty = int(abs(d))
        orders.append(Order(make_client_order_id(strategy, asof, sym, side, qty), sym, side, qty))
    orders.sort(key=lambda o: (o.side != "SELL", o.symbol))    # sells first: free cash before buying
    return orders
