"""Broker abstraction, a paper broker, and the safety rails around real trading.

A fill is NEVER assumed: `submit_order` only returns SUBMITTED. The status becomes FILLED only
after the broker confirms it (for the paper broker: `process_fills`). The caller must poll the
status and reconcile positions against what the strategy expected.

No real-broker adapter exists in this prototype. Live trading is off unless BOTH the config
flag and an explicit environment confirmation are set (see `assert_live_allowed`).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum

from ..backtest.engine import CostModel
from .orders import Order

LIVE_CONFIRM_VALUE = "YES_I_AUTHORIZE_REAL_ORDERS"


class OrderStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class OrderState:
    order: Order
    status: OrderStatus
    filled_qty: int = 0
    avg_price: float | None = None
    reason: str | None = None


class BrokerAdapter(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> OrderState: ...

    @abstractmethod
    def cancel_order(self, client_order_id: str) -> OrderState: ...

    @abstractmethod
    def get_order(self, client_order_id: str) -> OrderState: ...

    @abstractmethod
    def get_positions(self) -> dict[str, int]: ...

    @abstractmethod
    def get_cash(self) -> float: ...


class PaperBroker(BrokerAdapter):
    """In-memory simulated broker. Fills happen only when `process_fills(prices)` is called."""

    def __init__(self, cash: float, cost: CostModel | None = None):
        self.cash = float(cash)
        self.cost = cost or CostModel()
        self.positions: dict[str, int] = {}
        self.orders: dict[str, OrderState] = {}

    def submit_order(self, order: Order) -> OrderState:
        if order.client_order_id in self.orders:        # duplicate-order prevention
            return self.orders[order.client_order_id]
        if order.quantity <= 0:
            st = OrderState(order, OrderStatus.REJECTED, reason="non-positive quantity")
        else:
            st = OrderState(order, OrderStatus.SUBMITTED)
        self.orders[order.client_order_id] = st
        return st

    def cancel_order(self, client_order_id: str) -> OrderState:
        st = self.orders[client_order_id]
        if st.status == OrderStatus.SUBMITTED:
            st.status = OrderStatus.CANCELLED
        return st

    def get_order(self, client_order_id: str) -> OrderState:
        return self.orders[client_order_id]

    def get_positions(self) -> dict[str, int]:
        return {k: v for k, v in self.positions.items() if v != 0}

    def get_cash(self) -> float:
        return self.cash

    def process_fills(self, prices: Mapping[str, float]) -> None:
        slip = self.cost.one_way_bps / 1e4
        for st in self.orders.values():
            if st.status != OrderStatus.SUBMITTED:
                continue
            o = st.order
            if o.symbol not in prices:
                continue                                  # no price => no fill, stays SUBMITTED
            px = prices[o.symbol] * (1 + slip if o.side == "BUY" else 1 - slip)
            if o.side == "BUY":
                if px * o.quantity > self.cash:
                    st.status, st.reason = OrderStatus.REJECTED, "insufficient cash"
                    continue
                self.cash -= px * o.quantity
                self.positions[o.symbol] = self.positions.get(o.symbol, 0) + o.quantity
            else:
                held = self.positions.get(o.symbol, 0)
                if o.quantity > held:
                    st.status, st.reason = OrderStatus.REJECTED, "would create a short position"
                    continue
                self.cash += px * o.quantity
                self.positions[o.symbol] = held - o.quantity
            st.status, st.filled_qty, st.avg_price = OrderStatus.FILLED, o.quantity, px


def reconcile(expected: Mapping[str, int], actual: Mapping[str, int]) -> dict[str, int]:
    """Return {symbol: actual - expected} for every mismatch. Empty dict = reconciled."""
    diffs = {}
    for sym in set(expected) | set(actual):
        d = int(actual.get(sym, 0)) - int(expected.get(sym, 0))
        if d != 0:
            diffs[sym] = d
    return diffs


class LiveTradingDisabled(RuntimeError):
    pass


@dataclass(frozen=True)
class LiveTradingSettings:
    enabled: bool = False       # default OFF; set to true in the config only at Stage 4


def assert_live_allowed(settings: LiveTradingSettings, env: Mapping[str, str] | None = None) -> None:
    """Raise unless real-money trading was explicitly enabled twice (config AND environment)."""
    env = os.environ if env is None else env
    if not settings.enabled:
        raise LiveTradingDisabled("live trading is disabled in the configuration (default).")
    if env.get("QUANTLAB_LIVE_CONFIRM") != LIVE_CONFIRM_VALUE:
        raise LiveTradingDisabled("missing explicit environment confirmation for live trading.")


def require_approval(
    orders: list[Order], approver: Callable[[list[Order]], list[str]]
) -> list[Order]:
    """Stage 3 (supervised): only orders whose client_order_id the human approver returns pass."""
    approved_ids = set(approver(orders))
    return [o for o in orders if o.client_order_id in approved_ids]
