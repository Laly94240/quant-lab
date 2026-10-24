"""Decorator that notifies (e.g. on Discord) every order event of any BrokerAdapter.

Events: order submitted, filled, rejected, cancelled. A status is notified once; replaying the
same order (duplicate submission) or restoring a persisted broker does not re-notify.
"""
from __future__ import annotations

from .broker import BrokerAdapter, OrderState, OrderStatus
from .orders import Order

_TITLES = {
    OrderStatus.SUBMITTED: ("Ordre soumis", "info"),
    OrderStatus.FILLED: ("Ordre exécuté", "success"),
    OrderStatus.REJECTED: ("Ordre rejeté", "error"),
    OrderStatus.CANCELLED: ("Ordre annulé", "warning"),
}


class NotifyingBroker(BrokerAdapter):
    def __init__(self, inner: BrokerAdapter, notifier, mode: str = "PAPER"):
        self.inner, self.notifier, self.mode = inner, notifier, mode.upper()
        # orders already known to the inner broker (e.g. restored state) are not re-announced
        self._seen: dict[str, OrderStatus] = {
            oid: st.status for oid, st in getattr(inner, "orders", {}).items()
        }

    def _notify(self, st: OrderState) -> None:
        label, level = _TITLES[st.status]
        o = st.order
        banner = "SIMULATION (paper)" if self.mode == "PAPER" else "ORDRE RÉEL"
        fields = [("Mode", self.mode), ("Symbole", o.symbol), ("Sens", o.side), ("Quantité", str(o.quantity)),
                  ("Type", o.order_type), ("Id", o.client_order_id)]
        if st.avg_price is not None:
            fields.append(("Prix moyen", f"{st.avg_price:.4f}"))
        if st.reason:
            fields.append(("Raison", st.reason))
        self.notifier.send(f"[{self.mode}] {label} : {o.side} {o.quantity} {o.symbol}", banner,
                           fields=fields, level=level)

    def submit_order(self, order: Order) -> OrderState:
        is_new = order.client_order_id not in self._seen
        st = self.inner.submit_order(order)
        if is_new:
            self._seen[order.client_order_id] = st.status
            self._notify(st)
        return st

    def cancel_order(self, client_order_id: str) -> OrderState:
        st = self.inner.cancel_order(client_order_id)
        self._sync_one(client_order_id, st)
        return st

    def get_order(self, client_order_id: str) -> OrderState:
        st = self.inner.get_order(client_order_id)
        self._sync_one(client_order_id, st)
        return st

    def _sync_one(self, oid: str, st: OrderState) -> None:
        if self._seen.get(oid) != st.status:
            self._seen[oid] = st.status
            self._notify(st)

    def poll(self) -> int:
        """Check every tracked order for a status change (call after fills are processed)."""
        n = 0
        for oid in list(self._seen):
            st = self.inner.get_order(oid)
            if self._seen[oid] != st.status:
                self._seen[oid] = st.status
                self._notify(st)
                n += 1
        return n

    def get_positions(self) -> dict[str, int]:
        return self.inner.get_positions()

    def get_cash(self) -> float:
        return self.inner.get_cash()
