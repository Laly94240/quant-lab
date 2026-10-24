import pandas as pd
import pytest

from quantlab.backtest import CostModel
from quantlab.execution import (
    LiveTradingDisabled,
    LiveTradingSettings,
    Order,
    OrderStatus,
    PaperBroker,
    assert_live_allowed,
    generate_orders,
    make_client_order_id,
    reconcile,
    require_approval,
)
from quantlab.risk import (
    RiskLimits,
    apply_weight_limits,
    check_data_freshness,
    check_halt,
    check_order_count,
)


def test_weight_limits_cap_positions_and_gross():
    lim = RiskLimits(max_position_weight=0.4, max_gross_exposure=1.0)
    w = pd.DataFrame({"A": [0.9, 0.3], "B": [0.5, 0.3], "C": [0.5, 0.3]})
    out = apply_weight_limits(w, lim)
    assert out.max().max() <= 0.4 + 1e-12 and out.sum(axis=1).max() <= 1.0 + 1e-12
    assert out.iloc[1].tolist() == pytest.approx([0.3, 0.3, 0.3])            # untouched when compliant


def test_shorts_are_clipped_unless_allowed():
    w = pd.Series({"A": -0.2, "B": 0.2})
    assert apply_weight_limits(w, RiskLimits())["A"] == 0.0
    assert apply_weight_limits(w, RiskLimits(allow_short=True))["A"] == -0.2


def test_halt_on_daily_loss_and_drawdown():
    lim = RiskLimits(max_daily_loss=0.03, max_drawdown=0.2)
    assert check_halt(pd.Series([100, 101, 102]), lim) is None
    assert "daily loss" in check_halt(pd.Series([100, 101, 97]), lim)
    slow_slide = pd.Series([100, 120, 117, 114, 111, 108, 105, 102, 99, 96, 95])   # each day > -3%
    assert "drawdown" in check_halt(slow_slide, lim)


def test_stale_data_and_order_count_guards():
    lim = RiskLimits(max_data_age_bdays=3, max_orders_per_day=5)
    assert check_data_freshness(pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-22"), lim) is None
    assert "stale" in check_data_freshness(pd.Timestamp("2026-10-12"), pd.Timestamp("2026-10-22"), lim)
    assert check_order_count(5, lim) is None and "exceed" in check_order_count(6, lim)


def test_orders_round_toward_zero_sell_first_and_respect_limits():
    w = pd.Series({"A": 0.9, "B": 0.1, "C": 0.0})        # 0.9 is above the 0.35 cap
    pos = pd.Series({"C": 40})
    px = pd.Series({"A": 100.0, "B": 33.0, "C": 50.0})
    orders = generate_orders(w, pos, px, 10_000, "2026-10-23", "s")
    assert [o.symbol for o in orders] == ["C", "A", "B"] and orders[0].side == "SELL"
    by = {o.symbol: o for o in orders}
    assert by["A"].quantity == 35 and by["B"].quantity == 30 and by["C"].quantity == 40
    assert sum(o.quantity * px[o.symbol] for o in orders if o.side == "BUY") <= 10_000


def test_client_order_id_is_deterministic():
    a = make_client_order_id("s", "2026-10-23", "A", "BUY", 5)
    assert a == make_client_order_id("s", "2026-10-23", "A", "BUY", 5)
    assert a != make_client_order_id("s", "2026-10-24", "A", "BUY", 5)


def test_paper_broker_never_assumes_a_fill_and_blocks_duplicates():
    b = PaperBroker(10_000, CostModel(0, 0, 0))
    o = Order("id1", "A", "BUY", 10)
    assert b.submit_order(o).status == OrderStatus.SUBMITTED
    assert b.get_positions() == {} and b.get_cash() == 10_000               # nothing before confirmation
    b.submit_order(o)                                                       # duplicate submission
    assert len(b.orders) == 1
    b.process_fills({"A": 100.0})
    assert b.get_order("id1").status == OrderStatus.FILLED
    assert b.get_positions() == {"A": 10} and b.get_cash() == 9_000
    b.process_fills({"A": 100.0})                                           # replay does not refill
    assert b.get_positions() == {"A": 10}


def test_paper_broker_rejections_and_cancel():
    b = PaperBroker(500, CostModel(0, 0, 0))
    b.submit_order(Order("big", "A", "BUY", 10))
    b.submit_order(Order("short", "B", "SELL", 1))
    b.submit_order(Order("nopx", "C", "BUY", 1))
    b.process_fills({"A": 100.0, "B": 10.0})
    assert b.get_order("big").status == OrderStatus.REJECTED
    assert b.get_order("short").status == OrderStatus.REJECTED
    assert b.get_order("nopx").status == OrderStatus.SUBMITTED              # no price => still pending
    assert b.cancel_order("nopx").status == OrderStatus.CANCELLED


def test_reconcile_reports_every_mismatch():
    assert reconcile({"A": 10, "B": 5}, {"A": 10, "B": 5}) == {}
    assert reconcile({"A": 10, "B": 5}, {"A": 8, "C": 2}) == {"A": -2, "B": -5, "C": 2}


def test_live_trading_is_off_by_default_and_needs_two_independent_confirmations():
    on = LiveTradingSettings(enabled=True)
    with pytest.raises(LiveTradingDisabled):
        assert_live_allowed(LiveTradingSettings(), env={})
    with pytest.raises(LiveTradingDisabled):
        assert_live_allowed(LiveTradingSettings(), env={"QUANTLAB_LIVE_CONFIRM": "YES_I_AUTHORIZE_REAL_ORDERS"})
    with pytest.raises(LiveTradingDisabled):
        assert_live_allowed(on, env={})
    with pytest.raises(LiveTradingDisabled):
        assert_live_allowed(on, env={"QUANTLAB_LIVE_CONFIRM": "yes"})
    assert_live_allowed(on, env={"QUANTLAB_LIVE_CONFIRM": "YES_I_AUTHORIZE_REAL_ORDERS"})  # only both


def test_supervised_stage_only_lets_approved_orders_through():
    orders = [Order("a", "A", "BUY", 1), Order("b", "B", "BUY", 1)]
    assert require_approval(orders, lambda os: []) == []
    assert require_approval(orders, lambda os: ["b"]) == [orders[1]]
