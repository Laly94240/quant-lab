import numpy as np
import pandas as pd
import pytest
from conftest import make_prices

from quantlab.backtest import (
    BacktestConfig,
    CostModel,
    compute_metrics,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    run_backtest,
    simulate,
)
from quantlab.risk import RiskLimits
from quantlab.strategies import Strategy

ZERO = CostModel(0, 0, 0)


def test_hand_computed_turnover_and_return_with_drift():
    p = make_prices({"A": [100, 110, 110], "B": [100, 100, 100]})
    t = pd.DataFrame(0.5, index=p.index, columns=p.columns)
    res = simulate(p, t, BacktestConfig(cost=ZERO, execution_delay=0))
    assert res.turnover.round(4).tolist() == [0.0, 1.0, 0.0476]   # buy-in, then rebalance of drift
    assert res.returns.round(4).tolist() == [0.0, 0.05, 0.0]


def test_costs_are_charged_exactly_on_turnover():
    p = make_prices({"A": [100.0] * 5})                     # flat price: only costs move equity
    t = pd.DataFrame(1.0, index=p.index, columns=p.columns)
    cfg = BacktestConfig(cost=CostModel(10, 0, 0), execution_delay=0,
                         risk=RiskLimits(max_position_weight=1.0))
    res = simulate(p, t, cfg)
    assert res.turnover.sum() == pytest.approx(1.0)
    assert res.costs.sum() == pytest.approx(0.0010)
    assert res.returns.sum() == pytest.approx(-0.0010)


def test_cost_model_components():
    c = CostModel(commission_bps=2, spread_bps=3, slippage_bps=2)
    assert c.one_way_bps == pytest.approx(5.5) and c.scaled(2).one_way_bps == pytest.approx(11.0)


def test_execution_delay_shifts_positions_by_one_more_bar():
    p = make_prices({"A": [100, 101, 102, 103, 104, 105]})
    t = pd.DataFrame(0.0, index=p.index, columns=p.columns)
    t.iloc[1] = 1.0                                         # decided at close of bar 1
    cfg = lambda d: BacktestConfig(cost=ZERO, execution_delay=d, risk=RiskLimits(max_position_weight=1.0))
    w0 = simulate(p, t, cfg(0)).weights["A"]
    w1 = simulate(p, t, cfg(1)).weights["A"]
    assert w0.iloc[2] == 1.0 and w0.iloc[1] == 0.0          # earns r[2] onwards
    assert w1.iloc[3] == 1.0 and w1.iloc[2] == 0.0          # one bar later


def test_same_bar_information_cannot_be_traded_on_the_same_bar(rw_prices):
    """A rule that buys after an up day (known at the close) must not earn that same day."""
    class AfterUpDay(Strategy):
        name = "after_up"
        def target_weights(self, prices):
            return (prices.pct_change() > 0).astype(float).fillna(0.0) / prices.shape[1]
    res = run_backtest(rw_prices, AfterUpDay(), BacktestConfig(cost=ZERO, execution_delay=0))
    t = (rw_prices.pct_change() > 0).astype(float) / rw_prices.shape[1]
    r = rw_prices.pct_change().fillna(0)
    leaky = (t * r).sum(axis=1)                              # what a look-ahead engine would book
    assert leaky.mean() > 0.0004 and abs(res.returns.mean()) < leaky.mean() / 3


def test_risk_limits_are_enforced_inside_the_backtest():
    p = make_prices({"A": np.linspace(100, 120, 30), "B": np.linspace(100, 110, 30)})
    class AllIn(Strategy):
        name = "allin"
        def target_weights(self, prices):
            return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)  # 200% gross
    res = run_backtest(p, AllIn(), BacktestConfig(risk=RiskLimits(max_position_weight=0.6)))
    assert res.exposure.max() <= 1.0 + 1e-9 and res.weights.max().max() <= 0.6 + 1e-9


def test_metrics_known_values():
    r = pd.Series([0.10, -0.50, 0.20, 0.0])
    m = compute_metrics(r)
    assert m["max_drawdown"] == pytest.approx(-0.5)
    assert m["total_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)

    daily = pd.Series(np.tile([0.01, -0.005], 126))             # 252 obs
    m = compute_metrics(daily)
    expected_sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(252)
    assert m["sharpe"] == pytest.approx(expected_sharpe)
    assert m["cagr"] == pytest.approx((1 + daily).prod() - 1)    # exactly one year
    down = np.sqrt(np.mean(np.minimum(daily, 0) ** 2))
    assert m["sortino"] == pytest.approx(daily.mean() / down * np.sqrt(252))
    assert m["daily_win_rate"] == pytest.approx(0.5)
    assert m["avg_win_over_avg_loss"] == pytest.approx(2.0)


def test_metrics_handles_degenerate_series():
    m = compute_metrics(pd.Series([0.0] * 50))
    assert m["sharpe"] is None and m["max_drawdown"] == 0.0


def test_psr_and_dsr_properties():
    assert probabilistic_sharpe_ratio(0.1, 0.1, 500) == pytest.approx(0.5)
    assert probabilistic_sharpe_ratio(0.1, 0.0, 1000) > probabilistic_sharpe_ratio(0.1, 0.0, 200)
    e = [expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
    assert e == sorted(e) and e[0] > 0
    d = [deflated_sharpe_ratio(0.08, 1500, n, 0.0004) for n in (1, 10, 100, 1000)]
    assert d == sorted(d, reverse=True)                      # more trials => less credit
    assert deflated_sharpe_ratio(0.08, 1500, 1, 0.0004) == pytest.approx(
        probabilistic_sharpe_ratio(0.08, 0.0, 1500))
    assert probabilistic_sharpe_ratio(0.1, 0.0, 1000, skew=-1.0, kurt=8.0) < \
        probabilistic_sharpe_ratio(0.1, 0.0, 1000)           # negative skew / fat tails penalised
