"""Vectorised daily backtest engine with explicit timing and transaction costs.

Timing convention (the single most important thing to get right):
  * target[t] is decided with information up to the close of bar t.
  * With execution_delay = d, the position held during bar u (earning r[u] = close[u]/close[u-1]-1)
    is target[u - 1 - d].  d = 0 -> trade at the signal close (market-on-close, optimistic);
    d = 1 (default) -> trade at the NEXT close (conservative).
  * Costs are charged on turnover, measured against the weights as they have DRIFTED with
    market moves since the last rebalance (not against the previous target).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..risk import RiskLimits, apply_weight_limits


@dataclass(frozen=True)
class CostModel:
    """One-way proportional costs in basis points of traded notional (assumptions to calibrate
    against real broker statements; see docs/DECISIONS.md)."""
    commission_bps: float = 2.0
    spread_bps: float = 3.0      # full quoted spread; half of it is paid per one-way trade
    slippage_bps: float = 2.0

    @property
    def one_way_bps(self) -> float:
        return self.commission_bps + self.spread_bps / 2.0 + self.slippage_bps

    def scaled(self, k: float) -> CostModel:
        return CostModel(self.commission_bps * k, self.spread_bps * k, self.slippage_bps * k)


@dataclass(frozen=True)
class BacktestConfig:
    cost: CostModel = field(default_factory=CostModel)
    execution_delay: int = 1
    cash_rate: float = 0.0       # annual rate earned on uninvested cash
    borrow_rate: float = 0.0     # annual rate paid on negative cash (leverage)
    periods_per_year: int = 252
    risk: RiskLimits = field(default_factory=RiskLimits)


@dataclass
class BacktestResult:
    returns: pd.Series           # net of costs
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    exposure: pd.Series          # gross exposure held during the bar
    weights: pd.DataFrame        # weights held during the bar

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()

    def slice_from(self, start: pd.Timestamp) -> BacktestResult:
        m = self.returns.index >= start
        return BacktestResult(
            self.returns[m], self.gross_returns[m], self.costs[m], self.turnover[m],
            self.exposure[m], self.weights[m],
        )


def simulate(prices: pd.DataFrame, target: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    if not target.index.equals(prices.index):
        raise ValueError("target weights must be indexed like prices")
    r = prices.pct_change().fillna(0.0)
    pos = target.shift(1 + cfg.execution_delay).fillna(0.0)

    cash = 1.0 - pos.sum(axis=1)
    cash_ret = (
        cfg.cash_rate / cfg.periods_per_year * cash.clip(lower=0.0)
        - cfg.borrow_rate / cfg.periods_per_year * (-cash).clip(lower=0.0)
    )
    gross_ret = (pos * r).sum(axis=1) + cash_ret

    # weights at the end of each bar, after market drift and before any rebalancing
    drifted_end = pos.mul(1.0 + r).div(1.0 + gross_ret, axis=0)
    drifted_prev = drifted_end.shift(1).fillna(0.0)
    turnover = (pos - drifted_prev).abs().sum(axis=1)
    costs = turnover * cfg.cost.one_way_bps / 1e4

    return BacktestResult(
        returns=gross_ret - costs,
        gross_returns=gross_ret,
        costs=costs,
        turnover=turnover,
        exposure=pos.abs().sum(axis=1),
        weights=pos,
    )


def run_backtest(prices: pd.DataFrame, strategy, cfg: BacktestConfig | None = None) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    target = apply_weight_limits(strategy.target_weights(prices), cfg.risk)
    return simulate(prices, target, cfg)
