"""Pre-registered acceptance gates. A candidate that fails any gate is REJECTED; the answer to a
rejection is a new hypothesis, not a bigger parameter grid.

Passing the gates only means 'worth paper trading' (Stage 2). It is never a statement that the
strategy is profitable, and it never authorises live trading.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .engine import BacktestResult
from .metrics import compute_metrics, sharpe_per_period
from .stats import deflated_sharpe_ratio, probabilistic_sharpe_ratio


@dataclass(frozen=True)
class GateCriteria:
    min_oos_years: float = 5.0
    min_oos_sharpe: float = 0.4
    min_dsr: float = 0.95                 # probability true Sharpe beats the best-of-N null
    max_drawdown: float = 0.35            # absolute, on the OOS period
    min_psr_vs_benchmark: float = 0.95    # P(true Sharpe > benchmark's observed Sharpe)
    min_cost_stress_sharpe: float = 0.0   # OOS Sharpe must stay above this with costs x stress


def evaluate_candidate(
    oos: BacktestResult,
    benchmark: BacktestResult,
    stressed: BacktestResult,
    criteria: GateCriteria,
    n_trials_total: int,
    var_sr_trials: float,
    ppy: int = 252,
    rf_annual: float = 0.0,
) -> dict:
    m = compute_metrics(oos.returns, oos.turnover, oos.costs, oos.exposure, rf_annual, ppy)
    bm = compute_metrics(benchmark.returns, benchmark.turnover, benchmark.costs,
                         benchmark.exposure, rf_annual, ppy)
    sm = compute_metrics(stressed.returns, rf_annual=rf_annual, periods_per_year=ppy)

    sr_pp = sharpe_per_period(oos.returns, rf_annual / ppy)
    dsr = deflated_sharpe_ratio(
        sr_pp, len(oos.returns), n_trials_total, var_sr_trials,
        skew=m.get("skew") or 0.0, kurt=m.get("kurtosis") or 3.0,
    )
    dsr = None if dsr is None or not np.isfinite(dsr) else float(dsr)

    bm_sr_pp = sharpe_per_period(benchmark.returns, rf_annual / ppy)
    psr_bm = probabilistic_sharpe_ratio(
        sr_pp, bm_sr_pp, len(oos.returns),
        skew=m.get("skew") or 0.0, kurt=m.get("kurtosis") or 3.0,
    )
    psr_bm = None if psr_bm is None or not np.isfinite(psr_bm) else float(psr_bm)

    def nz(x, default=-np.inf):
        return default if x is None else x

    gates = [
        dict(name="oos_length_years", value=len(oos.returns) / ppy,
             threshold=criteria.min_oos_years, passed=len(oos.returns) / ppy >= criteria.min_oos_years),
        dict(name="oos_sharpe", value=m.get("sharpe"), threshold=criteria.min_oos_sharpe,
             passed=nz(m.get("sharpe")) >= criteria.min_oos_sharpe),
        dict(name="deflated_sharpe_ratio", value=dsr, threshold=criteria.min_dsr,
             passed=nz(dsr) >= criteria.min_dsr),
        dict(name="oos_max_drawdown", value=m.get("max_drawdown"), threshold=-criteria.max_drawdown,
             passed=nz(m.get("max_drawdown")) >= -criteria.max_drawdown),
        dict(name="cost_stress_sharpe", value=sm.get("sharpe"), threshold=criteria.min_cost_stress_sharpe,
             passed=nz(sm.get("sharpe")) > criteria.min_cost_stress_sharpe),
    ]
    gates.append(dict(name="psr_vs_benchmark", value=psr_bm, threshold=criteria.min_psr_vs_benchmark,
                      passed=nz(psr_bm) >= criteria.min_psr_vs_benchmark))
    failed = [g["name"] for g in gates if not g["passed"]]
    return dict(
        verdict="REJECT" if failed else "PASS_TO_PAPER",
        failed_gates=failed,
        gates=gates,
        oos_metrics=m,
        benchmark_metrics=bm,
        stress_metrics=sm,
        n_trials_total=n_trials_total,
        criteria=asdict(criteria),
    )
