"""End-to-end research pipeline: validate data -> check look-ahead -> walk-forward -> gates -> save."""
from __future__ import annotations

import pandas as pd

from .backtest import (
    BacktestConfig,
    CostModel,
    GateCriteria,
    evaluate_candidate,
    simulate,
    walk_forward,
)
from .backtest.stats import sample_var
from .backtest.walkforward import param_combinations
from .data import dataframe_hash, validate_prices
from .experiments import ExperimentStore, config_hash
from .risk import RiskLimits
from .strategies import REGISTRY, check_no_lookahead


def build_backtest_config(cfg: dict) -> BacktestConfig:
    bt = cfg.get("backtest", {})
    return BacktestConfig(
        cost=CostModel(**cfg.get("costs", {})),
        execution_delay=bt.get("execution_delay", 1),
        cash_rate=bt.get("cash_rate", 0.0),
        borrow_rate=bt.get("borrow_rate", 0.0),
        risk=RiskLimits(**cfg.get("risk", {})),
    )


def run_research(
    cfg: dict,
    prices: pd.DataFrame,
    *,
    is_synthetic: bool,
    data_source: str,
    store: ExperimentStore | None = None,
    force: bool = False,
) -> dict:
    prices, dq = validate_prices(prices)
    dq.raise_if_errors()
    data_hash = dataframe_hash(prices)
    data_info = {
        "source": data_source, "is_synthetic": is_synthetic, "hash": data_hash,
        "start": dq.start, "end": dq.end, "tickers": list(prices.columns),
        "quality": dq.to_dict(),
    }

    bt = build_backtest_config(cfg)
    wf_cfg = cfg.get("walk_forward", {})
    criteria = GateCriteria(**cfg.get("gates", {}))
    stress_k = cfg.get("cost_stress_multiplier", 2.0)
    rf = cfg.get("backtest", {}).get("rf_annual", 0.0)
    research_id = cfg["research_id"]
    bench_spec = cfg.get("benchmark", {"name": "equal_weight_bh", "params": {"rebalance": "M"}})

    # 1) decide what actually runs (identical past experiments are not repeated)
    pending, skipped = [], []
    for spec in cfg["strategies"]:
        run_cfg = {
            "research_id": research_id, "strategy": spec, "costs": cfg.get("costs", {}),
            "backtest": cfg.get("backtest", {}), "risk": cfg.get("risk", {}), "walk_forward": wf_cfg,
            "benchmark": bench_spec, "gates": cfg.get("gates", {}), "stress": stress_k,
            "data": {"hash": data_hash, "start": dq.start, "end": dq.end},
        }
        h = config_hash(run_cfg)
        prior = store.has_run(h) if store else None
        if prior and not force:
            skipped.append({"strategy": spec["name"], "reason": "identical run already stored",
                            "run_id": prior["run_id"], "verdict": prior["verdict"]})
        else:
            pending.append((spec, run_cfg))

    # 2) run the walk-forward for each family (with the look-ahead check as a hard precondition)
    wf_results = []
    for spec, _ in pending:
        cls = REGISTRY[spec["name"]]
        combos = param_combinations(spec.get("grid", {}), spec.get("fixed", {}))
        for p in combos:
            check_no_lookahead(cls(**p), prices)
        wf_results.append(walk_forward(
            prices, cls, combos, bt,
            train_years=wf_cfg.get("train_years", 5), test_years=wf_cfg.get("test_years", 1),
            anchored=wf_cfg.get("anchored", True),
        ))

    # 3) multiple-testing bookkeeping: every parameter set ever tried under this research_id counts
    new_trials = sum(w.n_trials for w in wf_results)
    prior_trials = store.cumulative_trials(research_id) if store else 0
    n_trials_total = prior_trials + new_trials
    var_trials = sample_var([s for w in wf_results for s in w.trial_sharpes])

    # 4) evaluate against the benchmark, with a cost stress test, then save
    reports = []
    for (spec, run_cfg), wf in zip(pending, wf_results, strict=True):
        bench_w = REGISTRY[bench_spec["name"]](**bench_spec.get("params", {})).target_weights(prices)
        bench_w.loc[prices.index < wf.oos_start] = 0.0     # benchmark starts with the OOS window
        bench = simulate(prices, bench_w, bt).slice_from(wf.oos_start)
        stress_cfg = BacktestConfig(cost=bt.cost.scaled(stress_k), execution_delay=bt.execution_delay,
                                    cash_rate=bt.cash_rate, borrow_rate=bt.borrow_rate, risk=bt.risk)
        stressed = simulate(prices, wf.stitched_target, stress_cfg).slice_from(wf.oos_start)

        report = evaluate_candidate(
            wf.oos, bench, stressed, criteria, n_trials_total, var_trials,
            ppy=bt.periods_per_year, rf_annual=rf,
        )
        report.update(
            strategy=spec["name"], is_synthetic=is_synthetic,
            oos_start=str(wf.oos_start.date()), folds=[f.__dict__ for f in wf.folds],
            n_trials_this_strategy=wf.n_trials,
        )
        run_id = None
        if store:
            run_id = store.save(research_id=research_id, config=run_cfg, report=report,
                                oos_returns=wf.oos.returns, data_info=data_info,
                                n_trials=wf.n_trials)
        report["run_id"] = run_id
        reports.append(report)

    return {"data": data_info, "reports": reports, "skipped": skipped,
            "n_trials_total": n_trials_total}


def format_summary(result: dict) -> str:
    lines = []
    d = result["data"]
    if d["is_synthetic"]:
        lines += ["=" * 78,
                  "SYNTHETIC DATA: pipeline validation only. Numbers below have NO financial meaning.",
                  "=" * 78]
    lines.append(f"data: {d['source']} | {d['start']} -> {d['end']} | {len(d['tickers'])} assets | "
                 f"hash {d['hash']}")
    w = result and result["data"]["quality"]["issues"]
    for i in w:
        lines.append(f"  [{i['severity']}] {i['code']}: {i['message']} {i['tickers'] or ''}")
    lines.append(f"hypotheses tested so far under this research_id: {result['n_trials_total']}")
    lines.append("")
    hdr = f"{'strategy':<14}{'OOS Sharpe':>11}{'CAGR':>8}{'MaxDD':>8}{'Turn/yr':>9}{'DSR':>7}{'vs bench':>10}  verdict"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    def f(x, pct=False, nd=2):
        return "n/a" if x is None else (f"{x:.1%}" if pct else f"{x:.{nd}f}")

    for r in result["reports"]:
        m, g = r["oos_metrics"], {x["name"]: x for x in r["gates"]}
        lines.append(
            f"{r['strategy']:<14}{f(m.get('sharpe')):>11}{f(m.get('cagr'), True):>8}"
            f"{f(m.get('max_drawdown'), True):>8}{f(m.get('annual_turnover'), nd=1):>9}"
            f"{f(g['deflated_sharpe_ratio']['value']):>7}{f(g['psr_vs_benchmark']['value']):>10}  "
            f"{r['verdict']}" + (f"  (failed: {', '.join(r['failed_gates'])})" if r["failed_gates"] else "")
        )
    if result["reports"]:
        b = result["reports"][0]["benchmark_metrics"]
        lines.append(f"{'benchmark':<14}{f(b.get('sharpe')):>11}{f(b.get('cagr'), True):>8}"
                     f"{f(b.get('max_drawdown'), True):>8}{f(b.get('annual_turnover'), nd=1):>9}")
    for s in result["skipped"]:
        lines.append(f"skipped {s['strategy']}: {s['reason']} ({s['run_id']}, {s['verdict']})")
    lines += ["", "PASS_TO_PAPER = eligible for paper trading only. It is not a profitability claim "
              "and never authorises live trading."]
    return "\n".join(lines)
