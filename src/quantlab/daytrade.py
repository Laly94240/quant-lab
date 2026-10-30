"""Day trading on DAILY bars: enter at the open, exit at the close, flat overnight.

What daily bars allow: a trade is the open->close return of one session, decided BEFORE the open from
information up to the previous close (plus today's opening price for gap rules).
What they do NOT allow: stops, targets, opening-range or any intraday path logic (the order of the
day's high and low is unknown), so none is modelled. Fills at the official open are optimistic: an
extra per-side cost is charged, and gap rules (which need the opening print) are the most optimistic.
Intraday data (1-minute bars) is needed for anything finer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import (
    BacktestConfig,
    BacktestResult,
    CostModel,
    GateCriteria,
    compute_metrics,
    evaluate_candidate,
    simulate,
)
from .backtest.metrics import sharpe_per_period
from .backtest.stats import sample_var
from .backtest.walkforward import Fold, WalkForwardResult, fold_windows, param_combinations
from .data import dataframe_hash, validate_prices
from .experiments import ExperimentStore, config_hash
from .risk import RiskLimits, apply_weight_limits
from .strategies import BuyAndHold


@dataclass(frozen=True)
class OHLC:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame

    def truncate(self, k: int, hide_session: bool = True) -> OHLC:
        """History up to row k. With hide_session, row k keeps ONLY its open (what is known at 9:30)."""
        parts = {f: getattr(self, f).iloc[: k + 1].copy() for f in ("open", "high", "low", "close")}
        if hide_session:
            for f in ("high", "low", "close"):
                parts[f].iloc[-1] = np.nan
        return OHLC(**parts)


def clean_ohlc(raw: dict[str, pd.DataFrame]) -> tuple[OHLC, list[str], dict]:
    """Align on the validated close index. Bars where high/low contradict open/close are REPAIRED and
    counted (never silently); too many of them is an error."""
    close, dq = validate_prices(raw["close"])
    dq.raise_if_errors()
    warnings = [f"{i.message} {i.tickers or ''}".strip() for i in dq.warnings]
    o, h, low = (raw[f].reindex(close.index) for f in ("open", "high", "low"))
    if ((o <= 0) | (h <= 0) | (low <= 0)).any().any():
        raise ValueError("non-positive open/high/low prices")
    hi_ref, lo_ref = np.maximum(o, close), np.minimum(o, close)
    bad = (h < hi_ref * (1 - 1e-9)) | (low > lo_ref * (1 + 1e-9))
    if bad.to_numpy().mean() > 0.02:
        raise ValueError(f"{bad.to_numpy().mean():.1%} of bars have high/low inconsistent with open/close")
    if bad.any().any():
        warnings.append(f"{int(bad.to_numpy().sum())} bars with inconsistent high/low were repaired")
        h, low = np.maximum(h, hi_ref), np.minimum(low, lo_ref)
    n_missing_open = int(o.isna().to_numpy().sum())
    if n_missing_open:
        warnings.append(f"{n_missing_open} missing opens: no trade on those days")
    return OHLC(o, h, low, close), warnings, dq.to_dict()


# --------------------------------------------------------------------------- strategies
class DayStrategy:
    """signals(ohlc)[t, i] in {0, 1}: 1 = buy at the open of day t, sell at its close.

    Contract: row t may use data up to the close of day t-1 and the OPEN of day t, nothing else."""
    name = "day_strategy"

    def __init__(self, **params):
        self.params = params

    def signals(self, ohlc: OHLC) -> pd.DataFrame:
        raise NotImplementedError


def _trend_ok(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return close.shift(1) > close.rolling(lookback, min_periods=lookback).mean().shift(1)


class IntradayTrend(DayStrategy):
    """H1: the existing trend filter, but exposed to the open->close part of the day only."""
    name = "intraday_trend"

    def __init__(self, lookback: int = 200):
        super().__init__(lookback=lookback)
        self.lookback = lookback

    def signals(self, ohlc: OHLC) -> pd.DataFrame:
        return _trend_ok(ohlc.close, self.lookback).astype(float)


class GapFade(DayStrategy):
    """H2: in an uptrend, a large gap DOWN at the open is bought and sold at the close (overreaction)."""
    name = "gap_fade"

    def __init__(self, k: float = 1.0, vol_window: int = 20, trend_lookback: int = 200):
        super().__init__(k=k, vol_window=vol_window, trend_lookback=trend_lookback)
        self.k, self.vol_window, self.trend_lookback = k, vol_window, trend_lookback

    def signals(self, ohlc: OHLC) -> pd.DataFrame:
        gap = ohlc.open / ohlc.close.shift(1) - 1.0
        sigma = ohlc.close.pct_change().rolling(self.vol_window, min_periods=self.vol_window).std().shift(1)
        return ((gap <= -self.k * sigma) & _trend_ok(ohlc.close, self.trend_lookback)).astype(float)


class IBSReversal(DayStrategy):
    """H3: in an uptrend, a close near the day's low (internal bar strength) is bought at the next open."""
    name = "ibs_reversal"

    def __init__(self, theta: float = 0.2, trend_lookback: int = 200):
        super().__init__(theta=theta, trend_lookback=trend_lookback)
        self.theta, self.trend_lookback = theta, trend_lookback

    def signals(self, ohlc: OHLC) -> pd.DataFrame:
        rng = (ohlc.high - ohlc.low).replace(0.0, np.nan)
        ibs = ((ohlc.close - ohlc.low) / rng).clip(0.0, 1.0)
        return ((ibs.shift(1) < self.theta) & _trend_ok(ohlc.close, self.trend_lookback)).astype(float)


DAY_REGISTRY = {"intraday_trend": IntradayTrend, "gap_fade": GapFade, "ibs_reversal": IBSReversal}


class LookaheadError(AssertionError):
    pass


def check_no_lookahead_day(strategy: DayStrategy, ohlc: OHLC, n_checks: int = 6, seed: int = 0,
                           min_history: int = 260) -> None:
    """Recompute the signals with the high/low/close of day k erased (only its open kept) and the future
    removed: rows <= k must be identical, otherwise the rule uses information it could not have."""
    full = strategy.signals(ohlc)
    rng = np.random.default_rng(seed)
    n = len(ohlc.close)
    for k in sorted(rng.choice(np.arange(min_history, n - 1), size=n_checks, replace=False)):
        partial = strategy.signals(ohlc.truncate(int(k)))
        if not np.allclose(partial.to_numpy(), full.iloc[: k + 1].to_numpy(), atol=1e-12, equal_nan=True):
            raise LookaheadError(f"{strategy.name}: signals change when the day's high/low/close are hidden")


# --------------------------------------------------------------------------- simulation
def simulate_day(ohlc: OHLC, weights: pd.DataFrame, cfg: BacktestConfig, extra_bps: float = 0.0) -> BacktestResult:
    """Each day: buy at the open, sell at the close. Round trip = 2 x (one-way cost + extra open/close bps)."""
    r = ohlc.close / ohlc.open - 1.0
    w = weights.where(r.notna(), 0.0)
    gross = (w * r.fillna(0.0)).sum(axis=1)
    used = w.abs().sum(axis=1)
    costs = used * 2.0 * (cfg.cost.one_way_bps + extra_bps) / 1e4
    return BacktestResult(returns=gross - costs, gross_returns=gross, costs=costs, turnover=2.0 * used,
                          exposure=used, weights=w)


def day_weights(strategy: DayStrategy, ohlc: OHLC, cfg: BacktestConfig) -> pd.DataFrame:
    n = ohlc.close.shape[1]
    return apply_weight_limits(strategy.signals(ohlc) / n, cfg.risk)


def walk_forward_day(ohlc: OHLC, strategy_cls, combos: list[dict], cfg: BacktestConfig, extra_bps: float,
                     train_years: int = 5, test_years: int = 1, anchored: bool = True,
                     min_test_obs: int = 60) -> WalkForwardResult:
    idx, ppy = ohlc.close.index, cfg.periods_per_year
    targets = [day_weights(strategy_cls(**p), ohlc, cfg) for p in combos]
    runs = [simulate_day(ohlc, w, cfg, extra_bps) for w in targets]
    trial_sharpes = [sharpe_per_period(r.returns) for r in runs]
    stitched = pd.DataFrame(0.0, index=idx, columns=ohlc.close.columns)
    folds, oos_start = [], None
    for tr_s, te_s, te_e in fold_windows(idx, train_years, test_years, anchored):
        train_m, test_m = (idx >= tr_s) & (idx < te_s), (idx >= te_s) & (idx < te_e)
        if test_m.sum() < min_test_obs or train_m.sum() < ppy:
            continue
        train_sh = [sharpe_per_period(r.returns[train_m]) for r in runs]
        best = int(np.argmax([-np.inf if not np.isfinite(s) else s for s in train_sh]))
        stitched.loc[test_m] = targets[best].loc[test_m].to_numpy()
        oos_start = oos_start or idx[test_m][0]
        test_sh = sharpe_per_period(runs[best].returns[test_m])
        folds.append(Fold(str(idx[train_m][0].date()), str(idx[train_m][-1].date()), str(idx[test_m][0].date()),
                          str(idx[test_m][-1].date()), combos[best],
                          float(train_sh[best] * np.sqrt(ppy)) if np.isfinite(train_sh[best]) else None,
                          float(test_sh * np.sqrt(ppy)) if np.isfinite(test_sh) else None))
    if not folds:
        raise ValueError("no valid walk-forward fold: history too short")
    oos = simulate_day(ohlc, stitched, cfg, extra_bps).slice_from(oos_start)
    return WalkForwardResult(folds, oos_start, stitched, oos, trial_sharpes, len(combos))


# --------------------------------------------------------------------------- analysis
def decompose(ohlc: OHLC, ppy: int = 252) -> dict:
    """Where does the return come from: the session (open->close) or the night (close->open)?"""
    intra = ohlc.close / ohlc.open - 1.0
    night = ohlc.open / ohlc.close.shift(1) - 1.0
    total = ohlc.close / ohlc.close.shift(1) - 1.0
    per_asset = {s: {"intraday": float(intra[s].mean() * ppy), "overnight": float(night[s].mean() * ppy),
                     "total": float(total[s].mean() * ppy)} for s in ohlc.close.columns}
    port = {"Séance (ouverture → clôture)": intra.mean(axis=1).dropna(),
            "Nuit (clôture → ouverture)": night.mean(axis=1).dropna(),
            "Total (clôture → clôture)": total.mean(axis=1).dropna()}
    return {"per_asset": per_asset, "portfolio": port}


def trade_stats(res: BacktestResult, ohlc: OHLC, cfg: BacktestConfig, extra_bps: float) -> dict:
    w = res.weights
    r = (ohlc.close / ohlc.open - 1.0).reindex(w.index)
    mask = w.abs() > 1e-12
    n_trades = int(mask.to_numpy().sum())
    years = max(len(w) / cfg.periods_per_year, 1e-9)
    gross_bps = float((np.sign(w) * r)[mask].stack().mean() * 1e4) if n_trades else None
    rt = 2.0 * (cfg.cost.one_way_bps + extra_bps)
    return {"n_trades": n_trades, "trades_per_year": n_trades / years,
            "pct_days_in_market": float((res.exposure > 1e-12).mean()),
            "gross_edge_bps": gross_bps, "round_trip_cost_bps": rt,
            "net_edge_bps": None if gross_bps is None else gross_bps - rt,
            "win_rate": float((np.sign(w) * r)[mask].stack().gt(0).mean()) if n_trades else None}


# --------------------------------------------------------------------------- research pipeline
def run_daytrade_research(cfg: dict, ohlc: OHLC, *, is_synthetic: bool, data_source: str,
                          store: ExperimentStore | None = None, force: bool = False,
                          data_warnings: list[str] | None = None, dq: dict | None = None) -> dict:
    ppy = 252
    bt = BacktestConfig(cost=CostModel(**cfg.get("costs", {})), execution_delay=0, periods_per_year=ppy,
                        risk=RiskLimits(**cfg.get("risk", {})))
    extra = cfg.get("daytrade", {}).get("open_close_extra_bps", 2.0)
    wf_cfg, criteria = cfg.get("walk_forward", {}), GateCriteria(**cfg.get("gates", {}))
    stress_k, research_id = cfg.get("cost_stress_multiplier", 2.0), cfg["research_id"]
    data_hash = dataframe_hash(pd.concat([ohlc.open, ohlc.close], axis=1))
    data_info = {"source": data_source, "is_synthetic": is_synthetic, "hash": data_hash,
                 "start": str(ohlc.close.index[0].date()), "end": str(ohlc.close.index[-1].date()),
                 "tickers": list(ohlc.close.columns), "quality": dq or {}, "warnings": data_warnings or []}

    pending, skipped = [], []
    for spec in cfg["strategies"]:
        run_cfg = {"research_id": research_id, "kind": "daytrade", "strategy": spec, "costs": cfg.get("costs", {}),
                   "daytrade": cfg.get("daytrade", {}), "risk": cfg.get("risk", {}), "walk_forward": wf_cfg,
                   "gates": cfg.get("gates", {}), "stress": stress_k,
                   "data": {"hash": data_hash, "start": data_info["start"], "end": data_info["end"]}}
        prior = store.has_run(config_hash(run_cfg)) if store else None
        if prior and not force:
            skipped.append({"strategy": spec["name"], "reason": "identical run already stored",
                            "run_id": prior["run_id"], "verdict": prior["verdict"]})
        else:
            pending.append((spec, run_cfg))

    wfs = []
    for spec, _ in pending:
        cls = DAY_REGISTRY[spec["name"]]
        combos = param_combinations(spec.get("grid", {}), spec.get("fixed", {}))
        for p in combos:
            check_no_lookahead_day(cls(**p), ohlc)
        wfs.append((combos, walk_forward_day(ohlc, cls, combos, bt, extra, wf_cfg.get("train_years", 5),
                                             wf_cfg.get("test_years", 1), wf_cfg.get("anchored", True))))

    n_trials_total = (store.cumulative_trials(research_id) if store else 0) + sum(w.n_trials for _, w in wfs)
    # Dispersion of Sharpe ratios WITHIN each hypothesis family (its parameter sets), averaged. Pooling all
    # families would inflate the variance with the cost-drag gap between families (e.g. a rule that trades
    # every day versus one that trades rarely) and over-penalise good candidates.
    var_trials = float(np.mean([sample_var(w.trial_sharpes) for _, w in wfs])) if wfs else 0.0
    reports, series = [], {}
    for (spec, run_cfg), (_, wf) in zip(pending, wfs, strict=True):
        bench_w = BuyAndHold(rebalance="M").target_weights(ohlc.close)
        bench_w.loc[ohlc.close.index < wf.oos_start] = 0.0
        bench = simulate(ohlc.close, bench_w, BacktestConfig(cost=bt.cost, execution_delay=1, periods_per_year=ppy,
                                                             risk=bt.risk)).slice_from(wf.oos_start)
        stress_cfg = BacktestConfig(cost=bt.cost.scaled(stress_k), execution_delay=0, periods_per_year=ppy,
                                    risk=bt.risk)
        stressed = simulate_day(ohlc, wf.stitched_target, stress_cfg, extra * stress_k).slice_from(wf.oos_start)
        report = evaluate_candidate(wf.oos, bench, stressed, criteria, n_trials_total, var_trials, ppy=ppy)
        always_w = apply_weight_limits(pd.DataFrame(1.0 / ohlc.close.shape[1], index=ohlc.close.index,
                                                    columns=ohlc.close.columns), bt.risk)
        always_w.loc[ohlc.close.index < wf.oos_start] = 0.0
        always = simulate_day(ohlc, always_w, bt, extra).slice_from(wf.oos_start)
        n5 = 5 * ppy
        report.update(
            strategy=spec["name"], kind="daytrade", is_synthetic=is_synthetic, oos_start=str(wf.oos_start.date()),
            folds=[f.__dict__ for f in wf.folds], n_trials_this_strategy=wf.n_trials,
            trade_stats=trade_stats(wf.oos, ohlc, bt, extra),
            last_5y={"years": min(len(wf.oos.returns), n5) / ppy,
                     "strategy": compute_metrics(wf.oos.returns.iloc[-n5:], None, None,
                                                 wf.oos.exposure.iloc[-n5:], 0.0, ppy),
                     "benchmark": compute_metrics(bench.returns.iloc[-n5:], None, None,
                                                  bench.exposure.iloc[-n5:], 0.0, ppy)})
        report["run_id"] = (store.save(research_id=research_id, config=run_cfg, report=report,
                                       oos_returns=wf.oos.returns, data_info=data_info, n_trials=wf.n_trials,
                                       benchmark_returns=bench.returns) if store else None)
        reports.append(report)
        series[spec["name"]] = {"oos": wf.oos.returns, "bench": bench.returns, "always": always.returns,
                                "weights": wf.oos.weights}
    return {"data": data_info, "reports": reports, "skipped": skipped, "n_trials_total": n_trials_total,
            "decomposition": decompose(ohlc, ppy), "series": series, "params": {"extra_bps": extra}}


def format_daytrade_summary(result: dict) -> str:
    d = result["data"]
    lines = []
    if d["is_synthetic"]:
        lines += ["=" * 78, "SYNTHETIC DATA: pipeline validation only. Numbers below have NO financial meaning.",
                  "=" * 78]
    lines.append(f"data: {d['source']} | {d['start']} -> {d['end']} | {len(d['tickers'])} assets | hash {d['hash']}")
    lines += [f"  [warning] {w}" for w in d.get("warnings", [])]
    lines.append(f"hypotheses tested so far under this research_id: {result['n_trials_total']}")
    pa = result["decomposition"]["per_asset"]
    lines += ["", "Average annual return by part of the day (arithmetic):",
              f"{'asset':<8}{'session':>9}{'night':>9}{'total':>9}"]
    lines += [f"{s:<8}{v['intraday']:>9.1%}{v['overnight']:>9.1%}{v['total']:>9.1%}" for s, v in pa.items()]
    f = lambda x, p=False: "n/a" if x is None else (f"{x:.1%}" if p else f"{x:.2f}")   # noqa: E731
    h = (f"{'strategy':<15}{'Sharpe':>7}{'CAGR':>8}{'MaxDD':>8}{'inMkt':>7}{'tr/yr':>7}{'edge bp':>9}{'cost bp':>9}"
         f"{'DSR':>6}  verdict")
    lines += ["", h, "-" * len(h)]
    for r in result["reports"]:
        m, t, g = r["oos_metrics"], r["trade_stats"], {x["name"]: x for x in r["gates"]}
        lines.append(f"{r['strategy']:<15}{f(m.get('sharpe')):>7}{f(m.get('cagr'), True):>8}"
                     f"{f(m.get('max_drawdown'), True):>8}{f(t['pct_days_in_market'], True):>7}"
                     f"{t['trades_per_year']:>7.0f}{f(t['gross_edge_bps']):>9}"
                     f"{t['round_trip_cost_bps']:>9.1f}{f(g['deflated_sharpe_ratio']['value']):>6}  {r['verdict']}"
                     + (f"  (failed: {', '.join(r['failed_gates'])})" if r["failed_gates"] else ""))
    if result["reports"]:
        b = result["reports"][0]["benchmark_metrics"]
        lines.append(f"{'benchmark':<15}{f(b.get('sharpe')):>7}{f(b.get('cagr'), True):>8}"
                     f"{f(b.get('max_drawdown'), True):>8}")
    lines += [f"skipped {s['strategy']}: {s['reason']} ({s['run_id']}, {s['verdict']})" for s in result["skipped"]]
    lines += ["", "edge bp = average GROSS return per trade; cost bp = round-trip cost. A strategy needs edge > cost.",
              "PASS_TO_PAPER = eligible for paper trading only; never a profitability claim."]
    return "\n".join(lines)
