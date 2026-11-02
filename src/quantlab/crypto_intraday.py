"""Intraday research on crypto perpetual futures, with a protocol that cannot be gamed by iteration.

Three blocks of history, used in this order and never mixed:
  development : free exploration, but every parameter set tried is counted (trial registry);
  validation  : only the finalists of development are evaluated, the best one that passes is the winner;
  lockbox     : the last year. The single winner is evaluated ONCE. A second look is refused
                (and any result obtained after a refusal is labelled burned / exploratory).
A strategy is a hypothesis written in advance; an entry is decided at the close of a bar and executed at the
OPEN of the next bar; costs are charged on every trade; the win rate is a reported gate, never the objective.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from .backtest.metrics import max_drawdown, sharpe_per_period
from .backtest.stats import deflated_sharpe_ratio
from .backtest.walkforward import param_combinations

PPY = 365


# ----------------------------------------------------------------------------- data
def to_bars(df1m: pd.DataFrame, minutes: int = 15) -> pd.DataFrame:
    agg = df1m.resample(f"{minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return agg.dropna(subset=["open", "close"])


def generate_synthetic_bars(n_days: int = 365 * 4, seed: int = 0, scenario: str = "random_walk",
                            start: str = "2020-01-01", minutes: int = 15, annual_vol: float = 0.6,
                            beta: float = 1.0, x_hours: int = 4) -> pd.DataFrame:
    """Synthetic 24/7 bars (TESTING ONLY). 'open_momentum' plants an edge: the first x hours of each UTC day
    predict the rest of the day (coefficient beta)."""
    bpd = 24 * 60 // minutes
    rng = np.random.default_rng(seed)
    vol_bar = annual_vol / np.sqrt(PPY * bpd)
    ret = rng.standard_normal((n_days, bpd)) * vol_bar
    if scenario == "open_momentum":
        xb = x_hours * 60 // minutes
        ret[:, xb:] += (beta * ret[:, :xb].sum(axis=1) / (bpd - xb))[:, None]
    elif scenario != "random_walk":
        raise ValueError(f"unknown scenario {scenario!r}")
    close = 100.0 * np.exp(np.cumsum(ret.ravel()))
    open_ = np.r_[100.0, close[:-1]]
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.standard_normal(close.size)) * vol_bar * 0.5)
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.standard_normal(close.size)) * vol_bar * 0.5)
    idx = pd.date_range(start, periods=n_days * bpd, freq=f"{minutes}min")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": 1.0}, index=idx)


# ----------------------------------------------------------------------------- strategies
class IntradayStrategy:
    """signals(bars, minutes) -> DataFrame[side in {-1,0,1}, hold]. Row t is decided at the CLOSE of bar t
    (past data only) and executed at the OPEN of bar t+1. hold = number of bars kept, 0 = until the end of the UTC day."""
    name = "intraday"

    def __init__(self, **params):
        self.params = params

    def signals(self, bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
        raise NotImplementedError


def _frame(bars: pd.DataFrame, side, hold) -> pd.DataFrame:
    return pd.DataFrame({"side": np.asarray(side, dtype=int), "hold": np.asarray(hold, dtype=int)}, index=bars.index)


class OpenMomentum(IntradayStrategy):
    """M1: the move of the first x hours of the UTC day continues until the end of the day (if large enough)."""
    name = "open_momentum"

    def __init__(self, x_hours: int = 4, min_z: float = 0.5):
        super().__init__(x_hours=x_hours, min_z=min_z)
        self.x_hours, self.min_z = x_hours, min_z

    def signals(self, bars, minutes):
        idx, day = bars.index, bars.index.normalize()
        decision = (idx - day) == pd.Timedelta(hours=self.x_hours) - pd.Timedelta(minutes=minutes)
        r = bars["close"] / bars["open"].groupby(day).transform("first") - 1.0
        rd = r[decision]
        z = (rd / rd.rolling(30, min_periods=20).std().shift(1)).reindex(idx)
        sel = decision & (z.abs() >= self.min_z).fillna(False).to_numpy()
        return _frame(bars, np.sign(r).where(sel, 0).fillna(0), np.zeros(len(bars)))


class HourReversal(IntradayStrategy):
    """M2: a hourly move more than k standard deviations is faded for n hours."""
    name = "hour_reversal"

    def __init__(self, k: float = 2.0, n_hours: int = 2):
        super().__init__(k=k, n_hours=n_hours)
        self.k, self.n_hours = k, n_hours

    def signals(self, bars, minutes):
        bph = 60 // minutes
        r1 = bars["close"] / bars["close"].shift(bph) - 1.0
        z = r1 / r1.rolling(7 * 24 * bph, min_periods=3 * 24 * bph).std().shift(1)
        side = (-np.sign(r1)).where(z.abs() >= self.k, 0).fillna(0)
        return _frame(bars, side, np.full(len(bars), self.n_hours * bph))


class RangeBreakout(IntradayStrategy):
    """M3: a close beyond the previous n-hour range is followed for h hours."""
    name = "range_breakout"

    def __init__(self, n_hours: int = 12, h_hours: int = 2):
        super().__init__(n_hours=n_hours, h_hours=h_hours)
        self.n_hours, self.h_hours = n_hours, h_hours

    def signals(self, bars, minutes):
        bph = 60 // minutes
        n = self.n_hours * bph
        hi = bars["high"].rolling(n, min_periods=n).max().shift(1)
        lo = bars["low"].rolling(n, min_periods=n).min().shift(1)
        side = np.where(bars["close"] > hi, 1, np.where(bars["close"] < lo, -1, 0))
        return _frame(bars, side, np.full(len(bars), self.h_hours * bph))


REGISTRY = {"open_momentum": OpenMomentum, "hour_reversal": HourReversal, "range_breakout": RangeBreakout}


class LookaheadError(AssertionError):
    pass


def check_no_lookahead_bars(strategy: IntradayStrategy, bars: pd.DataFrame, minutes: int, n_checks: int = 4,
                            seed: int = 0, min_history: int = 3000) -> None:
    """Signals up to bar k must not change when every bar after k is removed."""
    full = strategy.signals(bars, minutes)
    rng = np.random.default_rng(seed)
    for k in sorted(rng.choice(np.arange(min_history, len(bars) - 1), size=n_checks, replace=False)):
        part = strategy.signals(bars.iloc[: k + 1], minutes)
        if not np.array_equal(part.to_numpy(), full.iloc[: k + 1].to_numpy()):
            raise LookaheadError(f"{strategy.name}: signals change when the future is removed")


# ----------------------------------------------------------------------------- engine
def run_trades(bars: pd.DataFrame, sig: pd.DataFrame, minutes: int, fee_bps: float, slip_bps: float) -> pd.DataFrame:
    """Non-overlapping trades. Entry: open of the bar after the signal. Exit: close of bar entry+hold-1, or the
    last bar of the UTC day when hold == 0. Each trade uses the whole equity at 1x (no leverage)."""
    o, c, idx = bars["open"].to_numpy(), bars["close"].to_numpy(), bars.index
    n = len(bars)
    last_of_day = pd.Series(np.arange(n), index=idx).groupby(idx.normalize()).transform("max").to_numpy()
    side, hold = sig["side"].to_numpy(), sig["hold"].to_numpy()
    step = pd.Timedelta(minutes=minutes)
    rt = 2.0 * (fee_bps + slip_bps) / 1e4
    rows, busy = [], -1
    for t in np.flatnonzero(side != 0):
        if t <= busy or t + 1 >= n:
            continue
        e = t + 1
        x = last_of_day[t] if hold[t] == 0 else e + hold[t] - 1
        if x < e or x >= n or idx[x] - idx[e] != (x - e) * step:       # signal on the day's last bar, or a data gap
            continue
        gross = side[t] * (c[x] / o[e] - 1.0)
        rows.append((idx[e], idx[x], int(side[t]), gross, gross - rt, x - e + 1))
        busy = x
    return pd.DataFrame(rows, columns=["entry_time", "exit_time", "side", "gross", "net", "bars"])


def block_slice(trades: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    m = (trades["entry_time"] >= pd.Timestamp(start)) & (trades["entry_time"] < pd.Timestamp(end) + pd.Timedelta(days=1))
    return trades[m]


def daily_returns(trades: pd.DataFrame, start: str, end: str) -> pd.Series:
    days = pd.date_range(start, end, freq="D")
    if trades.empty:
        return pd.Series(0.0, index=days)
    s = trades.groupby(trades["exit_time"].dt.normalize())["net"].sum()
    return s.reindex(days, fill_value=0.0)


def trade_stats(trades: pd.DataFrame, start: str, end: str, minutes: int) -> dict:
    d = daily_returns(trades, start, end)
    n = len(trades)
    out = {"n_trades": n, "n_days": len(d)}
    if n == 0:
        return out | {"win_rate": None, "expectancy_bps": None, "gross_expectancy_bps": None, "profit_factor": None,
                      "sharpe": None, "total_return": 0.0, "max_drawdown": 0.0, "time_in_market": 0.0}
    net, wins, losses = trades["net"], trades.loc[trades["net"] > 0, "net"], trades.loc[trades["net"] <= 0, "net"]
    sd = d.std(ddof=1)
    out.update(win_rate=float((net > 0).mean()), expectancy_bps=float(net.mean() * 1e4),
               gross_expectancy_bps=float(trades["gross"].mean() * 1e4),
               profit_factor=float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else None,
               avg_win_bps=float(wins.mean() * 1e4) if len(wins) else None,
               avg_loss_bps=float(losses.mean() * 1e4) if len(losses) else None,
               sharpe=float(d.mean() / sd * np.sqrt(PPY)) if sd > 0 else None,
               total_return=float((1 + d).prod() - 1), max_drawdown=max_drawdown(d),
               time_in_market=float(trades["bars"].sum() * minutes / (len(d) * 1440)))
    return out


@dataclass(frozen=True)
class Gates:
    min_trades_per_year: int = 100
    min_win_rate: float = 0.51
    min_expectancy_bps: float = 0.0
    min_profit_factor: float = 1.10
    min_dsr: float = 0.95
    max_drawdown: float = 0.30
    min_stress_expectancy_bps: float = 0.0


def evaluate_gates(trades: pd.DataFrame, start: str, end: str, minutes: int, g: Gates, rt_cost: float,
                   n_trials: int) -> dict:
    """Deflated Sharpe: the dispersion of Sharpe ratios under 'no skill' is the sampling variance of one Sharpe
    estimate, about 1 / (number of days). It does NOT use the observed dispersion across parameter sets, which
    would be inflated precisely when a real effect exists (and shrink the credit of the true winner)."""
    st = trade_stats(trades, start, end, minutes)
    d = daily_returns(trades, start, end)
    stress = float((trades["gross"] - 2 * rt_cost).mean() * 1e4) if len(trades) else None
    sr_pp = sharpe_per_period(d) if len(d) > 2 else float("nan")
    dsr = None
    if len(trades) and np.isfinite(sr_pp):
        v = deflated_sharpe_ratio(sr_pp, len(d), n_trials, 1.0 / len(d), float(sps.skew(d)),
                                  float(sps.kurtosis(d, fisher=False)))
        dsr = float(v) if np.isfinite(v) else None
    nz = lambda x: -np.inf if x is None else x   # noqa: E731
    gates = [
        dict(name="n_trades", value=st["n_trades"], threshold=round(g.min_trades_per_year * len(d) / 365),
             passed=st["n_trades"] >= g.min_trades_per_year * len(d) / 365),
        dict(name="win_rate", value=st.get("win_rate"), threshold=g.min_win_rate, passed=nz(st.get("win_rate")) >= g.min_win_rate),
        dict(name="expectancy_bps_net", value=st.get("expectancy_bps"), threshold=g.min_expectancy_bps,
             passed=nz(st.get("expectancy_bps")) > g.min_expectancy_bps),
        dict(name="profit_factor", value=st.get("profit_factor"), threshold=g.min_profit_factor,
             passed=nz(st.get("profit_factor")) >= g.min_profit_factor),
        dict(name="deflated_sharpe_ratio", value=dsr, threshold=g.min_dsr, passed=nz(dsr) >= g.min_dsr),
        dict(name="max_drawdown", value=st.get("max_drawdown"), threshold=-g.max_drawdown,
             passed=st.get("max_drawdown", -1.0) >= -g.max_drawdown and st["n_trades"] > 0),
        dict(name="cost_stress_expectancy_bps", value=stress, threshold=g.min_stress_expectancy_bps,
             passed=nz(stress) > g.min_stress_expectancy_bps),
    ]
    return {"stats": st, "gates": gates, "passed": all(x["passed"] for x in gates),
            "failed": [x["name"] for x in gates if not x["passed"]]}


# ----------------------------------------------------------------------------- registry and lockbox
class TrialRegistry:
    """Every parameter set ever evaluated counts, forever (unique keys per research_id)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _keys(self, research_id: str) -> set[str]:
        if not self.path.exists():
            return set()
        rows = [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]
        return {r["key"] for r in rows if r["research_id"] == research_id}

    def register(self, research_id: str, keys: list[str]) -> int:
        known = self._keys(research_id)
        new = [k for k in dict.fromkeys(keys) if k not in known]
        if new:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                for k in new:
                    f.write(json.dumps({"research_id": research_id, "key": k,
                                        "date": datetime.now(timezone.utc).isoformat()}) + "\n")
        return len(known) + len(new)


class LockboxBurned(RuntimeError):
    pass


class Lockbox:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def used(self, research_id: str, symbol: str) -> dict | None:
        return self._read().get(f"{research_id}|{symbol}")

    def record(self, research_id: str, symbol: str, info: dict) -> None:
        data = self._read()
        data[f"{research_id}|{symbol}"] = info | {"date": datetime.now(timezone.utc).isoformat()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------- protocol
def combos_of(cfg: dict) -> list[tuple[str, dict]]:
    out = []
    for spec in cfg["strategies"]:
        out += [(spec["name"], p) for p in param_combinations(spec.get("grid", {}), spec.get("fixed", {}))]
    return out


def run_protocol(bars: pd.DataFrame, cfg: dict, symbol: str, *, through: str = "validate", state_dir: str | Path = "runs",
                 allow_burned: bool = False, all_symbols: list[str] | None = None, check_lookahead: bool = True) -> dict:
    minutes, rid = cfg.get("bar_minutes", 15), cfg["research_id"]
    b = cfg["blocks"]
    first = str(bars.index[0].date())
    last = str(bars.index[-1].date())
    dev = (first, b["dev_end"])
    val = (str((pd.Timestamp(b["dev_end"]) + pd.Timedelta(days=1)).date()), b["validation_end"])
    box = (str((pd.Timestamp(b["validation_end"]) + pd.Timedelta(days=1)).date()), b.get("lockbox_end") or last)
    fee, slip = cfg["costs"]["fee_bps"], cfg["costs"]["slippage_bps"]
    rt = 2.0 * (fee + slip) / 1e4
    gates, k = Gates(**cfg.get("gates", {})), cfg.get("finalists", 3)
    registry = TrialRegistry(Path(state_dir) / "trial_registry.jsonl")

    combos = combos_of(cfg)
    all_trades = {}
    for name, p in combos:
        strat = REGISTRY[name](**p)
        if check_lookahead:
            check_no_lookahead_bars(strat, bars, minutes)
        all_trades[(name, json.dumps(p, sort_keys=True))] = run_trades(bars, strat.signals(bars, minutes), minutes, fee, slip)

    symbols_run = all_symbols or [symbol]
    keys = [f"{sy}|{n}|{json.dumps(p, sort_keys=True)}" for sy in symbols_run for n, p in combos]
    rows = list(all_trades)
    # every parameter set of every symbol ever tried counts, plus the finalist selection of each symbol
    n_trials = registry.register(rid, keys) + k * len(symbols_run)

    dev_table = []
    for name, pj in rows:
        ev = evaluate_gates(block_slice(all_trades[(name, pj)], *dev), *dev, minutes, gates, rt, n_trials)
        dev_table.append({"strategy": name, "params": json.loads(pj)} | ev)
    passing = [r for r in dev_table if r["passed"]]
    passing.sort(key=lambda r: r["stats"].get("sharpe") or -np.inf, reverse=True)
    finalists = passing[:k]
    result = {"symbol": symbol, "blocks": {"dev": dev, "validation": val, "lockbox": box}, "n_trials": n_trials,
              "dev": dev_table, "finalists": [], "winner": None, "lockbox": None, "status": "no_candidate_in_development",
              "costs_round_trip_bps": rt * 1e4}
    if not finalists or through == "dev":
        result["status"] = "no_candidate_in_development" if not finalists else "dev_only"
        return result

    registry.register(rid, [f"validation|{symbol}|{r['strategy']}|{json.dumps(r['params'], sort_keys=True)}" for r in finalists])
    val_table = []
    for r in finalists:
        tr = all_trades[(r["strategy"], json.dumps(r["params"], sort_keys=True))]
        val_table.append({"strategy": r["strategy"], "params": r["params"],
                          **evaluate_gates(block_slice(tr, *val), *val, minutes, gates, rt, n_trials)})
    result["finalists"] = val_table
    ok = [r for r in val_table if r["passed"]]
    ok.sort(key=lambda r: r["stats"].get("sharpe") or -np.inf, reverse=True)
    if not ok:
        result["status"] = "no_winner_in_validation"
        return result
    win = ok[0]
    result["winner"] = {"strategy": win["strategy"], "params": win["params"]}
    result["status"] = "winner_found_lockbox_not_used"
    tr = all_trades[(win["strategy"], json.dumps(win["params"], sort_keys=True))]
    result["_trades"] = {"dev": block_slice(tr, *dev), "validation": block_slice(tr, *val)}
    if through != "lockbox":
        return result

    lock = Lockbox(Path(state_dir) / "lockbox_used.json")
    prior = lock.used(rid, symbol)
    if prior and not allow_burned:
        raise LockboxBurned(f"the lockbox for {symbol} was already used on {prior['date']}: a second look is refused. "
                            "Any new result on it is exploratory (--allow-burned).")
    result["lockbox"] = evaluate_gates(block_slice(tr, *box), *box, minutes, gates, rt, n_trials)
    result["lockbox"]["burned"] = bool(prior)
    result["status"] = "lockbox_passed" if result["lockbox"]["passed"] else "lockbox_failed"
    if result["lockbox"]["burned"]:
        result["status"] += "_EXPLORATORY_burned"
    else:
        lock.record(rid, symbol, {"strategy": win["strategy"], "params": win["params"], "status": result["status"]})
    result["_trades"] = {"dev": block_slice(tr, *dev), "validation": block_slice(tr, *val), "lockbox": block_slice(tr, *box)}
    return result
