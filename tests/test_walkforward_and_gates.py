from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from quantlab.backtest import BacktestConfig, walk_forward
from quantlab.backtest.walkforward import fold_windows, param_combinations
from quantlab.data import generate_synthetic_prices
from quantlab.pipeline import run_research
from quantlab.strategies import TrendFollowing

CFG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "etf_trend.yaml").read_text())
TICKERS = CFG["universe"]
COMBOS = param_combinations({"lookback": [50, 100, 150, 200]}, {"rebalance": "M"})


def test_param_combinations():
    assert len(param_combinations({"a": [1, 2], "b": [3, 4, 5]}, {"c": 0})) == 6
    assert param_combinations({}, {"c": 0}) == [{"c": 0}]


def test_fold_windows_never_overlap_and_test_follows_train(rw_prices):
    wins = list(fold_windows(rw_prices.index, 5, 1, anchored=True))
    for (tr_s, te_s, te_e), nxt in zip(wins, wins[1:] + [None]):
        assert tr_s < te_s < te_e
        if nxt:
            assert nxt[1] == te_e                         # tests tile the timeline, no overlap


def test_walk_forward_is_not_affected_by_future_data(rw_prices):
    """Truncating the history after fold k must leave folds <= k (choices and OOS returns) intact."""
    cfg = BacktestConfig()
    full = walk_forward(rw_prices, TrendFollowing, COMBOS, cfg)
    cut = pd.Timestamp(full.folds[3].test_end) + pd.Timedelta(days=1)
    part = walk_forward(rw_prices.loc[:cut], TrendFollowing, COMBOS, cfg)
    for a, b in zip(full.folds[:4], part.folds[:4]):
        assert a.params == b.params and a.test_start == b.test_start
    end = pd.Timestamp(full.folds[3].test_end)
    pd.testing.assert_series_equal(
        full.oos.returns.loc[:end], part.oos.returns.loc[:end], check_exact=False, rtol=1e-12)


def test_parameters_are_selected_on_train_data_only(rw_prices):
    """Rewriting the TEST window of fold 0 with garbage must not change fold 0's parameters."""
    cfg = BacktestConfig()
    base = walk_forward(rw_prices, TrendFollowing, COMBOS, cfg)
    f0 = base.folds[0]
    mutated = rw_prices.copy()
    m = (mutated.index >= f0.test_start)
    mutated.loc[m] = mutated.loc[m] * np.exp(np.linspace(0, 1.5, m.sum()))[:, None]
    other = walk_forward(mutated, TrendFollowing, COMBOS, cfg)
    assert other.folds[0].params == f0.params and other.folds[0].train_sharpe_ann == f0.train_sharpe_ann


def test_oos_is_flat_before_first_test_window(rw_prices):
    wf = walk_forward(rw_prices, TrendFollowing, COMBOS, BacktestConfig())
    assert wf.oos.returns.index[0] >= pd.Timestamp(wf.folds[0].test_start)
    assert (wf.stitched_target.loc[: pd.Timestamp(wf.folds[0].train_end)] == 0).all().all()


def _gate_run(scenario, seed):
    prices = generate_synthetic_prices(TICKERS, seed=seed, scenario=scenario)
    cfg = dict(CFG, strategies=[CFG["strategies"][0]])
    res = run_research(cfg, prices, is_synthetic=True, data_source="synthetic", store=None)
    return res["reports"][0]


def test_false_positive_control_on_random_walks():
    """No-edge data: the gates must (almost) never wave a strategy through."""
    verdicts = [_gate_run("random_walk", s)["verdict"] for s in range(12)]
    assert verdicts.count("PASS_TO_PAPER") <= 1, verdicts


def test_detection_power_on_planted_regime_edge():
    """Planted persistent-drift edge: the same gates must be able to detect it."""
    verdicts = [_gate_run("regime", s)["verdict"] for s in range(12)]
    assert verdicts.count("PASS_TO_PAPER") >= 6, verdicts


def test_gate_report_structure_and_cost_stress():
    r = _gate_run("regime", 1)
    names = {g["name"] for g in r["gates"]}
    assert names == {"oos_length_years", "oos_sharpe", "deflated_sharpe_ratio",
                     "oos_max_drawdown", "cost_stress_sharpe", "psr_vs_benchmark"}
    assert r["stress_metrics"]["sharpe"] <= r["oos_metrics"]["sharpe"]     # 2x costs never help
    assert r["is_synthetic"] is True and r["verdict"] in {"REJECT", "PASS_TO_PAPER"}
