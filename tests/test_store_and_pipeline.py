import json
from pathlib import Path

import pytest
import yaml

from quantlab.cli import main
from quantlab.data import DataQualityError, generate_synthetic_prices
from quantlab.experiments import ExperimentStore, config_hash, git_state
from quantlab.pipeline import run_research

CFG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "etf_trend.yaml").read_text())
CONFIG_PATH = str(Path(__file__).parents[1] / "configs" / "etf_trend.yaml")


def test_config_hash_is_stable_and_sensitive():
    a = {"x": 1, "y": {"b": 2, "a": 1}}
    assert config_hash(a) == config_hash({"y": {"a": 1, "b": 2}, "x": 1})
    assert config_hash(a) != config_hash({"x": 2, "y": {"b": 2, "a": 1}})


def test_git_state_outside_repo_does_not_crash(tmp_path):
    assert set(git_state(tmp_path)) == {"commit", "dirty"}


def test_pipeline_saves_reproducible_runs_and_skips_identical_ones(tmp_path):
    prices = generate_synthetic_prices(CFG["universe"], seed=2, scenario="regime")
    store = ExperimentStore(tmp_path)
    r1 = run_research(CFG, prices, is_synthetic=True, data_source="synthetic", store=store)
    assert len(r1["reports"]) == 2 and not r1["skipped"]
    run_dir = tmp_path / r1["reports"][0]["run_id"]
    for name in ("config.json", "manifest.json", "report.json", "oos_returns.csv"):
        assert (run_dir / name).exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["data"]["is_synthetic"] is True and len(manifest["data"]["hash"]) == 16
    assert "commit" in manifest["git"]

    r2 = run_research(CFG, prices, is_synthetic=True, data_source="synthetic", store=store)
    assert len(r2["skipped"]) == 2 and not r2["reports"]                    # no repeated experiment
    r3 = run_research(CFG, prices, is_synthetic=True, data_source="synthetic", store=store, force=True)
    assert len(r3["reports"]) == 2


def test_multiple_testing_counter_accumulates_across_runs(tmp_path):
    prices = generate_synthetic_prices(CFG["universe"], seed=2, scenario="random_walk")
    store = ExperimentStore(tmp_path)
    r1 = run_research(CFG, prices, is_synthetic=True, data_source="synthetic", store=store)
    assert r1["n_trials_total"] == 4 + 4 == store.cumulative_trials(CFG["research_id"])
    cfg2 = dict(CFG, strategies=[dict(CFG["strategies"][0], grid={"lookback": [75, 125, 175]})])
    r2 = run_research(cfg2, prices, is_synthetic=True, data_source="synthetic", store=store)
    assert r2["n_trials_total"] == 8 + 3                                     # history is never forgotten


def test_more_trials_can_only_lower_the_deflated_sharpe(tmp_path):
    prices = generate_synthetic_prices(CFG["universe"], seed=2, scenario="regime")
    cfg = dict(CFG, strategies=[CFG["strategies"][0]])
    dsr = lambda res: {g["name"]: g["value"] for g in res["reports"][0]["gates"]}["deflated_sharpe_ratio"]
    fresh = run_research(cfg, prices, is_synthetic=True, data_source="synthetic", store=None)
    store = ExperimentStore(tmp_path)
    for i in range(3):                                                       # earlier, failed attempts
        other = dict(CFG, strategies=[dict(CFG["strategies"][0], grid={"lookback": [60 + i, 90 + i]})])
        run_research(other, prices, is_synthetic=True, data_source="synthetic", store=store)
    burdened = run_research(cfg, prices, is_synthetic=True, data_source="synthetic", store=store)
    assert dsr(burdened) <= dsr(fresh)


def test_bad_data_stops_the_pipeline():
    prices = generate_synthetic_prices(CFG["universe"], seed=2)
    prices.iloc[100, 0] = -1.0
    with pytest.raises(DataQualityError):
        run_research(CFG, prices, is_synthetic=True, data_source="synthetic")


def test_cli_end_to_end_on_synthetic_data(tmp_path, capsys):
    code = main(["run", "--config", CONFIG_PATH, "--provider", "synthetic", "--scenario", "regime",
                 "--seed", "1", "--store-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0 and "SYNTHETIC DATA" in out and "trend_sma" in out and "benchmark" in out
    assert (tmp_path / "index.jsonl").exists()
