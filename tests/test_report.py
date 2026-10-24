import json
from pathlib import Path

import pytest

from quantlab.cli import main
from quantlab.report import collect_runs, render_report, write_report

CONFIG_PATH = str(Path(__file__).parents[1] / "configs" / "etf_trend.yaml")


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("runs")
    main(["run", "--config", CONFIG_PATH, "--provider", "synthetic", "--scenario", "regime", "--seed", "3",
          "--store-dir", str(d)])
    return d


def test_report_is_self_contained_flags_synthetic_and_has_no_script(runs_dir, tmp_path):
    out = write_report(runs_dir, tmp_path / "r" / "index.html")
    html = out.read_text()
    assert "Données synthétiques" in html and "trend_sma" in html and "xs_momentum" in html
    assert "<script" not in html and "http://" not in html and "src=" not in html
    assert "<polyline" in html and "Mois gagnants" in html and "5 dernières années" in html


def test_report_escapes_everything_read_from_disk(runs_dir):
    runs = collect_runs(runs_dir)
    runs[0]["report"]["strategy"] = "<script>alert(1)</script>"
    runs[0]["manifest"]["data"]["source"] = '"><img src=x onerror=alert(1)>'
    html = render_report(runs)
    assert "<script>alert" not in html and "<img src=x" not in html
    assert "&lt;script&gt;" in html


def test_report_shows_paper_state(runs_dir, tmp_path):
    state = {"last_bar": "2026-10-23", "halted": "drawdown <script>", "equity_history": [
        {"date": f"2026-10-{d:02d}", "equity": 10_000 + 10 * d} for d in range(1, 24)],
        "broker": {"cash": 1.0, "positions": {"SPY": 3}, "orders": [
            {"order": {"side": "BUY", "symbol": "SPY", "quantity": 3}, "status": "FILLED", "avg_price": 101.5}]}}
    p = tmp_path / "state.json"
    p.write_text(json.dumps(state))
    html = render_report(collect_runs(runs_dir), p)
    assert "Paper trading" in html and "SIMULATION" in html and "FILLED" in html
    assert "<script>" not in html and "Arrêt de sécurité" in html


def test_cli_report_command(runs_dir, tmp_path, capsys):
    assert main(["report", "--runs", str(runs_dir), "--out", str(tmp_path / "x.html")]) == 0
    assert (tmp_path / "x.html").exists() and "report written" in capsys.readouterr().out
