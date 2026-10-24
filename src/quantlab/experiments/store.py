"""Append-only experiment store: every run is saved with its config, code version, data
fingerprint and results, so that it can be reproduced and so that the number of hypotheses
tried (needed for multiple-testing corrections) is never lost."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def dumps(obj, **kw) -> str:
    return json.dumps(obj, default=_default, sort_keys=True, **kw)


def config_hash(config: dict) -> str:
    return hashlib.sha256(dumps(config).encode()).hexdigest()[:12]


def git_state(cwd: str | Path | None = None) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)

    try:
        head = run("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"commit": None, "dirty": None}
        dirty = bool(run("status", "--porcelain").stdout.strip())
        return {"commit": head.stdout.strip(), "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


class ExperimentStore:
    def __init__(self, root: str | Path = "runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"

    def _read_index(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        return [json.loads(line) for line in self.index_path.read_text().splitlines() if line.strip()]

    def has_run(self, cfg_hash: str) -> dict | None:
        for row in self._read_index():
            if row["config_hash"] == cfg_hash:
                return row
        return None

    def cumulative_trials(self, research_id: str) -> int:
        """Total parameter sets ever evaluated under this research question (multiple testing)."""
        return sum(r.get("n_trials", 0) for r in self._read_index() if r["research_id"] == research_id)

    def save(self, *, research_id: str, config: dict, report: dict, oos_returns: pd.Series,
             data_info: dict, n_trials: int) -> str:
        cfg_hash = config_hash(config)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        run_id = f"{stamp}_{config['strategy']['name']}_{cfg_hash[:6]}"
        d = self.root / run_id
        d.mkdir(parents=True, exist_ok=False)
        manifest = {
            "run_id": run_id,
            "created_utc": stamp,
            "config_hash": cfg_hash,
            "git": git_state(self.root),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {m.__name__: getattr(m, "__version__", None) for m in (np, pd)},
            "data": data_info,
        }
        (d / "config.json").write_text(dumps(config, indent=2))
        (d / "manifest.json").write_text(dumps(manifest, indent=2))
        (d / "report.json").write_text(dumps(report, indent=2))
        oos_returns.rename("net_return").to_csv(d / "oos_returns.csv")
        row = {
            "run_id": run_id, "config_hash": cfg_hash, "research_id": research_id,
            "strategy": config["strategy"]["name"], "n_trials": n_trials,
            "verdict": report["verdict"], "oos_sharpe": report["oos_metrics"].get("sharpe"),
            "is_synthetic": data_info.get("is_synthetic"),
        }
        with self.index_path.open("a") as f:
            f.write(dumps(row) + "\n")
        return run_id
