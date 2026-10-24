"""Command line: `quantlab run --config configs/etf_trend.yaml` / `quantlab fetch ...`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .data import CSVProvider, SyntheticProvider, YahooProvider
from .experiments import ExperimentStore
from .pipeline import format_summary, run_research


def _load(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _provider(name: str, cfg: dict, args):
    d = cfg.get("data", {})
    if name == "csv":
        return CSVProvider(d.get("raw_dir", "data/raw"))
    if name == "yahoo":
        return YahooProvider(cache_dir=d.get("raw_dir", "data/raw"))
    if name == "synthetic":
        return SyntheticProvider(seed=args.seed, scenario=args.scenario)
    raise SystemExit(f"unknown provider {name!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="quantlab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download daily prices from Yahoo and snapshot them to CSV")
    f.add_argument("--config", required=True)

    r = sub.add_parser("run", help="run the research pipeline (walk-forward + gates + save)")
    r.add_argument("--config", required=True)
    r.add_argument("--provider", choices=["csv", "yahoo", "synthetic"], default=None)
    r.add_argument("--scenario", choices=["random_walk", "regime"], default="random_walk",
                   help="synthetic provider only")
    r.add_argument("--seed", type=int, default=0, help="synthetic provider only")
    r.add_argument("--store-dir", default=None)
    r.add_argument("--force", action="store_true", help="rerun even if an identical run exists")
    args = ap.parse_args(argv)

    cfg = _load(args.config)
    d = cfg.get("data", {})
    tickers = cfg["universe"]

    if args.cmd == "fetch":
        prices = YahooProvider(cache_dir=d.get("raw_dir", "data/raw")).get_prices(
            tickers, d.get("start"), d.get("end"))
        print(f"saved {len(tickers)} CSV files to {d.get('raw_dir', 'data/raw')} "
              f"({prices.index[0].date()} -> {prices.index[-1].date()})")
        return 0

    name = args.provider or d.get("provider", "csv")
    provider = _provider(name, cfg, args)
    prices = provider.get_prices(tickers, d.get("start"), d.get("end"))
    store = ExperimentStore(args.store_dir or cfg.get("store_dir", "runs"))
    result = run_research(cfg, prices, is_synthetic=provider.is_synthetic,
                          data_source=provider.name, store=store, force=args.force)
    print(format_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
