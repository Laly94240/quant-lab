"""Command line.

  quantlab fetch       --config CFG                 snapshot Yahoo prices to CSV
  quantlab run         --config CFG                 research pipeline (walk-forward + gates + save)
  quantlab report      --runs runs --out report/index.html
  quantlab paper-run   --config CFG                 one daily paper-trading step (Discord alerts)
  quantlab news-brief  --config CFG                 news -> validated, advisory briefing
  quantlab notify-test                              send a test message to the Discord webhook
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from .data import CSVProvider, SyntheticProvider, YahooProvider
from .experiments import ExperimentStore
from .notify import notifier_from_env
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


def _cmd_news_brief(cfg: dict, args) -> int:
    from .news import AnthropicClient, RSSProvider, XProvider, build_prompt, make_briefing
    from .news.sanitize import looks_suspicious

    ncfg = cfg.get("news", {})
    since = datetime.now(timezone.utc) - timedelta(hours=ncfg.get("lookback_hours", 24))
    items = []
    if ncfg.get("rss_feeds"):
        items += RSSProvider(ncfg["rss_feeds"]).fetch_latest(since)
    xcfg = ncfg.get("x", {})
    if xcfg.get("enabled"):
        items += XProvider(queries=xcfg.get("queries", []), max_results=xcfg.get("max_results", 20),
                           max_posts_per_run=xcfg.get("max_posts_per_run", 40)).fetch_latest(since)
    items.sort(key=lambda i: i.published or since, reverse=True)
    print(f"{len(items)} news items fetched "
          f"({sum(looks_suspicious(i.title + ' ' + i.text) for i in items)} flagged as suspicious, excluded)")
    if not items:
        return 0
    if args.dry_run:
        clean = [(n, it) for n, it in enumerate(items, 1) if not looks_suspicious(f"{it.title} {it.text}")]
        print(build_prompt(clean[: ncfg.get("max_items", 40)], cfg["universe"])[1])
        return 0

    llm_cfg = ncfg.get("llm", {})
    llm = AnthropicClient(model=llm_cfg.get("model", "claude-sonnet-5-5"), max_tokens=llm_cfg.get("max_tokens", 1500))
    briefing = make_briefing(items, cfg["universe"], llm, ncfg.get("max_items", 40))
    out = Path(ncfg.get("briefing_file", "paper/latest_briefing.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(briefing.to_dict(), indent=2, ensure_ascii=False))

    lines = [briefing.regime_note, ""]
    lines += [f"• [{f.severity}] {f.theme} → {', '.join(f.affected_assets)}" for f in briefing.risk_flags]
    lines += [f"→ {i.symbol}: {i.direction} ({i.horizon}, confiance {i.confidence:.0%}) {i.rationale}"
              for i in briefing.ideas]
    lines.append(f"\n{briefing.items_used} sources lues, {len(briefing.suspicious_ids)} écartées, "
                 f"{len(briefing.rejected)} sorties du modèle rejetées.")
    text = "\n".join(lines)
    print(text)
    notifier = notifier_from_env(enabled=cfg.get("notify", {}).get("discord", {}).get("enabled", True)
                                 and not args.no_discord)
    notifier.send("Briefing marchés : conseil, AUCUN ordre passé", text, level="info")
    return 0


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

    rp = sub.add_parser("report", help="build the static HTML report from saved runs")
    rp.add_argument("--runs", default="runs")
    rp.add_argument("--out", default="report/index.html")
    rp.add_argument("--paper-state", default=None)

    pp = sub.add_parser("paper-run", help="one daily paper-trading step (simulated money)")
    pp.add_argument("--config", required=True)
    pp.add_argument("--provider", choices=["csv", "yahoo"], default="yahoo")
    pp.add_argument("--no-discord", action="store_true")
    pp.add_argument("--clear-halt", action="store_true", help="resume after a safety halt (human decision)")

    nb = sub.add_parser("news-brief", help="fetch news and build an advisory briefing")
    nb.add_argument("--config", required=True)
    nb.add_argument("--no-discord", action="store_true")
    nb.add_argument("--dry-run", action="store_true", help="fetch and print the prompt, no LLM call")

    sub.add_parser("notify-test", help="send a test message to DISCORD_WEBHOOK_URL")

    args = ap.parse_args(argv)

    if args.cmd == "notify-test":
        notifier = notifier_from_env()
        ok = notifier.send("quantlab : test de notification", "Si tu lis ceci, le webhook fonctionne. "
                           "Aucun ordre n'a été passé.", level="success")
        print("message sent" if ok else "NOT sent: DISCORD_WEBHOOK_URL missing or rejected by Discord")
        return 0 if ok else 1

    if args.cmd == "report":
        from .report import write_report
        out = write_report(args.runs, args.out, args.paper_state)
        print(f"report written to {out}")
        return 0

    cfg = _load(args.config)
    d = cfg.get("data", {})
    tickers = cfg["universe"]

    if args.cmd == "news-brief":
        return _cmd_news_brief(cfg, args)

    if args.cmd == "fetch":
        prices = YahooProvider(cache_dir=d.get("raw_dir", "data/raw")).get_prices(
            tickers, d.get("start"), d.get("end"))
        print(f"saved {len(tickers)} CSV files to {d.get('raw_dir', 'data/raw')} "
              f"({prices.index[0].date()} -> {prices.index[-1].date()})")
        return 0

    if args.cmd == "paper-run":
        from .news import Briefing
        from .paper import run_paper_day
        prices = _provider(args.provider, cfg, args).get_prices(tickers, d.get("start"), None)
        notifier = notifier_from_env(enabled=cfg.get("notify", {}).get("discord", {}).get("enabled", True)
                                     and not args.no_discord)
        briefing = None
        bf = Path(cfg.get("news", {}).get("briefing_file", "paper/latest_briefing.json"))
        if cfg.get("news", {}).get("overlay", {}).get("enabled") and bf.exists():
            briefing = Briefing.from_dict(json.loads(bf.read_text()))
        summary = run_paper_day(cfg, prices, notifier, briefing=briefing, clear_halt=args.clear_halt)
        print(json.dumps(summary, indent=2, default=str))
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
