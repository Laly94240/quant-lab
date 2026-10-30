"""HTML report for the daily-bar day-trading study (self-contained, no JavaScript)."""
from __future__ import annotations

import json
from html import escape

from .crypto_report import CSS, EXTRA_CSS, bar_chart, drawdown_chart, line_chart

LABELS = {"intraday_trend": "H1 tendance intraday", "gap_fade": "H2 gap fade", "ibs_reversal": "H3 IBS"}
CLS = {"intraday_trend": "c-strat", "gap_fade": "c-ok", "ibs_reversal": "c-btc"}


def _p(x, pct=False):
    if x is None:
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:.2f}"


def render_daytrade_html(result: dict) -> str:
    d, reports, series = result["data"], result["reports"], result["series"]
    dec = result["decomposition"]
    banner = ('<div class="banner"><b>Données synthétiques.</b> Aperçu du format : ces chiffres n\'ont aucune valeur financière.</div>'
              if d["is_synthetic"] else "")
    warn = "".join(f"<li>{escape(w)}</li>" for w in d.get("warnings", []))
    assets = list(dec["per_asset"])
    day_vals = [dec["per_asset"][a]["intraday"] for a in assets]
    night_vals = [dec["per_asset"][a]["overnight"] for a in assets]
    port = dec["portfolio"]
    port_curves = [(k, (1 + v).cumprod(), c) for (k, v), c in zip(port.items(), ["c-strat", "c-btc", "c-bh"], strict=True)]

    names = [r["strategy"] for r in reports]
    eq, dd = [], []
    for n in names:
        s = series[n]
        eq.append((LABELS.get(n, n), (1 + s["oos"]).cumprod(), CLS.get(n, "c-strat")))
        dd.append((LABELS.get(n, n), (1 + s["oos"]).cumprod() / (1 + s["oos"]).cumprod().cummax() - 1, CLS.get(n, "c-strat")))
    if names:
        s0 = series[names[0]]
        eq += [("Panier équipondéré (acheter et garder)", (1 + s0["bench"]).cumprod(), "c-bh"),
               ("Long chaque jour, ouverture → clôture (trader naïf)", (1 + s0["always"]).cumprod(), "c-ko")]
        dd.append(("Panier équipondéré", (1 + s0["bench"]).cumprod() / (1 + s0["bench"]).cumprod().cummax() - 1, "c-bh"))

    rows = ""
    for r in reports:
        m, t = r["oos_metrics"], r["trade_stats"]
        cls = "pass" if r["verdict"] == "PASS_TO_PAPER" else "fail"
        rows += (f"<tr><td>{escape(LABELS.get(r['strategy'], r['strategy']))}</td><td>{_p(m.get('sharpe'))}</td>"
                 f"<td>{_p(m.get('cagr'), True)}</td><td>{_p(m.get('max_drawdown'), True)}</td>"
                 f"<td>{_p(t['pct_days_in_market'], True)}</td><td>{t['trades_per_year']:.0f}</td>"
                 f"<td>{_p(t['win_rate'], True)}</td><td>{_p(t['gross_edge_bps'])}</td><td>{t['round_trip_cost_bps']:.1f}</td>"
                 f"<td>{_p(t['net_edge_bps'])}</td><td class=\"{cls}\">{'ÉLIGIBLE PAPER' if cls == 'pass' else 'REJETÉE'}</td></tr>")
    bm = reports[0]["benchmark_metrics"] if reports else {}
    rows += (f"<tr><td>Panier équipondéré</td><td>{_p(bm.get('sharpe'))}</td><td>{_p(bm.get('cagr'), True)}</td>"
             f"<td>{_p(bm.get('max_drawdown'), True)}</td><td colspan=\"7\"></td></tr>")
    net = [(r["trade_stats"]["net_edge_bps"] or 0.0) for r in reports]
    gates = "".join(
        f"<h3>{escape(LABELS.get(r['strategy'], r['strategy']))}</h3><table><tr><th>Critère</th><th>Valeur</th><th>Seuil</th><th>Résultat</th></tr>"
        + "".join(f"<tr><td>{escape(g['name'])}</td><td>{_p(g['value'])}</td><td>{_p(g['threshold'])}</td>"
                  f'<td class="{"pass" if g["passed"] else "fail"}">{"OK" if g["passed"] else "ÉCHEC"}</td></tr>' for g in r["gates"])
        + "</table>" for r in reports)

    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Day trading sur barres journalières</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>Day trading sur données journalières</h1>
<p class="sub">Entrée à l'ouverture, sortie à la clôture, aucune position la nuit. Données : {escape(d['source'])}, {escape(d['start'])} → {escape(d['end'])}.</p>{banner}
<section class="card"><h2>Ce qu'on peut tester, et ce qu'on ne peut pas</h2>
<p>Avec une barre par jour (ouverture, plus haut, plus bas, clôture), un trade est le rendement d'une séance, décidé <b>avant l'ouverture</b> avec les informations de la veille (et le prix d'ouverture pour les règles de gap).
On ne peut <b>pas</b> tester d'ordres stop, d'objectifs de gain ni de logique intra-séance : l'ordre du plus haut et du plus bas est inconnu. Cela demande des données à la minute.
Chaque trade paie les coûts deux fois (ouverture et clôture) : {result['params']['extra_bps']:.0f} pb de plus par côté qu'en journalier, car les prix d'ouverture et de clôture sont moins bien exécutés.
Les règles de gap supposent d'obtenir un prix proche de l'ouverture après l'avoir vu, ce qui est optimiste.</p>
{('<ul class="note">' + warn + '</ul>') if warn else ''}</section>

<section class="card"><h2>1. D'où vient le rendement : la séance ou la nuit ?</h2>
<p class="note">Rendement annuel moyen par actif, séparé entre la séance (ouverture → clôture) et la nuit (clôture → ouverture). C'est l'information clé pour décider s'il vaut la peine d'être exposé pendant la séance.</p>
<h3>Séance</h3>{bar_chart(assets, day_vals, lambda v: f"{v:.1%}")}
<h3>Nuit</h3>{bar_chart(assets, night_vals, lambda v: f"{v:.1%}")}
<h3>Panier équipondéré, capital cumulé (base 1)</h3>{line_chart(port_curves, log=True)}</section>

<section class="card"><h2>2. Les trois hypothèses, hors échantillon</h2>
<p class="note">Enregistrées avant tout test (docs/JOURNAL.md). Paramètres choisis sur l'entraînement seul (walk-forward). « Edge » = rendement moyen brut par trade ; « coût » = coût d'un aller-retour. Il faut edge &gt; coût.</p>
<table><tr><th>Stratégie</th><th>Sharpe</th><th>Rendement/an</th><th>Pire baisse</th><th>Temps en position</th><th>Trades/an</th><th>Trades gagnants</th><th>Edge (pb)</th><th>Coût (pb)</th><th>Net (pb)</th><th>Verdict</th></tr>{rows}</table>
<h3>Gain net moyen par trade (points de base)</h3>{bar_chart([LABELS.get(n, n) for n in names], net, lambda v: f"{v:.0f}") if names else ''}</section>

<section class="card"><h2>3. Valeur du capital hors échantillon (base 1)</h2>
{line_chart(eq, log=True) if eq else ''}</section>

<section class="card"><h2>4. Les baisses</h2>{drawdown_chart(dd) if dd else ''}</section>

<section class="card"><h2>5. Critères de validation</h2>
<p class="note">Fixés avant le test. Un « éligible » ne veut dire qu'éligible au paper trading.</p>{gates}</section>

<section class="card"><h2>Limites</h2><ul class="note">
<li>Barres journalières seulement : pas de stop, pas d'objectif, pas de logique intra-séance.</li>
<li>Prix d'ouverture et de clôture officiels (enchères) : l'exécution réelle est moins bonne que ce prix. Les coûts supposés sont des hypothèses à calibrer.</li>
<li>ETF américains utilisés comme proxys : un particulier de l'EEE ne peut en général pas les acheter (règle PRIIPs). Les produits accessibles (ETF UCITS, CFD) ont d'autres horaires, spreads et coûts.</li>
<li>Peu de trades pour certaines règles : les chiffres sont fragiles. Les critères tiennent compte du nombre d'essais, pas de tous les biais possibles.</li>
<li>Information technique, pas un conseil en investissement.</li></ul></section>
</main></body></html>"""


def daytrade_summary_json(result: dict) -> str:
    out = {"data": {k: v for k, v in result["data"].items() if k != "quality"}, "n_trials_total": result["n_trials_total"],
           "decomposition": result["decomposition"]["per_asset"],
           "reports": [{k: r[k] for k in ("strategy", "verdict", "failed_gates", "oos_metrics", "benchmark_metrics",
                                          "trade_stats", "folds", "gates", "last_5y")} for r in result["reports"]]}
    return json.dumps(out, indent=2, default=str)

