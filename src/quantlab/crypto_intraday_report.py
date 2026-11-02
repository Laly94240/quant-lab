"""HTML report of the three-block intraday protocol (self-contained, no JavaScript)."""
from __future__ import annotations

import json
from html import escape

import pandas as pd

from .crypto_intraday import daily_returns
from .crypto_report import CSS, EXTRA_CSS, bar_chart, line_chart

STATUS = {
    "no_candidate_in_development": ("Aucune hypothèse ne passe les critères en développement : on s'arrête, personne n'est retenu.", "fail"),
    "dev_only": ("Développement seul exécuté.", "note"),
    "no_winner_in_validation": ("Des finalistes existaient, aucun ne passe la validation : on s'arrête, la dernière année n'est pas touchée.", "fail"),
    "winner_found_lockbox_not_used": ("Un gagnant est sorti de la validation. La dernière année (lockbox) n'a PAS encore été utilisée.", "pass"),
    "lockbox_passed": ("Le gagnant passe aussi la dernière année, vue une seule fois. Éligible au paper trading, pas rentable pour autant.", "pass"),
    "lockbox_failed": ("Le gagnant échoue sur la dernière année. Le test est consommé : on n'itère pas dessus.", "fail"),
}


def _f(x, pct=False, nd=2):
    return "n/a" if x is None else (f"{x:.1%}" if pct else f"{x:.{nd}f}")


def _label(r):
    return f"{r['strategy']} " + ",".join(f"{v}" for v in r["params"].values())


def _table(rows, limit=12):
    head = ("<tr><th>Hypothèse</th><th>Paramètres</th><th>Trades</th><th>Gagnants</th><th>Gain net (pb)</th><th>Brut (pb)</th>"
            "<th>Facteur de profit</th><th>Sharpe</th><th>Résultat</th></tr>")
    body = ""
    for r in rows[:limit]:
        s = r["stats"]
        res = '<td class="pass">OK</td>' if r["passed"] else f'<td class="fail">{escape(", ".join(r["failed"][:3]))}</td>'
        body += (f"<tr><td>{escape(r['strategy'])}</td><td>{escape(json.dumps(r['params']))}</td><td>{s['n_trades']}</td>"
                 f"<td>{_f(s.get('win_rate'), True)}</td><td>{_f(s.get('expectancy_bps'), nd=1)}</td><td>{_f(s.get('gross_expectancy_bps'), nd=1)}</td>"
                 f"<td>{_f(s.get('profit_factor'))}</td><td>{_f(s.get('sharpe'))}</td>{res}</tr>")
    return f"<table>{head}{body}</table>"


def _gates(block):
    rows = "".join(f"<tr><td>{escape(g['name'])}</td><td>{_f(g['value'], nd=3)}</td><td>{_f(g['threshold'], nd=3)}</td>"
                   f'<td class="{"pass" if g["passed"] else "fail"}">{"OK" if g["passed"] else "ÉCHEC"}</td></tr>' for g in block["gates"])
    return f"<table><tr><th>Critère</th><th>Valeur</th><th>Seuil</th><th>Résultat</th></tr>{rows}</table>"


def _symbol_section(res: dict) -> str:
    b = res["blocks"]
    status, cls = STATUS.get(res["status"].split("_EXPLORATORY")[0], (res["status"], "note"))
    if "EXPLORATORY" in res["status"]:
        status += " Résultat EXPLORATOIRE : la dernière année avait déjà été consommée."
    dev_sorted = sorted(res["dev"], key=lambda r: (r["stats"].get("sharpe") is None, -(r["stats"].get("sharpe") or 0)))
    chart_rows = dev_sorted[:12]
    out = [f'<section class="card"><h2>{escape(res["symbol"])}</h2><p class="{cls}"><b>{escape(status)}</b></p>',
           f'<p class="note">Développement {b["dev"][0]} → {b["dev"][1]} · validation {b["validation"][0]} → {b["validation"][1]} · '
           f'lockbox {b["lockbox"][0]} → {b["lockbox"][1]} · {res["n_trials"]} essais comptés · coût aller-retour {res["costs_round_trip_bps"]:.0f} pb.</p>',
           "<h3>1. Développement : toutes les hypothèses, classées par Sharpe</h3>", _table(dev_sorted),
           "<p class=\"note\">Gain brut moyen par trade (pb) pour les 12 premières, à comparer au coût d'un aller-retour :</p>",
           bar_chart([f"{i + 1}" for i in range(len(chart_rows))], [r["stats"].get("gross_expectancy_bps") or 0.0 for r in chart_rows], lambda v: f"{v:.0f}")]
    if res["finalists"]:
        out += ["<h3>2. Validation : seuls les finalistes</h3>", _table(res["finalists"])]
    if res["winner"]:
        w = res["winner"]
        out.append(f"<h3>3. Gagnant : {escape(w['strategy'])} {escape(json.dumps(w['params']))}</h3>")
        tr = res.get("_trades", {})
        parts, shade = [], []
        for name in ("dev", "validation", "lockbox"):
            if name in tr:
                s, e = b[name]
                parts.append(daily_returns(tr[name], s, e))
                if name != "dev":
                    shade.append((pd.Timestamp(s), pd.Timestamp(e)))
        if parts:
            eq = (1 + pd.concat(parts)).cumprod()
            out += ['<p class="note">Capital cumulé du gagnant (1x, sans levier). Zones ombrées : validation, puis dernière année.</p>',
                    line_chart([("Gagnant", eq, "c-strat")], shade=shade, log=True)]
    if res.get("lockbox"):
        out += ["<h3>4. Dernière année (lockbox), vue une seule fois</h3>", _gates(res["lockbox"])]
    out.append("</section>")
    return "".join(out)


def render_crypto_intraday_html(results: list[dict], cfg: dict, is_synthetic: bool) -> str:
    banner = ('<div class="banner"><b>Données synthétiques.</b> Aperçu du format et du protocole : aucun chiffre n\'a de valeur financière.</div>'
              if is_synthetic else "")
    g = cfg.get("gates", {})
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Recherche intraday sur futures crypto</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>Recherche intraday sur futures crypto</h1>
<p class="sub">Barres de {cfg.get('bar_minutes', 15)} minutes, entrée à l'ouverture de la barre suivante, coûts à chaque trade.</p>{banner}
<section class="card"><h2>Le protocole</h2>
<p>L'historique est coupé en trois blocs, utilisés dans l'ordre et jamais mélangés : <b>développement</b> (exploration libre, chaque essai est compté),
<b>validation</b> (seuls les finalistes ; le meilleur qui passe devient le gagnant), <b>dernière année</b> (le gagnant, une seule fois ; une deuxième consultation est refusée).
Un test consulté pour choisir n'est plus un test : sur une seule année, l'erreur d'un Sharpe est d'environ 1, donc le meilleur de 100 essais affiche souvent un Sharpe de 2 ou plus par pur hasard.</p>
<p>Critères identiques à chaque bloc : au moins {g.get('min_trades_per_year', 100)} trades par an de bloc, <b>trades gagnants ≥ {g.get('min_win_rate', 0.51):.0%}</b>, gain net moyen &gt; 0,
facteur de profit ≥ {g.get('min_profit_factor', 1.10)}, Sharpe déflaté ≥ {g.get('min_dsr', 0.95)}, baisse maximale ≤ {g.get('max_drawdown', 0.30):.0%}, gain net &gt; 0 avec coûts doublés.
Le taux de trades gagnants est un critère, jamais l'objectif : on l'obtient facilement avec de petits gains et de grosses pertes, sans gagner d'argent.</p></section>
{''.join(_symbol_section(r) for r in results)}
<section class="card"><h2>Limites</h2><ul class="note">
<li>Frais pris à 5 pb par côté (niveau standard, à vérifier sur ton compte) plus 1 pb de glissement ; le glissement réel dépend de la taille et du moment.</li>
<li>Le financement (funding) n'est pas modélisé : il est faible sur des positions de quelques heures, mais non nul.</li>
<li>Un seul grand cycle crypto depuis 2020, peu d'actifs corrélés, risque de plateforme non modélisé.</li>
<li>Réussir le protocole veut dire « éligible au paper trading ». Les derniers mois ne sont jamais un test aveugle de l'avenir.</li>
<li>Information technique, pas un conseil en investissement.</li></ul></section></main></body></html>"""


def results_json(results: list[dict]) -> str:
    return json.dumps([{k: v for k, v in r.items() if not k.startswith("_")} for r in results], indent=2, default=str)
