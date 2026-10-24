"""Static HTML report (no JavaScript, no external resource): open it locally, attach it to a CI
run, or publish it with GitHub Pages. Everything read from disk is HTML-escaped."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd

CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;--ok:#15803d;--ko:#b91c1c;
--warn:#b45309;--accent:#2563eb;--bench:#a8a29e}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--fg:#f5f5f4;--muted:#a8a29e;--card:#1c1917;
--line:#292524;--ok:#4ade80;--ko:#f87171;--warn:#fbbf24;--accent:#60a5fa;--bench:#78716c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 18px 60px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:0}h3{font-size:13px;color:var(--muted);
text-transform:uppercase;letter-spacing:.06em;margin:20px 0 8px}
.sub{color:var(--muted);margin:0 0 20px}
.banner{border:1px solid var(--warn);color:var(--warn);border-radius:10px;padding:10px 14px;margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin:0 0 18px}
.head{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.badge{font-size:12px;font-weight:600;padding:3px 10px;border-radius:999px;border:1px solid}
.pass{color:var(--ok)}.fail{color:var(--ko)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px}
.kpi{border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.kpi b{display:block;font-size:18px}.kpi span{color:var(--muted);font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:500}
svg{width:100%;height:auto;display:block}.legend{font-size:12px;color:var(--muted)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:var(--muted)}
footer{color:var(--muted);font-size:12px;margin-top:26px}
"""


def _f(x, kind="num"):
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return escape(str(x))
    return f"{x:.1%}" if kind == "pct" else f"{x:.2f}"


def _kpi(label, value):
    return f'<div class="kpi"><b>{escape(value)}</b><span>{escape(label)}</span></div>'


def equity_svg(series: list[tuple[str, pd.Series, str]], w: int = 720, h: int = 220) -> str:
    curves = []
    for label, r, cls in series:
        if r is None or len(r) < 2:
            continue
        eq = (1 + r).cumprod()
        step = max(1, len(eq) // 500)                       # keep the file small: <= ~500 points / curve
        keep = list(range(0, len(eq), step))
        if keep[-1] != len(eq) - 1:
            keep.append(len(eq) - 1)
        curves.append((label, eq.iloc[keep], cls))
    if not curves:
        return ""
    lo = min(float(c.min()) for _, c, _ in curves)
    hi = max(float(c.max()) for _, c, _ in curves)
    pad = 0.05 * (hi - lo or 1.0)
    lo, hi = lo - pad, hi + pad
    paths = []
    for _, c, cls in curves:
        n = len(c)
        pts = " ".join(f"{i / (n - 1) * (w - 60) + 50:.1f},{h - 24 - (float(v) - lo) / (hi - lo) * (h - 40):.1f}"
                       for i, v in enumerate(c.to_numpy()))
        color = "var(--accent)" if cls == "strat" else "var(--bench)"
        paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{pts}"/>')
    first, last = curves[0][1].index[0], curves[0][1].index[-1]
    axis = (f'<text x="50" y="{h - 6}" font-size="11" fill="var(--muted)">{escape(str(first)[:10])}</text>'
            f'<text x="{w - 10}" y="{h - 6}" font-size="11" text-anchor="end" fill="var(--muted)">'
            f'{escape(str(last)[:10])}</text>'
            f'<text x="44" y="{h - 24}" font-size="11" text-anchor="end" fill="var(--muted)">{lo:.2f}</text>'
            f'<text x="44" y="16" font-size="11" text-anchor="end" fill="var(--muted)">{hi:.2f}</text>')
    legend = " &nbsp; ".join(
        f'<span style="color:{"var(--accent)" if c == "strat" else "var(--bench)"}">&#9632;</span> {escape(label)}'
        for label, _, c in curves)
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="equity curves (growth of 1)">'
            f'<rect x="50" y="8" width="{w - 60}" height="{h - 32}" fill="none" stroke="var(--line)"/>'
            f'{"".join(paths)}{axis}</svg><div class="legend">{legend} (base 1)</div>')


def _read_returns(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0]


def collect_runs(runs_dir: str | Path) -> list[dict]:
    out = []
    for d in sorted(Path(runs_dir).glob("*/report.json")):
        run = d.parent
        out.append({
            "id": run.name,
            "report": json.loads(d.read_text()),
            "manifest": json.loads((run / "manifest.json").read_text()),
            "oos": _read_returns(run / "oos_returns.csv"),
            "bench": _read_returns(run / "benchmark_returns.csv"),
        })
    return out


def _run_card(run: dict) -> str:
    rep, man = run["report"], run["manifest"]
    m, b = rep["oos_metrics"], rep["benchmark_metrics"]
    ok = rep["verdict"] == "PASS_TO_PAPER"
    badge = (f'<span class="badge {"pass" if ok else "fail"}">'
             f'{"ÉLIGIBLE AU PAPER TRADING" if ok else "REJETÉE"}</span>')
    kpis = "".join([
        _kpi("Sharpe OOS", _f(m.get("sharpe"))), _kpi("CAGR", _f(m.get("cagr"), "pct")),
        _kpi("Max drawdown", _f(m.get("max_drawdown"), "pct")), _kpi("Volatilité", _f(m.get("ann_vol"), "pct")),
        _kpi("Jours gagnants", _f(m.get("daily_win_rate"), "pct")),
        _kpi("Mois gagnants", _f(m.get("monthly_win_rate"), "pct")),
        _kpi("Rotation / an", _f(m.get("annual_turnover"))), _kpi("Sharpe benchmark", _f(b.get("sharpe"))),
    ])
    l5 = rep.get("last_5y")
    last5 = ""
    if l5:
        s5, b5 = l5["strategy"], l5["benchmark"]
        last5 = ("<h3>5 dernières années hors échantillon</h3><table><tr><th></th><th>Sharpe</th><th>CAGR</th>"
                 "<th>Max DD</th><th>Jours gagnants</th><th>Mois gagnants</th></tr>" + "".join(
                     f"<tr><td>{n}</td><td>{_f(x.get('sharpe'))}</td><td>{_f(x.get('cagr'), 'pct')}</td>"
                     f"<td>{_f(x.get('max_drawdown'), 'pct')}</td><td>{_f(x.get('daily_win_rate'), 'pct')}</td>"
                     f"<td>{_f(x.get('monthly_win_rate'), 'pct')}</td></tr>"
                     for n, x in (("Stratégie", s5), ("Benchmark", b5))) + "</table>")
    gates = "".join(
        f"<tr><td>{escape(g['name'])}</td><td>{_f(g['value'])}</td><td>{_f(g['threshold'])}</td>"
        f'<td class="{"pass" if g["passed"] else "fail"}">{"OK" if g["passed"] else "ÉCHEC"}</td></tr>'
        for g in rep["gates"])
    folds = "".join(
        f"<tr><td>{escape(f['test_start'])} → {escape(f['test_end'])}</td><td>{escape(json.dumps(f['params']))}</td>"
        f"<td>{_f(f.get('train_sharpe_ann'))}</td><td>{_f(f.get('test_sharpe_ann'))}</td></tr>"
        for f in rep["folds"])
    chart = equity_svg([("Stratégie (nette de coûts)", run["oos"], "strat"), ("Benchmark équipondéré", run["bench"], "bench")])
    data = man.get("data", {})
    return f"""<section class="card">
<div class="head"><h2>{escape(rep['strategy'])}</h2>{badge}</div>
<p class="mono">{escape(run['id'])} · données : {escape(str(data.get('source')))} {escape(str(data.get('start')))} → {escape(str(data.get('end')))} · hash {escape(str(data.get('hash')))} · essais cumulés : {escape(str(rep.get('n_trials_total')))}</p>
<div class="grid">{kpis}</div>{chart}{last5}
<h3>Critères de validation (fixés avant le test)</h3>
<table><tr><th>Critère</th><th>Valeur</th><th>Seuil</th><th>Résultat</th></tr>{gates}</table>
<h3>Walk-forward : paramètres choisis sur l'entraînement seul</h3>
<table><tr><th>Fenêtre de test</th><th>Paramètres</th><th>Sharpe entraînement</th><th>Sharpe test</th></tr>{folds}</table>
</section>"""


def _paper_card(state_path: Path) -> str:
    s = json.loads(state_path.read_text())
    hist = s.get("equity_history", [])
    eq = pd.Series([e["equity"] for e in hist], index=pd.to_datetime([e["date"] for e in hist]), dtype=float)
    ret = eq.pct_change().dropna()
    chart = equity_svg([("Portefeuille papier", ret, "strat")]) if len(ret) > 2 else ""
    pos = s["broker"].get("positions", {})
    pos_rows = "".join(f"<tr><td>{escape(k)}</td><td>{escape(str(v))}</td></tr>" for k, v in sorted(pos.items() ) if v)
    orders = s["broker"].get("orders", [])[-10:]
    ord_rows = "".join(
        f"<tr><td>{escape(o['order']['side'])}</td><td>{escape(o['order']['symbol'])}</td>"
        f"<td>{escape(str(o['order']['quantity']))}</td><td>{escape(o['status'])}</td>"
        f"<td>{_f(o.get('avg_price'))}</td></tr>" for o in reversed(orders))
    halt = f'<p class="fail">Arrêt de sécurité actif : {escape(str(s["halted"]))}</p>' if s.get("halted") else ""
    last = f"{eq.iloc[-1]:,.2f}" if len(eq) else "n/a"
    return f"""<section class="card"><div class="head"><h2>Paper trading</h2><span class="badge">SIMULATION</span></div>
<p class="mono">argent fictif · dernière barre : {escape(str(s.get('last_bar')))} · valeur : {escape(last)}</p>{halt}{chart}
<h3>Positions</h3><table><tr><th>Symbole</th><th>Quantité</th></tr>{pos_rows or '<tr><td colspan=2>aucune</td></tr>'}</table>
<h3>Derniers ordres</h3><table><tr><th>Sens</th><th>Symbole</th><th>Qté</th><th>Statut</th><th>Prix moyen</th></tr>{ord_rows or '<tr><td colspan=5>aucun</td></tr>'}</table></section>"""


def render_report(runs: list[dict], paper_state: Path | None = None) -> str:
    synthetic = any(r["manifest"].get("data", {}).get("is_synthetic") for r in runs)
    banner = ('<div class="banner"><b>Données synthétiques.</b> Ce rapport valide le pipeline, ses chiffres n\'ont '
              "aucune valeur financière.</div>") if synthetic else ""
    body = "".join(_run_card(r) for r in runs) or '<p class="sub">Aucun run trouvé.</p>'
    paper = _paper_card(paper_state) if paper_state and Path(paper_state).exists() else ""
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>quantlab — rapport</title>
<style>{CSS}</style></head><body><main>
<h1>quantlab</h1><p class="sub">Recherche systématique : validation hors échantillon, coûts réalistes, tests multiples.</p>
{banner}{body}{paper}
<footer>« Éligible au paper trading » ne signifie ni rentable ni autorisé en réel. Information technique, pas un conseil en investissement.</footer>
</main></body></html>"""


def write_report(runs_dir: str | Path, out: str | Path, paper_state: str | Path | None = None) -> Path:
    html = render_report(collect_runs(runs_dir), Path(paper_state) if paper_state else None)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
