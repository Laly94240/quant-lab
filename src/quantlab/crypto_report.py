"""'How the strategy would have been applied' report for perpetual futures (long/flat, 1x).

This is an EXPLANATORY replay of ONE fixed parameter set (no search, no optimisation, no
verdict): it shows when the rule was invested, what each trade did, what costs and funding took,
and how it compares with simply holding. It does not validate the strategy: use `quantlab run`
(walk-forward + gates) for that.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, CostModel, compute_metrics, simulate
from .report import CSS
from .risk import RiskLimits, apply_weight_limits
from .strategies import BuyAndHold, TrendFollowing

PPY = 365            # crypto trades every calendar day
EXTRA_CSS = """
.heat td{text-align:right;padding:4px 8px;font-size:12px}.heat th{font-size:12px}
.note{color:var(--muted);font-size:13px;margin:6px 0 0}
.c-strat{stroke:var(--accent)}.c-bh{stroke:var(--bench)}.c-btc{stroke:var(--warn)}
.c-ok{stroke:var(--ok)}.c-ko{stroke:var(--ko)}
"""
COLOR_VAR = {"c-strat": "accent", "c-bh": "bench", "c-btc": "warn", "c-ok": "ok", "c-ko": "ko"}


@dataclass
class Trade:
    symbol: str
    entry_date: str
    exit_date: str
    entry_px: float
    exit_px: float
    days: int
    ret: float
    open: bool


def extract_trades(prices: pd.DataFrame, held: pd.DataFrame) -> list[Trade]:
    """Contiguous invested runs per asset. A position held during bar t was bought at close t-1."""
    trades = []
    idx = prices.index
    for sym in prices.columns:
        on = (held[sym] > 1e-12).to_numpy()
        i = 0
        while i < len(on):
            if not on[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            e = max(i - 1, 0)
            entry, exit_ = float(prices[sym].iloc[e]), float(prices[sym].iloc[j])
            trades.append(Trade(sym, str(idx[e].date()), str(idx[j].date()), entry, exit_,
                                j - e, exit_ / entry - 1.0, j == len(on) - 1))
            i = j + 1
    return sorted(trades, key=lambda t: t.entry_date)


def _fmt(x, kind="num"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.1%}" if kind == "pct" else f"{x:.2f}"


def _ds(s: pd.Series, n: int = 600) -> pd.Series:
    step = max(1, len(s) // n)
    keep = list(range(0, len(s), step))
    if keep[-1] != len(s) - 1:
        keep.append(len(s) - 1)
    return s.iloc[keep]


def line_chart(curves, shade=None, log=False, h=250, w=760) -> str:
    """curves: [(label, Series, css_class)]; shade: [(start_ts, end_ts)] invested periods."""
    curves = [(lab, _ds(s.dropna()), cls) for lab, s, cls in curves if len(s.dropna()) > 1]
    if not curves:
        return ""
    t0, t1 = curves[0][1].index[0], curves[0][1].index[-1]
    tf = lambda ts: 50 + (ts - t0).total_seconds() / max((t1 - t0).total_seconds(), 1) * (w - 60)   # noqa: E731
    vals = np.concatenate([c.to_numpy() for _, c, _ in curves])
    vals = np.log(vals) if log else vals
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.05 * ((hi - lo) or 1.0)
    lo, hi = lo - pad, hi + pad
    yf = lambda v: h - 24 - ((np.log(v) if log else v) - lo) / (hi - lo) * (h - 40)   # noqa: E731
    out = [f'<rect x="50" y="8" width="{w - 60}" height="{h - 32}" fill="none" stroke="var(--line)"/>']
    for a, b in shade or []:
        x0, x1 = tf(max(a, t0)), tf(min(b, t1))
        out.append(f'<rect x="{x0:.1f}" y="8" width="{max(x1 - x0, 1):.1f}" height="{h - 32}" fill="var(--ok)" opacity="0.12"/>')
    for lab, s, cls in curves:
        pts = " ".join(f"{tf(t):.1f},{yf(v):.1f}" for t, v in s.items())
        out.append(f'<polyline class="{cls}" fill="none" stroke-width="1.5" points="{pts}"/>')
    fmt = (lambda v: f"{np.exp(v):.2f}") if log else (lambda v: f"{v:.2f}")
    out.append(f'<text x="44" y="16" font-size="11" text-anchor="end" fill="var(--muted)">{fmt(hi)}</text>'
               f'<text x="44" y="{h - 24}" font-size="11" text-anchor="end" fill="var(--muted)">{fmt(lo)}</text>'
               f'<text x="50" y="{h - 6}" font-size="11" fill="var(--muted)">{t0.date()}</text>'
               f'<text x="{w - 10}" y="{h - 6}" font-size="11" text-anchor="end" fill="var(--muted)">{t1.date()}</text>')
    legend = " &nbsp; ".join(f'<span style="color:var(--{COLOR_VAR.get(cls, "fg")})">&#9632;</span> {escape(lab)}'
                             for lab, _, cls in curves)
    return f'<svg viewBox="0 0 {w} {h}" role="img">{"".join(out)}</svg><div class="legend">{legend}</div>'


def drawdown_chart(dds, h=200, w=760) -> str:
    dds = [(lab, _ds(s), cls) for lab, s, cls in dds]
    t0, t1 = dds[0][1].index[0], dds[0][1].index[-1]
    lo = min(float(s.min()) for _, s, _ in dds) * 1.05 or -0.01
    tf = lambda ts: 50 + (ts - t0).total_seconds() / max((t1 - t0).total_seconds(), 1) * (w - 60)   # noqa: E731
    yf = lambda v: 8 + (v / lo) * (h - 32)   # noqa: E731
    out = [f'<rect x="50" y="8" width="{w - 60}" height="{h - 32}" fill="none" stroke="var(--line)"/>']
    for lab, s, cls in dds:
        pts = " ".join(f"{tf(t):.1f},{yf(min(v, 0)):.1f}" for t, v in s.items())
        poly = f"50,8 {pts} {w - 10},8"
        col = f"var(--{COLOR_VAR.get(cls, 'bench')})"
        out.append(f'<polygon points="{poly}" fill="{col}" opacity="0.25"/><polyline fill="none" stroke="{col}" stroke-width="1.2" points="{pts}"/>')
    out.append(f'<text x="44" y="16" font-size="11" text-anchor="end" fill="var(--muted)">0%</text>'
               f'<text x="44" y="{h - 24}" font-size="11" text-anchor="end" fill="var(--muted)">{lo:.0%}</text>')
    legend = " &nbsp; ".join(f'<span style="color:var(--{COLOR_VAR.get(c, "bench")})">&#9632;</span> {escape(lab)}' for lab, _, c in dds)
    return f'<svg viewBox="0 0 {w} {h}" role="img">{"".join(out)}</svg><div class="legend">{legend}</div>'


def timeline_chart(prices: pd.DataFrame, trades: list[Trade], w=760) -> str:
    syms = list(prices.columns)
    t0, t1 = prices.index[0], prices.index[-1]
    span = max((t1 - t0).total_seconds(), 1)
    row_h, h = 26, 26 * len(syms) + 30
    out = []
    for r, sym in enumerate(syms):
        y = 10 + r * row_h
        out.append(f'<text x="44" y="{y + 15}" font-size="11" text-anchor="end" fill="var(--muted)">{escape(sym)}</text>'
                   f'<rect x="50" y="{y}" width="{w - 60}" height="{row_h - 6}" fill="none" stroke="var(--line)"/>')
        for t in (x for x in trades if x.symbol == sym):
            a, b = pd.Timestamp(t.entry_date), pd.Timestamp(t.exit_date)
            x0 = 50 + (a - t0).total_seconds() / span * (w - 60)
            x1 = 50 + (b - t0).total_seconds() / span * (w - 60)
            col = "var(--ok)" if t.ret > 0 else "var(--ko)"
            out.append(f'<rect x="{x0:.1f}" y="{y + 1}" width="{max(x1 - x0, 1.5):.1f}" height="{row_h - 8}" fill="{col}" opacity="0.8"/>')
    out.append(f'<text x="50" y="{h - 4}" font-size="11" fill="var(--muted)">{t0.date()}</text>'
               f'<text x="{w - 10}" y="{h - 4}" font-size="11" text-anchor="end" fill="var(--muted)">{t1.date()}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" role="img">{"".join(out)}</svg>'
            '<div class="legend"><span style="color:var(--ok)">&#9632;</span> position investie, trade gagnant &nbsp; '
            '<span style="color:var(--ko)">&#9632;</span> trade perdant</div>')


def bar_chart(labels, values, fmt, w=760, h=200, colors=None) -> str:
    n = len(values)
    hi, lo = max(max(values), 0.0), min(min(values), 0.0)
    rng = (hi - lo) or 1.0
    zero_y = 10 + (hi / rng) * (h - 50)
    bw = (w - 70) / n
    out = [f'<line x1="50" y1="{zero_y:.1f}" x2="{w - 10}" y2="{zero_y:.1f}" stroke="var(--line)"/>']
    for i, (lab, v) in enumerate(zip(labels, values, strict=True)):
        bh = abs(v) / rng * (h - 50)
        y = zero_y - bh if v >= 0 else zero_y
        col = (colors[i] if colors else ("var(--ok)" if v >= 0 else "var(--ko)"))
        x = 55 + i * bw
        out.append(f'<rect x="{x + 2:.1f}" y="{y:.1f}" width="{max(bw - 4, 1):.1f}" height="{max(bh, 0.5):.1f}" fill="{col}" opacity="0.85"/>'
                   f'<text x="{x + bw / 2:.1f}" y="{h - 6}" font-size="10" text-anchor="middle" fill="var(--muted)">{escape(str(lab))}</text>'
                   f'<text x="{x + bw / 2:.1f}" y="{(y - 3) if v >= 0 else (y + bh + 11):.1f}" font-size="10" text-anchor="middle" fill="var(--fg)">{escape(fmt(v))}</text>')
    return f'<svg viewBox="0 0 {w} {h}" role="img">{"".join(out)}</svg>'


def monthly_heatmap(returns: pd.Series) -> str:
    m = (1 + returns).groupby([returns.index.year, returns.index.month]).prod() - 1
    years = sorted({y for y, _ in m.index})
    head = "".join(f"<th>{n}</th>" for n in ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"])
    rows = []
    scale = max(float(m.abs().quantile(0.9)), 0.01)
    for y in years:
        cells = []
        for mo in range(1, 13):
            v = m.get((y, mo))
            if v is None:
                cells.append("<td></td>")
                continue
            a = min(abs(v) / scale, 1.0) * 0.55
            col = f"rgba(22,163,74,{a:.2f})" if v >= 0 else f"rgba(220,38,38,{a:.2f})"
            cells.append(f'<td style="background:{col}">{v:.1%}</td>')
        yr = (1 + m[[k for k in m.index if k[0] == y]]).prod() - 1
        rows.append(f"<tr><th>{y}</th>{''.join(cells)}<td><b>{yr:.1%}</b></td></tr>")
    return f'<table class="heat"><tr><th></th>{head}<th>Année</th></tr>{"".join(rows)}</table>'


def build_report(prices: pd.DataFrame, *, lookback: int = 200, rebalance: str = "M", weighting: str = "equal",
                 cost: CostModel | None = None, funding: pd.DataFrame | None = None,
                 source: str = "", is_synthetic: bool = False, warnings: list[str] | None = None) -> dict:
    cost = cost or CostModel(commission_bps=5.0, spread_bps=1.0, slippage_bps=2.0)
    cfg = BacktestConfig(cost=cost, execution_delay=1, periods_per_year=PPY, risk=RiskLimits())
    strat = TrendFollowing(lookback=lookback, rebalance=rebalance, weighting=weighting)
    target = apply_weight_limits(strat.target_weights(prices), cfg.risk)
    res = simulate(prices, target, cfg)

    funding_cost = pd.Series(0.0, index=prices.index)
    if funding is not None:
        f = funding.reindex(prices.index).fillna(0.0).reindex(columns=prices.columns).fillna(0.0)
        funding_cost = (res.weights.clip(lower=0) * f).sum(axis=1)       # positive funding: longs pay
    net = res.returns - funding_cost

    first_on = (res.exposure > 1e-12).idxmax() if (res.exposure > 1e-12).any() else prices.index[0]
    sl = slice(first_on, None)

    def bh(weights):
        r = simulate(prices, weights, BacktestConfig(cost=cost, execution_delay=1, periods_per_year=PPY,
                                                     risk=RiskLimits(max_position_weight=1.0)))
        return r

    ew = bh(BuyAndHold(rebalance="M").target_weights(prices))
    btc_sym = prices.columns[0]
    btc_w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    btc_w[btc_sym] = 1.0
    btc = bh(btc_w)
    ew_net = ew.returns - (funding_cost * 0 if funding is None else
                           (ew.weights.clip(lower=0) * funding.reindex(prices.index).fillna(0.0).reindex(columns=prices.columns).fillna(0.0)).sum(axis=1))
    btc_net = btc.returns - (0 if funding is None else
                             btc.weights[btc_sym] * funding.reindex(prices.index).fillna(0.0).get(btc_sym, 0.0))

    series = {"Stratégie tendance": net[sl], "Panier équipondéré (acheter et garder)": ew_net[sl],
              f"{btc_sym} (acheter et garder)": btc_net[sl]}
    metrics = {}
    for name, r in series.items():
        src = res if name.startswith("Strat") else (ew if "panier" in name.lower() else btc)
        metrics[name] = compute_metrics(r, src.turnover[sl], None, src.exposure[sl], 0.0, PPY)
    metrics["Stratégie tendance"]["annual_cost_drag"] = float(res.costs[sl].mean() * PPY)
    metrics["Stratégie tendance"]["annual_funding_drag"] = float(funding_cost[sl].mean() * PPY)

    trades = extract_trades(prices, res.weights)
    closed = [t for t in trades if not t.open]
    wins = [t.ret for t in closed if t.ret > 0]
    losses = [t.ret for t in closed if t.ret <= 0]
    tstats = {
        "n_trades": len(trades), "n_closed": len(closed),
        "win_rate": len(wins) / len(closed) if closed else None,
        "avg_win": float(np.mean(wins)) if wins else None, "avg_loss": float(np.mean(losses)) if losses else None,
        "avg_days": float(np.mean([t.days for t in closed])) if closed else None,
        "best": max(trades, key=lambda t: t.ret).__dict__ if trades else None,
        "worst": min(trades, key=lambda t: t.ret).__dict__ if trades else None,
    }

    mult = [0, 1, 2, 3, 5, 10]
    finals = []
    for k in mult:
        r = simulate(prices, target, BacktestConfig(cost=cost.scaled(k), execution_delay=1, periods_per_year=PPY, risk=cfg.risk))
        finals.append(float((1 + (r.returns - funding_cost)[sl]).prod() - 1))

    return {"prices": prices, "target": target, "res": res, "net": net, "series": series, "metrics": metrics,
            "trades": trades, "tstats": tstats, "cost_mult": mult, "cost_finals": finals, "first_on": first_on,
            "params": {"lookback": lookback, "rebalance": rebalance, "weighting": weighting,
                       "cost_one_way_bps": cost.one_way_bps, "funding_included": funding is not None},
            "source": source, "is_synthetic": is_synthetic, "warnings": warnings or [], "sl": sl,
            "dd": {n: (1 + r).cumprod() / (1 + r).cumprod().cummax() - 1 for n, r in series.items()}}


def render_html(rep: dict) -> str:
    prices, res, p = rep["prices"], rep["res"], rep["params"]
    btc = prices.columns[0]
    sma = prices[btc].rolling(p["lookback"], min_periods=p["lookback"]).mean()
    runs = [(pd.Timestamp(t.entry_date), pd.Timestamp(t.exit_date)) for t in rep["trades"] if t.symbol == btc]
    sl = rep["sl"]
    eq = {n: (1 + r).cumprod() for n, r in rep["series"].items()}
    cls = ["c-strat", "c-bh", "c-btc"]
    names = list(eq)
    m = rep["metrics"]

    def row(label, key, kind="num"):
        return f"<tr><td>{label}</td>" + "".join(f"<td>{_fmt(m[n].get(key), kind)}</td>" for n in names) + "</tr>"

    table = ("<table><tr><th></th>" + "".join(f"<th>{escape(n)}</th>" for n in names) + "</tr>"
             + row("Rendement total", "total_return", "pct") + row("Rendement annualisé", "cagr", "pct")
             + row("Volatilité annualisée", "ann_vol", "pct") + row("Sharpe", "sharpe") + row("Sortino", "sortino")
             + row("Pire baisse (drawdown)", "max_drawdown", "pct") + row("Calmar", "calmar")
             + row("Temps investi", "avg_gross_exposure", "pct") + "</table>")
    ts = rep["tstats"]
    banner = ('<div class="banner"><b>Données synthétiques.</b> Aperçu du format du rapport : ces chiffres n\'ont aucune valeur financière.</div>'
              if rep["is_synthetic"] else "")
    warns = "".join(f"<li>{escape(w)}</li>" for w in rep["warnings"])
    cd, fd = m[names[0]].get("annual_cost_drag"), m[names[0]].get("annual_funding_drag")
    top = sorted(rep["trades"], key=lambda t: t.ret)
    pick = top[:3] + top[-3:] if len(top) > 6 else top
    trade_rows = "".join(f"<tr><td>{escape(t.symbol)}</td><td>{t.entry_date}</td><td>{t.exit_date}{' (en cours)' if t.open else ''}</td>"
                         f"<td>{t.days}</td><td>{_fmt(t.ret, 'pct')}</td></tr>" for t in pick)
    hist = np.histogram([t.ret for t in rep["trades"] if not t.open] or [0.0], bins=10)
    hist_lab = [f"{(hist[1][i] + hist[1][i + 1]) / 2:.0%}" for i in range(10)]
    hist_col = ["var(--ok)" if (hist[1][i] + hist[1][i + 1]) / 2 > 0 else "var(--ko)" for i in range(10)]
    exp = res.exposure[sl].rolling(30, min_periods=1).mean()

    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stratégie de tendance sur futures Binance</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>Comment la stratégie de tendance aurait été appliquée</h1>
<p class="sub">Futures perpétuels USDT-M, position longue ou neutre (jamais vendeuse), sans levier. Source : {escape(rep['source'])}.</p>{banner}
<section class="card"><h2>La règle</h2>
<p>Pour chaque actif, on est investi quand son prix de clôture dépasse sa moyenne mobile de <b>{p['lookback']} jours</b> ; sinon on reste en liquidités.
Le capital est découpé en parts égales (1/{prices.shape[1]} par actif). La décision est revue le premier jour de chaque mois ({'mensuelle' if p['rebalance'] == 'M' else p['rebalance']}) et exécutée à la <b>clôture du lendemain</b>.
Coûts supposés : {p['cost_one_way_bps']:.1f} points de base par transaction (hypothèse : frais « taker » standard + glissement, à vérifier sur ta grille de frais).
Frais de financement (funding) : {'inclus' if p['funding_included'] else '<b>NON inclus</b>'}.
Ce n'est <b>qu'un jeu de paramètres fixe</b>, ni optimisé ni validé.</p>
{('<ul class="note">' + warns + '</ul>') if warns else ''}</section>

<section class="card"><h2>1. Quand la règle est investie ({escape(btc)})</h2>
<p class="note">Le prix passe au-dessus de sa moyenne mobile : la règle achète (zone verte). Il repasse dessous : elle vend. Ces signaux arrivent avec retard, c'est le coût de la méthode.</p>
{line_chart([(f"{btc} (échelle logarithmique)", prices[btc][sl], 'c-btc'), (f"Moyenne mobile {p['lookback']} j", sma[sl], 'c-strat')], shade=runs, log=True)}</section>

<section class="card"><h2>2. Quels actifs, quand</h2>
<p class="note">Chaque ligne est un actif. Chaque barre est une période investie, verte si le trade a gagné, rouge sinon.</p>
{timeline_chart(prices[sl], rep['trades'])}</section>

<section class="card"><h2>3. Valeur du capital (base 1)</h2>
{line_chart([(n, eq[n], cls[i]) for i, n in enumerate(names)], log=True)}{table}</section>

<section class="card"><h2>4. Les baisses</h2>
<p class="note">Part du capital perdue depuis le dernier sommet. La tendance cherche à les réduire, pas à battre le marché.</p>
{drawdown_chart([(n, rep['dd'][n], cls[i]) for i, n in enumerate(names)])}</section>

<section class="card"><h2>5. Exposition au marché</h2>
<p class="note">Part du capital investie (moyenne glissante sur 30 jours).</p>
{line_chart([("Exposition", exp, 'c-strat')])}</section>

<section class="card"><h2>6. Rendements mensuels de la stratégie</h2>{monthly_heatmap(rep['series'][names[0]])}</section>

<section class="card"><h2>7. Les trades</h2>
<div class="grid">
<div class="kpi"><b>{ts['n_trades']}</b><span>trades</span></div><div class="kpi"><b>{_fmt(ts['win_rate'], 'pct')}</b><span>trades gagnants</span></div>
<div class="kpi"><b>{_fmt(ts['avg_win'], 'pct')}</b><span>gain moyen</span></div><div class="kpi"><b>{_fmt(ts['avg_loss'], 'pct')}</b><span>perte moyenne</span></div>
<div class="kpi"><b>{_fmt(ts['avg_days'])}</b><span>jours par trade</span></div></div>
<p class="note">Rendements de prix par trade, avant coûts et funding.</p>
{bar_chart(hist_lab, [int(x) for x in hist[0]], lambda v: str(int(v)), colors=hist_col)}
<h3>Trois pires et trois meilleurs trades</h3><table><tr><th>Actif</th><th>Entrée</th><th>Sortie</th><th>Jours</th><th>Rendement</th></tr>{trade_rows}</table></section>

<section class="card"><h2>8. Sensibilité aux coûts</h2>
<p class="note">Rendement total de la stratégie si les coûts étaient multipliés par 0, 1, 2, 3, 5 et 10. Coût actuel : {_fmt(cd, 'pct')} par an ; funding : {_fmt(fd, 'pct')} par an.</p>
{bar_chart([f"x{k}" for k in rep['cost_mult']], rep['cost_finals'], lambda v: f"{v:.0%}")}</section>

<section class="card"><h2>Limites</h2><ul class="note">
<li>Un seul jeu de paramètres, choisi d'avance : aucune validation hors échantillon ici. Pour un verdict, utiliser la recherche walk-forward avec les critères.</li>
<li>Crypto : peu d'historique (depuis 2019-2020), un seul grand cycle haussier, forte corrélation entre actifs, seuls des actifs qui existent encore sont testés (biais de survie).</li>
<li>Pas de levier, donc pas de liquidation modélisée. Risque de plateforme (faillite, gel de retraits) non modélisé.</li>
<li>Vérifie que Binance Futures t'est accessible légalement : les dérivés crypto à effet de levier sont encadrés pour les particuliers dans l'UE.</li>
<li>Information technique, pas un conseil en investissement.</li></ul></section>
</main></body></html>"""


def summary_json(rep: dict) -> str:
    ts = {k: v for k, v in rep["tstats"].items()}
    out = {"params": rep["params"], "source": rep["source"], "is_synthetic": rep["is_synthetic"],
           "period": [str(rep["prices"].index[0].date()), str(rep["prices"].index[-1].date())],
           "metrics": rep["metrics"], "trade_stats": ts, "cost_sensitivity": dict(zip(map(str, rep["cost_mult"]), rep["cost_finals"], strict=True)),
           "warnings": rep["warnings"], "first_invested": str(rep["first_on"].date())}
    return json.dumps(out, indent=2, default=str)
