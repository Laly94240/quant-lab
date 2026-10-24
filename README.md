# quantlab

Cadre modulaire pour **rechercher, valider, suivre en paper trading** des stratégies systématiques, avec garde-fous
déterministes, notifications Discord et une couche « news » **conseillère** (jamais exécutante).

> **Le trading réel est désactivé par défaut et n'est branché sur aucun broker.** Aucune performance réelle n'est annoncée :
> le dépôt n'a été exécuté que sur des données synthétiques de contrôle. « Éligible au paper trading » ne veut pas dire « rentable ».

## Ce que contient le dépôt

| Brique | Rôle | État |
|---|---|---|
| Recherche (`quantlab run`) | Walk-forward, coûts, Sharpe déflaté, gates, historique des essais | testé (données synthétiques) |
| Rapport (`quantlab report`) | Page HTML statique : résultats, 5 dernières années, jours/mois gagnants, paper trading | testé (structure et échappement) |
| Paper trading (`quantlab paper-run`) | Un pas quotidien, argent fictif, arrêt de sécurité | testé (données synthétiques) |
| Discord (`quantlab notify-test`) | Message à chaque ordre soumis / exécuté / rejeté / annulé, et à chaque arrêt de sécurité | testé avec transport simulé |
| News (`quantlab news-brief`) | RSS + X → briefing validé → Discord, **sans pouvoir d'ordre** | testé avec réponses simulées |
| Broker réel | — | **non écrit** |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,yahoo]"
pytest -q                          # ~100 tests
```

## 1. Tester sur de vraies données Yahoo

```bash
quantlab fetch --config configs/etf_trend.yaml       # snapshot CSV dans data/raw/ (une fois)
quantlab run   --config configs/etf_trend.yaml       # walk-forward + gates + sauvegarde dans runs/
quantlab report --runs runs --out report/index.html  # ouvre report/index.html
```

Sans rien installer en local : onglet **Actions → research-real-data → Run workflow** sur GitHub (les serveurs de GitHub ont
accès à Internet). Le résumé s'affiche dans la page du run et le rapport HTML est téléchargeable. Yahoo peut limiter les adresses
partagées des serveurs de CI (erreur 429) : dans ce cas, lance `quantlab fetch` chez toi.

La sortie contient, pour la période hors échantillon **et** pour les 5 dernières années : Sharpe, CAGR, drawdown, **jours gagnants,
mois gagnants**, rotation, Sharpe déflaté, PSR contre benchmark. Voir `docs/DECISIONS.md` §9 pour ce que « plus de 50 % » peut
signifier et pourquoi il n'est pas garanti.

## 2. Notifications Discord

1. Discord : *Paramètres du salon → Intégrations → Webhooks → Nouveau webhook*, copie l'URL.
2. `export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/…"` (secret : jamais dans le code, `.env` est ignoré par git).
3. `quantlab notify-test` puis `quantlab paper-run --config configs/etf_trend.yaml`.

Chaque message indique clairement `[PAPER]` (simulation) ou `[LIVE]` (ordre réel, indisponible dans cette version).

## 3. Paper trading quotidien

```bash
quantlab paper-run --config configs/etf_trend.yaml    # à lancer après la clôture, par ex. via cron
```

Les ordres décidés à la clôture du jour D sont exécutés à la clôture de D+1 (même convention que le backtest). Données invalides ou
périmées, perte journalière ou drawdown excessifs : aucun ordre, alerte Discord ; l'arrêt de sécurité reste actif jusqu'à
`--clear-halt` (décision humaine).

## 4. Agent news (X + RSS), conseil uniquement

Voir `docs/NEWS_AGENT.md` : défenses contre l'injection de prompt, coût de l'API X, surcouche « réduction seule » (désactivée).

## Publier sur GitHub

```bash
git remote add origin git@github.com:<toi>/quant-lab.git     # ou : gh repo create quant-lab --private --source=. --push
git push -u origin main
```

Dépôt **privé** conseillé. Ajoute les secrets éventuels dans *Settings → Secrets and variables → Actions* ; ne commite jamais
`.env`, `data/raw/` (conditions d'usage de Yahoo : usage personnel) ni `paper/`.

## Structure

```
configs/          configuration YAML (univers, coûts, risque, walk-forward, gates, stratégies, paper, news)
docs/             ARCHITECTURE.md, DECISIONS.md, NEWS_AGENT.md
src/quantlab/
  data/ strategies/ risk/ backtest/ experiments/   recherche
  execution/      ordres, BrokerAdapter, PaperBroker, NotifyingBroker, garde-fous live
  notify/         Discord
  news/           RSS, X, nettoyage, briefing LLM validé, surcouche réduction seule
  paper.py report.py pipeline.py cli.py
tests/
.github/workflows/  ci.yml (tests + lint), research.yml (recherche sur données réelles, manuel)
```

## Limites connues

Pas de données réelles testées ici ; barres journalières ; coûts proportionnels ; pas de fiscalité ni de change ; aucun
adaptateur broker réel ; X, RSS, Anthropic et Discord testés avec des réponses simulées, pas contre les vrais services.
Information technique, pas un conseil en investissement.
