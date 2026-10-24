# quantlab

Cadre modulaire pour **rechercher, backtester et valider** des stratégies systématiques, avec des
garde-fous d'exécution déterministes. État : **prototype de la phase 0** (pipeline de recherche complet,
exécution limitée à un broker papier interne).

> Le trading réel est **désactivé par défaut** et n'est branché sur aucun broker. Aucune performance réelle n'est
> annoncée : le prototype n'a été exécuté que sur des données synthétiques de contrôle.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # + ".[yahoo]" pour télécharger depuis Yahoo
pytest -q                         # 55 tests
```

## Démarrage rapide

```bash
# 1. Vérifier que le pipeline est sain (données SYNTHÉTIQUES, aucun sens financier)
quantlab run --config configs/etf_trend.yaml --provider synthetic --scenario random_walk --seed 5
quantlab run --config configs/etf_trend.yaml --provider synthetic --scenario regime      --seed 5

# 2. Données réelles : figer un snapshot CSV (une seule fois), puis rejouer hors-ligne
pip install -e ".[yahoo]"
quantlab fetch --config configs/etf_trend.yaml      # écrit data/raw/<TICKER>.csv
quantlab run   --config configs/etf_trend.yaml      # provider csv par défaut
```

Sortie : tableau OOS (Sharpe, CAGR, drawdown, rotation, Sharpe déflaté, PSR vs benchmark) et un verdict
`REJECT` / `PASS_TO_PAPER`. Chaque run est sauvegardé dans `runs/<id>/` (config, manifeste avec hash git et hash des
données, rapport, rendements OOS). Un run identique n'est pas relancé (`--force` pour l'imposer).

## Lire un résultat avec méfiance

- `PASS_TO_PAPER` veut dire « mérite d'être suivi en paper trading », jamais « rentable ».
- Le Sharpe déflaté tient compte de **tous** les jeux de paramètres déjà essayés sous le même `research_id` : changer
  `research_id` pour repartir de zéro revient à tricher.
- Modifie les gates **avant** de lancer sur données réelles, pas après.

## Structure

```
configs/          configuration YAML (univers, coûts, risque, walk-forward, gates, stratégies)
docs/             ARCHITECTURE.md, DECISIONS.md (marché, données, brokers, feuille de route)
src/quantlab/
  data/           providers CSV / Yahoo / synthétique, validation qualité
  strategies/     contrat anti look-ahead, tendance SMA, momentum, benchmark
  risk/           limites de poids, levier, perte jour, drawdown, données périmées
  backtest/       moteur (retard d'exécution, dérive, coûts), métriques, walk-forward, PSR/DSR, gates
  experiments/    historique append-only des runs
  execution/      ordres, BrokerAdapter, PaperBroker, garde-fous live, validation humaine
  pipeline.py cli.py
tests/            55 tests (moteur chiffré à la main, absence de fuite, contrôle des faux positifs, ...)
```

## Ajouter une stratégie

1. Créer une classe `Strategy` avec `target_weights(prices)` (la ligne *t* n'utilise que `prices[: t+1]`).
2. L'enregistrer dans `strategies/__init__.py` (`REGISTRY`).
3. L'ajouter dans `configs/*.yaml` avec sa `grid`. Le test anti look-ahead est lancé automatiquement.

## Secrets

Jamais dans le code. Variables d'environnement ou gestionnaire de secrets ; `.env` est ignoré par git (voir `.env.example`).

## Limites connues

Voir `docs/DECISIONS.md` §8. En résumé : pas de données réelles testées ici, barres journalières, coûts proportionnels,
pas de fiscalité ni de change, aucun adaptateur broker réel.
