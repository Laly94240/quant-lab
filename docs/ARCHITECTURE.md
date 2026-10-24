# Architecture

## Principe directeur

**L'IA propose, le code déterministe décide.** L'agent (LLM) intervient sur la recherche : formuler des
hypothèses, écrire/modifier du code de stratégie, lancer des expériences, analyser les résultats.
Il n'a **aucun chemin** vers un ordre réel : les contrôles de risque et l'exécution sont du logiciel
classique, testé, qui ne dépend pas d'une décision générée.

## Choix : application Python modulaire + agent connecté par des outils (pas un plugin)

| Option | Pour | Contre | Verdict |
|---|---|---|---|
| Plugin dans une plateforme existante | Démarrage rapide | Dépendance à la plateforme, tests et reproductibilité difficiles, contrôle limité du risque | Non |
| Application Python autonome (CLI + librairie) | Testable, reproductible, versionnée, portable | Il faut construire l'infrastructure | **Base** |
| Agent LLM qui appelle un moteur Python | Recherche autonome (écrire du code, lancer, analyser) | Risque de sur-optimisation par l'agent s'il n'est pas contraint | **Couche ajoutée en phase 4**, avec gates imposées par le code |
| Workflow GitHub (CI) | Historique, revue de code, tests automatiques | Pas d'exécution live | **Oui**, autour du dépôt |

## Composants et frontières

```
 data/            ingestion + validation (DataProvider: CSV, Yahoo, Synthetic)
   │  prix ajustés propres + rapport qualité + empreinte (hash)
 strategies/      Strategy.target_weights(prices) : poids cibles, SANS regard vers le futur
   │  (check_no_lookahead : contrôle automatique par troncature)
 risk/            limites déterministes (poids, levier, perte jour, drawdown, données périmées)
   │  mêmes limites en backtest et en exécution
 backtest/        engine (délai d'exécution, dérive des poids, coûts), metrics, walk-forward,
   │              stats (PSR/DSR), gates (critères pré-enregistrés)
 experiments/     historique append-only : config, hash de code et de données, résultats, nb d'essais
 pipeline.py      orchestration de bout en bout ; cli.py : `quantlab fetch | run`
 execution/       Order, génération d'ordres, BrokerAdapter, PaperBroker, NotifyingBroker, garde-fous live
 notify/          Discord (webhook) : notifications d'ordres, d'arrêts de sécurité et de briefings
 news/            RSS / X, nettoyage, briefing LLM validé, surcouche « réduction seule » (conseil uniquement)
 paper.py         un pas quotidien de paper trading, état persisté, arrêt de sécurité
 report.py        rapport HTML statique (recherche + paper trading)
```

Règle de dépendance : `execution` → `risk`, `backtest.engine` (coûts) ; `strategies` ne connaît ni le
broker ni les coûts ; `data` ne connaît personne. Un adaptateur broker réel (ex. IBKR) se branche en
implémentant `BrokerAdapter` sans toucher au reste.

## Flux de données d'un run de recherche

1. `DataProvider.get_prices` → `validate_prices` (erreurs bloquantes, avertissements tracés).
2. `check_no_lookahead` sur chaque jeu de paramètres (précondition dure).
3. `walk_forward` : sélection des paramètres sur l'entraînement **uniquement**, poids de test assemblés,
   **une** simulation avec coûts (les coûts de changement de paramètres sont comptés).
4. Benchmark simulé avec les mêmes règles (retard, coûts) sur la même fenêtre OOS.
5. `evaluate_candidate` : Sharpe OOS, Sharpe déflaté (nombre d'essais cumulé), PSR contre le benchmark,
   drawdown, test de stress des coûts (×2) → `REJECT` ou `PASS_TO_PAPER`.
6. `ExperimentStore.save` : tout est conservé ; une expérience identique n'est pas relancée.

## Modes de défaillance traités

| Défaillance | Parade |
|---|---|
| Regard vers le futur dans un signal | `check_no_lookahead` (troncature) ; retard d'exécution par défaut = 1 barre |
| Données corrompues / manquantes | `validate_prices` : doublons, prix ≤ 0, trous > 5 barres, rendements extrêmes (splits), séries figées |
| Surapprentissage / tests multiples | Walk-forward ; compteur d'essais cumulé par `research_id` ; DSR ; gates fixées avant |
| Coûts sous-estimés | Spread + commission + slippage explicites ; stress ×2 ; rotation annuelle reportée |
| Ordre dupliqué | `client_order_id` déterministe + idempotence du broker |
| Ordre supposé exécuté | Statut `SUBMITTED` jusqu'à confirmation ; `reconcile` des positions |
| Trading réel involontaire | Désactivé par défaut ; double confirmation (config **et** variable d'environnement) |
| Fuite de secrets | Aucun secret dans le code ; `.env` ignoré par git ; `.env.example` fourni |

## Ce qui n'existe pas encore (à ne pas supposer)

- Aucun adaptateur broker réel (IBKR ou autre) : non écrit, non testé contre une vraie API.
- Le runner de paper trading est un pas quotidien à lancer (cron local) ; pas de planification ni de supervision intégrées.
- Le briefing LLM, X, les flux RSS et le webhook Discord n'ont pas été testés contre les vrais services (réponses simulées).
- L'agent de recherche autonome (hypothèse → code → backtest) n'existe pas : seule la couche « news » est conseillère.
