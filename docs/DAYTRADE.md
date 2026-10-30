# Day trading sur données journalières

## Ce que c'est

Un trade = le rendement d'une séance : on **achète à l'ouverture, on vend à la clôture**, aucune position la nuit.
La décision est prise **avant l'ouverture** avec les informations de la veille (et le prix d'ouverture pour les règles de gap).
Le code est dans `src/quantlab/daytrade.py` ; la commande est `quantlab daytrade`.

## Ce que les barres journalières ne permettent pas

Pas d'ordres stop, pas d'objectifs de gain, pas de « range d'ouverture », pas de logique intra-séance : on connaît le plus haut et
le plus bas du jour, pas **dans quel ordre** ils ont eu lieu. Pour cela il faut des données à la minute (étape suivante, non faite).
Les prix d'ouverture et de clôture sont ceux des enchères officielles : on ne les obtient pas exactement. Chaque côté d'un trade
paie donc 2 pb en plus des coûts habituels (paramètre `daytrade.open_close_extra_bps`). Les règles de gap sont les plus optimistes :
elles supposent d'acheter près de l'ouverture après l'avoir vue.

## Les trois hypothèses (enregistrées avant tout test, voir `docs/JOURNAL.md`)

| | Règle | Pourquoi on l'essaie | Paramètres |
|---|---|---|---|
| H1 `intraday_trend` | Acheter à l'ouverture si le cours de la veille est au-dessus de sa moyenne mobile | C'est le filtre de tendance déjà utilisé, exposé à la seule partie « séance » | moyenne 100, 150, 200, 250 j |
| H2 `gap_fade` | En tendance haussière, acheter après un gap baissier supérieur à k écarts-types, vendre à la clôture | Surréaction à l'ouverture qui se corrige dans la séance | k = 0,5 ; 1 ; 1,5 |
| H3 `ibs_reversal` | En tendance haussière, acheter après une clôture proche du plus bas du jour | Rebond de court terme | seuil 0,1 ; 0,2 ; 0,3 |

10 jeux de paramètres au total. Long ou neutre, pas de vente à découvert, pas de levier.

## Lire les résultats

- **Section 1 du rapport : séance contre nuit.** Où se fait le rendement de chaque actif : entre l'ouverture et la clôture, ou pendant la nuit ?
  C'est le premier test de bon sens avant tout day trading.
- **Edge contre coût.** « Edge » = rendement moyen **brut** par trade (en points de base). « Coût » = coût d'un aller-retour (15 pb avec
  les hypothèses actuelles). Une règle qui trade tous les jours dépense environ 15 pb × 250 jours par an en coûts ; il faut un edge nettement supérieur.
- **« Trader naïf »** (courbe rouge) : acheter chaque jour à l'ouverture, vendre à la clôture. C'est ce que fait un day trader sans edge ; la courbe montre l'effet des seuls coûts.
- **Verdict** : mêmes critères que l'étude journalière, évalués avec le nombre total d'essais. `PASS_TO_PAPER` veut dire éligible au paper trading, rien d'autre.

## Choix de méthode fixé avant les résultats réels

Le Sharpe déflaté utilise la dispersion des Sharpe **à l'intérieur de chaque famille** de paramètres (moyenne des familles) et non celle de tous
les essais mélangés : l'écart de coûts entre une règle qui trade tous les jours et une qui trade rarement gonflerait la variance et pénaliserait à tort
les bons candidats. Vérifié sur données synthétiques : 0 faux positif sur 40 marchés aléatoires, détection d'un effet de gap planté dans 15 cas sur 15.

## Lancer

```bash
quantlab fetch --config configs/etf_daytrade.yaml     # snapshot Yahoo OUVERTURE/HAUT/BAS/CLÔTURE (jusqu'à fin 2023, période verrouillée)
quantlab daytrade --config configs/etf_daytrade.yaml  # walk-forward, critères, rapport report/daytrade.html
```

Les anciens fichiers de `data/raw` ne contiennent que « Adj Close » : refais `quantlab fetch`. Sur GitHub : onglet Actions → `daytrade-research` → Run workflow.
Le compteur d'essais est lu dans `runs/index.jsonl` (versionné) ; après un run sur GitHub, copie le `runs/index.jsonl` de l'artefact dans le dépôt pour que les essais comptent.

## Limites

ETF américains comme proxys (un particulier de l'EEE ne peut en général pas les acheter, règle PRIIPs ; les produits accessibles, ETF UCITS ou CFD, ont
d'autres horaires, spreads et frais). Peu de trades pour certaines règles : chiffres fragiles. Aucun résultat réel n'a été produit au moment de l'écriture.
Information technique, pas un conseil en investissement.
