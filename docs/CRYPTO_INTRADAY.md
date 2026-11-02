# Recherche intraday sur futures crypto (Binance USD-M)

## Le protocole en trois blocs

| Bloc | Période (config) | Usage |
|---|---|---|
| Développement | 2020-01 → 2023-12-31 | Exploration libre. Chaque jeu de paramètres essayé est compté (`runs/trial_registry.jsonl`). |
| Validation | 2024-01-01 → 2025-10-31 | Seuls les finalistes du développement (3 au plus). Le meilleur qui passe les critères devient le **gagnant**. |
| Dernière année (lockbox) | 2025-11-01 → dernière donnée | Le gagnant, **une seule fois**. Une deuxième consultation est refusée (`runs/lockbox_used.json`). |

Pourquoi trois blocs et pas deux : si on itère jusqu'à trouver « la meilleure stratégie d'après le test » de la dernière année, ce test sert à choisir et n'est plus un test.
Sur un an, l'erreur d'un Sharpe est d'environ 1 : le meilleur de 100 essais affiche souvent un Sharpe de 2 ou plus par pur hasard. La validation intermédiaire sert à
choisir « la meilleure d'après les données », la dernière année à confirmer une fois.

## Ce que fait chaque stratégie

Barres de 15 minutes, 24h/24. Décision à la **clôture** d'une barre, exécution à l'**ouverture de la suivante**, jamais au prix du signal. Un trade utilise tout le capital sans levier,
les trades ne se chevauchent pas, et chacun paie les coûts (5 pb de frais + 1 pb de glissement par côté, soit 12 pb l'aller-retour, à vérifier sur ton compte).

| | Règle | Grille |
|---|---|---|
| M1 `open_momentum` | Le mouvement des x premières heures du jour UTC se poursuit jusqu'à la fin du jour | x = 2, 4, 6, 8 h ; seuil z = 0 ou 0,5 |
| M2 `hour_reversal` | Un mouvement horaire de plus de k écarts-types est « corrigé » pendant n heures | k = 2 ou 3 ; n = 1, 2, 4 h |
| M3 `range_breakout` | La cassure du range des n dernières heures est suivie pendant h heures | n = 6, 12, 24 h ; h = 1, 2, 4 h |

23 jeux de paramètres par symbole (BTCUSDT, ETHUSDT). Un test automatique vérifie qu'aucun signal n'utilise une information postérieure.

## Les critères (identiques dans chaque bloc)

Au moins 100 trades par an de bloc · **trades gagnants ≥ 51 %** · gain net moyen par trade > 0 · facteur de profit ≥ 1,10 · Sharpe déflaté ≥ 0,95 (tient compte de tous les essais) ·
baisse maximale ≤ 30 % · gain net > 0 avec coûts doublés.

Le taux de trades gagnants est un critère, pas l'objectif. On l'obtient trivialement avec de petits gains et de grosses pertes (60 % de trades gagnants peuvent perdre de l'argent) :
le critère de gain net par trade et le facteur de profit protègent de ce piège.

## Itérer sans se tromper

- On peut **ajouter des hypothèses** en développement : les écrire d'abord dans `docs/JOURNAL.md` avec leur justification, les ajouter dans `configs/crypto_intraday.yaml`, relancer. Chaque nouveau jeu augmente le nombre d'essais et durcit le Sharpe déflaté.
- Si aucun candidat ne passe le développement, on s'arrête : personne n'est promu, la dernière année n'est pas touchée. Ce résultat est normal et fréquent.
- Si le gagnant échoue sur la dernière année, ce test est consommé : on n'itère pas dessus. La suite est du paper trading, qui produit de vraies données neuves.
- Modifier les critères est possible entre deux séries, jamais après avoir vu un résultat, et chaque modification est notée au journal.

## Lancer

```bash
quantlab crypto-intraday --config configs/crypto_intraday.yaml --through dev        # développement seul
quantlab crypto-intraday --config configs/crypto_intraday.yaml --through validate   # + validation (défaut)
quantlab crypto-intraday --config configs/crypto_intraday.yaml --through lockbox    # + dernière année, UNE fois
quantlab crypto-intraday --config configs/crypto_intraday.yaml --synthetic --scenario open_momentum   # aperçu sur fausses données
```

Sur GitHub : Actions → `crypto-intraday` → Run workflow (`through` = `dev` ou `validate` ; pour `lockbox`, taper `OPEN_LOCKBOX_ONCE` dans `confirm`). Après un run, copie
`runs/trial_registry.jsonl` (et `runs/lockbox_used.json` si la dernière année a été ouverte) depuis l'artefact dans le dépôt : sinon le compteur d'essais et le verrou repartent de zéro.
Le verrou de la dernière année repose sur ce fichier et sur ta discipline : rien n'empêche techniquement de supprimer le fichier.

## Limites

- Aucun résultat réel au moment de l'écriture : le code est testé sur des données synthétiques (0 faux gagnant sur 12 marchés aléatoires, effet planté détecté 6 fois sur 6). Les fichiers Binance n'ont pas été téléchargés par mes outils.
- Le financement (funding) n'est pas modélisé : faible sur quelques heures, non nul. Le glissement réel dépend de la taille et du moment.
- Un seul grand cycle crypto depuis 2020, deux actifs très corrélés, risque de plateforme (faillite, gel des retraits) non modélisé.
- Les futures crypto à effet de levier sont encadrés pour les particuliers de l'Union européenne : vérifie que Binance Futures t'est accessible légalement avant tout passage en réel.
- Passer le protocole veut dire « éligible au paper trading ». Information technique, pas un conseil en investissement.
