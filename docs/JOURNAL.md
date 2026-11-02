# Journal de recherche (etf_gtaa_daily_v1)

## 2026-10-25 - Mise en place et premier passage
- Période verrouillée : data.end = 2023-12-31 dans configs/etf_trend.yaml. Les données 2024 et après ne sont pas lues pendant le développement.
- Critères (gates) inchangés par rapport au fichier d'origine : min_oos_sharpe 0.4, min_dsr 0.95, max_drawdown 0.35, min_psr_vs_benchmark 0.95.
- Premier passage sur 2007-01-03 -> 2023-12-29 : trend_sma (Sharpe OOS 0.53, DSR 0.92) et xs_momentum (0.37, DSR 0.79), toutes deux REJECT. Panier passif : Sharpe 0.53.
- Compteur d'essais cumulé : 8 (4 par stratégie). Aucun critère ni paramètre modifié après lecture des résultats.
- Résultat enregistré dans docs/RESULT_01.txt. Le compteur (runs/index.jsonl) est désormais versionné dans Git.

## 2026-10-28 - Hypothèses de day trading sur barres journalières (enregistrées avant tout test)
- Question (research_id etf_daytrade_daily_v1) : des règles simples, exécutées uniquement entre l'ouverture et la clôture (aucune position la nuit), battent-elles le panier passif après coûts de day trading, hors échantillon ?
- Données : mêmes 8 ETF ; data.end = 2023-12-31 (période verrouillée). Rappel d'honnêteté : l'étude journalière a été lue jusqu'en 2026-10 avant le verrouillage, donc la période 2024+ n'est pas vierge pour ces actifs.
- H1 intraday_trend : le filtre de tendance existant, exposé à la seule séance. Grille : moyenne mobile 100, 150, 200, 250.
- H2 gap_fade : en tendance haussière, achat après un gap baissier supérieur à k écarts-types (k = 0,5 ; 1,0 ; 1,5), vente à la clôture (surréaction à l'ouverture).
- H3 ibs_reversal : en tendance haussière, achat après une clôture proche du plus bas du jour (seuil 0,1 ; 0,2 ; 0,3).
- 10 essais au total. Coûts : 5,5 pb par côté + 2 pb à l'ouverture/clôture, soit 15 pb l'aller-retour. Critères (gates) identiques à l'étude journalière.
- Choix de méthode fixé avant les résultats réels : variance des Sharpe estimée par famille (moyenne), pas sur tous les essais mélangés. Contrôle sur données synthétiques : 0 faux positif sur 40 marchés aléatoires, détection d'un effet de gap planté 15 fois sur 15.
- Résultat réel : aucun à ce jour.


## 2026-10-30 - Intraday futures crypto : protocole et hypothèses (enregistrés avant tout test)
- Question (research_id crypto_intraday_v1) : des règles intraday sur futures perpétuels Binance (BTCUSDT, ETHUSDT, barres de 15 minutes) ont-elles un gain net par trade positif, avec au moins 51 % de trades gagnants, validé hors échantillon ?
- Blocs : développement 2020-01 → 2023-12-31 ; validation 2024-01-01 → 2025-10-31 ; dernière année (lockbox) 2025-11-01 → dernière donnée, utilisée une seule fois pour le gagnant.
- Demande initiale : itérer jusqu'à la meilleure stratégie d'après le test de la dernière année. Écartée : un test qui sert à choisir n'est plus un test. Remplacée par une validation intermédiaire.
- M1 open_momentum (x = 2, 4, 6, 8 h ; z = 0, 0,5), M2 hour_reversal (k = 2, 3 ; n = 1, 2, 4 h), M3 range_breakout (n = 6, 12, 24 h ; h = 1, 2, 4 h) : 23 jeux par symbole, 46 au total.
- Critères : 100 trades par an de bloc, trades gagnants ≥ 51 %, gain net > 0, facteur de profit ≥ 1,10, Sharpe déflaté ≥ 0,95, baisse maximale ≤ 30 %, gain net > 0 avec coûts doublés. Coûts : 5 pb de frais et 1 pb de glissement par côté.
- Choix de méthode fixé avant les résultats réels : variance du Sharpe sous hypothèse nulle prise égale à 1 / (nombre de jours du bloc), pas la dispersion observée entre jeux de paramètres. Contrôle synthétique : 0 faux gagnant sur 12 marchés aléatoires, effet planté passé 6 fois sur 6.
- Résultat réel : aucun à ce jour.
