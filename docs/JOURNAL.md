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
