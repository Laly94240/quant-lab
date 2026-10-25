# Journal de recherche (etf_gtaa_daily_v1)

## 2026-10-25 - Mise en place et premier passage
- Période verrouillée : data.end = 2023-12-31 dans configs/etf_trend.yaml. Les données 2024 et après ne sont pas lues pendant le développement.
- Critères (gates) inchangés par rapport au fichier d'origine : min_oos_sharpe 0.4, min_dsr 0.95, max_drawdown 0.35, min_psr_vs_benchmark 0.95.
- Premier passage sur 2007-01-03 -> 2023-12-29 : trend_sma (Sharpe OOS 0.53, DSR 0.92) et xs_momentum (0.37, DSR 0.79), toutes deux REJECT. Panier passif : Sharpe 0.53.
- Compteur d'essais cumulé : 8 (4 par stratégie). Aucun critère ni paramètre modifié après lecture des résultats.
- Résultat enregistré dans docs/RESULT_01.txt. Le compteur (runs/index.jsonl) est désormais versionné dans Git.