# Décisions de départ : marché, données, broker, stratégie

> État des informations : vérifiées en octobre 2026 sur des sources publiques (documentation IBKR,
> forum et pages Alpaca, PyPI, articles tiers). Les offres, tarifs et règles de ces acteurs changent :
> **revérifier avant tout engagement**. Les coûts du backtest sont des hypothèses, pas des mesures.
> Ce document n'est pas un conseil en investissement.

## 1. Marché recommandé pour une première version

**ETF liquides multi-classes d'actifs, barres journalières, long-only, sans levier.**

Pourquoi :
- *Données* : historique profond et propre en journalier (prix ajustés dividendes), gratuit pour prototyper.
- *Liquidité et coûts* : spreads de quelques points de base sur les gros ETF → l'hypothèse de coûts est
  crédible et un test de stress ×2 la borne.
- *Coûts maîtrisés* : à basse fréquence (rebalancement mensuel) la rotation reste faible, donc les coûts pèsent beaucoup moins
  qu'en intraday.
- *Complexité* : pas de carnet d'ordres, pas de latence, pas de marge/roll de futures, pas de borrow.
- *Univers* : 8 ETF (actions US, actions développées hors US, émergents, obligations 7-10 ans et 20 ans+, or,
  matières premières, immobilier coté) = diversification réelle, mais assez petit pour comprendre chaque ligne.

Ce que ce choix **n'est pas** : une promesse de rentabilité. Une stratégie de tendance sur ETF est un
bon terrain pédagogique et un bon test du pipeline ; l'edge, s'il existe, est modeste et instable.

### Contrainte réglementaire à connaître (résidents de l'UE)

D'après la documentation d'Interactive Brokers, le règlement PRIIPs impose un document d'information (KID) pour
vendre un ETF à un client de détail de l'EEE ; les ETF cotés aux États-Unis n'en ont en général pas, donc un
client de détail de l'EEE **ne peut pas les acheter** (alternatives : équivalents UCITS européens, ou CFD, ou statut
professionnel, avec perte de protections). Conséquences pour ce projet :

- Le backtest utilise des **ETF US comme proxys de recherche** (historique long) ;
- l'exécution devra passer par des **équivalents UCITS** (Euronext Paris, Xetra, LSE...) dont il faut **re-tester**
  l'historique (plus court, devise, spreads, calendrier) ;
- il y a un **risque de change** (proxys en USD) et un **risque de suivi** proxy → UCITS à mesurer.

## 2. Fournisseurs de données

| Option | Atouts | Limites | Usage conseillé |
|---|---|---|---|
| Yahoo Finance via `yfinance` | Gratuit, prix ajustés, très simple | Non officiel, endpoints non documentés qui peuvent changer ou être limités sans préavis ; usage personnel seulement | Prototype. **Toujours figer en CSV** (`quantlab fetch`) |
| EODHD | Large couverture (actions, ETF, bourses européennes), prix d'entrée annoncés autour de 20 $/mois | Payant (essai court) ; vérifier les droits d'usage | Candidat n°1 pour passer de Yahoo à une source contractuelle |
| Tiingo / Alpha Vantage / Twelve Data / Finnhub | Offres gratuites ou bon marché, API documentées | Profondeur historique ou volume gratuit limités selon l'offre | À comparer sur l'univers UCITS réel |
| Données du broker (IBKR) | Même source que l'exécution | Limites de requêtes, abonnements de marché, historique parfois moins pratique | Contrôle croisé et données live |

Décision : **Yahoo pour amorcer, snapshot CSV versionné (empreinte dans chaque run), puis une source
payante contractuelle avant tout paper trading sérieux.** Biais à surveiller : survivance (on ne teste que des ETF
qui existent aujourd'hui) et ajustements (dividendes/splits).

## 3. Brokers

| Option | Atouts | Limites / points à vérifier |
|---|---|---|
| **Interactive Brokers** (API TWS/Gateway, bibliothèque Python communautaire) | Accès à Euronext/Xetra/LSE et aux ETF UCITS ; API complète (ordres, annulation, statut, positions) ; compte papier | Nécessite TWS ou IB Gateway ouvert ; API plus lourde ; règles PRIIPs ; **grille de commissions à relire** (non reprise ici) ; vérifier l'éligibilité PEA/CTO |
| **Alpaca** | API moderne, paper trading ouvert à tous | Orienté actions/ETF US ; d'après leurs pages, comptes live hors US ouverts progressivement avec KYC et dépôt réduit, mais **statut réglementaire en France à vérifier** (non régulé AMF d'après un comparateur) ; ETF UCITS non couverts |
| Courtiers français « classiques » | Fiscalité simple (PEA) | Peu ou pas d'API de trading |

Décision : **Interactive Brokers comme cible d'exécution** (cohérent avec l'univers UCITS), via une classe qui implémente
`BrokerAdapter`. D'ici là : `PaperBroker` interne. Alpaca peut servir de bac à sable d'API pour tester du code, pas de cible
pour des ETF UCITS.

## 4. Stratégies de départ (simples, interprétables, comparables)

1. **Tendance SMA** (`trend_sma`) : investi si prix > moyenne mobile (100 à 250 j), sinon cash ; 1/8 du capital par actif.
2. **Momentum transversal + filtre absolu** (`xs_momentum`) : top-k des rendements 6-12 mois (dernier mois exclu), uniquement si > 0.
3. **Benchmark** : portefeuille équipondéré rééquilibré mensuellement (même moteur, mêmes coûts, même retard).

Pas de machine learning, pas de pairs trading ni de long/short à ce stade : chacun ajoute des biais (emprunt, financement,
sélection) et des données à valider avant d'avoir prouvé que le pipeline est fiable.

## 5. Hypothèses de coûts (à calibrer)

Par transaction (aller simple) : commission 2 bps + demi-spread 1,5 bp + slippage 2 bps = **5,5 bps** ; stress ×2 dans les gates.
Si les commissions réelles ont un minimum fixe par ordre, l'effet dépend de la taille du compte : **à modéliser avec ton capital réel**.

## 6. Gates (à figer avant de regarder les résultats réels)

Fenêtre OOS ≥ 5 ans · Sharpe OOS ≥ 0,4 · Sharpe déflaté ≥ 0,95 (nombre d'essais cumulé) · PSR contre le benchmark ≥ 0,95 ·
drawdown OOS ≤ 35 % · Sharpe > 0 avec coûts ×2. Valeurs initiales **arbitraires** : discute-les maintenant, puis ne les
ajuste pas après coup. Réussir = « éligible au paper trading », jamais « rentable ».

## 7. Feuille de route

| Phase | Contenu | Critère de sortie |
|---|---|---|
| 0 (fait) | Pipeline recherche → backtest → métriques → sauvegarde, tests, garde-fous | Tests verts ; tests de contrôle sur données synthétiques (rejet sur marche aléatoire, détection d'un edge planté) |
| 1 | `quantlab fetch` sur l'univers réel ; run réel ; lecture critique (régimes, coûts, sensibilité) | Résultats reproduits depuis les CSV figés |
| 2 | Source de données contractuelle ; mapping UCITS ; re-test ; analyse de change | Écart proxy/UCITS mesuré |
| 3 | Adaptateur IBKR **paper** ; runner quotidien ; réconciliation ; alertes | ≥ 3 mois de paper sans écart non expliqué |
| 4 | Exécution supervisée : ordres proposés, validation humaine (`require_approval`) | Journal d'ordres propre, coûts réels vs modèle |
| 5 | Agent de recherche autonome (hypothèse → code → backtest → décision), gates imposées par le code | Compteur d'essais respecté, aucun contournement |
| 6 | Automatisation contrôlée, petits montants, limites strictes, arrêt d'urgence | Décision humaine explicite et écrite |

## 8. Limites connues du prototype

- Aucune donnée réelle n'a été testée dans cet environnement (accès aux fournisseurs de données impossible) : **aucun résultat de performance réel n'est annoncé**.
- Barres journalières, exécution à la clôture suivante : pas de modèle d'impact de marché ni de carnet d'ordres.
- Coûts proportionnels uniquement (pas de minimum par ordre), pas de fiscalité, pas de change.
- Le test de détection du look-ahead est empirique (troncature) : il attrape les fuites courantes, pas toutes.
- Les gates fixent un compromis faux positifs / puissance ; sur marche aléatoire synthétique elles n'ont rien laissé passer (60 tirages), et sur un edge planté la détection n'est que partielle : un rejet ne prouve pas l'absence d'edge.
