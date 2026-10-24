# Agent « news » (X + flux RSS) : ce qu'il fait, et ce qu'il ne peut pas faire

## Principe

L'agent lit des actualités (flux RSS ; X/Twitter en option), demande à un LLM un **briefing structuré**, et
envoie ce briefing sur Discord **pour décision humaine**. Il ne peut pas passer d'ordre.

```
RSS / X  ──►  nettoyage + détection d'injection  ──►  LLM (Anthropic API)  ──►  validation stricte JSON
                                                                                   │
                         Discord : « conseil, AUCUN ordre passé »  ◄───────────────┤
                         surcouche optionnelle « réduction seule » (OFF par défaut) ◄┘
```

## Pourquoi l'agent n'« agit » pas comme un trader autonome

Un trader discrétionnaire s'appuie sur des années d'expérience, un carnet d'ordres, un cadre de risque et une
responsabilité. Un LLM branché sur des tweets n'a ni validation historique ni garde-fou propre, et les contenus qu'il lit
sont **non fiables par nature** (un tweet peut contenir des instructions destinées au modèle). La règle du projet,
posée dans ta spécification, est donc appliquée ici :

- l'IA **propose** (synthèse, thèmes de risque, idées avec preuves) ;
- le code **déterministe** décide et borne (limites de risque, ordres, exécution) ;
- aucune idée n'est transformée automatiquement en ordre.

## Défenses contre l'injection de prompt (le texte des news est une donnée, jamais une consigne)

| Couche | Mesure | Test |
|---|---|---|
| Entrée | Nettoyage (balises, caractères de contrôle/invisibles, URL, `@everyone`), longueur bornée | `test_sanitize_*` |
| Entrée | Les éléments qui ressemblent à une injection ne sont **jamais envoyés** au modèle | `test_suspicious_items_never_reach_the_model` |
| Prompt | Règles de sécurité en tête, données encadrées dans `<untrusted_data>`, liste blanche d'actifs | `test_prompt_declares_news_untrusted...` |
| Sortie | JSON strict ; symboles hors liste blanche, valeurs hors énumération, preuves inexistantes ou absentes : **rejetés** et journalisés ; clés inconnues (`orders`, `send_order`...) ignorées | `test_a_compromised_model_cannot_smuggle...` |
| Architecture | Le briefing n'a aucun lien avec `Order`, le broker ou les limites de risque | structure du code |
| Sortie Discord | `allowed_mentions` vide : un texte hostile ne peut pas notifier `@everyone` | `test_payload_disables_mentions...` |
| Flux | `defusedxml` (entités externes, « XML bombs »), HTTPS seulement, taille limitée | `test_xml_bombs...` |

La détection par mots-clés n'est **qu'une heuristique** : la vraie protection est l'architecture (sortie validée,
lecture seule, aucun pouvoir d'exécution).

## Surcouche « réduction seule » (désactivée par défaut)

Si tu l'actives (`news.overlay.enabled: true`), un drapeau de risque `high` peut réduire de 50 % maximum les
positions des actifs concernés, pendant 24 h. Elle ne peut **jamais** augmenter ni inverser une position (propriété
testée sur 200 tirages aléatoires). Elle n'a **aucune validation historique** : il n'existe pas d'archive de news
exploitable pour la backtester. À juger en paper trading, sur plusieurs mois, avant de s'y fier.

## X (Twitter) : accès et coût

D'après des synthèses de tarifs publiées en 2026 (à vérifier sur la page officielle de X), l'accès à l'API est
désormais **payant à l'usage** (environ 0,005 $ par post lu selon ces sources, plafond mensuel de lecture ; les
chiffres divergent d'une source à l'autre) et la recherche récente ne couvre que **7 jours**. Le fournisseur impose donc
un plafond dur `max_posts_per_run` et refuse les requêtes qui le dépassent. Sans jeton `X_BEARER_TOKEN`, seuls les flux RSS sont utilisés.

Le code X est testé sur une réponse imitant la documentation v2, **pas contre l'API réelle**.

## Mise en route

```bash
export ANTHROPIC_API_KEY=...        # briefing (API payante)
export DISCORD_WEBHOOK_URL=...      # optionnel
export X_BEARER_TOKEN=...           # optionnel (payant)
quantlab news-brief --config configs/etf_trend.yaml --dry-run   # récupère les news et affiche le prompt
quantlab news-brief --config configs/etf_trend.yaml             # appelle le LLM, écrit paper/latest_briefing.json
```

Limites : le client Anthropic et les flux RSS configurés n'ont pas été testés en réel depuis l'environnement de
développement (pas d'accès réseau) ; vérifie les URL des flux avant usage.
