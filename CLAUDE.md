\# Règles du projet quantlab (mode recherche étendue)



\## Sécurité (non négociable)

\- Trading réel désactivé. Jamais d'ordre réel, jamais de clé ni de webhook dans le code ou le dépôt.

\- Travailler sur une branche, jamais sur main. Lancer pytest et ruff avant de terminer.

\- Ne pas commiter data/raw ni .env.



\## Recherche : ce qui est libre

\- Tester autant de stratégies, de paramètres et de critères de sélection que nécessaire pour trouver la meilleure stratégie.

\- Les critères (gates) peuvent être modifiés entre deux séries de tests, pas pendant : chaque changement est noté dans docs/JOURNAL.md avec la date, la raison et le résultat qui l'a motivé.



\## Recherche : ce qui est obligatoire

\- Période verrouillée : tout le développement utilise data.end = 2023-12-31. Les données à partir de 2024-01-01 ne sont lues qu'une fois, sur les finalistes, en fin de série. Si une finaliste échoue, on ne s'en sert plus pour ajuster.

\- Chaque essai (stratégie x paramètres, y compris les essais abandonnés) est enregistré. Versionner runs/index.jsonl (modifier .gitignore en conséquence). Ne jamais supprimer l'historique ni changer research\_id pour réinitialiser le compteur.

\- Le critère de sélection final est évalué avec le nombre total d'essais (Sharpe déflaté), jamais sur le Sharpe brut.

\- Préférer la stratégie la plus simple à performance proche. Plafond : 300 essais par research\_id.

\- Un résultat obtenu après modification des critères sur les mêmes données est étiqueté « exploratoire », jamais « validé ».

\- Une stratégie n'est candidate au paper trading qu'après avoir passé la période verrouillée.

\- Répondre en français.

