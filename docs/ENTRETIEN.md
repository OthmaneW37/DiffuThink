# Expliquer DiffuThink en entretien

## Présentation courte

« DiffuThink est un petit modèle de langage dont les poids et le vocabulaire ont été entraînés dans ce projet, sans modèle préentraîné externe. Le projet a commencé par du débruitage de texte. Les métriques de reconstruction étaient meilleures que la qualité des histoires générées : j'ai donc fait évoluer le protocole vers une continuation autorégressive, en conservant un objectif auxiliaire de débruitage. L'expérience suivante étend le corpus et le contexte. Je compare les versions sur les mêmes cibles et je conserve aussi les sorties qui échouent. Le développement a été largement assisté par IA. »

Cette présentation décrit le projet. Avant de la reprendre à ton compte, exécute les commandes, lis les fonctions centrales et assure-toi de pouvoir justifier leurs choix.

## Le chemin des données au texte

1. **Données.** Télécharger une révision précise de TinyStories. Réserver d'abord les documents de validation et de test. Exclure les doublons exacts normalisés de l'entraînement. Conserver les empreintes pour détecter un changement de données.
2. **Tokenisation.** Le BPE appris sur le train transforme du texte en sous-mots. Un token n'est pas nécessairement un mot. Le vocabulaire de 8 192 entrées est gelé lors de la reprise : changer les IDs changerait le sens des embeddings déjà appris.
3. **Fenêtres.** Un récit court devient `[BOS] + tokens + [EOS] + [PAD]…`. Avec 512 positions, 97,2 % des récits du corpus élargi tiennent dans une seule fenêtre. Les récits plus longs sont segmentés ; leur fin réelle seule reçoit EOS.
4. **Représentation.** Chaque ID est associé à un vecteur de taille 320. On ajoute un vecteur de position appris. Le réseau utilise six blocs Transformer, huit têtes d'attention, RoPE, RMSNorm et SwiGLU. Les embeddings d'entrée servent aussi à calculer les scores de sortie.
5. **Apprentissage causal.** À chaque position, prédire le token suivant en ne voyant que le passé. L'attention triangulaire interdit de consulter la cible future. La cross-entropy ignore les cibles PAD. Les gradients mettent à jour les poids avec AdamW.
6. **Apprentissage auxiliaire.** Un lot sur dix utilise des trous à reconstruire avec les contextes gauche et droit. Ce sont 90 % / 10 % des lots, pas nécessairement des tokens supervisés ni une pondération explicite des deux losses.
7. **Génération.** Réinjecter chaque token produit pour prédire le suivant. La température et le top-p contrôlent l'échantillonnage. Les pénalités de répétition sont des règles de décodage déclarées ; elles ne prouvent pas que le réseau a appris une meilleure logique.
8. **Arrêt.** EOS est une décision du modèle. La marge de fin de phrase de la démo est une règle d'interface, bornée à 32 tokens. L'interface indique si le modèle s'est arrêté, si une ponctuation a été atteinte ou si le plafond a coupé la sortie.

Code à lire : `diffuthink/v2/data.py`, `model.py`, `hybrid.py`, puis `scripts/compare_context.py`.

## Questions à savoir traiter

**Pourquoi une bonne reconstruction ne suffit-elle pas ?**

Un mot masqué peut être déduit grâce aux mots situés à sa droite. En continuation, cette information n'existe pas. Une erreur générée devient en plus une partie du contexte des prochaines prédictions. La tâche et les conditions d'inférence diffèrent.

**Est-ce encore un modèle de diffusion ?**

La continuation recommandée est autorégressive. Le réseau conserve une capacité de débruitage bidirectionnel. Présenter le progrès de continuation comme une percée de diffusion serait incorrect.

**Pourquoi augmenter le contexte ?**

Le réseau peut apprendre davantage de récits complets, y compris leurs fins, et consulter un historique plus long. Cela augmente son champ d'information, sans lui donner automatiquement une représentation fiable des personnages ou de leurs intentions. « Il atteint le sommet, puis monte encore » reste une erreur possible.

**Pourquoi ne pas comparer directement les deux perplexités publiées ?**

Le découpage en fenêtres a changé. Des positions reçoivent plus de contexte, et les premières 512 fenêtres ne représentent plus les mêmes cibles. La comparaison principale réévalue les deux checkpoints sur toutes les mêmes fenêtres anciennes de 192 tokens. Le score des fenêtres longues est publié séparément.

**Comment sais-tu que l'amélioration vient du contexte ?**

Cette expérience ne l'isole pas : corpus, contexte et quantité de calcul changent ensemble. Il faudrait des ablations à budgets comparables : ancien corpus/contexte long, nouveau corpus/contexte court, puis combinaison. De même, le bénéfice du débruitage auxiliaire demanderait une comparaison causal seul / hybride.

**Que prouvent les tests automatisés ?**

Ils vérifient notamment l'absence d'accès au futur, les cibles supervisées, la conservation des anciens vecteurs de position, l'équivalence des gradients de l'attention optimisée, les fins de documents, les limites de génération et la reprise CPU. Ils ne prouvent pas que les histoires sont bonnes.

**Quelles limites annonces-tu ?**

Corpus synthétique anglais, petit modèle spécialisé, contradictions possibles, contrôle des quasi-doublons incomplet et absence d'évaluation humaine aveugle. Ni assistant généraliste, ni modèle factuel, ni architecture Transformer inventée ici.

## Exercices pratiques avant présentation

- Générer le même prompt à température zéro avec deux seeds et expliquer le résultat.
- Retirer les contrôles de répétition, garder les sorties et observer ce qui change.
- Retrouver dans les métadonnées le corpus, sa révision, le tokenizer et l'historique des poids.
- Lire une paire avant/après où la nouvelle version échoue. Expliquer pourquoi une meilleure NLL peut coexister avec une histoire incohérente.
- Relancer un petit entraînement CPU, l'interrompre et vérifier sa reprise avec le test fourni.

Les résultats chiffrés de l'expérience doivent être pris dans `reports/context512/RESULTATS.md`, une fois le calcul terminé, et non anticipés.
