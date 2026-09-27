# StoryPatch : un usage contrôlé du modèle

Un atelier pour explorer un mot ou un court passage dans un mini-récit anglais. L'utilisateur choisit la zone, compare les propositions, applique une modification, peut l'annuler et exporter le texte. La continuation libre reste accessible dans `/lab`.

L'originalité du projet est dans l'assemblage du parcours, ses garanties vérifiables et son caractère inspectable. Il ne revendique pas l'invention de la réécriture, du text infilling ou du débruitage.

## Algorithme réel

1. Recevoir les bornes de sélection en caractères Unicode. Le navigateur convertit ses positions UTF-16 en indices de points de code Python.
2. Préserver séparément le préfixe et le suffixe originaux, y compris les espaces autour de la zone. Le contexte gauche de génération est débarrassé des espaces finaux avant tokenisation pour éviter des espaces BPE isolés ; le texte affiché original n'est pas modifié.
3. Générer six essais : longueur originale en sous-mots et deux longueurs voisines. Le premier est glouton, les suivants utilisent une température de 0,8 et des seeds successives. EOS est interdit dans les trous.
4. Retirer les propositions vides, identiques, dupliquées ou contenant un caractère de remplacement Unicode. Toutes les variantes distinctes restantes sont affichées ; le nombre d'essais est annoncé.
5. Recomposer avec `préfixe_original + remplacement + suffixe_original`. Aucun token généré ne peut remplacer le contexte protégé.
6. Classer par NLL causale moyenne sur la zone et jusqu'à 16 tokens suivants. Le score original est fourni. La normalisation et les longueurs variables peuvent favoriser certaines formulations ; ce n'est pas un score de vérité, de grammaire ou de fidélité sémantique.
7. Appliquer uniquement sur décision de l'utilisateur. Une réponse calculée sur une ancienne version du texte est refusée à l'application. Une annulation restaure le texte précédent.

Code principal : `diffuthink/v2/editor.py` et `workshop.html`. Le réseau n'a pas été remplacé par un LLM externe. C'est un nouvel usage des poids du projet, pas un fine-tuning spécifique à la correction.

## Garanties et limites

Les tests vérifient la conservation du contexte et des espaces, les bornes de sélection, le refus de mots partiels alphanumériques et l'interdiction d'EOS dans les trous. Les propositions peuvent rester maladroites. Le modèle peut préférer l'original à toutes ses variantes ; l'interface le dit.

Les exemples intégrés sont des démonstrations connues, pas un benchmark caché. Les métriques de `reports/context512` concernent la génération libre, pas un taux de correction. Aucun gain pédagogique ni taux de correction n'est revendiqué sans évaluation dédiée.

La NLL ne produit pas une explication linguistique fiable. L'interface montre le mécanisme et les scores, sans inventer de justification grammaticale.

## Hébergement

Le paquet `artifacts/StoryPatch-Space` utilise CPU, deux threads PyTorch et un serveur à requêtes séquentielles. Il contient les poids et ne télécharge pas de modèle au démarrage. Aucune donnée soumise n'est écrite par l'application ; aucun service de génération externe n'est appelé.

La configuration suit la [documentation officielle des Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker) : SDK Docker, port 7860 et utilisateur 1000. Le paquet reste local jusqu'à une publication explicite. Le serveur de démonstration n'est pas présenté comme un service de production multi-utilisateur.
