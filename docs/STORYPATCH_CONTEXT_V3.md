# StoryPatch v3 : apprendre des récits plus variés

La version précédente apprenait surtout des gabarits. Ses bons résultats sur
le corpus procédural ne se retrouvaient pas sur les anciens diagnostics.
Cette expérience change les données, l’initialisation et la façon de classer
les variantes. Les résultats chiffrés sont dans
`reports/storypatch-context-v3/RESULTATS.md`.

## Modèle et provenance

Le générateur de 13 439 680 paramètres conserve ses poids. Le nouveau classeur
possède 13 645 761 paramètres. Il reprend les embeddings, positions, six blocs
Transformer, normalisation et conditionnement à bruit nul de notre propre
générateur. Une tête de classement et les marqueurs de sélection sont ajoutés.
Un test vérifie que les représentations initiales sont les mêmes que celles
du générateur à bruit nul. L’ancien classeur v2 avait 3 528 769 paramètres.

Ce classeur n’est donc **pas nouvellement initialisé entièrement au hasard** :
il réutilise le modèle appris dans ce projet. Toute la lignée des poids
remonte à l’initialisation aléatoire de DiffuThink. Aucun modèle externe,
poids préentraîné externe ou professeur distant n’est utilisé.

## Données

24 000 contextes d’entraînement : 12 000 exemples procéduraux de la phase v2,
et 12 000 extraits de TinyStories, comprenant jusqu’à quatre phrases et au
plus 128 tokens avant altération. La validation mélange 600 contextes de
chaque type ; le test mélange 300 contextes de chaque type.

« Extraits » signifie des passages du corpus plutôt que des gabarits de notre
code. **TinyStories lui-même est synthétique.** Leurs passages originaux
sont positifs, et des altérations mécaniques servent de contrastes : mots
répétés, inversés, remplacés, ou lettres transposées. Ce sont des références
de reconstruction, pas des préférences validées par des personnes. Certaines
altérations restent grammaticales et certaines alternatives valides peuvent
être rejetées par les listes fermées.

Les documents du corpus conservent leurs splits d’origine. Les extraits sont
dédupliqués par empreinte de texte normalisé entre splits, avec priorité aux
splits de test et de validation. Il ne s’agit pas d’une déduplication générale
des paraphrases. Les extraits peuvent commencer au milieu d’un récit.

Les 300 contextes procéduraux et les 36 anciens diagnostics sont réutilisés
pour suivre les régressions ; ils ne sont pas présentés comme de nouveaux
tests indépendants. Les 300 contextes issus du corpus sont nouveaux pour
l’évaluation de ce classeur, mais leurs documents appartiennent au test du
corpus déjà utilisé lors des expériences antérieures de génération.

## Entraînement

24 paires par lot, AdamW à un maximum de 0,0001, décroissance cosinus,
100 étapes de chauffe, clipping à 1, dropout de 10 %. L’objectif combine BCE
et préférence par paire. Jusqu’à 3 200 mises à jour sont prévues ; arrêt après
six validations sans amélioration. Le checkpoint est choisi par BCE sur les
1 200 contextes de validation, jamais par les diagnostics finaux.

## Classement et conservation

Le score final est :

`-NLL_locale + poids × clip(score_appris, -8, 8)`

Le coefficient est choisi dans une grille fixée de 0 à 1,6 sur les premières
240 paires de contextes de validation (120 de chaque domaine). Il doit
améliorer le classement moyen sans régresser sur la première proposition
dans aucun des deux domaines. En cas d’égalité, le poids le plus faible est
retenu. Un poids nul revient exactement au classement NLL.

Ensuite, sur les mêmes 480 entrées de validation, une marge et un score
minimal choisissent entre remplacement et conservation. Les seuils imposent
au plus 5 % de modifications des entrées correctes et au plus 10 % de
remplacements hors référence sur les entrées altérées, sur cet échantillon
de validation uniquement. Ces contraintes ne sont pas des garanties futures.
Le score combiné n’est pas une probabilité de correction.

## Test et interface

Le test comprend 1 200 entrées : une version altérée et une version correcte
pour chacun des 600 contextes. Les méthodes reçoivent les mêmes propositions
du générateur et les mêmes seeds. On rapporte séparément les deux domaines,
la couverture des réponses de référence, la première variante, les
recommandations de réparation et de conservation, et les anciens diagnostics.

Une réponse valide absente des références compte comme un échec : ces mesures
ne sont pas des taux de correction linguistique évalués par des personnes.
Le bootstrap par contexte décrit l’incertitude sur cet ensemble seulement.

L’interface propose « Classement contextuel · expérimental » et conserve
l’exploration NLL par défaut. Les modifications restent manuelles et
réversibles. Tout ce qui se trouve hors de la sélection est conservé par
composition exacte du texte. Ni soumissions ni retours utilisateur ne sont
enregistrés automatiquement.

## Reproduction

```powershell
python -m diffuthink.storypatch.context_data
python -m diffuthink.storypatch.train_ranker --data data/processed/storypatch-context-v3 --output runs/storypatch-context-ranker-v3 --initialize-from runs/storypatch-13m/best --steps 3200 --batch-pairs 24 --lr 0.0001
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL='1'
python -m diffuthink.storypatch.context_evaluate
python scripts/context_ranker_report.py
python scripts/package_context_ranker.py
python -m diffuthink.v2 serve --model artifacts/StoryPatch-context-v3 --ranker artifacts/StoryPatch-context-v3/ranker --device cpu --port 7862
```

Les corpus précédents et le checkpoint DiffuThink doivent déjà être présents.
Les scripts refusent d’écraser les données figées. `--resume` reprend un
entraînement interrompu avec les mêmes arguments. Un test de reprise CPU
vérifie l’égalité exacte des poids. Le matériel et les noyaux GPU peuvent
modifier les résultats numériques.

Code et poids : Apache-2.0. Extraits TinyStories : CDLA-Sharing-1.0, source
et révision dans le manifeste. Les documents, le code et les expériences
ont été préparés avec assistance IA.
