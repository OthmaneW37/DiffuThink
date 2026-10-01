# StoryPatch : classement appris et choix de garder l’original

Le classement NLL reste le mode par défaut. Le nouveau classeur est un mode
expérimental sélectionnable, car il progresse sur le corpus procédural mais
régresse sur les 36 anciens diagnostics : première variante acceptée 52,8 %
contre 72,2 % avec la NLL. Cette décision d’interface évite de remplacer
automatiquement un classement plus utile sur ces textes. Les poids et seuils
restent ceux choisis sur la validation, sans réentraînement sur ces diagnostics.

Le générateur reste l’adaptation DiffuThink de 13 439 680 paramètres choisie
par l’expérience v2. Cette étape ajoute un réseau indépendant de 3 528 769
paramètres. Ses poids partent d’une initialisation aléatoire ; aucun poids
préentraîné externe, service de génération ou modèle professeur n’intervient.
Le réseau classe des propositions. Il ne crée pas de nouvelles réponses.

## Ce que le système fait

1. L’utilisateur sélectionne un passage dans un petit récit anglais.
2. Le générateur produit six essais de reconstruction bidirectionnelle.
3. En mode « Correction ciblée · expérimental », le classeur lit chaque texte
   reconstruit, avec des marqueurs sur le passage.
   Trois blocs Transformer avec RoPE, RMSNorm et SwiGLU calculent ses
   représentations. Une moyenne sur le passage et une moyenne sur le texte
   alimentent une petite tête de score.
4. Le texte original est évalué de la même manière. Une variante n’est
   conseillée que si son score dépasse celui de l’original avec une marge
   suffisante et passe un seuil de score absolu. Les seuils viennent de la
   validation. Le score n’est pas une probabilité d’avoir raison.
5. L’utilisateur décide. Appliquer, annuler, conserver et exporter restent
   des actions explicites. Le préfixe et le suffixe sont recopiés exactement.

Le contexte du classeur est de 512 tokens. Aucun contexte n’est coupé en silence :
si une entrée déborde, le système conserve l’original et explique le refus.
Le générateur utilise BF16 sur le GPU testé, le classeur FP32 sur CPU et GPU.

## Données et limites des étiquettes

Le corpus est **synthétique et procédural**, sans annotation humaine. Il
contient 12 000 contextes d’entraînement, 1 200 de validation et 300 de test.
Six catégories : couleur, emplacement, émotion, état, identité et accord.
Les identifiants de gabarits sont séparés entre splits (0–7, 8–9, 10–11).
Les empreintes des contextes sont dédupliquées entre splits. Les constructions
restent apparentées : cette séparation ne démontre pas une généralisation à
des textes libres, à d’autres langues ou à des catégories nouvelles.

Chaque contexte comporte une liste explicite de formulations acceptées et de
contrastes moins appropriés selon les règles du corpus. Ces règles simplifient
la langue. Une émotion peut coexister avec une autre, un pronom peut être
ambigu, et une alternative raisonnable peut manquer à la liste.

La première expérience apprend seulement ces contrastes. Sur la validation
avec les véritables propositions du générateur, elle ne respecte pas les
limites de modifications erronées ; son calibrage choisit de tout garder.
Cette expérience est conservée dans `reports/storypatch-ranker-v1`.

Le corpus v2 explicite aussi certaines variantes acceptables, par exemple
« very sad », « underneath » ou « close behind ». Une nouvelle seed fige
un nouveau corpus avant le second entraînement. Pour 1 200 contextes du
**split d’entraînement seulement**, le générateur produit des propositions.
Les propositions absentes de la liste de réponses deviennent des contrastes
supplémentaires. Chaque proposition et sa provenance sont conservées. Il
s’agit d’un étiquetage fermé imparfait, pas de corrections validées par une
personne. Ni les contextes de validation ni ceux du test ne sont utilisés
pour cette extraction.

## Entraînement et choix des seuils

La perte combine une BCE sur la compatibilité et une perte de préférence :
`BCE + 0.5 * softplus(score_négatif - score_positif)`.
Chaque lot tire 48 paires, donc 96 textes. AdamW, décroissance cosinus,
100 étapes de chauffe, clipping du gradient, dropout de 10 %. Le meilleur
checkpoint est celui ayant la BCE minimale sur les 1 200 contextes de
validation. Les courbes et paramètres exacts sont conservés dans les rapports.

Les premiers 120 contextes de validation donnent 240 cas d’édition : une
version correcte à conserver, une version altérée à réparer. Une grille de
seuils maximise la moyenne du taux de réparation et du taux de conservation,
avec au plus 5 % de changements sur les versions correctes et au plus 10 %
de remplacements hors référence sur les versions altérées. Ce sont des
contraintes mesurées sur ce petit échantillon ; elles ne garantissent pas les
mêmes taux sur d’autres textes. Toujours garder l’original constitue une
solution de repli explicite, avec une réussite équilibrée de 50 %.

## Évaluation

Les 300 contextes de test donnent **600 cas**, moitié corrects, moitié
altérés. Ce sont 300 contextes indépendants, pas 600 observations
indépendantes. Chaque méthode reçoit **les mêmes variantes** et la même seed.
Le classement NLL est comparé au classement appris ; on mesure séparément :

- réponse de référence disponible parmi les propositions ;
- première variante acceptée, sans option de conservation ;
- correction conseillée sur les entrées altérées ;
- conservation exacte sur les entrées correctes ;
- remplacement hors référence et préservation du contexte.

Un bootstrap de 1 000 tirages, groupé par contexte, donne un intervalle
descriptif de la différence de réussite équilibrée. Cet intervalle ne couvre
pas l’incertitude des étiquettes ni la validité sur des textes réels.
Les sorties complètes, poids et données sont reliés par SHA-256.

Un CSV de revue masque l’ordre des deux méthodes. Les colonnes de jugement
restent vides. Une évaluation humaine n’aura lieu qu’une fois ces colonnes
remplies par des personnes ; aucun résultat humain n’est annoncé ici.

## Reproduire

Depuis la racine du dépôt avec l’environnement de recherche installé et
le générateur `runs/storypatch-13m/best` disponible :

```powershell
python -c "from diffuthink.storypatch.corrections import build; build('data/processed/storypatch-corrections-v2',seed=2841,version=2)"
python -m diffuthink.storypatch.mine_ranker
python -m diffuthink.storypatch.train_ranker --data data/processed/storypatch-corrections-mined-v2 --output runs/storypatch-ranker-v2
python -m diffuthink.storypatch.evaluate_ranker --data data/processed/storypatch-corrections-mined-v2 --ranker runs/storypatch-ranker-v2/best --output reports/storypatch-ranker-v2
python -m diffuthink.v2 serve --model runs/storypatch-13m/best --ranker runs/storypatch-ranker-v2/best --device cpu --port 7862
```

Pour le GPU AMD local, les mesures de génération ont utilisé
`$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL='1'`. La compilation des noyaux
peut allonger le premier démarrage. Les dossiers de données et de poids
doivent être nouveaux pour préserver les expériences figées. Un entraînement
interrompu reprend avec `--resume runs/storypatch-ranker-v2/resume.pt` et les
mêmes options. Les seeds fixent les tirages ; les résultats GPU peuvent varier
avec l’environnement logiciel et les noyaux.

Code, corpus procédural et poids : Apache-2.0. Le générateur conserve la
provenance et l’attribution TinyStories documentées dans l’expérience v2.
Développement et rédaction assistés par IA.
