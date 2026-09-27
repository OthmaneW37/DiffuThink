# Pourquoi DiffuThink devient hybride

## Le problème observé

La V2 pouvait reconstruire des sous-mots tout en générant des suites telles que « The man was so happy that he had forgot to the tall tree ». Une bonne accuracy de reconstruction ne suffisait donc pas à établir une bonne qualité de continuation.

Deux écarts expliquent le changement de méthode : les tokens d'un bloc étaient proposés à partir de trous encore inconnus puis figés ; et l'objectif de reconstruction de suffixes ne formait pas explicitement le réseau à prédire chaque prochain token avec un historique propre. Le modèle apprenait aussi à placer EOS à la fin d'une fenêtre visible, ce qui pouvait terminer trop tôt une continuation par blocs.

## Changement réel de modèle

Le réseau conserve ses 13 337 280 paramètres et son tokenizer maison. Il repart uniquement des poids appris dans ce projet. Aucun modèle préentraîné externe, correcteur grammatical ou texte préécrit n'est utilisé.

Une nouvelle phase de 16 000 mises à jour alterne 9 batches autorégressifs et 1 batch de débruitage. Pour le chemin autorégressif :

1. L'entrée est la fenêtre sans son dernier token.
2. La cible est la même fenêtre décalée d'un token vers la gauche.
3. Un masque triangulaire empêche chaque position de voir la suite.
4. La cross-entropy ignore PAD et supervise tous les prochains tokens valides.
5. Le conditionnement de bruit est fixé à zéro.

Pour le débruitage, on conserve l'attention bidirectionnelle, les masques variables et la loss sur les trous. Le partage des poids peut dégrader certaines métriques de reconstruction : elles sont évaluées séparément et restent dans le rapport final.

**La continuation recommandée est désormais autorégressive.** L'interface ne la présente pas comme de la diffusion. Le débruitage reste disponible pour les trous et comme comparaison expérimentale.

## Inférence

La continuation ajoute un token à la fois. Température par défaut 0,5, nucleus sampling à 0,9, pénalité de répétition de 1,12 sur les 64 derniers tokens et interdiction de reproduire un même groupe de quatre tokens. Ces contrôles modifient la distribution de sortie ; ils ne corrigent pas la grammaire a posteriori. Le rapport fournit aussi les sorties sans les deux protections contre la répétition pour distinguer leur effet de celui de l'entraînement. Température zéro sélectionne l'argmax et ne dépend pas de la seed. Le contexte est borné à 192 tokens ; au-delà, une fenêtre glissante est utilisée. Il n'y a pas encore de cache KV.

EOS est interdit pendant les huit premiers nouveaux tokens pour éviter les réponses immédiatement vides. Le budget maximal peut couper une phrase ; les sorties brutes ne sont pas réécrites. Les espaces finaux du prompt sont retirés avant tokenisation afin d'éviter un token d'espace isolé suivi d'un second espace appris. Les autres caractères du prompt restent inchangés.

## Évaluation

Le checkpoint est choisi sur la NLL causale de 256 fenêtres de validation fixes. Le test utilise 512 fenêtres distinctes de la partition test. Contrairement à la cross-entropy masquée, `exp(NLL)` constitue ici une perplexité autorégressive, car chaque prédiction dépend uniquement du passé.

Douze débuts de récits fixés sont générés avant/après, avec le même budget de 96 tokens et température 0,5. Les algorithmes et leurs règles de sampling diffèrent explicitement ; une seed commune ne représente pas des tirages identiques entre eux. Le cas fourni par l'utilisateur est un cas de diagnostic connu, pas un test caché. Les exemples ne servent pas à sélectionner le checkpoint ; la validation numérique le fait.

Toutes les sorties sont conservées dans `reports/hybrid/continuations.json`. La fréquence de trigrammes répétés est un indicateur de répétition, pas une mesure de grammaire ou de fidélité narrative. Aucune évaluation humaine en aveugle n'est prétendue.

## Reproduire

```powershell
python -m diffuthink.v2.hybrid
python scripts/compare_continuations.py
```

Reprise d'un run interrompu : même commande avec `--resume`. Le dossier conserve l'état de l'optimiseur et les générateurs explicitement utilisés pour sélectionner les batches et les masques. Les paramètres critiques et le manifeste doivent correspondre. La reproductibilité bit à bit entre GPU ou versions PyTorch différentes n'est pas garantie.

## Limites

Le modèle reste petit, spécialisé dans des récits anglais synthétiques et limité à 192 tokens de contexte. Une amélioration grammaticale ne prouve ni raisonnement général ni cohérence parfaite. Les objectifs masqué et causal peuvent entrer en concurrence. Les contrôles de sampling réduisent certains défauts sans les éliminer. Une évaluation multigraines et une revue humaine aveugle restent à faire.
