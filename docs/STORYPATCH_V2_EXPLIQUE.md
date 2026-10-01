# Comprendre et présenter StoryPatch v2

## Le problème concret

Un utilisateur sélectionne quelques mots dans une histoire. StoryPatch propose
des remplacements qui tiennent compte du texte avant **et** après la sélection.
L'utilisateur compare les variantes et décide d'en appliquer une. Le code recolle
le préfixe et le suffixe originaux sans les modifier : cette garantie vient de la
composition de chaînes, pas d'une promesse du réseau de neurones.

Le modèle publié avait surtout appris à prédire le prochain token. Seulement une
petite partie de ses mises à jour portait sur le débruitage, avec des masques qui
ne correspondaient pas précisément aux sélections de mots de l'interface. La v2
cherche à réduire cet écart entre la tâche d'entraînement et l'usage.

## D'où viennent les exemples ?

Nous réutilisons le tokenizer BPE du projet et les 500 000 documents TinyStories.
Un tokenizer transforme une phrase en identifiants de sous-mots. Un mot rare peut
donc occuper plusieurs tokens. La sélection respecte les limites des mots, puis
est convertie en positions de tokens à masquer.

Par exemple, dans un récit source :

```
Lily lost her favorite toy. She felt sad and began to cry.
```

Le système peut remplacer les tokens de `sad` par des masques, laisser les deux
côtés visibles et demander au modèle de retrouver les tokens d'origine. La cible
provient du récit. Il ne s'agit pas d'une correction grammaticale annotée par un
humain, ni d'une réponse fournie par une API extérieure.

Chaque fenêtre possède jusqu'à huit sélections précomputées de 1 à 12 mots. Les
plus petits exemples peuvent répéter une sélection : quatre millions de positions
disponibles ne signifient pas quatre millions de récits distincts.

Les documents de validation et de test sont réservés avant entraînement. Les
masques d'évaluation sont fixes. Les données de test ne servent pas à choisir les
checkpoints. Le filtrage retire certains textes très répétitifs ou mal encodés ;
il ne certifie pas leur cohérence sémantique.

## Qu'apprend le modèle ?

Deux objectifs partagent les mêmes poids du Transformer :

1. **Prédiction causale** : prédire le token suivant en regardant uniquement le
   passé. Cela conserve la capacité à poursuivre une histoire et à calculer la
   vraisemblance locale utilisée pour classer les variantes.
2. **Reconstruction bidirectionnelle** : prédire les tokens masqués avec du
   contexte à gauche et à droite. Certaines mises à jour révèlent déjà une partie
   du passage, pour représenter les étapes intermédiaires de l'inférence.

La loss est une entropie croisée sur les positions supervisées. Les caractères
visibles ne sont pas des cibles de reconstruction. Le nombre de tokens présentés
au réseau est donc différent du nombre de cibles contribuant à la loss.

Le modèle adapté reprend nos propres poids de 13,44 M de paramètres. Le challenger
de 58,47 M a commencé avec des poids aléatoires. Tous deux conservent le tokenizer
du projet. « Depuis zéro » signifie qu'aucun poids de modèle externe n'est utilisé ;
cela ne signifie pas que le corpus TinyStories est écrit par nous. Ce corpus est
lui-même synthétique, généré à l'origine à l'aide de modèles externes.

## Pourquoi le grand modèle a-t-il été entraîné par phases ?

Le premier essai mélangeait immédiatement les deux objectifs. La validation
montrait une progression causale, mais une stagnation de la reconstruction. Un
diagnostic sur CPU montrait la même prédiction sur presque tous les masques.

Cela constitue un symptôme d'apprentissage insuffisant de cette tâche, pas la
preuve qu'un Transformer de cette taille ne peut pas la résoudre. L'essai a été
arrêté au dernier checkpoint conservé. Nous avons ensuite testé une consolidation
causale suivie d'une spécialisation avec un taux d'apprentissage réduit.

Cette modification a été décidée après observation de la validation. Il faut la
présenter comme une adaptation du protocole, et conserver les traces de l'essai
initial. Elle empêche de prétendre à une comparaison parfaitement contrôlée de la
taille des modèles.

## Comment lire les résultats ?

- Une **NLL plus basse** indique que le modèle attribue davantage de probabilité
  aux tokens de référence. Cela ne démontre pas à lui seul une meilleure logique.
- L'**accuracy token** compte les sous-mots retrouvés. L'exact match demande de
  retrouver le passage entier, et peut rejeter une reformulation pourtant valable.
- Les **36 cas ciblés** testent quelques contradictions, accords, pronoms et
  relations. Ils servent au diagnostic ; ils ne représentent pas tout l'anglais.
- Le **top-3** indique qu'au moins l'une des trois premières propositions appartient
  aux réponses acceptées. Ce n'est pas la qualité moyenne de toutes les sorties.
- La **préservation du contexte** est une propriété du système de composition.
  Il faut la distinguer de la cohérence du nouveau passage avec ce contexte.

Le score local de classement n'est pas un vérificateur de faits. Le modèle peut
préférer une phrase plausible mais fausse dans l'histoire. Un véritable reranker
sémantique demanderait ses propres annotations et une évaluation séparée.

## Présentation courte possible

« J'ai construit, avec une assistance IA documentée, un pipeline d'édition de
petits récits anglais. J'ai adapté l'entraînement à la tâche de sélection de mots,
comparé un modèle de 13 M et un challenger de 58 M, et mesuré la reconstruction,
la génération et les corrections sur des données réservées. Le système permet
d'inspecter les variantes et garantit que le reste du texte ne change pas. Je
présente aussi les échecs et les changements de protocole, notamment l'essai
initial où le grand modèle ne reconstruisait pas correctement les passages. »

Ajouter uniquement les chiffres du [rapport final](../reports/storypatch-v2/RESULTATS.md)
et les étapes que tu sais personnellement expliquer et reproduire.
