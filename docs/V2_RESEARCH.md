# DiffuThink V2 — protocole de recherche

## Question expérimentale

Un petit Transformer appris depuis zéro peut-il reconstruire des sous-mots dans des récits inconnus, et quel compromis qualité/coût offrent différentes politiques de révélation ? Le projet étudie cette question sur un domaine limité ; il ne prétend pas entraîner un assistant généraliste.

## Différences avec V1

| Composant | V1 | V2 |
| --- | --- | --- |
| Paramètres | 180 800 | 13 337 280 |
| Corpus | 1 152 phrases à gabarit | 100 000 récits TinyStories |
| Tokenizer | Octets | Byte-level BPE, 8 192 tokens |
| Contexte | 64 octets | 192 sous-mots |
| Réseau | 2 blocs, largeur 64 | 6 blocs, largeur 320, 8 têtes |
| Positions | RoPE | RoPE + embeddings absolus appris |
| Objectif | Masques aléatoires | Masques aléatoires + suffixes masqués |
| Calcul | CPU FP32 | GPU BF16, accumulation de gradients |
| Checkpoint | Poids d'inférence | Poids Safetensors + état complet de reprise |
| Baseline | Unigramme | Unigramme + bigramme bidirectionnel |

La V1 affichait une accuracy en octets sur une tâche très facile. La V2 mesure des sous-mots sur d'autres données : les pourcentages ne sont pas comparables directement.

## Données et séparation

Source : [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), révision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
Il s'agit de récits synthétiques générés par GPT-3.5/4, pas d'un corpus de textes humains. La fiche du corpus déclare CDLA-Sharing-1.0. Le manifeste conserve cette provenance ; les textes ne sont pas inclus dans le dépôt de code ou le dossier de publication du modèle.

On sélectionne 100 000 documents uniques du fichier train. Les 2 000 premiers documents uniques du fichier validation sont répartis alternativement en 1 000 validation et 1 000 test. Les doublons exacts, après normalisation des espaces, sont exclus entre partitions. Ce filtrage ne constitue pas une détection des quasi-doublons ou de la proximité sémantique.

Le BPE est appris uniquement sur les 50 000 premiers documents d'entraînement. Chaque document est ensuite découpé en fenêtres non chevauchantes ; deux partitions ne partagent jamais une fenêtre d'un même document. Les fenêtres intermédiaires ne reçoivent pas de faux EOS ; seule la fin du document en reçoit un. BOS indique le début de chaque fenêtre, qui n'est donc pas toujours le début d'une histoire.

Le corpus préparé contient 21 134 655 tokens textuels de train, 152 676 fenêtres de train, 1 396 de validation et 1 380 de test. Les données restent sur disque en tableaux uint16 ; seuls les batches sont transférés sur GPU. L'ordre de sélection suit la source, sans prétendre être un échantillon représentatif aléatoire de tout TinyStories.

## Architecture et objectif

Le réseau réutilise les blocs du projet : RMSNorm, QKV explicites, RoPE, attention bidirectionnelle et SwiGLU. Les poids entrée/sortie sont partagés. Les positions absolues fournissent un signal distinct même avec beaucoup de masques identiques ; l'information de bruit entre dans chaque bloc.

Pour chaque exemple, on choisit soit des masques indépendants à probabilité variable, soit un suffixe masqué. BOS et PAD ne sont jamais des cibles ; EOS peut l'être. Le conditionnement représente la **fraction réelle** de tokens masqués, en entraînement comme en inférence. La cross-entropy s'applique aux positions masquées et la projection vers le vocabulaire n'est calculée que sur ces positions pendant l'entraînement.

Cet objectif est une reconstruction masquée, sans pondération 1/t ni démonstration d'une borne variationnelle. Le sampler est heuristique ; « diffusion » décrit ici le cadre de corruption/débruitage discret et ne garantit pas l'équivalence à un algorithme publié.

## Entraînement

AdamW, clipping de norme à 1, warmup linéaire puis décroissance cosinus, microbatches de 16 et accumulation de 2, calcul BF16 et paramètres/états d'optimiseur en FP32. Une première phase de 6 000 mises à jour utilise un pic de learning rate de 0,0006. Une seconde phase repart du checkpoint de cette première phase avec un nouvel optimiseur et un pic à 0,0002. Ce sont toujours les poids appris dans le projet, jamais des poids externes.

Le fichier `last.pt` sauvegarde modèle, optimiseur, états des générateurs, position dans le calendrier, manifestes et compteurs. `--resume` reprend le même calendrier, avec les mêmes paramètres critiques. `--initialize-from` commence explicitement une nouvelle phase d'optimisation avec une filiation enregistrée. Ce ne sont pas deux noms pour la même opération.

Les checkpoints sont sélectionnés par la moyenne de cross-entropy de validation sur 128 fenêtres et trois taux de masquage : 15 %, 50 % et 85 %. Le rapport final évalue 512 fenêtres de test. La sélection du checkpoint ne lit pas le test. Les rapports de phases intermédiaires existent aussi : toute comparaison ultérieure reste exploratoire, sans transformer un test consulté en nouveau jeu caché.

## Génération et comparaisons

La continuation remplit des blocs de 16 sous-mots. Chaque bloc démarre masqué, puis les candidats sont fixés suivant un calendrier cosinus. Un token déjà fixé n'est pas remasqué. Le texte est tronqué à EOS ; la génération peut donc s'arrêter avant le budget maximal.

La politique par confiance classe les candidats selon leur probabilité avant application de la température. À température zéro, utiliser les probabilités après division par une température quasi nulle détruirait le classement en produisant des scores proches de 1 partout : la V2 évite cette erreur.

Quatre budgets/politiques sont mesurés sur les mêmes masques : une passe parallèle, révélation gauche à droite, révélation par confiance et révélation adaptative. L'adaptatif fixe en plus les candidats de confiance ≥0,85. Ce seuil est une heuristique explicite, pas un module de raisonnement appris. Aucun gain de qualité ou de vitesse n'est garanti.

Le bigramme estime les candidats à partir des voisins immédiats visibles, avec lissage et repli unigramme. Il ne consulte jamais les tokens propres cachés. C'est une référence contextuelle plus informative qu'un unigramme, mais pas une comparaison à un Transformer autorégressif entraîné avec le même budget.

## Ce qui reste à démontrer

La reconstruction n'est pas la cohérence d'un récit complet. Les générations qualitatives utilisent des prompts fixes et une seed annoncée ; les échecs sont conservés. Il manque notamment plusieurs graines d'entraînement, des intervalles de confiance, une évaluation humaine en aveugle et un modèle autorégressif témoin. La démo locale est utile en entretien mais ne constitue pas une validation de production.

## Contribution à défendre en entretien

Expliquer le choix du tokenizer, la séparation avant fenêtrage, les masques mixtes, la condition de bruit, les positions absolues, l'optimisation mémoire et la différence entre reprise et nouvelle phase. Montrer une comparaison qui contredit une intuition, puis les limites. Le développement est assisté par IA ; ne pas présenter les architectures standards comme une invention personnelle.
