# DiffuThink

Un modèle de langage miniature à débruitage itératif, entraîné depuis des poids aléatoires. Le projet couvre données, entraînement, évaluation, checkpoints, inférence et démonstration locale.

**Périmètre :** reconstruction de phrases anglaises courtes sur une grammaire synthétique. Ce modèle n'est pas un chatbot généraliste. Le réseau est implémenté ici ; PyTorch fournit tenseurs, autograd, optimiseur et noyau d'attention. Les anciennes expériences sont conservées dans `experiments/`.

## Exécution

Python 3.10+ avec PyTorch 2.2+, depuis la racine :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m diffuthink prepare
python -m diffuthink train --steps 600 --output runs/demo
python -m diffuthink evaluate
python -m diffuthink sample --template "The cat ~~~~~ in the garden today." --temperature 0
python -m diffuthink serve
```

Ouvrir http://127.0.0.1:7860. Sous Linux/macOS : `source .venv/bin/activate`. CPU par défaut ; `--device cuda` utilise un backend CUDA/ROCm déjà fonctionnel. Le projet ne configure pas les pilotes. Le run de référence est mesuré sur CPU.

Le checkpoint local est `runs/demo/best.pt`, ignoré par Git. Un nouveau clone doit relancer l'entraînement. Depuis la racine, les commandes fonctionnent également sans installation avec un environnement où PyTorch est disponible.

Chaque `~` représente **un octet manquant**. La longueur de sortie est imposée par le template. Le contexte visible reste inchangé. Température zéro : argmax ; température positive : échantillonnage. Un `~` littéral ne peut pas être fixé. Une sortie UTF-8 invalide affiche des caractères de remplacement.

## Architecture

Octets UTF-8 → embedding → 2 blocs Transformer bidirectionnels avec RMSNorm, RoPE, SwiGLU et conditionnement par le bruit → projection vers 256 octets. Largeur 64, 4 têtes, longueur maximale 64 octets. PAD=256 et MASK=257 ne sont que des tokens d'entrée. La perte porte uniquement sur les positions masquées ; les clés de padding sont exclues de l'attention.

Le sampler révèle les octets les plus confiants selon un calendrier linéaire, sans remasquage. C'est une diffusion masquée simplifiée, sans revendication de reproduction exacte d'un article ou de borne variationnelle.

## Évaluation

Corpus original de 1 152 phrases générées localement. Déduplication puis split déterministe : 922 train, 115 validation, 115 test. Aucune phrase complète partagée ; vocabulaire et grammaire restent communs. Le checkpoint est choisi sur la validation uniquement. Le test mesure de nouvelles combinaisons dans ce domaine, pas une compréhension générale.

`reports/reference_run.json` contient les résultats réels, configuration, versions, durée, graines et empreintes SHA-256. L'évaluation refuse un corpus différent du manifeste d'entraînement. La baseline unigramme utilise uniquement les fréquences du train avec lissage. La cross-entropy masquée n'est pas une perplexité autorégressive. Les résultats peuvent varier selon la plateforme et la version de PyTorch.

```powershell
python -m unittest discover -s tests -v
python scripts/benchmark.py --output reports/sampling.json
```

Le benchmark compare 1, 4 et 12 étapes sur les mêmes masques du test. La démo web est un serveur local loopback, pas un service de production. Les checkpoints sont destinés à l'inférence, sans reprise de l'état de l'optimiseur.

## Organisation et appropriation

- `diffuthink/model.py` : réseau, corruption, perte et sampler.
- `diffuthink/data.py` : corpus et partitions.
- `diffuthink/engine.py` : entraînement, checkpoints et évaluation.
- `diffuthink/demo.py` : interface locale.
- `tests/` : invariants et apprentissage sur mini-batch.
- [Guide technique et entretien](docs/EXPLICATION.md).
- [Model card](docs/MODEL_CARD.md).

Le projet a été développé avec une assistance IA importante, depuis les expériences initiales jusqu'au pipeline. Présentation honnête : « J'ai développé et expérimenté ce projet avec une assistance IA ; voici les choix que je comprends, les mesures que j'ai reproduites et mes modifications personnelles. » Pour se l'approprier : reproduire un entraînement, expliquer une passe avant/arrière, modifier une hypothèse et documenter son effet. Aucune revendication de recherche originale ou de niveau production.
