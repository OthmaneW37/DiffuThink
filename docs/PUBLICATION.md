# Préparer GitHub, Hugging Face et LinkedIn

## Version actuelle : Hybrid

La version recommandée pour la continuation est désormais **DiffuThink Hybrid**, avec un entraînement autorégressif supplémentaire. Ne pas décrire ses continuations comme des générations par diffusion. Les modes restent séparés dans la démo : continuation causale, continuation par diffusion expérimentale et texte à trous.

```powershell
python -m diffuthink.v2 export --model runs/stories-hybrid/best --report runs/stories-hybrid/report.json --output artifacts/DiffuThink-Hybrid-13M
```

Le rapport actuel est `reports/hybrid/RESULTATS.md`. Il compare toutes les sorties sur douze prompts, avec et sans protections contre les répétitions. Le code garde les anciennes versions et leurs résultats pour documenter pourquoi l'objectif a changé. Les instructions V2 ci-dessous restent historiques.

## Dépôt GitHub

Le code, les tests, les manifestes et les rapports peuvent être versionnés. Les données préparées, checkpoints d'optimiseur, environnements et poids sont ignorés. Ne pas ajouter `runs/`, `data/processed/`, `.venv/` ou `artifacts/` avec une commande qui contourne `.gitignore`.

Avant publication : choisir la licence du code et du modèle, conserver les attributions, relire les résultats et vérifier que le modèle recharge depuis le bundle seul. Le projet ne publie rien automatiquement et n'ajoute aucun identifiant personnel à un dépôt distant.

## Bundle Hugging Face

```powershell
python -m diffuthink.v2 export --model runs/stories-v2-refined/best --report runs/stories-v2-refined/report.json --output artifacts/DiffuThink-13M
```

Le dossier contient `model.safetensors`, `config.json`, `tokenizer.json`, les informations d'entraînement, l'évaluation, une model card, le code d'inférence, les dépendances et les checksums. C'est un checkpoint PyTorch personnalisé avec son API documentée ; ne pas promettre une compatibilité `AutoModel` inexistante. Les poids externes préentraînés ne sont pas nécessaires.

Pour tester le bundle seul, ouvrir un terminal dans le dossier, installer `requirements.txt`, puis suivre l'exemple de sa model card. Les poids pourront être déposés dans le dépôt Hugging Face choisi après relecture. Aucune publication ou création de compte n'est effectuée par l'export.

## Présentation LinkedIn — texte à personnaliser

« J'ai développé DiffuThink, un projet expérimental de génération de texte par débruitage, avec une assistance IA documentée.

Le modèle de 13,3 millions de paramètres et son tokenizer BPE ont été entraînés depuis zéro sur 100 000 récits TinyStories. J'ai construit la préparation des données, l'entraînement GPU, la reprise des checkpoints, les métriques sur données séparées et une interface qui montre les étapes de génération.

J'ai comparé plusieurs politiques de débruitage à une référence contextuelle bigramme. Le rapport joint présente les mesures, les échecs et les limites : le modèle reste spécialisé dans des récits anglais simples et ne prétend pas être un assistant généraliste.

Ce projet m'a permis d'étudier [ajouter uniquement les points réellement compris et reproduits personnellement]. Code, model card et résultats : [ajouter les liens après publication]. »

Ajouter les chiffres du rapport final effectivement livré, avec le taux de masquage, le jeu d'évaluation et l'unité « sous-mots ». Ne pas reprendre les 89,1 % de la V1 : cette mesure portait sur une autre tâche et un autre tokenizer. Ne pas affirmer un état de l'art ou un gain adaptatif qui n'a pas été mesuré.
