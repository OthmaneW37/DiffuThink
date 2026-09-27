# DiffuThink — model card

Transformer bidirectionnel miniature, initialisé aléatoirement, entraîné avec masquage variable et génération par révélation itérative. Implémentation PyTorch personnalisée et développement assisté par IA. Version pédagogique 0.1.0.

**Usage :** apprentissage et reconstruction de phrases anglaises courtes. Configuration de référence : 1..64 octets, longueur de sortie imposée, `~` pour les trous. Pas de token de fin ni d'entraînement conversationnel. Les expériences historiques ont un autre format de modèle.

**Données :** 1 152 phrases originales générées localement (8 sujets × 6 verbes × 6 lieux × 4 expressions temporelles), sans téléchargement ni collecte personnelle. Splits 922/115/115, sans phrases identiques mais avec vocabulaire et grammaire communs. `data/tiny.txt` n'est pas utilisé dans le run de référence.

**Mesures :** `reports/reference_run.json` donne configuration, paramètres, versions, graines, SHA-256, durée et résultats. Checkpoint choisi sur validation, test à 25/50/75/100 % de masquage. `reports/sampling.json` compare l'inférence itérative.

**Limites :** corpus artificiel minuscule, baseline unigramme faible, aucune preuve de raisonnement ou de généralisation à une nouvelle grammaire. Pas de revendication de supériorité sur les modèles préentraînés. UTF-8 invalide possible en sortie. Confiance non calibrée, erreurs précoces figées, entrée entièrement masquée sans repères de position absolue. Pas d'arrêt adaptatif appris, de reprise de l'optimiseur ou de service de production. Checkpoints d'inférence locaux, ignorés par Git.
