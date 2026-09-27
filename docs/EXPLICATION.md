# Comprendre et présenter DiffuThink

## Le pitch

« J'étudie la génération de texte par débruitage. Je transforme une phrase en octets, j'en masque une partie et j'entraîne un Transformer bidirectionnel à les reconstruire. Pour générer, je pars d'un texte à trous et révèle les prédictions les plus confiantes. Le pipeline inclut partitions de données, sélection sur validation, test séparé, baseline et interface locale. Le domaine est contrôlé et le développement assisté par IA. »

## Une passe d'entraînement

1. Supprimer les lignes vides, dédupliquer les phrases puis séparer train/validation/test. Une phrase trop longue déclenche une erreur, sans tronquage silencieux.
2. Encoder en octets 0..255 et compléter avec PAD=256. Forme du batch : `[B, T]`.
3. Tirer un taux t uniforme entre 0,05 et 1 par exemple. Remplacer chaque octet réel par MASK=257 avec probabilité t. Si aucun octet n'est sélectionné, choisir une position valide au hasard pour éviter une perte vide. Cette règle modifie légèrement la distribution des masques.
4. L'embedding produit `[B, T, C]`. Un encodage sinusoïdal de t, suivi d'un MLP, conditionne chaque bloc par addition.
5. RMSNorm normalise les amplitudes. Projeter Q, K et V, puis diviser C entre H têtes. RoPE fait tourner les paires de coordonnées de Q et K en fonction de la position.
6. L'attention fait une somme pondérée de V en regardant les deux côtés des trous. Les clés PAD sont exclues. Une connexion résiduelle ajoute la sortie à l'entrée.
7. Une seconde RMSNorm précède SwiGLU : `down(silu(gate(x)) * up(x))`, puis une autre connexion résiduelle.
8. La tête finale produit `[B, T, 256]` logits. PAD et MASK ne peuvent pas être prédits.
9. La perte moyenne `-log p(octet correct)` porte sur les positions masquées seulement. Autograd calcule les gradients, leur norme est bornée à 1 et AdamW met à jour les poids.
10. Évaluer à intervalles réguliers sur des masques fixes. Garder le meilleur checkpoint sur validation, puis mesurer le test après sélection.

L'objectif est une reconstruction masquée à bruit variable. Ce n'est pas une borne variationnelle de diffusion ; il n'utilise pas de pondération théorique 1/t.

## Inférence

Avec N trous, effectuer au plus `min(étapes, N)` passes. Le taux fourni au réseau est la fraction restante de trous. Chaque position reçoit un candidat, dont la probabilité sert de confiance. Fixer les candidats les plus confiants selon un calendrier linéaire. Les positions fixées ne sont pas remasquées.

Une erreur précoce peut se propager et la confiance n'est pas calibrée. Davantage d'étapes ne garantit pas un meilleur résultat. Le benchmark compare les budgets sur les mêmes masques. Avec une entrée entièrement masquée et seulement des positions relatives, la première passe manque de repères pour distinguer les positions : essayer des tokens de frontière ou des positions absolues est une amélioration pertinente.

## Lire les métriques

- Accuracy masquée : fraction des octets cachés corrects, espaces compris.
- Cross-entropy : qualité de la probabilité de l'octet correct, en nats ; plus bas est meilleur.
- Unigramme : distribution fixe calculée sur le train, référence faible sans contexte.
- Exact match : phrase entièrement identique à l'original. Une autre phrase valide est comptée comme incorrecte.
- Latence : mesure locale CPU après chauffe, dépendante de la machine et du nombre de trous.

Le test partage la grammaire du train. Il ne prouve pas une compréhension générale. Ajouter une baseline n-gramme et un modèle autorégressif à budget comparable renforcerait l'évaluation.

## Présentation en cinq minutes

1. Annoncer le périmètre et l'assistance IA ; montrer une phrase à trous.
2. Expliquer les étapes visibles et le contexte conservé.
3. Montrer les formes des tenseurs, l'attention et la loss dans le code.
4. Montrer le rapport, la baseline et la séparation des données.
5. Montrer un échec réel et proposer une expérience pour tester sa cause.

## Exercices d'appropriation

1. Calculer une cross-entropy à la main, puis vérifier avec PyTorch.
2. Ajouter des tokens de frontière et mesurer la génération entièrement masquée.
3. Comparer une et deux couches avec trois graines, sans choisir sur le test.
4. Comparer deux calendriers avec mêmes trous et budgets.
5. Utiliser un corpus naturel aux droits connus, séparé par document avant les fenêtres.
6. Consigner une modification personnelle : hypothèse, diff, commande, mesures, analyse et limite.

## Questions à maîtriser

Pourquoi des octets ? Aucun vocabulaire appris ni mot inconnu, mais des séquences longues et un risque d'UTF-8 invalide. Pourquoi bidirectionnel ? Les deux côtés sont disponibles en reconstruction. Pourquoi masquer la loss ? Copier les octets visibles ne suffit pas. Pourquoi validation/test ? Choisir sur le test biaise l'estimation. Pourquoi pas une perplexité classique ? Le conditionnement inclut le futur et dépend du masque. Pourquoi pas production ? Il manque des données représentatives, une évaluation forte, de la robustesse et un déploiement surveillé adapté à un usage concret.
