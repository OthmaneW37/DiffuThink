# Résultats de référence

Exécution locale : 600 mises à jour sur CPU, 180 800 paramètres, environ 172 secondes d'entraînement et validation, graine 42. Checkpoint retenu : étape 600. Voir `reference_run.json` pour toutes les données et versions.

| Fraction masquée | Accuracy du modèle | Accuracy unigramme | CE modèle |
| --- | ---: | ---: | ---: |
| 25 % | 96,18 % | 17,07 % | 0,137 |
| 50 % | 89,10 % | 17,83 % | 0,326 |
| 75 % | 69,72 % | 17,79 % | 0,890 |
| 100 % | 17,64 % | 17,64 % | 2,869 |

Ces résultats portent sur 115 phrases de test d'une grammaire synthétique commune aux partitions. Ils mesurent une reconstruction en une passe, pas une qualité conversationnelle.

## Budget d'inférence

Sur 24 phrases du test, avec les mêmes trous tirés à 50 %, température zéro :

| Étapes demandées | Accuracy sur les trous | Phrases exactes | Latence moyenne CPU |
| --- | ---: | ---: | ---: |
| 1 | 90,00 % | 29,17 % | 12,5 ms |
| 4 | 84,89 % | 25,00 % | 50,7 ms |
| 12 | 84,44 % | 25,00 % | 146,5 ms |

La révélation itérative n'améliore pas ce run. Le sampler fixe les erreurs et modifie le contexte utilisé lors des passes suivantes. Ces mesures ne permettent pas d'isoler la cause ; un remasquage et un meilleur conditionnement sont des hypothèses à tester sur validation. Les temps sont des mesures locales indicatives, sans intervalle de confiance. Tous les exemples, y compris les échecs, sont dans `sampling.json`.

Exemple vérifié : `The c~t runs in the g~rden today.` devient `The cat runs in the garden today.`. Un trou couvrant tout un mot ambigu peut au contraire produire un mot invalide. La reconstruction entièrement masquée ne dépasse pas l'unigramme ; les repères de position et l'objectif d'entraînement nécessitent une nouvelle expérience avant de revendiquer une génération libre satisfaisante.

## Vérifications

Neuf tests passent : UTF-8, masquage hors padding, conservation des normes RoPE, invariance au padding, perte sélective, contexte/seed/calendrier du sampler, checkpoint, séparation des données/évaluation reproductible et apprentissage sur un mini-batch. La CLI d'évaluation reproduit les métriques du rapport. L'API locale et la démonstration de reconstruction ont été exécutées.
