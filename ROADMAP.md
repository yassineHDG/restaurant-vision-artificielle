# Roadmap — Restaurant Vision

Cette roadmap transforme le dépôt de démonstration en projet de vision artificielle
évaluable. La règle principale est simple : **on n'affiche comme “live” que ce qui
est réellement mesuré**. La file d'attente et les prévisions restent signalées comme
planifiées jusqu'à leur connexion au pipeline.

## État actuel

- [x] Dépôt cloné et dépendances installées.
- [x] Démonstration YOLO et flux vidéo accessibles sur `localhost:5000`.
- [x] Comptage des clients assis par zones de tables.
- [x] Interface de supervision et endpoint live `/api/status`.
- [x] Simulateur reproductible et export d'événements pour les futurs modèles.
- [x] RTX 3050 configurée avec PyTorch `cu130`, contrôle CUDA strict et benchmark.
- [x] Pipeline deux caméras, modèle partagé, lecteurs parallèles et trackers séparés.
- [x] File d'attente détectée après calibration : confirmé/incertain, staff et passants exclus.
- [x] Interface de calibration normalisée et proposition de tables par YOLO Seg.
- [ ] Validation de la file sur la vraie caméra d'entrée du groupe.
- [ ] Modèles ML entraînés et comparés.
- [ ] Prédictions intégrées à l'interface.

## Phase 1 — Notre caméra et la vérité terrain

**Objectif :** remplacer la vidéo de démonstration par une vidéo du restaurant et
mesurer le point de départ du système.

1. Filmer une séquence fixe couvrant les tables et une séquence couvrant l'entrée.
2. Dessiner les polygones des tables et de la zone d'entrée dans `/calibration`.
3. Annoter un petit jeu de test : personne assise/debout, table occupée/libre,
   entrée/sortie de la file.
4. Mesurer le comptage MAE, le F1 d'occupation, le délai de transition et les FPS.
5. Tester une taille de modèle plus petite si le flux reste lent.

**Terminé lorsque :** la vidéo du groupe fonctionne sans modifier le code, les zones
sont versionnées et les métriques de référence sont consignées.

## Phase 2 — File d'attente réelle

**Objectif :** savoir combien de personnes attendent, depuis combien de temps et dans
quel ordre approximatif.

1. [x] Ajouter une zone polygonale de file distincte des tables.
2. [x] Utiliser un ByteTrack indépendant par caméra.
3. [x] Compter les identifiants après trois secondes et filtrer staff/passants.
4. [ ] Produire et persister les événements `queue_entered` et `queue_left`.
5. [x] Exposer `/api/queue` et connecter le module “File d'attente” du frontend.

Pour le MVP, l'hôte saisit la taille des groupes. Une caméra peut compter des
personnes, mais ne peut pas toujours déduire correctement quelles personnes forment
un même groupe.

**Terminé lorsque :** le compteur de file et les temps d'attente sont corrects sur
une vidéo annotée, y compris lorsque des personnes se croisent.

## Phase 3 — États métier et alertes

**Objectif :** convertir les observations vidéo en décisions utiles au personnel.

1. Implémenter `READY`, `OCCUPIED`, `NEEDS_CLEANING`, `CLEANING`, `UNKNOWN`.
2. Ajouter les confirmations manuelles “servi” et “table nettoyée”.
3. Déclencher une alerte si une table attend un serveur plus de cinq minutes.
4. Déclencher une alerte de nettoyage après un départ confirmé.
5. Stocker l'historique append-only dans SQLite avec `run_id` et la provenance.

**Terminé lorsque :** un scénario complet arrivée → repas → départ → nettoyage →
table prête est rejouable et visible dans l'interface.

## Phase 4 — Entraînement et comparaison ML

Les tâches doivent avoir des jeux de données, des cibles et des métriques séparés.

| Prédiction | Baseline | Modèles à comparer | Métrique principale |
| --- | --- | --- | --- |
| Arrivées par 15 min | moyenne même jour/créneau | Ridge, Random Forest, HistGradientBoosting | MAE/RMSE, surtout au pic |
| Temps restant d'une table | médiane conditionnelle | Ridge, Random Forest, HistGradientBoosting | MAE en minutes |
| Durée du nettoyage | médiane historique | modèle seulement si assez de données | MAE en minutes |

Les découpages sont chronologiques. Toutes les observations d'une même visite restent
dans le même split. Les données simulées, publiques et réelles sont évaluées
séparément afin de ne pas présenter la simulation comme une preuve réelle.

**Terminé lorsque :** chaque modèle est reproductible par une commande, comparé à sa
baseline sur un test tenu à l'écart et sauvegardé avec ses métriques et paramètres.

## Phase 5 — Prédiction de file et limite dynamique

**Objectif :** recommander combien de personnes peuvent raisonnablement attendre en
fonction des tables qui vont bientôt se libérer, et non appliquer une limite fixe de 8.

1. Prédire l'heure de disponibilité de chaque table occupée, nettoyage inclus.
2. Ordonner les groupes en attente et les affecter aux tables compatibles.
3. Calculer le temps estimé de chaque groupe et la capacité à 15 minutes.
4. Afficher une fourchette d'incertitude seulement si elle est calibrée.
5. Tester les groupes mixtes, les retards de nettoyage et les pics de midi.

**Terminé lorsque :** le système explique la recommandation, par exemple : “8
personnes recommandées car deux tables de 4 seront prêtes dans 15 minutes”.

## Phase 6 — Intégration, évaluation et soutenance

1. Relier les prédictions, alertes et historiques au frontend.
2. Tester les scénarios calme, pic 12h–14h et nettoyage retardé.
3. Mesurer précision, latence, FPS et robustesse sur une vidéo jamais utilisée.
4. Protéger la vie privée : pas de reconnaissance faciale, conservation minimale,
   floutage des visages si la vidéo doit être montrée.
5. Préparer une démo enregistrée de secours, le rapport et les contributions Git.

## Répartition des quatre personnes

| Personne | Responsabilité | Livrable principal |
| --- | --- | --- |
| 1 | Vision, tracking et calibration | zones, suivi, benchmark vidéo |
| 2 | Données, annotation et simulation | événements, jeux de données, qualité |
| 3 | ML et évaluation | scripts d'entraînement, modèles, rapport comparatif |
| 4 | Backend, frontend et intégration | API, dashboard, scheduler et démonstration |

Tout le monde participe aux tests, au rapport et à la présentation. Les interfaces de
données (`event_type`, identifiants, temps et métriques) doivent être décidées ensemble
avant que chaque personne travaille en parallèle.

## Ordre recommandé

`Caméra + annotations → tracking de file → états métier → modèles ML → scheduler → intégration finale`

Commencer l'entraînement avant d'avoir des labels fiables produirait seulement des
résultats difficiles à interpréter. Le prochain sprint est donc **Phase 1**, puis la
zone d'entrée de la **Phase 2**.
