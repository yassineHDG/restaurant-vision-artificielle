# Validation datasets

The approved first review pack is declared in `data/datasets/manifest.json`.
No dataset is silently mixed into training data. The workflow is:

1. Review the source page and license statement.
2. Download only the selected assets into `data/datasets/<dataset-id>/raw/`.
3. Record every SHA-256 checksum and local size in the manifest.
4. Generate a contact sheet and metadata summary for user validation.
5. Keep public-dataset, simulated, and staged-team results separate.

CAVIAR is used for entrances, crossings, waiting, and the shop-assistant
scenario. Collective Activity supplies explicit waiting/queueing sequences.
Mall is a counting benchmark only because its sampling rate is too low for a
meaningful persistent-ID evaluation.

Staff/customer validation requires a short staged team recording with the
chosen visible apron. Public roles are not treated as ground truth unless the
dataset explicitly annotates them.

## Pack proposé pour validation

| Dataset | Aperçus officiels | Taille prévue | Décision avant téléchargement |
| --- | --- | ---: | --- |
| CAVIAR | [frame annotée](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/head_hands_image.jpg) | 20,407,994 octets pour les trois clips choisis | Vérifier CC BY-SA et l'adéquation des scénarios |
| Collective Activity | [waiting](https://cvgl.stanford.edu/projects/collective/images/exwait.jpg), [queueing](https://cvgl.stanford.edu/projects/collective/images/exqueue.jpg) | inconnue (serveur lent au test HEAD) | Confirmer les conditions d'utilisation, non indiquées sur la page |
| Mall | [frame](https://personal.ie.cuhk.edu.hk/~ccloy/images/shopping_mall.jpg), [annotation](https://personal.ie.cuhk.edu.hk/~ccloy/images/shopping_mall_annotated.jpg) | 91,833,998 octets | Usage recherche uniquement, pas commercial |

Les champs SHA-256 restent explicitement `null` avec l'état
`pending_download`. Ils seront calculés sur les octets effectivement reçus;
inventer un checksum avant téléchargement rendrait le manifeste trompeur.
