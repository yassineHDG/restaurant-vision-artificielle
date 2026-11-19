# Réentraînement staff / clients

Le modèle spécialisé est facultatif : le système continue d'utiliser le tablier
calibré, puis fusionne cette preuve avec le modèle `staff/customer` lorsque
`ROLE_MODEL_PATH` est défini. Aucune reconnaissance faciale n'est utilisée.

## 1. Préparer les annotations

Placez les images et labels YOLO sous `data/training/restaurant_roles` :

```text
images/train  images/val  images/test
labels/train  labels/val  labels/test
```

Les classes sont `0=customer_seated`, `1=customer_standing` et `2=staff`.
Gardez toutes les frames d'une même
séquence dans le même split. Le test doit contenir une scène réellement nouvelle.

## 2. Auditer avant l'entraînement

```powershell
uv run python src/training.py audit
```

La commande refuse les labels manquants, les classes inconnues et les coordonnées
invalides. Elle refuse aussi l'entraînement sans un minimum d'images et
d'instances dans chaque split.

## 3. Comparer plusieurs modèles sur la RTX 3050

```powershell
uv run python src/training.py train --models yolo26n.pt,yolo26s.pt --epochs 80 --batch 4 --device cuda:0
```

Chaque modèle est évalué sur `test`; les poids, le mAP50 et le mAP50-95 sont
enregistrés dans `runs/roles/experiment-*.json`. Commencez avec `batch=4`; réduisez
à 2 en cas de manque de VRAM sur la RTX 3050 Laptop.

## 4. Activer le meilleur modèle

Dans `.env` :

```dotenv
ROLE_MODEL_PATH=runs/roles/roles-yolo26n-seed42/weights/best.pt
ROLE_MODEL_CONF=0.45
```

Redémarrez ensuite le backend. Le modèle n'est activé que si ce chemin est rempli,
ce qui évite de présenter un réentraînement non validé comme opérationnel.
