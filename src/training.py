"""Audit, train, and evaluate a restaurant staff/customer detector."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "config" / "restaurant_roles.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_NAMES = {0: "customer_seated", 1: "customer_standing", 2: "staff"}


def load_dataset_config(path: Path) -> tuple[dict, Path, dict[int, str]]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    root = Path(config.get("path", ""))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    raw_names = config.get("names", {})
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    else:
        names = {int(index): str(name) for index, name in raw_names.items()}
    if names != EXPECTED_NAMES:
        raise ValueError(
            "Les classes attendues sont exactement "
            "0=customer_seated, 1=customer_standing et 2=staff"
        )
    return config, root, names


def audit_dataset(path: Path) -> dict:
    config, root, names = load_dataset_config(path)
    report = {
        "dataset": str(path.resolve()),
        "root": str(root),
        "classes": names,
        "splits": {},
        "errors": [],
    }
    for split in ("train", "val", "test"):
        image_path = Path(config.get(split, f"images/{split}"))
        if not image_path.is_absolute():
            image_path = root / image_path
        label_path = root / "labels" / split
        images = sorted(
            item for item in image_path.glob("**/*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ) if image_path.exists() else []
        counts: Counter[str] = Counter()
        missing_labels = 0
        invalid_labels = 0
        labelled_images = 0
        for image in images:
            relative = image.relative_to(image_path).with_suffix(".txt")
            label = label_path / relative
            if not label.exists():
                missing_labels += 1
                report["errors"].append(f"Label manquant: {label}")
                continue
            labelled_images += 1
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                try:
                    if len(parts) != 5:
                        raise ValueError("5 colonnes attendues")
                    class_id = int(parts[0])
                    coordinates = [float(value) for value in parts[1:]]
                    if class_id not in names:
                        raise ValueError(f"classe {class_id} inconnue")
                    if not all(0 <= value <= 1 for value in coordinates):
                        raise ValueError("coordonnées hors de [0,1]")
                    if coordinates[2] <= 0 or coordinates[3] <= 0:
                        raise ValueError("largeur/hauteur nulle")
                    counts[names[class_id]] += 1
                except ValueError as exc:
                    invalid_labels += 1
                    report["errors"].append(f"{label}:{line_number}: {exc}")
        if not images:
            report["errors"].append(f"Aucune image dans le split {split}: {image_path}")
        report["splits"][split] = {
            "images": len(images),
            "labelled_images": labelled_images,
            "missing_labels": missing_labels,
            "invalid_labels": invalid_labels,
            "instances": dict(counts),
        }
    return report


def assert_training_ready(report: dict) -> None:
    if report["errors"]:
        raise RuntimeError("Le dataset contient des erreurs; lancez d'abord la commande audit")
    for split, minimum in (("train", 20), ("val", 5), ("test", 5)):
        stats = report["splits"][split]
        if stats["images"] < minimum:
            raise RuntimeError(f"{split}: au moins {minimum} images sont requises")
        for class_name in EXPECTED_NAMES.values():
            if stats["instances"].get(class_name, 0) < minimum:
                raise RuntimeError(
                    f"{split}: au moins {minimum} instances {class_name} sont requises"
                )


def require_cuda(device: str) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("L'entraînement exige CUDA. Exécutez Test-GpuPython avant de continuer.")


def train_models(args, report: dict) -> list[dict]:
    assert_training_ready(report)
    require_cuda(args.device)
    project = Path(args.project).resolve()
    project.mkdir(parents=True, exist_ok=True)
    results = []
    for model_name in [item.strip() for item in args.models.split(",") if item.strip()]:
        stem = Path(model_name).stem
        run_name = f"roles-{stem}-seed{args.seed}"
        model = YOLO(model_name)
        model.train(
            data=str(Path(args.data).resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            patience=args.patience,
            project=str(project),
            name=run_name,
            exist_ok=False,
            plots=True,
            verbose=True,
        )
        best = Path(model.trainer.best).resolve()
        metrics = YOLO(str(best)).val(
            data=str(Path(args.data).resolve()), split="test", device=args.device
        )
        results.append(
            {
                "base_model": model_name,
                "best_weights": str(best),
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
            }
        )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": report,
        "device": args.device,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "results": results,
    }
    manifest_path = project / f"experiment-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return results


def evaluate(args) -> dict:
    report = audit_dataset(Path(args.data))
    assert_training_ready(report)
    require_cuda(args.device)
    metrics = YOLO(args.weights).val(
        data=str(Path(args.data).resolve()), split="test", device=args.device
    )
    return {
        "weights": str(Path(args.weights).resolve()),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_map50_95": [float(value) for value in metrics.box.maps],
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Restaurant Vision model training")
    commands = root.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Validate annotations and split integrity")
    audit.add_argument("--data", default=str(DEFAULT_DATA))
    train = commands.add_parser("train", help="Train and compare role detectors")
    train.add_argument("--data", default=str(DEFAULT_DATA))
    train.add_argument("--models", default="yolo26n.pt,yolo26s.pt")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=4)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "roles"))
    validation = commands.add_parser("evaluate", help="Evaluate one model on the held-out test set")
    validation.add_argument("--data", default=str(DEFAULT_DATA))
    validation.add_argument("--weights", required=True)
    validation.add_argument("--device", default="cuda:0")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "audit":
            result = audit_dataset(Path(args.data))
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 2 if result["errors"] else 0
        if args.command == "train":
            result = train_models(args, audit_dataset(Path(args.data)))
        else:
            result = evaluate(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Erreur: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
