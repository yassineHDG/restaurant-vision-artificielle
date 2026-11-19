"""Training dataset audit tests."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from training import audit_dataset


class TrainingAuditTests(unittest.TestCase):
    def make_dataset(
        self,
        root: Path,
        label=(
            "0 0.5 0.5 0.2 0.4\n"
            "1 0.4 0.5 0.2 0.4\n"
            "2 0.6 0.5 0.2 0.4\n"
        ),
    ):
        config = root / "roles.yaml"
        config.write_text(
            "path: dataset\ntrain: images/train\nval: images/val\ntest: images/test\n"
            "names:\n  0: customer_seated\n  1: customer_standing\n  2: staff\n",
            encoding="utf-8",
        )
        for split in ("train", "val", "test"):
            images = root / "dataset" / "images" / split
            labels = root / "dataset" / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / "frame.jpg").write_bytes(b"test")
            (labels / "frame.txt").write_text(label, encoding="utf-8")
        return config

    def test_valid_yolo_dataset_is_audited_by_split_and_class(self):
        with tempfile.TemporaryDirectory() as directory:
            report = audit_dataset(self.make_dataset(Path(directory)))
        self.assertEqual(report["errors"], [])
        self.assertEqual(
            report["splits"]["train"]["instances"],
            {"customer_seated": 1, "customer_standing": 1, "staff": 1},
        )

    def test_invalid_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            report = audit_dataset(
                self.make_dataset(Path(directory), label="3 0.5 0.5 0.2 0.4\n")
            )
        self.assertTrue(any("classe 3 inconnue" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
