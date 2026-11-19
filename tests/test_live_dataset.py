"""Authorized live-source configuration and preview tests."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from live_dataset import contact_sheet, extractor_options, load_sources, selected_sources


class LiveDatasetTests(unittest.TestCase):
    def test_config_requires_authorized_unique_https_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {"id": "patio", "url": "https://example.test/live", "authorized": True}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            sources = load_sources(path)
        self.assertEqual(selected_sources(sources, ["patio"])[0]["id"], "patio")

    def test_anonymous_mode_never_reads_browser_cookies(self):
        self.assertNotIn("cookiesfrombrowser", extractor_options(None))
        self.assertEqual(extractor_options("edge")["cookiesfrombrowser"][0], "edge")

    def test_contact_sheet_is_created_from_candidate_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = []
            for index in range(2):
                path = root / f"frame-{index}.jpg"
                cv2.imwrite(str(path), np.full((120, 200, 3), 40 + index * 80, dtype=np.uint8))
                images.append(path)
            output = root / "preview.jpg"
            contact_sheet(images, output, "Patio")
            preview = cv2.imread(str(output))
        self.assertIsNotNone(preview)
        self.assertGreater(preview.shape[1], preview.shape[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
