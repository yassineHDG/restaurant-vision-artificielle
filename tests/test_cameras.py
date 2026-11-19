"""Normalized multi-camera configuration tests."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cameras import resolve_camera_sources, scale_points, validate_camera_config


def camera_config():
    return {
        "version": 1,
        "cameras": [
            {
                "id": "dining",
                "role": "dining",
                "regions": {
                    "tables": [
                        {
                            "id": "T-01",
                            "capacity": 4,
                            "counting_polygon": [[0, 0], [0.5, 0], [0.5, 0.5]],
                        }
                    ]
                },
            },
            {
                "id": "entrance",
                "role": "entrance",
                "regions": {
                    "queue_polygon": [[0.1, 0.2], [0.5, 0.2], [0.5, 0.8]],
                    "transit_polygon": [],
                    "interior_polygon": [],
                    "entry_line": [[0.2, 0.5], [0.8, 0.5]],
                },
            },
        ],
    }


class CameraConfigTests(unittest.TestCase):
    def test_normalized_points_scale_to_any_resolution(self):
        self.assertEqual(scale_points([[0.25, 0.5], [1, 1]], 1920, 1080), [[480, 540], [1920, 1080]])

    def test_validates_required_roles_and_normalized_coordinates(self):
        self.assertEqual(len(validate_camera_config(camera_config())["cameras"]), 2)
        invalid = camera_config()
        invalid["cameras"][1]["regions"]["queue_polygon"][0][0] = 1.1
        with self.assertRaisesRegex(ValueError, "invalid normalized point"):
            validate_camera_config(invalid)

    def test_rejects_duplicate_camera_ids(self):
        invalid = camera_config()
        invalid["cameras"][1]["id"] = "dining"
        with self.assertRaisesRegex(ValueError, "Duplicate camera id"):
            validate_camera_config(invalid)

    def test_changed_source_disables_stale_table_geometry_without_deleting_it(self):
        config = camera_config()
        dining = config["cameras"][0]
        dining["source_setting"] = "DINING_VIDEO_PATH"
        dining["calibration_source"] = "videos/old-room.mp4"
        with patch(
            "cameras.settings",
            SimpleNamespace(DINING_VIDEO_PATH="https://www.youtube.com/watch?v=live"),
        ):
            resolved = resolve_camera_sources(config)
        self.assertTrue(resolved[0]["calibration_required"])
        self.assertEqual(resolved[0]["regions"]["tables"], [])
        self.assertEqual(config["cameras"][0]["regions"]["tables"][0]["id"], "T-01")


if __name__ == "__main__":
    unittest.main(verbosity=2)
