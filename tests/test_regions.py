"""Region loading and optional display-polygon behavior."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from counting.load_regions import load_zones
from services.video_processor import VideoProcessor


class RegionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "regions.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _write_regions(self, zone):
        self.path.write_text(json.dumps({"zones": [zone]}), encoding="utf-8")

    def _zone(self, **extra):
        return {
            "name": "T-01",
            "counting_polygon": [[0, 0], [30, 0], [30, 30]],
            "max": 4,
            **extra,
        }

    def test_display_polygon_may_be_omitted(self):
        self._write_regions(self._zone())
        zones = load_zones(self.path)
        self.assertNotIn("display_polygon", zones[0])

    def test_provided_display_polygon_is_validated(self):
        self._write_regions(self._zone(display_polygon=[[0, 0], [1, 1]]))
        with self.assertRaisesRegex(ValueError, "display_polygon"):
            load_zones(self.path)

    def test_video_processor_counts_without_display_polygon(self):
        config = self._zone()
        with patch("services.video_processor.YOLO") as yolo, patch(
            "services.video_processor.load_zones", return_value=[config]
        ):
            processor = VideoProcessor()

        self.assertIsNone(processor.zones[0]["annotator"])
        self.assertIsNotNone(processor.zones[0]["counting_zone"])

        yolo.return_value.predict.return_value = [SimpleNamespace(keypoints=None)]
        detections = sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            class_id=np.empty((0,), dtype=int),
        )
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch.object(sv.Detections, "from_ultralytics", return_value=detections):
            processed, stats = processor.process_frame(frame, 0.0)

        self.assertIs(processed, frame)
        self.assertEqual(stats["table_counts"], {"T-01": 0})

    def test_video_processor_keeps_annotator_when_display_polygon_is_present(self):
        config = self._zone(display_polygon=[[0, 0], [30, 0], [30, 30]])
        with patch("services.video_processor.YOLO"), patch(
            "services.video_processor.load_zones", return_value=[config]
        ):
            processor = VideoProcessor()

        self.assertIsNotNone(processor.zones[0]["annotator"])

    def test_normalized_camera_tables_scale_on_first_frame(self):
        camera = {
            "regions": {
                "tables": [
                    {
                        "id": "T-01",
                        "capacity": 4,
                        "counting_polygon": [[0, 0], [0.5, 0], [0.5, 0.5]],
                        "display_polygon": [[0, 0], [0.5, 0], [0.5, 0.5]],
                    }
                ]
            }
        }
        with patch("services.video_processor.YOLO"):
            processor = VideoProcessor(camera_config=camera)
        self.assertEqual(processor.zones, [])
        detections = sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            class_id=np.empty((0,), dtype=int),
        )
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        result = SimpleNamespace(keypoints=None, boxes=None)
        with patch.object(sv.Detections, "from_ultralytics", return_value=detections):
            processor.process_result(frame, result, 0.0)
        np.testing.assert_array_equal(
            processor.zones[0]["counting_zone"].polygon,
            np.array([[0, 0], [100, 0], [100, 50]]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
