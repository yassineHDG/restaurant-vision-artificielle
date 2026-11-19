"""Tracked-person assignment to calibrated seating zones."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services.table_assignment import TableAssignmentTracker


def zone(name, counting, display):
    return {
        "name": name,
        "counting_polygon": np.asarray(counting, dtype=np.int32),
        "display_polygon": np.asarray(display, dtype=np.int32),
    }


class TableAssignmentTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = TableAssignmentTracker(hold_seconds=1.5, margin_ratio=0)
        self.tracker.configure(
            [
                zone("T-01", [[0, 0], [70, 0], [70, 100], [0, 100]], [[10, 30], [30, 30], [30, 50], [10, 50]]),
                zone("T-02", [[30, 0], [100, 0], [100, 100], [30, 100]], [[70, 30], [90, 30], [90, 50], [70, 50]]),
            ],
            width=100,
            height=100,
        )

    def test_overlapping_zones_choose_the_nearest_table_exclusively(self):
        counts = self.tracker.update(
            [{"track_id": 7, "role": "unknown", "eligible": True, "anchor": [65, 40]}],
            0.0,
        )
        self.assertEqual(counts, {"T-01": 0, "T-02": 1})

    def test_previous_assignment_is_stable_and_survives_short_occlusion(self):
        self.tracker.update(
            [{"track_id": 7, "role": "unknown", "eligible": True, "anchor": [20, 40]}],
            0.0,
        )
        overlapping = self.tracker.update(
            [{"track_id": 7, "role": "unknown", "eligible": True, "anchor": [50, 40]}],
            0.5,
        )
        self.assertEqual(overlapping, {"T-01": 1, "T-02": 0})
        self.assertEqual(self.tracker.update([], 1.9), {"T-01": 1, "T-02": 0})
        self.assertEqual(self.tracker.update([], 2.01), {"T-01": 0, "T-02": 0})

    def test_staff_track_is_removed_immediately(self):
        self.tracker.update(
            [{"track_id": 3, "role": "unknown", "eligible": True, "anchor": [20, 40]}],
            0.0,
        )
        counts = self.tracker.update(
            [{"track_id": 3, "role": "staff", "eligible": False, "anchor": [20, 40]}],
            0.2,
        )
        self.assertEqual(counts, {"T-01": 0, "T-02": 0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
