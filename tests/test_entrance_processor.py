"""Queue dwell, passerby, and staff-exclusion tests."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services.entrance_processor import EntranceProcessor
from services.person_roles import RoleHistory


class FakeRoleTracker:
    def __init__(self, people):
        self.people = people

    def update(self, result, frame, video_time):
        return self.people


def person(track_id, center, role="unknown"):
    return {
        "track_id": track_id,
        "xyxy": [center[0] - 5, center[1] - 20, center[0] + 5, center[1]],
        "center": list(center),
        "score": 0.9,
        "role": role,
        "history": RoleHistory(first_seen=0, last_seen=0),
    }


class EntranceProcessorTests(unittest.TestCase):
    def setUp(self):
        self.processor = EntranceProcessor(
            {
                "regions": {
                    "queue_polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]],
                    "transit_polygon": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]],
                    "interior_polygon": [],
                    "entry_line": [[0.5, 0], [0.5, 1]],
                }
            }
        )
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_waiting_becomes_confirmed_only_after_dwell_threshold(self):
        waiting = person(1, (25, 60))
        self.processor.role_tracker = FakeRoleTracker([waiting])
        _, first = self.processor.process_result(self.frame, None, 0.0)
        _, later = self.processor.process_result(self.frame, None, 3.1)
        self.assertEqual(first["uncertain_queue_count"], 1)
        self.assertEqual(first["confirmed_queue_count"], 0)
        self.assertEqual(later["confirmed_queue_count"], 1)
        self.assertEqual(later["tracks"][0]["role"], "customer")

    def test_staff_in_queue_is_excluded(self):
        staff = person(2, (25, 60), role="staff")
        staff["history"].queue_entered = 0.0
        self.processor.role_tracker = FakeRoleTracker([staff])
        _, stats = self.processor.process_result(self.frame, None, 10.0)
        self.assertEqual(stats["staff_visible"], 1)
        self.assertEqual(stats["confirmed_queue_count"], 0)
        self.assertEqual(stats["tracks"][0]["queue_state"], "excluded_staff")

    def test_fast_person_in_transit_is_a_passerby(self):
        passerby = person(3, (75, 60))
        self.processor.role_tracker = FakeRoleTracker([passerby])
        _, stats = self.processor.process_result(self.frame, None, 1.0)
        self.assertEqual(stats["confirmed_queue_count"], 0)
        self.assertEqual(stats["uncertain_queue_count"], 0)
        self.assertEqual(stats["tracks"][0]["queue_state"], "passerby")

    def test_wait_is_averaged_per_group_and_completed_once(self):
        waiting = person(9, (25, 60))
        tracker = FakeRoleTracker([waiting])
        self.processor.role_tracker = tracker
        self.processor.process_result(self.frame, None, 0.0)
        _, active = self.processor.process_result(self.frame, None, 3.5)
        self.assertEqual(active["confirmed_group_count"], 1)
        self.assertEqual(active["average_wait_seconds"], 3.5)
        tracker.people = []
        _, completed = self.processor.process_result(self.frame, None, 6.0)
        self.assertEqual(len(completed["completed_groups"]), 1)
        self.assertEqual(completed["completed_groups"][0]["party_size"], 1)
        _, next_frame = self.processor.process_result(self.frame, None, 7.0)
        self.assertEqual(next_frame["completed_groups"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
