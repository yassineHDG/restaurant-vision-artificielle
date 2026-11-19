"""Operational states, alerts, settings, and history sampling tests."""

import sys
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack, closing
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from operations_config import OperationsSettings, load_operations_settings, save_operations_settings
from database.connection import get_connection, init_database
from database.repository import (
    close_table_visit,
    expected_table_duration,
    operations_history,
    start_table_visit,
    sync_tables,
)
from services.operations_service import OperationsService


REPOSITORY_CALLS = (
    "close_open_table_visits",
    "close_table_visit",
    "expected_table_duration",
    "record_alert_event",
    "record_operational_snapshot",
    "record_queue_group",
    "start_table_visit",
    "update_table_visit_max",
)


class OperationsServiceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.mocks = {
            name: self.stack.enter_context(patch(f"services.operations_service.{name}"))
            for name in REPOSITORY_CALLS
        }
        self.mocks["expected_table_duration"].return_value = 2700.0
        self.mocks["start_table_visit"].return_value = 11
        self.service = OperationsService([{"id": "T-01", "max": 4}])
        self.service.update_settings(
            OperationsSettings(
                service_alert_seconds=30,
                cleaning_alert_seconds=0,
                queue_alert_people=3,
                max_queue_wait_seconds=60,
                history_sample_seconds=1,
            )
        )

    def tearDown(self):
        self.stack.close()

    @staticmethod
    def row(guests, elapsed=None):
        return [{"id": "T-01", "guests": guests, "occupancy_seconds": elapsed}]

    def test_table_lifecycle_alerts_and_release_prediction(self):
        self.service.update_tables(self.row(2, 0), 0)
        occupied = self.service.live_status()["tables"]["T-01"]
        self.assertEqual(occupied["state"], "occupied")
        self.assertEqual(occupied["predicted_release_seconds"], 2700)

        self.service.update_tables(self.row(2, 31), 31)
        self.assertIn("service:T-01", self.service.alerts)
        self.service.table_action("T-01", "served")
        self.assertNotIn("service:T-01", self.service.alerts)

        self.service.update_tables(self.row(0), 60)
        self.assertEqual(self.service.live_status()["tables"]["T-01"]["state"], "needs_cleaning")
        self.assertIn("cleaning:T-01", self.service.alerts)
        self.service.table_action("T-01", "ready")
        self.assertEqual(self.service.live_status()["tables"]["T-01"]["state"], "ready")
        self.assertNotIn("cleaning:T-01", self.service.alerts)

    def test_queue_threshold_average_wait_and_completed_group(self):
        self.service.update_queue(
            {
                "confirmed_queue_count": 4,
                "confirmed_group_count": 2,
                "average_wait_seconds": 42,
                "max_wait_seconds": 61,
                "staff_visible": 1,
                "completed_groups": [
                    {"group_id": 7, "party_size": 3, "wait_seconds": 50, "outcome": "seated"}
                ],
            },
            100,
        )
        live = self.service.live_status()
        self.assertEqual(live["average_wait_seconds"], 42)
        self.assertIn("queue:capacity", self.service.alerts)
        self.assertIn("queue:wait", self.service.alerts)
        self.mocks["record_queue_group"].assert_called_once_with("7", 3, 50.0, "seated")

    def test_snapshots_are_rate_limited(self):
        self.service.update_tables(self.row(1, 0), 0)
        self.service.update_tables(self.row(1, 0.4), 0.4)
        self.service.update_tables(self.row(1, 1.1), 1.1)
        self.assertEqual(self.mocks["record_operational_snapshot"].call_count, 2)


class OperationsConfigTests(unittest.TestCase):
    def test_partial_updates_are_validated_and_preserve_other_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operations.json"
            saved = save_operations_settings({"queue_alert_people": 12}, path)
            self.assertEqual(saved.queue_alert_people, 12)
            self.assertEqual(load_operations_settings(path).stream_jpeg_quality, 95)
            with self.assertRaises(ValueError):
                save_operations_settings({"stream_jpeg_quality": 10}, path)


class TableArchiveTests(unittest.TestCase):
    def test_short_demo_visits_do_not_train_release_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restaurant.db"
            with patch("database.connection.DATABASE_PATH", path):
                init_database()
                sync_tables([("T-01", 4)])
                short_visit = start_table_visit("T-01", 0, 2)
                close_table_visit(short_visit, 45)
                self.assertEqual(expected_table_duration("T-01", 2700), 2700)
                real_visit = start_table_visit("T-01", 100, 2)
                close_table_visit(real_visit, 1000)
                self.assertEqual(expected_table_duration("T-01", 2700), 900)

    def test_removed_calibration_tables_are_archived_not_listed_in_turnover(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restaurant.db"
            with patch("database.connection.DATABASE_PATH", path):
                init_database()
                sync_tables([("T-01", 4), ("T-02", 4)])
                sync_tables([("T-01", 6)])
                with closing(get_connection()) as connection:
                    rows = connection.execute(
                        "SELECT name, capacity, active FROM tables ORDER BY name"
                    ).fetchall()
                history = operations_history(24)
        self.assertEqual(rows, [("T-01", 6, 1), ("T-02", 4, 0)])
        self.assertEqual([row["table_id"] for row in history["turnover"]], ["T-01"])

    def test_existing_database_is_migrated_with_active_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restaurant.db"
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE tables (id INTEGER PRIMARY KEY, name TEXT UNIQUE, capacity INTEGER)"
            )
            connection.commit()
            connection.close()
            with patch("database.connection.DATABASE_PATH", path):
                init_database()
                with closing(get_connection()) as connection:
                    columns = {row[1] for row in connection.execute("PRAGMA table_info(tables)")}
        self.assertIn("active", columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
