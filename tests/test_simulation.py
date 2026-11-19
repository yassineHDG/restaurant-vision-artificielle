"""Behavioral checks for synthetic data and leakage/censoring boundaries.

Runs without the video stack: python -m unittest discover -s tests -p test_simulation.py -v
"""

from collections import Counter, defaultdict
import csv
from dataclasses import replace
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.events import EventType as E, RestaurantEvent
from src.simulation.config import ArrivalBand, SimulationConfig, TableSpec
from src.simulation.dataset import arrival_rows, outcome_rows, split_for_day, write_dataset
from src.simulation.engine import Arrival, simulate_day


class EventTests(unittest.TestCase):
    def event(self, **kwargs):
        values = dict(restaurant_id="restaurant", run_id="run", event_id="one",
                      event_type=E.PARTY_ARRIVED, elapsed_seconds=0, source="manual",
                      is_synthetic=False, observation_method="host")
        return RestaurantEvent(**(values | kwargs))

    def test_dated_and_undated_observations_round_trip(self):
        for occurred in (None, datetime(2026, 11, 1, 12, tzinfo=timezone.utc)):
            event = self.event(occurred_at=occurred, video_time_seconds=1.5)
            self.assertEqual(RestaurantEvent.from_dict(json.loads(json.dumps(event.to_dict()))), event)

    def test_invalid_timestamps_confidence_and_censoring_are_rejected(self):
        cases = [dict(occurred_at=datetime(2026, 11, 1)), dict(elapsed_seconds=-1),
                 dict(elapsed_seconds=None), dict(elapsed_seconds=float("nan")),
                 dict(video_time_seconds=float("inf")), dict(confidence=1.1),
                 dict(party_id="party", party_size=0), dict(capacity=4),
                 dict(censored=True), dict(schema_version=2), dict(is_synthetic=1)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.event(**kwargs)


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = SimulationConfig.load(ROOT / "config/simulation.json")

    def small(self, **kwargs):
        values = dict(opens_at="12:00", closes_at="13:00", arrival_bands=(),
                      tables=(TableSpec("T-01", 4),), meal_minutes_median=10,
                      meal_minutes_per_extra_guest=0, meal_log_sigma=0,
                      cleaning_minutes_median=2, cleaning_log_sigma=0,
                      service_minutes_mean=0.05)
        return replace(self.base, **(values | kwargs))

    def run_day(self, config=None, arrivals=None, seed=42):
        return simulate_day(config or self.small(), date(2026, 11, 1), seed, arrivals)

    def test_seed_reproducibility_event_identity_and_chronology(self):
        first = self.run_day(self.base)
        self.assertEqual(first, self.run_day(self.base))
        self.assertNotEqual([e.elapsed_seconds for e in first.events],
                            [e.elapsed_seconds for e in self.run_day(self.base, seed=43).events])
        self.assertEqual(len({e.event_id for e in first.events}), len(first.events))
        self.assertEqual([e.elapsed_seconds for e in first.events],
                         sorted(e.elapsed_seconds for e in first.events))
        self.assertTrue(all(e.occurred_at.utcoffset() is not None and e.is_synthetic for e in first.events))

    def test_table_is_not_reused_until_cleaning_is_confirmed(self):
        day = self.run_day(arrivals=[Arrival(0, 4), Arrival(1, 4)])
        seated = [e for e in day.events if e.event_type == E.PARTY_SEATED]
        departure = next(e for e in day.events if e.event_type == E.PARTY_DEPARTED)
        ready = next(e for e in day.events if e.event_type == E.TABLE_READY and e.party_id)
        self.assertEqual(len(seated), 2)
        self.assertAlmostEqual(ready.elapsed_seconds - departure.elapsed_seconds, 120)
        self.assertEqual(seated[1].elapsed_seconds, ready.elapsed_seconds)

    def test_unsupported_party_does_not_block_the_queue(self):
        day = self.run_day(arrivals=[Arrival(0, 8), Arrival(1, 4)])
        rejected = [e for e in day.events if e.event_type == E.PARTY_REJECTED]
        seated = [e for e in day.events if e.event_type == E.PARTY_SEATED]
        self.assertEqual([e.party_size for e in rejected], [8])
        self.assertEqual([e.party_size for e in seated], [4])

    def test_smallest_fitting_table_and_compatible_queue_bypass(self):
        config = self.small(tables=(TableSpec("small", 2), TableSpec("large", 4)))
        day = self.run_day(config, [Arrival(0, 4), Arrival(1, 4), Arrival(2, 2)])
        seated = [e for e in day.events if e.event_type == E.PARTY_SEATED]
        self.assertEqual([(e.elapsed_seconds, e.table_id) for e in seated[:2]], [(0, "large"), (2, "small")])
        self.assertEqual(seated[2].party_id, next(e.party_id for e in day.events
                                                if e.event_type == E.PARTY_ARRIVED and e.elapsed_seconds == 1))

    def test_abandoned_party_is_never_seated(self):
        day = self.run_day(arrivals=[Arrival(0, 4), Arrival(1, 2, patience_seconds=30)])
        abandoned = next(e for e in day.events if e.event_type == E.PARTY_ABANDONED)
        self.assertEqual(abandoned.elapsed_seconds, 31)
        self.assertFalse(any(e.party_id == abandoned.party_id and e.event_type == E.PARTY_SEATED
                             for e in day.events))

    def test_table_ready_exactly_at_patience_deadline_allows_seating(self):
        config = self.small(service_minutes_mean=0.000001, meal_minutes_median=1,
                            cleaning_minutes_median=1)
        day = self.run_day(config, [Arrival(0, 4), Arrival(1, 4, patience_seconds=120)])
        seated = [e for e in day.events if e.event_type == E.PARTY_SEATED]
        self.assertEqual([e.elapsed_seconds for e in seated], [0, 121])
        self.assertFalse(any(e.event_type == E.PARTY_ABANDONED for e in day.events))

    def test_unfinished_meals_and_queue_are_censored_without_fake_departures(self):
        config = self.small(meal_minutes_median=120)
        day = self.run_day(config, [Arrival(0, 4), Arrival(1, 2, patience_seconds=10000)])
        counts = Counter(e.event_type for e in day.events)
        self.assertEqual(counts[E.VISIT_CENSORED], 1)
        self.assertEqual(counts[E.QUEUE_CENSORED], 1)
        self.assertEqual(counts[E.PARTY_DEPARTED], 0)
        parties, visits, snapshots = outcome_rows(day, "train", config)
        self.assertIsNone(parties[1]["target_wait_to_seat_seconds"])
        self.assertEqual(parties[1]["queue_censored"], 1)
        self.assertIsNone(visits[0]["target_dining_seconds"])
        self.assertEqual(visits[0]["observed_seated_seconds"], 3600)
        self.assertTrue(all(s["target_remaining_dining_seconds"] is None for s in snapshots))

    def test_completed_meal_with_unfinished_cleaning_has_only_dining_label(self):
        config = self.small(cleaning_minutes_median=120)
        day = self.run_day(config, [Arrival(0, 2)])
        _, visits, snapshots = outcome_rows(day, "train", config)
        self.assertEqual(visits[0]["dining_censored"], 0)
        self.assertEqual(visits[0]["ready_censored"], 1)
        self.assertIsNotNone(visits[0]["target_dining_seconds"])
        self.assertIsNone(visits[0]["target_seated_to_ready_seconds"])
        self.assertTrue(all(s["target_remaining_ready_seconds"] is None for s in snapshots))
        self.assertTrue(any(e.event_type == E.TURNOVER_CENSORED for e in day.events))

    def test_service_alert_once_at_threshold_only_before_first_contact(self):
        config = self.small(servers=1, service_minutes_mean=30,
                            tables=tuple(TableSpec(f"T-{i}", 4) for i in range(6)))
        day = self.run_day(config, [Arrival(0, 2) for _ in range(6)])
        alerts = [e for e in day.events if e.event_type == E.SERVICE_ALERT]
        self.assertTrue(alerts)
        self.assertEqual(len(alerts), len({e.party_id for e in alerts}))
        served = {e.party_id: e.elapsed_seconds for e in day.events if e.event_type == E.SERVICE_CONFIRMED}
        for event in alerts:
            self.assertEqual(event.elapsed_seconds, 300)
            self.assertGreater(served.get(event.party_id, float("inf")), event.elapsed_seconds)
        _, _, snapshots = outcome_rows(day, "train", config)
        for row in snapshots:
            elapsed = (row["snapshot_at"] - day.opens_at).total_seconds()
            self.assertEqual(row["service_received"], int(served.get(row["party_id"], float("inf")) <= elapsed))

    def test_replay_preserves_capacity_state_and_cleaner_limits(self):
        day = self.run_day(self.base)
        state = {}
        seated_parties = set()
        cleaning = set()
        for event in day.events:
            if event.event_type == E.PARTY_SEATED:
                self.assertEqual(state[event.table_id], "READY")
                self.assertLessEqual(event.party_size, event.capacity)
                self.assertNotIn(event.party_id, seated_parties)
                seated_parties.add(event.party_id)
                state[event.table_id] = "OCCUPIED"
            elif event.event_type == E.PARTY_DEPARTED:
                self.assertEqual(state[event.table_id], "OCCUPIED")
                state[event.table_id] = "NEEDS_CLEANING"
            elif event.event_type == E.CLEANING_STARTED:
                self.assertEqual(state[event.table_id], "NEEDS_CLEANING")
                state[event.table_id] = "CLEANING"
                cleaning.add(event.table_id)
            elif event.event_type == E.TABLE_READY:
                if event.end_reason != "opening":
                    self.assertEqual(state[event.table_id], "CLEANING")
                    cleaning.remove(event.table_id)
                state[event.table_id] = "READY"
            self.assertLessEqual(len(cleaning), self.base.cleaners)

    def test_arrival_buckets_include_zeroes_and_exclude_future_features(self):
        config = self.small()
        day = self.run_day(config, [Arrival(0, 2), Arrival(900, 4)])
        rows = arrival_rows(day, "train", config)
        self.assertEqual([r["target_arriving_people"] for r in rows], [2, 4, 0, 0])
        self.assertEqual(rows[0]["queue_people_start"], 0)
        self.assertEqual(rows[0]["occupied_tables_start"], 0)
        self.assertEqual(rows[0]["ready_tables_start"], 1)
        self.assertIsNone(rows[0]["lag_1_arriving_people"])
        self.assertEqual(rows[1]["lag_1_arriving_people"], 2)
        self.assertEqual(sum(r["target_arriving_parties"] for r in rows), 2)

    def test_quiet_day_exports_all_zero_buckets_and_no_spurious_parties(self):
        day = self.run_day(arrivals=[])
        self.assertEqual(outcome_rows(day, "train", self.small()), ([], [], []))
        self.assertEqual(sum(r["target_arriving_people"] for r in arrival_rows(day, "train", self.small())), 0)

    def test_bad_configs_and_scripted_arrivals_fail_early(self):
        cases = [dict(cleaners=0), dict(servers=True), dict(meal_minutes_median=-1),
                 dict(party_size_weights=(0, 0)), dict(patience_log_sigma=float("nan")),
                 dict(tables=(TableSpec("same", 2), TableSpec("same", 4))),
                 dict(arrival_bands=(ArrivalBand("12:00", "13:00", 5), ArrivalBand("12:30", "13:00", 2))),
                 dict(opens_at="22:00", closes_at="10:00")]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.small(**kwargs)
        for arrival in (Arrival(-1, 2), Arrival(3600, 2), Arrival(0, 0), Arrival(0, 2, 0)):
            with self.subTest(arrival=arrival), self.assertRaises(ValueError):
                self.run_day(arrivals=[arrival])

    def test_exports_are_reproducible_grouped_by_day_and_hash_verified(self):
        config = self.small(arrival_bands=(ArrivalBand("12:00", "13:00", 20),))
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "first", Path(temp) / "second"
            manifest = write_dataset(config, date(2026, 11, 1), 7, 42, first)
            write_dataset(config, date(2026, 11, 1), 7, 42, second)
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
            self.assertEqual(manifest["split_days"], {"train": 4, "validation": 1, "test": 2})
            groups = defaultdict(set)
            for filename, metadata in manifest["files"].items():
                payload = (first / filename).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), metadata["sha256"])
                if filename.endswith(".csv"):
                    with (first / filename).open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual(len(rows), metadata["rows"])
                    for row in rows:
                        groups[row["run_id"]].add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
            with self.assertRaises(FileExistsError):
                write_dataset(config, date(2026, 11, 1), 7, 42, first)

    def test_split_edge_cases_keep_days_in_order(self):
        self.assertEqual([split_for_day(i, 1) for i in range(1)], ["train"])
        self.assertEqual([split_for_day(i, 2) for i in range(2)], ["train", "test"])
        self.assertEqual([split_for_day(i, 3) for i in range(3)], ["train", "validation", "test"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
