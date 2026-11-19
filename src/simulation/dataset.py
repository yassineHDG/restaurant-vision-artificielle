"""Export observed events, outcomes, and training tables with day-level splits."""

from collections import Counter
from contextlib import ExitStack
import csv
from datetime import date, timedelta
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform

from ..events import EventType as E
from .config import SimulationConfig
from .engine import DayResult, simulate_day


IDENTITY = ["restaurant_id", "run_id", "date", "split", "source", "is_synthetic"]
PARTY_COLUMNS = IDENTITY + [
    "party_id", "party_size", "arrived_at", "queue_exit_at", "outcome", "table_id",
    "target_wait_to_seat_seconds", "observed_queue_seconds", "queue_censored",
]
VISIT_COLUMNS = IDENTITY + [
    "party_id", "table_id", "capacity", "party_size", "seated_at", "service_confirmed_at",
    "departed_at", "ready_at", "observed_until", "observed_seated_seconds",
    "target_dining_seconds", "target_seated_to_ready_seconds", "dining_censored",
    "ready_censored", "end_reason",
]
DURATION_FEATURES = ["party_size", "capacity", "elapsed_seated_seconds", "weekday", "hour",
                     "minute", "service_received", "seconds_since_service"]
SNAPSHOT_COLUMNS = IDENTITY + ["party_id", "table_id", "snapshot_at"] + DURATION_FEATURES + [
    "target_remaining_dining_seconds", "target_remaining_ready_seconds",
    "dining_censored", "ready_censored",
]
ARRIVAL_FEATURES = ["weekday", "hour", "minute", "bucket_duration_seconds", "queue_people_start",
                    "occupied_tables_start", "ready_tables_start", "lag_1_arriving_people",
                    "lag_4_arriving_people", "past_4_mean_arriving_people"]
BUCKET_COLUMNS = IDENTITY + ["bucket_start", "bucket_end"] + ARRIVAL_FEATURES + [
    "target_arriving_parties", "target_arriving_people",
]


def split_for_day(index: int, days: int) -> str:
    """70/15/15 chronological days, reserving nonempty holdouts when possible."""
    if days < 1 or not 0 <= index < days:
        raise ValueError("Invalid day index or day count")
    if days == 1:
        return "train"
    if days == 2:
        return "train" if index == 0 else "test"
    train_end = min(days - 2, max(1, days * 70 // 100))
    validation_end = min(days - 1, max(train_end + 1, days * 85 // 100))
    return "train" if index < train_end else "validation" if index < validation_end else "test"


def _seconds(end, start) -> float:
    return round((end - start).total_seconds(), 6)


def outcome_rows(day: DayResult, split: str, config: SimulationConfig) -> tuple[list, list, list]:
    """Labels use future outcomes; snapshot features only use observations available then."""
    common = dict(restaurant_id=config.restaurant_id, run_id=day.run_id, date=day.day.isoformat(),
                  split=split, source="simulation", is_synthetic=1)
    parties = {}
    visits = {}
    for event in day.events:
        if event.is_prediction:
            continue
        kind, party_id, at = event.event_type, event.party_id, event.occurred_at
        if kind == E.PARTY_ARRIVED:
            parties[party_id] = dict(common, party_id=party_id, party_size=event.party_size,
                                     arrived_at=at, queue_exit_at=None, outcome="queued",
                                     table_id=None, target_wait_to_seat_seconds=None,
                                     observed_queue_seconds=None, queue_censored=0)
        elif kind in (E.PARTY_SEATED, E.PARTY_ABANDONED, E.PARTY_REJECTED, E.QUEUE_CENSORED):
            row = parties[party_id]
            censored = kind == E.QUEUE_CENSORED
            row.update(queue_exit_at=None if censored else at,
                       observed_queue_seconds=_seconds(at, row["arrived_at"]),
                       queue_censored=int(censored),
                       outcome={E.PARTY_SEATED: "seated", E.PARTY_ABANDONED: "abandoned",
                                E.PARTY_REJECTED: "unsupported_party_size",
                                E.QUEUE_CENSORED: "censored"}[kind])
            if kind == E.PARTY_SEATED:
                row.update(table_id=event.table_id,
                           target_wait_to_seat_seconds=_seconds(at, row["arrived_at"]))
                visits[party_id] = dict(common, party_id=party_id, table_id=event.table_id,
                    capacity=event.capacity, party_size=event.party_size, seated_at=at,
                    service_confirmed_at=None, departed_at=None, ready_at=None)
        elif kind == E.SERVICE_CONFIRMED:
            visits[party_id]["service_confirmed_at"] = at
        elif kind == E.PARTY_DEPARTED:
            visits[party_id]["departed_at"] = at
        elif kind == E.TABLE_READY and party_id is not None:
            visits[party_id]["ready_at"] = at

    snapshots = []
    for visit in visits.values():
        seated = visit["seated_at"]
        departed, ready, served = (visit[k] for k in ("departed_at", "ready_at", "service_confirmed_at"))
        until = departed or day.closes_at
        visit.update(observed_until=until, observed_seated_seconds=_seconds(until, seated),
                     target_dining_seconds=_seconds(departed, seated) if departed else None,
                     target_seated_to_ready_seconds=_seconds(ready, seated) if ready else None,
                     dining_censored=int(departed is None), ready_censored=int(ready is None),
                     end_reason="observed_departure" if departed else "observation_end")
        snapshot = seated
        while snapshot < until:
            service_received = served is not None and served <= snapshot
            snapshots.append(dict(common, party_id=visit["party_id"], table_id=visit["table_id"],
                snapshot_at=snapshot, party_size=visit["party_size"], capacity=visit["capacity"],
                elapsed_seated_seconds=_seconds(snapshot, seated), weekday=snapshot.weekday(),
                hour=snapshot.hour, minute=snapshot.minute, service_received=int(service_received),
                seconds_since_service=_seconds(snapshot, served) if service_received else None,
                target_remaining_dining_seconds=_seconds(departed, snapshot) if departed else None,
                target_remaining_ready_seconds=_seconds(ready, snapshot) if ready else None,
                dining_censored=int(departed is None), ready_censored=int(ready is None)))
            snapshot += timedelta(minutes=config.snapshot_minutes)
    snapshots.sort(key=lambda row: (row["snapshot_at"], row["party_id"]))
    return list(parties.values()), list(visits.values()), snapshots


def arrival_rows(day: DayResult, split: str, config: SimulationConfig) -> list[dict]:
    """Targets are [bucket start, bucket end); features use events strictly before start."""
    common = dict(restaurant_id=config.restaurant_id, run_id=day.run_id, date=day.day.isoformat(),
                  split=split, source="simulation", is_synthetic=1)
    events = [e for e in day.events if not e.is_prediction]
    cursor = 0
    queue = {}
    occupied = set()
    ready = {table.table_id for table in config.tables}
    history = []
    result = []
    duration = int((day.closes_at - day.opens_at).total_seconds())
    for start in range(0, duration, config.bucket_minutes * 60):
        end = min(duration, start + config.bucket_minutes * 60)
        while cursor < len(events) and events[cursor].elapsed_seconds < start:
            event = events[cursor]
            if event.event_type == E.PARTY_ARRIVED:
                queue[event.party_id] = event.party_size
            elif event.event_type in (E.PARTY_SEATED, E.PARTY_REJECTED, E.PARTY_ABANDONED, E.QUEUE_CENSORED):
                queue.pop(event.party_id, None)
            if event.event_type == E.PARTY_SEATED:
                occupied.add(event.table_id)
                ready.discard(event.table_id)
            elif event.event_type == E.PARTY_DEPARTED:
                occupied.discard(event.table_id)
            elif event.event_type == E.TABLE_READY:
                ready.add(event.table_id)
            cursor += 1
        count_parties = count_people = 0
        target_cursor = cursor
        while target_cursor < len(events) and events[target_cursor].elapsed_seconds < end:
            event = events[target_cursor]
            if event.event_type == E.PARTY_ARRIVED:
                count_parties += 1
                count_people += event.party_size
            target_cursor += 1
        at = day.opens_at + timedelta(seconds=start)
        result.append(dict(common, bucket_start=at, bucket_end=day.opens_at + timedelta(seconds=end),
            weekday=at.weekday(), hour=at.hour, minute=at.minute, bucket_duration_seconds=end - start,
            queue_people_start=sum(queue.values()), occupied_tables_start=len(occupied),
            ready_tables_start=len(ready), lag_1_arriving_people=history[-1] if history else None,
            lag_4_arriving_people=history[-4] if len(history) >= 4 else None,
            past_4_mean_arriving_people=sum(history[-4:]) / 4 if len(history) >= 4 else None,
            target_arriving_parties=count_parties, target_arriving_people=count_people))
        history.append(count_people)
    return result


def write_dataset(config: SimulationConfig, start_date: date, days: int, seed: int,
                  output: Path | str) -> dict:
    """Write a new dataset directory. An existing directory is never overwritten."""
    if type(days) is not int or days < 1:
        raise ValueError("days must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    # Validate date arithmetic before creating any output files.
    start_date + timedelta(days=days - 1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    columns = {"parties.csv": PARTY_COLUMNS, "visits.csv": VISIT_COLUMNS,
               "duration_snapshots.csv": SNAPSHOT_COLUMNS, "arrivals_15min.csv": BUCKET_COLUMNS}
    # The default is 15 minutes; use a neutral filename for other bucket widths.
    if config.bucket_minutes != 15:
        columns[f"arrivals_{config.bucket_minutes}min.csv"] = columns.pop("arrivals_15min.csv")
    arrival_file = f"arrivals_{config.bucket_minutes}min.csv"
    row_counts = Counter()
    outcomes = Counter()
    totals = Counter()
    split_counts = Counter()
    day_metadata = []
    with ExitStack() as stack:
        event_file = stack.enter_context((output / "events.jsonl").open("w", encoding="utf-8", newline="\n"))
        writers = {}
        for filename, fieldnames in columns.items():
            handle = stack.enter_context((output / filename).open("w", encoding="utf-8", newline=""))
            writers[filename] = csv.DictWriter(handle, fieldnames=fieldnames)
            writers[filename].writeheader()
        for index in range(days):
            day = simulate_day(config, start_date + timedelta(days=index), seed)
            split = split_for_day(index, days)
            split_counts[split] += 1
            day_metadata.append(dict(date=day.day.isoformat(), run_id=day.run_id,
                                     day_seed=str(day.day_seed), split=split))
            for event in day.events:
                event_file.write(json.dumps(event.to_dict(), sort_keys=True, allow_nan=False) + "\n")
                row_counts["events.jsonl"] += 1
                totals["service_alerts"] += event.event_type == E.SERVICE_ALERT
            parties, visits, snapshots = outcome_rows(day, split, config)
            for row in parties:
                outcomes[row["outcome"]] += 1
                if row["outcome"] == "seated":
                    totals["seated_wait_seconds"] += row["target_wait_to_seat_seconds"]
            totals["completed_visits"] += sum(not row["dining_censored"] for row in visits)
            totals["censored_visits"] += sum(row["dining_censored"] for row in visits)
            for filename, rows in (("parties.csv", parties), ("visits.csv", visits),
                                   ("duration_snapshots.csv", snapshots),
                                   (arrival_file, arrival_rows(day, split, config))):
                for row in rows:
                    writers[filename].writerow({key: value.isoformat() if hasattr(value, "isoformat")
                                               else value for key, value in row.items()})
                    row_counts[filename] += 1
    file_metadata = {}
    for filename in ["events.jsonl", *columns]:
        checksum = hashlib.sha256()
        with (output / filename).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(block)
        file_metadata[filename] = dict(rows=row_counts[filename], sha256=checksum.hexdigest())
    dataset_key = f"v1:{config.fingerprint}:{start_date}:{days}:{seed}"
    try:
        timezone_data = version("tzdata")
    except PackageNotFoundError:
        timezone_data = "system IANA database"
    manifest = dict(
        dataset_version=1, event_schema_version=1, generator="restaurant-simulator-v1",
        dataset_id=hashlib.sha256(dataset_key.encode()).hexdigest()[:16],
        source="simulation", is_synthetic=True, seed=seed, start_date=start_date.isoformat(),
        environment=dict(python=platform.python_version(), tzdata=timezone_data),
        days=days, config=config.to_dict(), config_sha256=config.fingerprint,
        split_days=dict(split_counts), runs=day_metadata, files=file_metadata,
        summary=dict(parties_arrived=sum(outcomes.values()), outcomes=dict(outcomes),
                     completed_visits=totals["completed_visits"], censored_visits=totals["censored_visits"],
                     service_alerts=totals["service_alerts"],
                     mean_seated_wait_minutes=round(totals["seated_wait_seconds"] / outcomes["seated"] / 60, 3)
                     if outcomes["seated"] else None),
        features=dict(arrivals=ARRIVAL_FEATURES, remaining_duration=DURATION_FEATURES),
        notes=["Synthetic outcomes do not measure real-restaurant accuracy.",
               "Every run starts empty; observation stops at closing without forcing departures.",
               "Blank targets are unknown/censored, never zero-duration meals.",
               "Whole days are assigned chronologically before snapshots are made.",
               "Arrival features precede bucket start; lags reset each day.",
               "Targets, IDs, split, and censoring flags are not model input features.",
               "Reservations, table combining, and camera errors are not simulated in v1."])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    return manifest
