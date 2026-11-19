"""SQLite database initialization and connections for the restaurant."""

import sqlite3
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "restaurant.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tables (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    capacity INTEGER NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS occupancy_sessions (
    id INTEGER PRIMARY KEY,
    table_id INTEGER NOT NULL REFERENCES tables (id),
    started_at REAL NOT NULL,
    ended_at REAL,
    max_people INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_session_per_table
    ON occupancy_sessions (table_id)
    WHERE ended_at IS NULL;

CREATE TABLE IF NOT EXISTS table_visits (
    id INTEGER PRIMARY KEY,
    table_id INTEGER NOT NULL REFERENCES tables (id),
    started_at_utc TEXT NOT NULL,
    ended_at_utc TEXT,
    started_video_time REAL NOT NULL,
    ended_video_time REAL,
    max_people INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_visit_per_table
    ON table_visits (table_id)
    WHERE ended_at_utc IS NULL;

CREATE TABLE IF NOT EXISTS queue_group_visits (
    id INTEGER PRIMARY KEY,
    group_key TEXT NOT NULL,
    party_size INTEGER NOT NULL,
    wait_seconds REAL NOT NULL,
    outcome TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS operational_snapshots (
    id INTEGER PRIMARY KEY,
    recorded_at TEXT NOT NULL,
    video_time REAL NOT NULL,
    occupied_tables INTEGER NOT NULL,
    total_tables INTEGER NOT NULL,
    seated_guests INTEGER NOT NULL,
    queue_people INTEGER NOT NULL,
    queue_groups INTEGER NOT NULL,
    staff_visible INTEGER NOT NULL,
    average_wait_seconds REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS snapshots_recorded_at
    ON operational_snapshots (recorded_at);

CREATE TABLE IF NOT EXISTS alerts_history (
    id INTEGER PRIMARY KEY,
    alert_key TEXT NOT NULL,
    event TEXT NOT NULL,
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    table_name TEXT,
    video_time REAL NOT NULL,
    recorded_at TEXT NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    """Open a database connection with foreign-key enforcement enabled.

    ``PRAGMA foreign_keys`` is per-connection in SQLite, so it must be
    enabled on every connection (not only at initialization).
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database() -> None:
    """Create the database file and schema if they do not exist yet."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(get_connection()) as conn:
        with conn:
            conn.executescript(SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(tables)")}
            if "active" not in columns:
                conn.execute(
                    "ALTER TABLE tables ADD COLUMN active INTEGER NOT NULL DEFAULT 1"
                )
