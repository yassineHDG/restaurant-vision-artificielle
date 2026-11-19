"""Database operations for tables and occupancy sessions."""

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

from database.connection import get_connection


def sync_tables(tables: list[tuple[str, int]]) -> None:
    """Insert new tables and refresh capacities from the regions config.

    Existing tables are updated in place and missing tables are archived, never
    deleted, so historical occupancy sessions survive re-synchronization.
    """
    with closing(get_connection()) as conn:
        with conn:
            conn.execute("UPDATE tables SET active = 0")
            for name, capacity in tables:
                conn.execute(
                    """
                    INSERT INTO tables (name, capacity, active) VALUES (?, ?, 1)
                    ON CONFLICT (name) DO UPDATE SET
                        capacity = excluded.capacity,
                        active = 1
                    """,
                    (name, capacity),
                )


def _close_session(conn, name: str, ended_at: float) -> None:
    """Close the table's active session on an existing connection.

    ``ended_at`` is clamped to ``started_at`` so a session can never be
    closed with an end timestamp earlier than its start.
    """
    conn.execute(
        """
        UPDATE occupancy_sessions
        SET ended_at = MAX(started_at, ?)
        WHERE ended_at IS NULL
          AND table_id = (SELECT id FROM tables WHERE name = ?)
        """,
        (ended_at, name),
    )


def _close_all(conn, ended_at: float) -> None:
    """Close every active session on an existing connection (same clamp)."""
    conn.execute(
        """
        UPDATE occupancy_sessions
        SET ended_at = MAX(started_at, ?)
        WHERE ended_at IS NULL
        """,
        (ended_at,),
    )


def start_session(name: str, started_at: float, max_people: int) -> None:
    """Record a 0 -> positive occupancy transition as a new session.

    Timestamps are video-timeline seconds. Each processing run records its
    own rows: a session left active by a previous run (e.g. after a crash)
    is closed first, so old rows can never absorb or block the new run's
    session, and nothing about an old row is ever updated afterwards.
    """
    with closing(get_connection()) as conn:
        with conn:
            _close_session(conn, name, started_at)
            conn.execute(
                """
                INSERT INTO occupancy_sessions (table_id, started_at, max_people)
                SELECT id, ?, ? FROM tables WHERE name = ?
                """,
                (started_at, max_people, name),
            )


def update_session_max(name: str, max_people: int) -> None:
    """Raise ``max_people`` of the active session when the count exceeds it."""
    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                """
                UPDATE occupancy_sessions
                SET max_people = ?
                WHERE ended_at IS NULL
                  AND max_people < ?
                  AND table_id = (SELECT id FROM tables WHERE name = ?)
                """,
                (max_people, max_people, name),
            )


def close_session(name: str, ended_at: float) -> None:
    """Close the table's active session with a video-timeline end time."""
    with closing(get_connection()) as conn:
        with conn:
            _close_session(conn, name, ended_at)


def close_all_sessions(ended_at: float) -> None:
    """Close every active session at the end of a processing run.

    Called with the last processed video timestamp, so sessions of people
    still seated when the video ends are closed there and never left open.
    """
    with closing(get_connection()) as conn:
        with conn:
            _close_all(conn, ended_at)


def close_leftover_sessions() -> None:
    """Close active sessions left behind by an interrupted processing run.

    Called at the start of every processing run. Each leftover row is
    preserved and closed at its own start time, so ``ended_at`` is never
    earlier than ``started_at``.
    """
    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                """
                UPDATE occupancy_sessions
                SET ended_at = started_at
                WHERE ended_at IS NULL
                """
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def close_open_table_visits() -> None:
    """Close visits left open by a stopped process without deleting history."""
    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                """
                UPDATE table_visits
                SET ended_at_utc = started_at_utc,
                    ended_video_time = started_video_time
                WHERE ended_at_utc IS NULL
                """
            )


def start_table_visit(name: str, video_time: float, people: int) -> int | None:
    now = _utc_now()
    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                """
                UPDATE table_visits
                SET ended_at_utc = started_at_utc,
                    ended_video_time = started_video_time
                WHERE ended_at_utc IS NULL
                  AND table_id = (SELECT id FROM tables WHERE name = ?)
                """,
                (name,),
            )
            cursor = conn.execute(
                """
                INSERT INTO table_visits
                    (table_id, started_at_utc, started_video_time, max_people)
                SELECT id, ?, ?, ? FROM tables WHERE name = ?
                """,
                (now, float(video_time), int(people), name),
            )
            return int(cursor.lastrowid) if cursor.rowcount else None


def update_table_visit_max(visit_id: int | None, people: int) -> None:
    if visit_id is None:
        return
    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                """UPDATE table_visits
                   SET max_people = MAX(max_people, ?)
                   WHERE id = ? AND ended_at_utc IS NULL""",
                (int(people), int(visit_id)),
            )


def close_table_visit(visit_id: int | None, video_time: float) -> None:
    if visit_id is None:
        return
    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                """
                UPDATE table_visits
                SET ended_at_utc = ?,
                    ended_video_time = MAX(started_video_time, ?)
                WHERE id = ? AND ended_at_utc IS NULL
                """,
                (_utc_now(), float(video_time), int(visit_id)),
            )


def expected_table_duration(name: str, default_seconds: float) -> float:
    with closing(get_connection()) as conn:
        row = conn.execute(
            """
            SELECT AVG(ended_video_time - started_video_time)
            FROM table_visits AS v
            JOIN tables AS t ON t.id = v.table_id
            WHERE t.name = ?
              AND ended_video_time - started_video_time BETWEEN 300 AND 14400
            """,
            (name,),
        ).fetchone()
    value = row[0] if row else None
    return float(value) if value and value > 0 else float(default_seconds)


def record_queue_group(
    group_key: str, party_size: int, wait_seconds: float, outcome: str
) -> None:
    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO queue_group_visits
                    (group_key, party_size, wait_seconds, outcome, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (group_key, int(party_size), float(wait_seconds), outcome, _utc_now()),
            )


def record_operational_snapshot(snapshot: dict) -> None:
    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO operational_snapshots
                    (recorded_at, video_time, occupied_tables, total_tables,
                     seated_guests, queue_people, queue_groups, staff_visible,
                     average_wait_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_now(),
                    float(snapshot.get("video_time", 0)),
                    int(snapshot.get("occupied_tables", 0)),
                    int(snapshot.get("total_tables", 0)),
                    int(snapshot.get("seated_guests", 0)),
                    int(snapshot.get("queue_people", 0)),
                    int(snapshot.get("queue_groups", 0)),
                    int(snapshot.get("staff_visible", 0)),
                    float(snapshot.get("average_wait_seconds", 0)),
                ),
            )


def record_alert_event(alert: dict, event: str, video_time: float) -> None:
    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO alerts_history
                    (alert_key, event, kind, severity, message, table_name,
                     video_time, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert["id"],
                    event,
                    alert["kind"],
                    alert["severity"],
                    alert["message"],
                    alert.get("table_id"),
                    float(video_time),
                    _utc_now(),
                ),
            )


def operations_history(hours: int = 24) -> dict:
    hours = max(1, min(int(hours), 24 * 366))
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")
    with closing(get_connection()) as conn:
        hourly_rows = conn.execute(
            """
            SELECT strftime('%Y-%m-%dT%H:00:00Z', recorded_at) AS bucket,
                   AVG(CASE WHEN total_tables > 0
                       THEN occupied_tables * 100.0 / total_tables ELSE 0 END),
                   MAX(CASE WHEN total_tables > 0
                       THEN occupied_tables * 100.0 / total_tables ELSE 0 END),
                   AVG(queue_people), AVG(average_wait_seconds), AVG(staff_visible),
                   COUNT(*)
            FROM operational_snapshots
            WHERE recorded_at >= ?
            GROUP BY bucket ORDER BY bucket
            """,
            (since,),
        ).fetchall()
        daily_rows = conn.execute(
            """
            SELECT substr(recorded_at, 1, 10) AS bucket,
                   AVG(CASE WHEN total_tables > 0
                       THEN occupied_tables * 100.0 / total_tables ELSE 0 END),
                   MAX(CASE WHEN total_tables > 0
                       THEN occupied_tables * 100.0 / total_tables ELSE 0 END),
                   AVG(queue_people), AVG(average_wait_seconds), AVG(staff_visible),
                   COUNT(*)
            FROM operational_snapshots
            WHERE recorded_at >= ?
            GROUP BY bucket ORDER BY bucket
            """,
            (since,),
        ).fetchall()
        turnover_rows = conn.execute(
            """
            SELECT t.name, COUNT(v.id),
                   AVG(CASE WHEN v.ended_video_time > v.started_video_time
                       THEN v.ended_video_time - v.started_video_time END),
                   MAX(v.max_people)
            FROM tables AS t
            LEFT JOIN table_visits AS v
              ON v.table_id = t.id AND v.started_at_utc >= ?
            WHERE t.active = 1
            GROUP BY t.id, t.name ORDER BY t.name
            """,
            (since,),
        ).fetchall()
        queue_row = conn.execute(
            """
            SELECT COUNT(*), AVG(wait_seconds), MAX(wait_seconds)
            FROM queue_group_visits WHERE recorded_at >= ?
            """,
            (since,),
        ).fetchone()

    def series(rows):
        return [
            {
                "bucket": row[0],
                "occupancy_rate": round(float(row[1] or 0), 2),
                "peak_occupancy_rate": round(float(row[2] or 0), 2),
                "average_queue": round(float(row[3] or 0), 2),
                "average_wait_seconds": round(float(row[4] or 0), 2),
                "average_staff": round(float(row[5] or 0), 2),
                "samples": int(row[6]),
            }
            for row in rows
        ]

    hourly = series(hourly_rows)
    daily = series(daily_rows)
    peak_hours = []
    for item in sorted(
        hourly, key=lambda row: (row["peak_occupancy_rate"], row["average_queue"]), reverse=True
    )[:5]:
        extra_staff = int(item["peak_occupancy_rate"] >= 75) + int(
            item["average_queue"] >= 4
        )
        peak_hours.append(
            dict(
                item,
                recommended_staff=max(1, round(item["average_staff"]) + extra_staff),
            )
        )

    return {
        "hours": hours,
        "hourly": hourly,
        "daily": daily,
        "peak_hours": peak_hours,
        "turnover": [
            {
                "table_id": row[0],
                "groups_served": int(row[1] or 0),
                "average_visit_seconds": round(float(row[2] or 0), 2),
                "peak_party_size": int(row[3] or 0),
            }
            for row in turnover_rows
        ],
        "queue_groups": {
            "served": int(queue_row[0] or 0),
            "average_wait_seconds": round(float(queue_row[1] or 0), 2),
            "max_wait_seconds": round(float(queue_row[2] or 0), 2),
        },
    }


def export_snapshot_rows(hours: int = 24 * 30) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(hours=max(1, int(hours)))).isoformat(
        timespec="seconds"
    )
    with closing(get_connection()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT recorded_at, video_time, occupied_tables, total_tables,
                      seated_guests, queue_people, queue_groups, staff_visible,
                      average_wait_seconds
               FROM operational_snapshots
               WHERE recorded_at >= ? ORDER BY recorded_at""",
            (since,),
        ).fetchall()
    return [dict(row) for row in rows]
