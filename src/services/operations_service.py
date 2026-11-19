"""Operational state, alerts, predictions, and historical snapshots."""

from __future__ import annotations

import threading
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone

from database.repository import (
    close_open_table_visits,
    close_table_visit,
    expected_table_duration,
    record_alert_event,
    record_operational_snapshot,
    record_queue_group,
    start_table_visit,
    update_table_visit_max,
)
from operations_config import OperationsSettings, load_operations_settings


TABLE_STATES = {"free", "occupied", "needs_cleaning", "ready"}


class OperationsService:
    """Turn vision observations into actions a restaurant staff can use."""

    def __init__(self, tables: list[dict], source: str = ""):
        self.lock = threading.RLock()
        self.source = source
        self.settings = load_operations_settings()
        self.tables = {
            table["id"]: {
                "capacity": int(table.get("max", table.get("capacity", 0))),
                "state": "free",
                "guests": 0,
                "previous_guests": 0,
                "occupancy_seconds": None,
                "served": False,
                "cleaning_since": None,
                "visit_id": None,
                "max_people": 0,
            }
            for table in tables
        }
        self.expected_durations = {}
        self.alerts: dict[str, dict] = {}
        self.recent_alerts = deque(maxlen=40)
        self.queue = {
            "confirmed_queue_count": 0,
            "confirmed_group_count": 0,
            "average_wait_seconds": 0.0,
            "max_wait_seconds": 0.0,
            "staff_visible": 0,
        }
        self.dining_staff_visible = 0
        self.last_video_time = 0.0
        self._last_snapshot_time = float("-inf")
        self.begin_run()

    def begin_run(self) -> None:
        with self.lock:
            close_open_table_visits()
            self.alerts.clear()
            self.recent_alerts.clear()
            self.last_video_time = 0.0
            self._last_snapshot_time = float("-inf")
            default = self.settings.default_dining_minutes * 60
            for table_id, table in self.tables.items():
                table.update(
                    state="free",
                    guests=0,
                    previous_guests=0,
                    occupancy_seconds=None,
                    served=False,
                    cleaning_since=None,
                    visit_id=None,
                    max_people=0,
                )
                self.expected_durations[table_id] = expected_table_duration(
                    table_id, default
                )

    def end_run(self, video_time: float) -> None:
        with self.lock:
            for table in self.tables.values():
                close_table_visit(table.get("visit_id"), video_time)
                table["visit_id"] = None

    def update_settings(self, settings: OperationsSettings) -> None:
        with self.lock:
            self.settings = settings
            default = settings.default_dining_minutes * 60
            for table_id in self.tables:
                self.expected_durations[table_id] = expected_table_duration(
                    table_id, default
                )

    def update_tables(
        self, rows: list[dict], video_time: float, staff_visible: int = 0
    ) -> dict[str, dict]:
        with self.lock:
            self.last_video_time = float(video_time)
            self.dining_staff_visible = int(staff_visible)
            for row in rows:
                table_id = row["id"]
                if table_id not in self.tables:
                    continue
                table = self.tables[table_id]
                guests = int(row.get("guests", 0))
                previous = int(table["guests"])
                elapsed = row.get("occupancy_seconds")
                table["previous_guests"] = previous
                table["guests"] = guests
                table["occupancy_seconds"] = elapsed

                if previous == 0 and guests > 0:
                    table["state"] = "occupied"
                    table["served"] = False
                    table["cleaning_since"] = None
                    table["max_people"] = guests
                    table["visit_id"] = start_table_visit(
                        table_id, video_time, guests
                    )
                    self._resolve_alert(f"cleaning:{table_id}", video_time)
                elif previous > 0 and guests == 0:
                    close_table_visit(table.get("visit_id"), video_time)
                    table["visit_id"] = None
                    table["state"] = "needs_cleaning"
                    table["cleaning_since"] = float(video_time)
                    table["served"] = False
                    self.expected_durations[table_id] = expected_table_duration(
                        table_id, self.settings.default_dining_minutes * 60
                    )
                    self._open_alert(
                        alert_id=f"cleaning:{table_id}",
                        kind="table_cleaning",
                        severity="warning",
                        title=f"{table_id} à nettoyer",
                        message="Le groupe est parti. La table doit être nettoyée avant le prochain placement.",
                        video_time=video_time,
                        table_id=table_id,
                    )

                if guests > 0:
                    table["state"] = "occupied"
                    if guests > table["max_people"]:
                        table["max_people"] = guests
                        update_table_visit_max(table.get("visit_id"), guests)
                    if (
                        not table["served"]
                        and elapsed is not None
                        and elapsed >= self.settings.service_alert_seconds
                    ):
                        self._open_alert(
                            alert_id=f"service:{table_id}",
                            kind="service_delay",
                            severity="danger",
                            title=f"Service attendu à {table_id}",
                            message=f"Les clients attendent depuis {self._minutes(elapsed)} sans confirmation de service.",
                            video_time=video_time,
                            table_id=table_id,
                        )
                else:
                    self._resolve_alert(f"service:{table_id}", video_time)
                    if table["state"] == "needs_cleaning":
                        cleaning_elapsed = max(
                            0.0, video_time - float(table["cleaning_since"] or video_time)
                        )
                        if cleaning_elapsed >= self.settings.cleaning_alert_seconds:
                            self._open_alert(
                                alert_id=f"cleaning:{table_id}",
                                kind="table_cleaning",
                                severity="danger",
                                title=f"Nettoyage en retard — {table_id}",
                                message=f"La table attend depuis {self._minutes(cleaning_elapsed)}.",
                                video_time=video_time,
                                table_id=table_id,
                            )

            self._record_snapshot_if_due(video_time)
            return self._table_snapshot()

    def update_queue(self, queue_stats: dict, video_time: float) -> None:
        with self.lock:
            self.queue.update(
                confirmed_queue_count=int(queue_stats.get("confirmed_queue_count", 0)),
                confirmed_group_count=int(queue_stats.get("confirmed_group_count", 0)),
                average_wait_seconds=float(queue_stats.get("average_wait_seconds", 0)),
                max_wait_seconds=float(queue_stats.get("max_wait_seconds", 0)),
                staff_visible=int(queue_stats.get("staff_visible", 0)),
            )
            count = self.queue["confirmed_queue_count"]
            if count >= self.settings.queue_alert_people:
                self._open_alert(
                    alert_id="queue:capacity",
                    kind="queue_threshold",
                    severity="danger",
                    title="File d’attente élevée",
                    message=f"{count} personnes attendent, seuil réglé à {self.settings.queue_alert_people}.",
                    video_time=video_time,
                )
            else:
                self._resolve_alert("queue:capacity", video_time)

            max_wait = self.queue["max_wait_seconds"]
            if max_wait >= self.settings.max_queue_wait_seconds:
                self._open_alert(
                    alert_id="queue:wait",
                    kind="queue_wait",
                    severity="warning",
                    title="Attente excessive",
                    message=f"Le groupe le plus ancien attend depuis {self._minutes(max_wait)}.",
                    video_time=video_time,
                )
            else:
                self._resolve_alert("queue:wait", video_time)

            for group in queue_stats.get("completed_groups", []):
                record_queue_group(
                    str(group["group_id"]),
                    int(group["party_size"]),
                    float(group["wait_seconds"]),
                    str(group.get("outcome", "left_queue")),
                )

    def table_action(self, table_id: str, action: str) -> dict:
        with self.lock:
            if table_id not in self.tables:
                raise KeyError(table_id)
            table = self.tables[table_id]
            if action == "served":
                if table["guests"] <= 0:
                    raise ValueError("Une table vide ne peut pas être marquée comme servie")
                table["served"] = True
                self._resolve_alert(f"service:{table_id}", self.last_video_time)
            elif action == "ready":
                if table["guests"] > 0:
                    raise ValueError("Une table occupée ne peut pas être marquée comme prête")
                table["state"] = "ready"
                table["cleaning_since"] = None
                self._resolve_alert(f"cleaning:{table_id}", self.last_video_time)
            elif action == "free":
                if table["guests"] > 0:
                    raise ValueError("Une table occupée ne peut pas être marquée comme libre")
                table["state"] = "free"
            elif action == "needs_cleaning":
                if table["guests"] > 0:
                    raise ValueError("Une table occupée ne peut pas passer au nettoyage")
                table["state"] = "needs_cleaning"
                table["cleaning_since"] = self.last_video_time
            else:
                raise ValueError(f"Action inconnue: {action}")
            return deepcopy(self._table_snapshot()[table_id])

    def acknowledge_alert(self, alert_id: str) -> dict:
        with self.lock:
            alert = self.alerts.get(alert_id)
            if alert is None:
                raise KeyError(alert_id)
            alert["acknowledged"] = True
            alert["acknowledged_at"] = self._utc_now()
            record_alert_event(alert, "acknowledged", self.last_video_time)
            return deepcopy(alert)

    def live_status(self) -> dict:
        with self.lock:
            table_snapshot = self._table_snapshot()
            occupied = sum(item["state"] == "occupied" for item in table_snapshot.values())
            total = len(table_snapshot)
            return {
                "occupancy_rate": round(occupied / total * 100, 1) if total else 0.0,
                "staff_visible": self.dining_staff_visible + self.queue["staff_visible"],
                "average_wait_seconds": round(self.queue["average_wait_seconds"], 2),
                "estimated_wait_seconds": round(self._estimated_wait(), 2),
                "queue_people": self.queue["confirmed_queue_count"],
                "queue_groups": self.queue["confirmed_group_count"],
                "active_alerts": len(self.alerts),
                "alerts": sorted(
                    (deepcopy(alert) for alert in self.alerts.values()),
                    key=lambda item: (item["severity"] != "danger", item["opened_video_time"]),
                ),
                "recent_alerts": list(deepcopy(self.recent_alerts)),
                "tables": table_snapshot,
                "settings": self.settings.model_dump(),
            }

    def _table_snapshot(self) -> dict[str, dict]:
        result = {}
        for table_id, table in self.tables.items():
            elapsed = table.get("occupancy_seconds")
            expected = self.expected_durations.get(
                table_id, self.settings.default_dining_minutes * 60
            )
            remaining = (
                max(0.0, expected - float(elapsed or 0))
                if table["state"] == "occupied"
                else None
            )
            result[table_id] = {
                "id": table_id,
                "state": table["state"],
                "served": bool(table["served"]),
                "predicted_release_seconds": round(remaining, 2) if remaining is not None else None,
                "expected_visit_seconds": round(expected, 2),
            }
        return result

    def _estimated_wait(self) -> float:
        available = sum(
            table["state"] in {"free", "ready"} for table in self.tables.values()
        )
        if available > 0:
            return 0.0
        releases = []
        for table_id, table in self.tables.items():
            if table["state"] != "occupied":
                continue
            expected = self.expected_durations.get(
                table_id, self.settings.default_dining_minutes * 60
            )
            releases.append(max(0.0, expected - float(table.get("occupancy_seconds") or 0)))
        if not releases:
            return 0.0
        groups_ahead = max(0, int(self.queue["confirmed_group_count"]))
        cycles = groups_ahead // max(1, len(self.tables))
        return min(releases) + cycles * self.settings.default_dining_minutes * 60

    def _record_snapshot_if_due(self, video_time: float) -> None:
        if video_time - self._last_snapshot_time < self.settings.history_sample_seconds:
            return
        self._last_snapshot_time = float(video_time)
        occupied = sum(table["state"] == "occupied" for table in self.tables.values())
        record_operational_snapshot(
            {
                "video_time": video_time,
                "occupied_tables": occupied,
                "total_tables": len(self.tables),
                "seated_guests": sum(table["guests"] for table in self.tables.values()),
                "queue_people": self.queue["confirmed_queue_count"],
                "queue_groups": self.queue["confirmed_group_count"],
                "staff_visible": self.dining_staff_visible + self.queue["staff_visible"],
                "average_wait_seconds": self.queue["average_wait_seconds"],
            }
        )

    def _open_alert(
        self,
        *,
        alert_id: str,
        kind: str,
        severity: str,
        title: str,
        message: str,
        video_time: float,
        table_id: str | None = None,
    ) -> None:
        existing = self.alerts.get(alert_id)
        if existing is not None:
            existing.update(severity=severity, title=title, message=message)
            return
        alert = {
            "id": alert_id,
            "kind": kind,
            "severity": severity,
            "title": title,
            "message": message,
            "table_id": table_id,
            "opened_video_time": round(float(video_time), 3),
            "opened_at": self._utc_now(),
            "acknowledged": False,
        }
        self.alerts[alert_id] = alert
        record_alert_event(alert, "opened", video_time)

    def _resolve_alert(self, alert_id: str, video_time: float) -> None:
        alert = self.alerts.pop(alert_id, None)
        if alert is None:
            return
        alert["resolved_at"] = self._utc_now()
        self.recent_alerts.appendleft(alert)
        record_alert_event(alert, "resolved", video_time)

    @staticmethod
    def _minutes(seconds: float) -> str:
        minutes, secs = divmod(max(0, int(seconds)), 60)
        return f"{minutes} min {secs:02d} s"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
