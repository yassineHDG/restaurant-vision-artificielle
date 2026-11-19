"""Entrance-camera tracking, staff exclusion, and queue dwell analytics."""

from __future__ import annotations

import cv2
import numpy as np

from cameras import scale_points
from config import settings
from operations_config import OperationsSettings, load_operations_settings
from services.person_roles import PersonRoleTracker


def _inside(point, polygon) -> bool:
    if len(polygon) < 3:
        return False
    contour = np.asarray(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(contour, tuple(point), False) >= 0


class EntranceProcessor:
    def __init__(
        self,
        camera_config: dict,
        frame_rate: float | None = None,
        operations_settings: OperationsSettings | None = None,
    ):
        marker = camera_config.get("staff_marker", {})
        self.operations_settings = operations_settings or load_operations_settings()
        self.regions = camera_config.get("regions", {})
        self.dwell_seconds = settings.QUEUE_DWELL_SECONDS
        self.role_tracker = PersonRoleTracker(
            frame_rate=frame_rate or settings.TARGET_FPS,
            hsv_lower=marker.get(
                "hsv_lower", settings.parse_hsv(settings.STAFF_MARKER_HSV_LOWER)
            ),
            hsv_upper=marker.get(
                "hsv_upper", settings.parse_hsv(settings.STAFF_MARKER_HSV_UPPER)
            ),
            minimum_ratio=marker.get(
                "minimum_ratio", settings.STAFF_MARKER_MIN_RATIO
            ),
        )
        self._next_group_id = 1
        self._track_groups: dict[int, int] = {}
        self._groups: dict[int, dict] = {}

    def process_result(self, frame, result, video_time: float, role_result=None):
        height, width = frame.shape[:2]
        queue_polygon = scale_points(
            self.regions.get("queue_polygon", []), width, height
        )
        transit_polygon = scale_points(
            self.regions.get("transit_polygon", []), width, height
        )
        interior_polygon = scale_points(
            self.regions.get("interior_polygon", []), width, height
        )
        entry_line = scale_points(self.regions.get("entry_line", []), width, height)

        tracked_people = (
            self.role_tracker.update(result, frame, video_time)
            if role_result is None
            else self.role_tracker.update(
                result, frame, video_time, role_result=role_result
            )
        )
        visible_tracks = []
        confirmed = 0
        uncertain = 0
        staff = 0
        active_group_tracks: dict[int, set[int]] = {}

        annotated = frame.copy()
        for polygon, color, label in (
            (queue_polygon, (29, 143, 122), "FILE"),
            (transit_polygon, (143, 119, 76), "PASSAGE"),
            (interior_polygon, (119, 80, 143), "INTERIEUR"),
        ):
            if len(polygon) >= 3:
                cv2.polylines(
                    annotated,
                    [np.asarray(polygon, dtype=np.int32)],
                    True,
                    color,
                    2,
                )
                cv2.putText(
                    annotated,
                    label,
                    tuple(polygon[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        if len(entry_line) == 2:
            cv2.line(annotated, tuple(entry_line[0]), tuple(entry_line[1]), (230, 180, 55), 2)

        for person in tracked_people:
            track_id = person["track_id"]
            history = person["history"]
            in_queue = _inside(person["center"], queue_polygon)
            in_transit = _inside(person["center"], transit_polygon)
            in_interior = _inside(person["center"], interior_polygon)

            if in_queue and history.queue_entered is None:
                history.queue_entered = video_time
            elif not in_queue:
                history.queue_entered = None

            dwell = (
                max(0.0, video_time - history.queue_entered)
                if history.queue_entered is not None
                else 0.0
            )
            role = person["role"]
            queue_state = "outside"
            if role == "staff":
                staff += 1
                queue_state = "excluded_staff"
            elif in_queue and dwell >= self.dwell_seconds:
                role = "customer"
                queue_state = "confirmed"
                confirmed += 1
            elif in_queue:
                queue_state = "uncertain"
                uncertain += 1
            elif in_transit:
                queue_state = "passerby"

            group_id = None
            if in_queue and role != "staff":
                group_id = self._group_for_track(
                    track_id,
                    history.queue_entered if history.queue_entered is not None else video_time,
                    video_time,
                )
                active_group_tracks.setdefault(group_id, set()).add(track_id)
                group = self._groups[group_id]
                group["last_seen"] = video_time
                group["members"].add(track_id)
            elif in_interior:
                existing_group = self._track_groups.get(track_id)
                if existing_group in self._groups:
                    self._groups[existing_group]["entered_interior"] = True

            x1, y1, x2, y2 = [int(value) for value in person["xyxy"]]
            color = (47, 153, 93) if role == "customer" else (52, 97, 190)
            if role == "staff":
                color = (184, 98, 54)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            face_bottom = min(y2, y1 + max(10, int((y2 - y1) * 0.25)))
            face = annotated[max(0, y1) : face_bottom, max(0, x1) : max(0, x2)]
            if face.size:
                annotated[max(0, y1) : face_bottom, max(0, x1) : max(0, x2)] = cv2.GaussianBlur(
                    face, (21, 21), 0
                )
            label = f"#{track_id} {role}"
            if in_queue and role != "staff":
                label += f" {dwell:.1f}s"
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )
            visible_tracks.append(
                {
                    "track_id": track_id,
                    "role": role,
                    "queue_state": queue_state,
                    "dwell_seconds": round(dwell, 2),
                    "in_queue": in_queue,
                    "in_transit": in_transit,
                    "in_interior": in_interior,
                    "confidence": round(person["score"], 4),
                    "group_id": group_id,
                }
            )

        groups, completed_groups = self._update_groups(active_group_tracks, video_time)
        confirmed_groups = [
            group for group in groups if group["dwell_seconds"] >= self.dwell_seconds
        ]
        average_wait = (
            sum(group["dwell_seconds"] for group in confirmed_groups) / len(confirmed_groups)
            if confirmed_groups
            else 0.0
        )
        max_wait = max((group["dwell_seconds"] for group in confirmed_groups), default=0.0)

        stats = {
            "confirmed_queue_count": confirmed,
            "confirmed_group_count": len(confirmed_groups),
            "uncertain_queue_count": uncertain,
            "staff_visible": staff,
            "tracks": visible_tracks,
            "groups": groups,
            "completed_groups": completed_groups,
            "average_wait_seconds": round(average_wait, 2),
            "max_wait_seconds": round(max_wait, 2),
            "calibration_required": len(queue_polygon) < 3,
        }
        return annotated, stats

    def _group_for_track(
        self, track_id: int, entered_at: float, video_time: float
    ) -> int:
        existing = self._track_groups.get(track_id)
        if existing in self._groups:
            return existing

        join_window = self.operations_settings.queue_group_join_seconds
        candidates = [
            group
            for group in self._groups.values()
            if video_time - group["created_at"] <= join_window
            and video_time - group["last_seen"] <= join_window
        ]
        if candidates:
            selected = min(
                candidates, key=lambda group: abs(group["created_at"] - entered_at)
            )
            group_id = selected["group_id"]
        else:
            group_id = self._next_group_id
            self._next_group_id += 1
            self._groups[group_id] = {
                "group_id": group_id,
                "created_at": float(entered_at),
                "last_seen": float(video_time),
                "members": set(),
                "entered_interior": False,
            }
        self._track_groups[track_id] = group_id
        return group_id

    def _update_groups(
        self, active_group_tracks: dict[int, set[int]], video_time: float
    ) -> tuple[list[dict], list[dict]]:
        completed = []
        grace = self.operations_settings.queue_group_exit_grace_seconds
        for group_id, group in list(self._groups.items()):
            active = active_group_tracks.get(group_id, set())
            if active:
                group["last_seen"] = video_time
                continue
            if video_time - group["last_seen"] < grace:
                continue
            wait = max(0.0, group["last_seen"] - group["created_at"])
            completed.append(
                {
                    "group_id": group_id,
                    "party_size": len(group["members"]),
                    "wait_seconds": round(wait, 2),
                    "outcome": "seated" if group["entered_interior"] else "left_queue",
                }
            )
            for track_id in group["members"]:
                self._track_groups.pop(track_id, None)
            del self._groups[group_id]

        visible = []
        for group_id, group in self._groups.items():
            active = active_group_tracks.get(group_id, set())
            if not active:
                continue
            visible.append(
                {
                    "group_id": group_id,
                    "party_size": len(group["members"]),
                    "active_people": len(active),
                    "dwell_seconds": round(
                        max(0.0, video_time - group["created_at"]), 2
                    ),
                }
            )
        return visible, completed
