"""Exclusive, temporally stable assignment of tracked guests to tables."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import cv2
import numpy as np


@dataclass
class _Assignment:
    table_id: str
    last_seen: float


class TableAssignmentTracker:
    """Assign every anonymous track to at most one calibrated table.

    ``counting_polygon`` covers the chairs and guests, while
    ``display_polygon`` stays tight around the tabletop.  When seating zones
    overlap, the closest tabletop wins.  The previous table is preferred while
    it remains a valid candidate, and short detector occlusions retain the
    assignment without inventing a cross-camera identity.
    """

    def __init__(self, hold_seconds: float = 1.5, margin_ratio: float = 0.012):
        self.hold_seconds = float(hold_seconds)
        self.margin_ratio = float(margin_ratio)
        self.assignments: dict[int, _Assignment] = {}
        self.polygons: dict[str, np.ndarray] = {}
        self.centers: dict[str, tuple[float, float]] = {}
        self.margin_pixels = 0.0

    def configure(self, zones: list[dict], width: int, height: int) -> None:
        self.polygons = {
            zone["name"]: np.asarray(zone["counting_polygon"], dtype=np.float32)
            for zone in zones
        }
        self.centers = {}
        for zone in zones:
            display = zone.get("display_polygon")
            if display is None or len(display) == 0:
                display = zone["counting_polygon"]
            self.centers[zone["name"]] = self._centroid(
                np.asarray(display, dtype=np.float32)
            )
        self.margin_pixels = hypot(width, height) * self.margin_ratio
        valid = set(self.polygons)
        self.assignments = {
            track_id: assignment
            for track_id, assignment in self.assignments.items()
            if assignment.table_id in valid
        }

    def reset(self) -> None:
        self.assignments.clear()

    def update(self, people: list[dict], video_time: float) -> dict[str, int]:
        for person in people:
            track_id = int(person["track_id"])
            if person.get("role") == "staff":
                self.assignments.pop(track_id, None)
                continue
            if not person.get("eligible", False):
                continue

            point = tuple(float(value) for value in person["anchor"])
            candidates = self._candidates(point)
            if not candidates:
                continue

            previous = self.assignments.get(track_id)
            if previous is not None and previous.table_id in candidates:
                table_id = previous.table_id
            else:
                table_id = min(
                    candidates,
                    key=lambda name: self._distance_squared(point, self.centers[name]),
                )
            self.assignments[track_id] = _Assignment(table_id, float(video_time))

        cutoff = float(video_time) - self.hold_seconds
        stale = [
            track_id
            for track_id, assignment in self.assignments.items()
            if assignment.last_seen < cutoff
        ]
        for track_id in stale:
            del self.assignments[track_id]

        counts = {table_id: 0 for table_id in self.polygons}
        for assignment in self.assignments.values():
            counts[assignment.table_id] += 1
        return counts

    def _candidates(self, point: tuple[float, float]) -> list[str]:
        return [
            table_id
            for table_id, polygon in self.polygons.items()
            if cv2.pointPolygonTest(polygon, point, True) >= -self.margin_pixels
        ]

    @staticmethod
    def _centroid(polygon: np.ndarray) -> tuple[float, float]:
        moments = cv2.moments(polygon)
        if abs(moments["m00"]) > 1e-6:
            return (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )
        mean = polygon.mean(axis=0)
        return float(mean[0]), float(mean[1])

    @staticmethod
    def _distance_squared(
        left: tuple[float, float], right: tuple[float, float]
    ) -> float:
        return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2
