"""Anonymous per-camera tracking and staff-marker role evidence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace


@dataclass
class RoleHistory:
    observations: deque[bool] = field(default_factory=lambda: deque(maxlen=12))
    first_seen: float = 0.0
    last_seen: float = 0.0
    queue_entered: float | None = None


class PersonRoleTracker:
    """ByteTrack IDs plus temporal staff-marker voting for one camera."""

    def __init__(
        self,
        frame_rate: float,
        hsv_lower=(80, 80, 60),
        hsv_upper=(110, 255, 255),
        minimum_ratio=0.025,
    ):
        args = IterableSimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=0.25,
            track_low_thresh=0.10,
            new_track_thresh=0.25,
            track_buffer=max(15, int(frame_rate * 3)),
            match_thresh=0.80,
            fuse_score=True,
        )
        self.tracker = BYTETracker(args)
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.minimum_ratio = float(minimum_ratio)
        self.history: dict[int, RoleHistory] = {}

    def _marker_visible(self, frame, xyxy) -> tuple[bool, float]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return False, 0.0
        person_height = y2 - y1
        torso = frame[
            y1 + int(person_height * 0.18) : y1 + int(person_height * 0.72),
            x1:x2,
        ]
        if torso.size == 0:
            return False, 0.0
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        return ratio >= self.minimum_ratio, ratio

    @staticmethod
    def _iou(first, second) -> float:
        ax1, ay1, ax2, ay2 = [float(value) for value in first]
        bx1, by1, bx2, by2 = [float(value) for value in second]
        width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        height = max(0.0, min(ay2, by2) - max(ay1, by1))
        intersection = width * height
        union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(
            0.0, (bx2 - bx1) * (by2 - by1)
        ) - intersection
        return intersection / union if union > 0 else 0.0

    def _model_staff_visible(self, role_result, xyxy) -> tuple[bool, float]:
        """Match a custom staff/customer detection to a pose-person box."""
        boxes = getattr(role_result, "boxes", None) if role_result is not None else None
        if boxes is None or len(boxes) == 0:
            return False, 0.0
        names = getattr(role_result, "names", {}) or {}
        model_boxes = boxes.xyxy.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        best = 0.0
        for model_box, class_id, confidence in zip(model_boxes, classes, confidences):
            class_name = str(names.get(int(class_id), int(class_id))).lower()
            if class_name != "staff" or float(confidence) < settings.ROLE_MODEL_CONF:
                continue
            if self._iou(xyxy, model_box) >= 0.35:
                best = max(best, float(confidence))
        return best > 0, best

    def update(self, result, frame, video_time: float, role_result=None) -> list[dict]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            tracks = self.tracker.update(boxes.cpu().numpy() if boxes is not None else _EmptyBoxes())
        else:
            tracks = self.tracker.update(boxes.cpu().numpy(), frame)

        active = []
        for track in tracks:
            x1, y1, x2, y2, track_id, score, class_id, detection_index = track[:8]
            if int(class_id) != 0:
                continue
            track_id = int(track_id)
            history = self.history.setdefault(
                track_id,
                RoleHistory(first_seen=video_time, last_seen=video_time),
            )
            marker, marker_ratio = self._marker_visible(frame, (x1, y1, x2, y2))
            model_staff, model_score = self._model_staff_visible(
                role_result, (x1, y1, x2, y2)
            )
            history.observations.append(marker or model_staff)
            history.last_seen = video_time
            votes = sum(history.observations)
            required = max(2, int(np.ceil(len(history.observations) * 0.35)))
            role = "staff" if votes >= required else "unknown"
            active.append(
                {
                    "track_id": track_id,
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "center": [float((x1 + x2) / 2), float(y2)],
                    "score": float(score),
                    "detection_index": int(detection_index),
                    "role": role,
                    "marker_ratio": round(marker_ratio, 4),
                    "role_model_score": round(model_score, 4),
                    "history": history,
                }
            )

        stale = [track_id for track_id, item in self.history.items() if video_time - item.last_seen > 10]
        for track_id in stale:
            del self.history[track_id]
        return active


class _EmptyBoxes:
    conf = np.empty((0,), dtype=np.float32)
    cls = np.empty((0,), dtype=np.float32)
    xywh = np.empty((0, 4), dtype=np.float32)

    def __getitem__(self, item):
        return self

    def __len__(self):
        return 0
