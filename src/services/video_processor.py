"""Per-frame video processing: detection, filtering, counting, drawing."""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from cameras import scale_points
from config import REGIONS_CONFIG_PATH, settings
from counting.load_regions import load_zones
from counting.logged_count import LoggedCount
from filters.standing_filter import StandingFilter
from services.occupancy_service import OccupancyService
from services.person_roles import PersonRoleTracker
from services.table_assignment import TableAssignmentTracker


def _make_zone(polygon_points, triggering_anchors=(sv.Position.CENTER,)):
    return sv.PolygonZone(
        polygon=np.array(polygon_points, dtype=np.int32),
        triggering_anchors=triggering_anchors,
    )


class VideoProcessor:
    """Process individual frames of the restaurant video.

    Owns the YOLO model and the frame-processing dependencies (regions,
    annotators, ``StandingFilter``, one ``LoggedCount`` per table) and
    delegates occupancy-session updates to ``OccupancyService``. Standing
    people are removed from counting only; the original detections are
    always drawn.
    """

    def __init__(
        self,
        model=None,
        frame_rate: float | None = None,
        camera_config: dict | None = None,
    ):
        self.model = model if model is not None else YOLO(settings.MODEL_PATH)
        self.conf = settings.CONF
        self.iou = settings.IOU
        self.imgsz = settings.IMGSZ

        self.normalized_tables = (
            camera_config.get("regions", {}).get("tables", [])
            if camera_config
            else []
        )
        zones_config = (
            [
                {
                    "name": table["id"],
                    "max": table["capacity"],
                    "normalized": table,
                }
                for table in self.normalized_tables
            ]
            if self.normalized_tables
            else load_zones(REGIONS_CONFIG_PATH)
        )
        self.tables = [
            {"id": config["name"], "max": config["max"]} for config in zones_config
        ]

        self._zones_config = zones_config
        self.zones = []
        if not self.normalized_tables:
            for config in zones_config:
                counting = np.asarray(config["counting_polygon"], dtype=np.int32)
                display = np.asarray(
                    config.get("display_polygon", config["counting_polygon"]),
                    dtype=np.int32,
                )
                self.zones.append(
                    {
                        "name": config["name"],
                        "max": config["max"],
                        "counting_polygon": counting,
                        "display_polygon": display,
                        "counting_zone": _make_zone(counting),
                        "annotator": (
                            sv.PolygonZoneAnnotator(
                                zone=_make_zone(display),
                                color=sv.Color.WHITE,
                                thickness=2,
                            )
                            if "display_polygon" in config
                            else None
                        ),
                    }
                )

        self.corner_annotator = sv.BoxCornerAnnotator(color=sv.Color.WHITE)
        self.standing_filter = StandingFilter()
        marker = (camera_config or {}).get("staff_marker", {})
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
        self.table_assignments = TableAssignmentTracker(
            hold_seconds=settings.TABLE_OCCLUSION_HOLD_SECONDS,
            margin_ratio=settings.TABLE_ASSIGNMENT_MARGIN_RATIO,
        )
        self._assignment_size = None
        self.count_stabilizers = {
            config["name"]: LoggedCount(
                increase_seconds=settings.COUNT_INCREASE_DELAY,
                decrease_seconds=settings.COUNT_DECREASE_DELAY,
                history_size=settings.COUNT_HISTORY_SIZE,
                empty_seconds=settings.COUNT_EMPTY_DELAY,
            )
            for config in zones_config
        }
        self.occupancy = OccupancyService([config["name"] for config in zones_config])

    def begin_run(self) -> None:
        """Called once when a processing run starts."""
        self.table_assignments.reset()
        self.occupancy.begin_run()

    def end_run(self, last_video_time: float) -> None:
        """Called once when a processing run ends (see ``OccupancyService``)."""
        self.occupancy.end_run(last_video_time)

    def process_frame(self, frame, video_time: float):
        """Process one frame; return the annotated frame and dashboard stats."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=settings.DEVICE,
            quantize=(
                16
                if settings.HALF_PRECISION and settings.DEVICE.startswith("cuda")
                else None
            ),
            verbose=False,
        )[0]

        return self.process_result(frame, results, video_time)

    def process_result(self, frame, results, video_time: float, role_result=None):
        """Apply dining-room analytics to an already computed YOLO result."""

        self._ensure_normalized_zones(frame)

        detections = sv.Detections.from_ultralytics(results)

        tracked_people = (
            self.role_tracker.update(results, frame, video_time)
            if role_result is None
            else self.role_tracker.update(
                results, frame, video_time, role_result=role_result
            )
        )
        staff_indexes = {
            person["detection_index"]
            for person in tracked_people
            if person["role"] == "staff"
        }
        non_staff_mask = np.array(
            [index not in staff_indexes for index in range(len(detections))],
            dtype=bool,
        )
        keypoints = results.keypoints
        keypoint_xy = (
            keypoints.xy.detach().cpu().numpy()
            if keypoints is not None and keypoints.xy is not None
            else None
        )
        keypoint_conf = (
            keypoints.conf.detach().cpu().numpy()
            if keypoints is not None and keypoints.conf is not None
            else None
        )
        seated_mask = self.standing_filter.keep_mask(
            detections,
            keypoint_xy,
            keypoint_conf,
        )
        eligible_mask = non_staff_mask & seated_mask

        assignment_people = []
        for person in tracked_people:
            index = int(person["detection_index"])
            x1, y1, x2, y2 = person["xyxy"]
            assignment_people.append(
                {
                    "track_id": person["track_id"],
                    "role": person["role"],
                    "eligible": 0 <= index < len(eligible_mask) and bool(eligible_mask[index]),
                    "anchor": [(x1 + x2) / 2, (y1 + y2) / 2],
                }
            )
        raw_counts = self.table_assignments.update(assignment_people, video_time)

        # Unfiltered detections are drawn; only counting uses the filtered set.
        annotated_frame = self._blur_faces(frame, detections)
        annotated_frame = self.corner_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
        )

        stats = {
            "table_counts": {},
            "seated_guests": 0,
            "occupancy_seconds": {},
            "staff_visible": sum(
                person["role"] == "staff" for person in tracked_people
            ),
            "people_tracked": len(tracked_people),
        }

        for zone in self.zones:
            raw_count = min(int(raw_counts.get(zone["name"], 0)), int(zone["max"]))
            count = self.count_stabilizers[zone["name"]].update(raw_count, video_time)
            maximum = zone["max"]
            stats["table_counts"][zone["name"]] = count

            stats["occupancy_seconds"][zone["name"]] = self.occupancy.update(
                zone["name"], count, video_time
            )

            if count == 0:
                color = sv.Color.GREEN
            elif count < maximum:
                color = sv.Color.YELLOW
            else:
                color = sv.Color.RED

            if zone["annotator"] is not None:
                zone["annotator"].color = color
                annotated_frame = zone["annotator"].annotate(
                    scene=annotated_frame,
                    label=f"{zone['name']}: {count}/{maximum}",
                )

        # SEATED GUESTS is the sum of the stabilized table counts, so the total
        # always matches the per-table numbers, colors, and occupancy.
        stats["seated_guests"] = sum(stats["table_counts"].values())

        return annotated_frame, stats

    def _ensure_normalized_zones(self, frame):
        height, width = frame.shape[:2]
        if self.normalized_tables and not self.zones:
            zones = []
            for config in self._zones_config:
                table = config["normalized"]
                counting = np.asarray(
                    scale_points(table["counting_polygon"], width, height),
                    dtype=np.int32,
                )
                display_source = table.get("display_polygon") or table["counting_polygon"]
                display = np.asarray(
                    scale_points(display_source, width, height), dtype=np.int32
                )
                zones.append(
                    {
                        "name": config["name"],
                        "max": config["max"],
                        "counting_polygon": counting,
                        "display_polygon": display,
                        "counting_zone": _make_zone(counting),
                        "annotator": sv.PolygonZoneAnnotator(
                            zone=_make_zone(display),
                            color=sv.Color.WHITE,
                            thickness=2,
                        ),
                    }
                )
            self.zones = zones

        size = (width, height)
        if self.zones and self._assignment_size != size:
            self.table_assignments.configure(self.zones, width, height)
            self._assignment_size = size

    @staticmethod
    def _blur_faces(frame, detections):
        """Blur only each person's upper face area for efficient privacy."""
        if len(detections) == 0:
            return frame
        annotated = frame.copy()
        height, width = frame.shape[:2]
        for x1, y1, x2, y2 in detections.xyxy:
            left = max(0, min(width, int(x1)))
            right = max(0, min(width, int(x2)))
            top = max(0, min(height, int(y1)))
            bottom = max(top, min(height, top + max(10, int((y2 - y1) * 0.25))))
            face = annotated[top:bottom, left:right]
            if face.size:
                annotated[top:bottom, left:right] = cv2.GaussianBlur(
                    face, (21, 21), 0
                )
        return annotated
