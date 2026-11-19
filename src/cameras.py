"""Camera-layout loading and normalized polygon helpers."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

from config import CAMERAS_CONFIG_PATH, PROJECT_ROOT, settings

CAMERA_ROLES = {"dining", "entrance"}


def _validate_polygon(name: str, points: object, *, line: bool = False) -> None:
    minimum = 2 if line else 3
    if points == []:
        return
    if not isinstance(points, list) or len(points) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} normalized points")
    for point in points:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point)
            or any(value < 0 or value > 1 for value in point)
        ):
            raise ValueError(f"{name} contains an invalid normalized point: {point}")


def _validate_staff_marker(camera_id: str, marker: object) -> None:
    if marker in (None, {}):
        return
    if not isinstance(marker, dict):
        raise ValueError(f"Camera {camera_id} staff_marker must be an object")
    for key in ("hsv_lower", "hsv_upper"):
        values = marker.get(key)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(type(value) is not int for value in values)
            or not 0 <= values[0] <= 179
            or any(not 0 <= value <= 255 for value in values[1:])
        ):
            raise ValueError(f"Camera {camera_id} {key} must be [H, S, V]")
    ratio = marker.get("minimum_ratio")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 <= ratio <= 1:
        raise ValueError(f"Camera {camera_id} minimum_ratio must be between 0 and 1")


def validate_camera_config(data: object) -> dict:
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Camera config requires version 1")
    cameras = data.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("Camera config requires a non-empty cameras list")
    ids = set()
    roles = set()
    for camera in cameras:
        if not isinstance(camera, dict):
            raise ValueError("Each camera must be an object")
        camera_id = camera.get("id")
        role = camera.get("role")
        if not isinstance(camera_id, str) or not camera_id:
            raise ValueError("Every camera requires an id")
        if camera_id in ids:
            raise ValueError(f"Duplicate camera id: {camera_id}")
        if role not in CAMERA_ROLES:
            raise ValueError(f"Unsupported camera role: {role}")
        ids.add(camera_id)
        roles.add(role)
        _validate_staff_marker(camera_id, camera.get("staff_marker", {}))
        regions = camera.get("regions", {})
        if not isinstance(regions, dict):
            raise ValueError(f"Camera {camera_id} regions must be an object")
        for key in ("queue_polygon", "transit_polygon", "interior_polygon"):
            _validate_polygon(f"{camera_id}.{key}", regions.get(key, []))
        _validate_polygon(f"{camera_id}.entry_line", regions.get("entry_line", []), line=True)
        tables = regions.get("tables", [])
        if not isinstance(tables, list):
            raise ValueError(f"Camera {camera_id} tables must be a list")
        table_ids = set()
        for table in tables:
            if not isinstance(table, dict):
                raise ValueError(f"Camera {camera_id} table entries must be objects")
            table_id = table.get("id")
            capacity = table.get("capacity")
            if not isinstance(table_id, str) or not table_id or table_id in table_ids:
                raise ValueError(f"Camera {camera_id} has an invalid or duplicate table id")
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
                raise ValueError(f"Camera {camera_id} table {table_id} needs a positive capacity")
            if not table.get("counting_polygon"):
                raise ValueError(f"Camera {camera_id} table {table_id} needs a counting polygon")
            _validate_polygon(
                f"{camera_id}.{table_id}.counting_polygon",
                table.get("counting_polygon", []),
            )
            _validate_polygon(
                f"{camera_id}.{table_id}.display_polygon",
                table.get("display_polygon", []),
            )
            table_ids.add(table_id)
    if "dining" not in roles or "entrance" not in roles:
        raise ValueError("Camera config requires dining and entrance roles")
    return deepcopy(data)


def load_camera_config(path: Path = CAMERAS_CONFIG_PATH) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Camera config not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid camera config: {exc}") from exc
    return validate_camera_config(data)


def save_camera_config(data: object, path: Path = CAMERAS_CONFIG_PATH) -> dict:
    validated = validate_camera_config(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="cameras-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(validated, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return validated


def resolve_camera_sources(config: dict) -> list[dict]:
    resolved = []
    for camera in config["cameras"]:
        item = deepcopy(camera)
        setting_name = item.get("source_setting", "")
        source = getattr(settings, setting_name, "") if setting_name else ""
        if source:
            source_path = Path(source)
            if not source_path.is_absolute() and "://" not in source:
                source = str(PROJECT_ROOT / source_path)
        calibration_source = str(item.get("calibration_source", "")).strip()
        if calibration_source and "://" not in calibration_source:
            calibration_path = Path(calibration_source)
            if not calibration_path.is_absolute():
                calibration_source = str(PROJECT_ROOT / calibration_path)
        calibration_required = bool(
            source and calibration_source and str(source) != calibration_source
        )
        item["calibration_required"] = calibration_required
        if calibration_required:
            # Keep the saved polygons in cameras.json, but never apply geometry
            # calibrated for another viewpoint to the active inference stream.
            item["regions"] = deepcopy(item.get("regions", {}))
            item["regions"]["tables"] = []
        item["source"] = source
        item["active"] = bool(item.get("enabled", True) and source)
        resolved.append(item)
    return resolved


def scale_points(points: list[list[float]], width: int, height: int):
    return [
        [int(round(point[0] * width)), int(round(point[1] * height))]
        for point in points
    ]
