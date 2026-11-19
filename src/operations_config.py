"""Validated, UI-editable operational thresholds and camera policies."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from config import PROJECT_ROOT

OPERATIONS_CONFIG_PATH = PROJECT_ROOT / "config" / "operations.json"


class OperationsSettings(BaseModel):
    version: int = Field(1, ge=1, le=1)
    queue_alert_people: int = Field(8, ge=1, le=100)
    max_queue_wait_seconds: float = Field(600, ge=10, le=14400)
    service_alert_seconds: float = Field(300, ge=10, le=7200)
    cleaning_alert_seconds: float = Field(180, ge=0, le=7200)
    default_dining_minutes: float = Field(45, ge=5, le=240)
    queue_group_join_seconds: float = Field(3, ge=0, le=30)
    queue_group_exit_grace_seconds: float = Field(2, ge=0, le=30)
    history_sample_seconds: float = Field(5, ge=1, le=3600)
    camera_reconnect_attempts: int = Field(5, ge=0, le=100)
    camera_reconnect_delay_seconds: float = Field(2, ge=0.1, le=120)
    stream_stale_seconds: float = Field(6, ge=1, le=300)
    stream_jpeg_quality: int = Field(95, ge=70, le=100)


def load_operations_settings(path: Path = OPERATIONS_CONFIG_PATH) -> OperationsSettings:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return OperationsSettings()
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid operations config: {exc}") from exc
    return OperationsSettings.model_validate(payload)


def save_operations_settings(
    payload: dict, path: Path = OPERATIONS_CONFIG_PATH
) -> OperationsSettings:
    current = load_operations_settings(path).model_dump()
    current.update(payload)
    settings = OperationsSettings.model_validate(current)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="operations-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(settings.model_dump(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return settings
