"""Portable event contract for simulation and future camera/host adapters."""

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
import math


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    PARTY_ARRIVED = "party_arrived"
    PARTY_SEATED = "party_seated"
    PARTY_REJECTED = "party_rejected"
    PARTY_ABANDONED = "party_abandoned"
    SERVICE_CONFIRMED = "service_confirmed"
    SERVICE_ALERT = "service_alert"
    PARTY_DEPARTED = "party_departed"
    CLEANING_STARTED = "cleaning_started"
    TABLE_READY = "table_ready"
    VISIT_CENSORED = "visit_censored"
    TURNOVER_CENSORED = "turnover_censored"
    QUEUE_CENSORED = "queue_censored"
    RUN_ENDED = "run_ended"


@dataclass(frozen=True)
class RestaurantEvent:
    restaurant_id: str
    run_id: str
    event_id: str
    event_type: EventType
    elapsed_seconds: float
    source: str
    is_synthetic: bool
    observation_method: str
    occurred_at: datetime | None = None
    video_time_seconds: float | None = None
    table_id: str | None = None
    capacity: int | None = None
    party_id: str | None = None
    party_size: int | None = None
    confidence: float | None = None
    end_reason: str | None = None
    censored: bool = False
    is_prediction: bool = False
    schema_version: int = 1

    def __post_init__(self):
        for name in ("restaurant_id", "run_id", "event_id", "source", "observation_method"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        object.__setattr__(self, "event_type", EventType(self.event_type))
        if self.schema_version != 1:
            raise ValueError("Unsupported event schema version")
        for name in ("is_synthetic", "censored", "is_prediction"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.elapsed_seconds is None:
            raise ValueError("elapsed_seconds is required")
        for name in ("elapsed_seconds", "video_time_seconds"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (float, int) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.occurred_at is not None and self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must have a timezone; leave unknown dates unset")
        if self.confidence is not None and not (0 <= self.confidence <= 1):
            raise ValueError("confidence must be between zero and one")
        if self.censored and not self.end_reason:
            raise ValueError("Censored events must explain their end_reason")
        for name, identity in (("capacity", "table_id"), ("party_size", "party_id")):
            value = getattr(self, name)
            if value is not None:
                if type(value) is not int or value <= 0 or not getattr(self, identity):
                    raise ValueError(f"{name} requires a positive integer and {identity}")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["event_type"] = self.event_type.value
        result["occurred_at"] = self.occurred_at.isoformat() if self.occurred_at else None
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "RestaurantEvent":
        values = dict(data)
        if values.get("occurred_at") is not None:
            values["occurred_at"] = datetime.fromisoformat(values["occurred_at"])
        return cls(**values)
