"""Validated, serializable restaurant and stochastic process settings."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def clock_minutes(value: str) -> int:
    hour, minute = map(int, value.split(":"))
    if hour == 24 and minute == 0:
        return 1440
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"Invalid clock time: {value}")
    return hour * 60 + minute


@dataclass(frozen=True)
class TableSpec:
    table_id: str
    capacity: int


@dataclass(frozen=True)
class ArrivalBand:
    start: str
    end: str
    parties_per_hour: float


@dataclass(frozen=True)
class SimulationConfig:
    restaurant_id: str
    timezone: str
    opens_at: str
    closes_at: str
    tables: tuple[TableSpec, ...]
    arrival_bands: tuple[ArrivalBand, ...]
    party_size_weights: tuple[float, ...]
    servers: int = 2
    cleaners: int = 1
    service_minutes_mean: float = 4
    meal_minutes_median: float = 40
    meal_minutes_per_extra_guest: float = 3
    meal_log_sigma: float = 0.30
    cleaning_minutes_median: float = 4
    cleaning_log_sigma: float = 0.25
    patience_minutes_median: float = 25
    patience_log_sigma: float = 0.35
    day_log_sigma: float = 0.20
    weekend_arrival_multiplier: float = 1.20
    service_alert_minutes: float = 5
    snapshot_minutes: int = 5
    bucket_minutes: int = 15

    def __post_init__(self):
        if not isinstance(self.restaurant_id, str) or not self.restaurant_id.strip():
            raise ValueError("restaurant_id must be nonempty")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                f"Timezone {self.timezone!r} unavailable. On Windows install IANA data: "
                "python -m pip install tzdata"
            ) from exc
        opening, closing = clock_minutes(self.opens_at), clock_minutes(self.closes_at)
        if opening >= closing:
            raise ValueError("Opening must precede closing; overnight shifts are not supported")
        if not self.tables or len({t.table_id for t in self.tables}) != len(self.tables):
            raise ValueError("Tables must have unique IDs and cannot be empty")
        for table in self.tables:
            if not isinstance(table.table_id, str) or not table.table_id.strip():
                raise ValueError("Table IDs must be nonempty")
            if type(table.capacity) is not int or table.capacity <= 0:
                raise ValueError("Table capacities must be positive integers")
        for name in ("servers", "cleaners", "snapshot_minutes", "bucket_minutes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("service_minutes_mean", "meal_minutes_median", "cleaning_minutes_median",
                     "patience_minutes_median", "weekend_arrival_multiplier", "service_alert_minutes"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("meal_minutes_per_extra_guest", "meal_log_sigma", "cleaning_log_sigma",
                     "patience_log_sigma", "day_log_sigma"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        weights = self.party_size_weights
        if not weights or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("party_size_weights must be nonnegative with positive total weight")
        previous = opening
        for band in self.arrival_bands:
            start, end = clock_minutes(band.start), clock_minutes(band.end)
            if not (previous <= start < end <= closing):
                raise ValueError("Arrival bands must be ordered, nonoverlapping, and inside opening hours")
            if not math.isfinite(band.parties_per_hour) or band.parties_per_hour < 0:
                raise ValueError("Arrival rates must be finite and nonnegative")
            previous = end

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationConfig":
        values = dict(data)
        values["tables"] = tuple(TableSpec(**table) for table in values["tables"])
        values["arrival_bands"] = tuple(ArrivalBand(**band) for band in values["arrival_bands"])
        values["party_size_weights"] = tuple(values["party_size_weights"])
        return cls(**values)

    @classmethod
    def load(cls, path: Path | str) -> "SimulationConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
