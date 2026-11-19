"""Discrete-event simulation. Hidden future durations never enter observation events."""

from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import hashlib
import heapq
import itertools
import math
import random
from zoneinfo import ZoneInfo

from ..events import EventType as E, RestaurantEvent
from .config import SimulationConfig, TableSpec, clock_minutes


@dataclass(frozen=True)
class Arrival:
    """An optional scripted arrival, in seconds after opening (useful for scenarios)."""

    elapsed_seconds: float
    party_size: int
    patience_seconds: float = 1800


@dataclass
class _Party:
    party_id: str
    size: int
    status: str = "queued"
    table_id: str | None = None
    served: bool = False


@dataclass
class _Table:
    spec: TableSpec
    state: str = "READY"
    party: _Party | None = None


@dataclass(frozen=True)
class DayResult:
    day: date
    run_id: str
    day_seed: int
    opens_at: datetime
    closes_at: datetime
    events: tuple[RestaurantEvent, ...]


def day_seed(seed: int, day: date) -> int:
    digest = hashlib.sha256(f"{seed}:{day.isoformat()}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def generate_arrivals(config: SimulationConfig, day: date, seed: int) -> list[Arrival]:
    """Piecewise Poisson arrivals with independent daily and weekend variation."""
    rng = random.Random(seed)
    daily_load = rng.lognormvariate(0, config.day_log_sigma)
    if day.weekday() >= 5:
        daily_load *= config.weekend_arrival_multiplier
    opening = clock_minutes(config.opens_at)
    result = []
    for band in config.arrival_bands:
        rate = band.parties_per_hour * daily_load / 3600
        if rate == 0:
            continue
        current = (clock_minutes(band.start) - opening) * 60
        stop = (clock_minutes(band.end) - opening) * 60
        while True:
            current += rng.expovariate(rate)
            if current >= stop:
                break
            size = rng.choices(range(1, len(config.party_size_weights) + 1),
                               weights=config.party_size_weights)[0]
            patience = rng.lognormvariate(math.log(config.patience_minutes_median),
                                         config.patience_log_sigma) * 60
            result.append(Arrival(current, size, patience))
    return result


class _DaySimulation:
    def __init__(self, config: SimulationConfig, day: date, seed: int):
        self.config = config
        self.day = day
        self.seed = day_seed(seed, day)
        identity = f"{config.fingerprint}:{seed}:{day.isoformat()}"
        self.run_id = f"sim-{day.isoformat()}-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
        midnight = datetime.combine(day, time(), ZoneInfo(config.timezone))
        self.opens_at = midnight + timedelta(minutes=clock_minutes(config.opens_at))
        self.closes_at = midnight + timedelta(minutes=clock_minutes(config.closes_at))
        self.duration = (self.closes_at - self.opens_at).total_seconds()
        self.rng = random.Random(self.seed ^ 0x5E471CE)
        self.tables = {spec.table_id: _Table(spec) for spec in config.tables}
        self.parties: dict[str, _Party] = {}
        self.queue: list[_Party] = []
        self.service_queue: deque[_Party] = deque()
        self.cleaning_queue: deque[_Table] = deque()
        self.busy_servers = 0
        self.busy_cleaners = 0
        self.calendar: list[tuple] = []
        self.sequence = itertools.count()
        self.events: list[RestaurantEvent] = []
        self.now = 0.0

    def emit(self, kind: E, party: _Party | None = None, table: _Table | None = None,
             *, reason: str | None = None, censored: bool = False):
        self.events.append(RestaurantEvent(
            restaurant_id=self.config.restaurant_id, run_id=self.run_id,
            event_id=f"{self.run_id}:{len(self.events) + 1:06d}", event_type=kind,
            elapsed_seconds=round(self.now, 6), source="simulation", is_synthetic=True,
            observation_method="simulator", occurred_at=self.opens_at + timedelta(seconds=self.now),
            table_id=table.spec.table_id if table else None,
            capacity=table.spec.capacity if table else None,
            party_id=party.party_id if party else None, party_size=party.size if party else None,
            end_reason=reason, censored=censored,
        ))

    def schedule(self, at: float, action: str, payload=None, priority: int = 0):
        heapq.heappush(self.calendar, (at, priority, next(self.sequence), action, payload))

    def seat_waiting(self):
        if self.now >= self.duration:
            return
        # Oldest compatible party first; choose the smallest ready table that fits.
        # A smaller party may pass an older one only when that older party cannot fit.
        for party in list(self.queue):
            available = [t for t in self.tables.values()
                         if t.state == "READY" and t.spec.capacity >= party.size]
            if not available:
                continue
            table = min(available, key=lambda t: (t.spec.capacity, t.spec.table_id))
            self.queue.remove(party)
            party.status, party.table_id = "seated", table.spec.table_id
            table.state, table.party = "OCCUPIED", party
            self.emit(E.PARTY_SEATED, party, table)
            self.service_queue.append(party)
            self.schedule(self.now + self.config.service_alert_minutes * 60,
                          "service_alert", party, priority=3)
        self.start_service()

    def start_service(self):
        while self.service_queue and self.busy_servers < self.config.servers:
            party = self.service_queue.popleft()
            party.served = True
            self.busy_servers += 1
            table = self.tables[party.table_id]
            self.emit(E.SERVICE_CONFIRMED, party, table)
            service_seconds = max(1.0, self.rng.expovariate(1 / (self.config.service_minutes_mean * 60)))
            meal_median = (self.config.meal_minutes_median
                           + (party.size - 1) * self.config.meal_minutes_per_extra_guest)
            meal_seconds = self.rng.lognormvariate(math.log(meal_median),
                                                  self.config.meal_log_sigma) * 60
            self.schedule(self.now + service_seconds, "server_available")
            self.schedule(self.now + service_seconds + meal_seconds, "depart", party)

    def start_cleaning(self):
        while self.cleaning_queue and self.busy_cleaners < self.config.cleaners:
            table = self.cleaning_queue.popleft()
            table.state = "CLEANING"
            self.busy_cleaners += 1
            self.emit(E.CLEANING_STARTED, table.party, table)
            duration = self.rng.lognormvariate(math.log(self.config.cleaning_minutes_median),
                                               self.config.cleaning_log_sigma) * 60
            self.schedule(self.now + duration, "ready", table)

    def run(self, arrivals: list[Arrival]) -> DayResult:
        self.emit(E.RUN_STARTED)
        for table in self.tables.values():
            self.emit(E.TABLE_READY, table=table, reason="opening")
        for index, arrival in enumerate(sorted(arrivals, key=lambda a: a.elapsed_seconds)):
            if (not math.isfinite(arrival.elapsed_seconds)
                    or not 0 <= arrival.elapsed_seconds < self.duration
                    or type(arrival.party_size) is not int or arrival.party_size <= 0
                    or not math.isfinite(arrival.patience_seconds) or arrival.patience_seconds <= 0):
                raise ValueError("Scripted arrivals need a valid time before closing, size, and patience")
            party = _Party(f"{self.run_id}:party-{index + 1:05d}", arrival.party_size)
            self.parties[party.party_id] = party
            self.schedule(arrival.elapsed_seconds, "arrive", party, priority=1)
            self.schedule(arrival.elapsed_seconds + arrival.patience_seconds,
                          "abandon", party, priority=2)
        while self.calendar and self.calendar[0][0] <= self.duration:
            self.now, _, _, action, payload = heapq.heappop(self.calendar)
            if action == "arrive":
                party = payload
                self.emit(E.PARTY_ARRIVED, party)
                if party.size > max(t.spec.capacity for t in self.tables.values()):
                    party.status = "rejected"
                    self.emit(E.PARTY_REJECTED, party, reason="unsupported_party_size")
                else:
                    self.queue.append(party)
                    self.seat_waiting()
            elif action == "abandon" and payload.status == "queued":
                payload.status = "abandoned"
                self.queue.remove(payload)
                self.emit(E.PARTY_ABANDONED, payload, reason="patience_exceeded")
            elif action == "server_available":
                self.busy_servers -= 1
                self.start_service()
            elif action == "service_alert" and payload.status == "seated" and not payload.served:
                self.emit(E.SERVICE_ALERT, payload, self.tables[payload.table_id])
            elif action == "depart":
                payload.status = "departed"
                table = self.tables[payload.table_id]
                table.state = "NEEDS_CLEANING"
                self.emit(E.PARTY_DEPARTED, payload, table, reason="observed_departure")
                self.cleaning_queue.append(table)
                self.start_cleaning()
            elif action == "ready":
                table = payload
                self.busy_cleaners -= 1
                table.state = "READY"
                self.emit(E.TABLE_READY, table.party, table, reason="cleaning_confirmed")
                table.party = None
                self.start_cleaning()
                self.seat_waiting()
        self.now = self.duration
        for party in self.queue:
            self.emit(E.QUEUE_CENSORED, party, reason="observation_end", censored=True)
        for table in self.tables.values():
            if table.state == "OCCUPIED":
                self.emit(E.VISIT_CENSORED, table.party, table,
                          reason="observation_end", censored=True)
            elif table.state in ("NEEDS_CLEANING", "CLEANING"):
                self.emit(E.TURNOVER_CENSORED, table.party, table,
                          reason="observation_end", censored=True)
        self.emit(E.RUN_ENDED, reason="observation_end")
        return DayResult(self.day, self.run_id, self.seed, self.opens_at, self.closes_at,
                         tuple(self.events))


def simulate_day(config: SimulationConfig, day: date, seed: int = 42,
                 arrivals: list[Arrival] | None = None) -> DayResult:
    simulation = _DaySimulation(config, day, seed)
    if arrivals is None:
        arrivals = generate_arrivals(config, day, simulation.seed)
    return simulation.run(arrivals)
