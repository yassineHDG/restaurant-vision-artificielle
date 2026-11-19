# Restaurant simulation and training data

The simulator is the first implemented extension of our four-person project. It
produces dated events and training tables without loading YOLO, OpenCV, or the
web application. No forecasting model is trained by this command.

## Run it

From the repository root, using Python 3.12 or newer:

```powershell
python -m src.simulation --days 90 --seed 42 --start-date 2026-11-01 --output data/simulation/my-first-dataset
```

Windows may need the small `tzdata` package for IANA timezones. This isolated
command supplies it without installing the project's video dependencies:

```powershell
uv run --no-project --python 3.12 --with tzdata python -m src.simulation --days 90 --seed 42 --start-date 2026-11-01 --output data/simulation/my-first-dataset
```

Use either command, not both for the same output path. An existing output folder
is refused to protect previous experiments. All generated data is under `data/`,
which is already ignored by Git. Share configuration, code, and seeds through
Git; generate the datasets locally or share them separately.

The checked local baseline is `data/simulation/baseline-90d`, seed 42, covering
2026-11-01 through 2027-01-29. It contains 9,562 arriving parties, 6,023 seated
visits, 62,155 duration snapshots, and 4,320 arrival intervals. These numbers
describe this artificial scenario, not a measured restaurant.

## Configure the restaurant

Edit a copy of `config/simulation.json`, then pass `--config path/to/config.json`.
All times are interpreted in the configured `Africa/Casablanca` timezone by
default, including the applicable offset for each simulated date.

| Setting | Meaning |
| --- | --- |
| `tables` | IDs and capacities; default six four-seat tables, matching the demo's capacities |
| `opens_at`, `closes_at` | Same-day observation window; overnight service is not supported |
| `arrival_bands` | Ordered, nonoverlapping times and expected arriving **parties per hour**; uncovered times have zero arrivals |
| `party_size_weights` | Weights for parties of size 1, 2, 3, etc.; normalization is automatic |
| `servers`, `cleaners` | Concurrent first-service tasks and concurrent cleaning tasks |
| `service_minutes_mean` | Mean duration of the first-service task, sampled exponentially with a one-second minimum |
| `meal_minutes_median`, `meal_minutes_per_extra_guest`, `meal_log_sigma` | Lognormal meal duration after service, with a party-size adjustment |
| `cleaning_minutes_median`, `cleaning_log_sigma` | Lognormal cleaning duration |
| `patience_minutes_median`, `patience_log_sigma` | Lognormal queue patience; exceeding it leads to abandonment |
| `day_log_sigma`, `weekend_arrival_multiplier` | Daily variation and a weekend multiplier on arrival rates |
| `service_alert_minutes` | Delay from seating to an unserved alert, default five minutes |
| `snapshot_minutes`, `bucket_minutes` | Duration snapshot spacing and arrival target interval size |

The configured lunch rush is an assumption supplied to the generator. Training
on it will not discover a real restaurant's lunch rush. Try changed arrival bands,
meal durations, table layouts, and staffing in separate held-out scenarios.

## Simulation behavior

- Each day starts with all tables ready and an empty queue. Arrivals follow a
  piecewise Poisson process, modulated by that day's stochastic load.
- Each party occupies one whole table. Choose the smallest ready table that fits.
  Among waiting parties, seat the oldest compatible party first. A younger party
  can pass an older one if the available table cannot fit the older party.
- There is no table combining. A party larger than every table is explicitly
  rejected, rather than blocking the queue. Abandoned parties are never seated.
- The first waiter contact is a `service_confirmed` event. The server remains
  busy for a sampled service duration; the party then eats for a sampled meal
  duration. Server shortages therefore affect waits and occupied duration.
- A party without first contact after five minutes produces one service alert.
  These are recorded events, not phone notifications.
- Departures put tables into `NEEDS_CLEANING`; cleaners move them through
  `CLEANING` to `READY`. Waiting parties can only use `READY` tables.
- Availability at exactly a party's patience deadline permits seating. No new
  party is seated exactly at the observation's closing boundary.
- At closing, observation stops. Occupied visits, unfinished cleaning, and
  unfinished queue waits are marked **censored** (their outcomes are unknown).
  The simulator does not invent a departure to close those labels.

Reservations, table merging, payment stages, repeat waiter visits, camera errors,
and a live queue-limit predictor are outside this first version. Server counts
represent first-contact service capacity, not a full model of restaurant labor.

## Files

| File | Contents |
| --- | --- |
| `events.jsonl` | One append-only observed event per JSON line, in event order within chronological runs |
| `parties.csv` | One row per arriving party, including seating, abandonment, unsupported size, or censoring |
| `visits.csv` | One row per seated visit, with service, departure, readiness, and label completeness |
| `duration_snapshots.csv` | Observations of ongoing visits every five minutes, with remaining-time targets |
| `arrivals_15min.csv` | All open-hours 15-minute intervals, including zero-arrival intervals; name follows configured bucket size |
| `manifest.json` | Dataset/config IDs, exact configuration, Python/timezone-data versions, per-day seeds and splits, row counts, checksums, summary, and model input columns |

Every CSV carries restaurant/run IDs, date, split, source, and a synthetic-data
flag. JSON timestamps use ISO 8601 with an offset. Blank CSV values mean unknown
or not yet available; they must not be automatically converted into zero.

The event schema lives in `src/events.py`. It supports optional real recording
dates and video-relative timestamps, so a future video adapter can leave an
unknown date unset. The simulator supplies known artificial dates and elapsed
seconds; it leaves `video_time_seconds` unset. Each event has a unique ID, source,
observation method, party/table information where relevant, and censoring metadata.
Predicted events have a separate flag and are excluded from observed-label exports.

## Training guidance

Splits are assigned to entire consecutive days before any snapshots are generated.
With 90 days there are **63 train, 13 validation, and 14 test days**. One-day
datasets contain training data only; two days produce train/test; three or more
reserve at least one validation and one test day. Do not shuffle snapshots from
the same party between splits.

Arrival prediction:

- At `bucket_start`, predict `target_arriving_parties` or `target_arriving_people`
  in `[bucket_start, bucket_end)`. This is total arriving demand, including parties
  that eventually abandon or cannot fit, rather than only seated customers.
- Queue/table features use events strictly before that boundary. Lag and rolling
  features use earlier intervals and reset at each day's opening.
- The manifest's `features.arrivals` list identifies input columns. The target
  counts are future information and must not be inputs.

Remaining-duration prediction:

- Use `features.remaining_duration` from the manifest as input columns. Service
  contact is available only after it has happened at the snapshot time.
- `target_remaining_dining_seconds` runs until departure. The distinct
  `target_remaining_ready_seconds` runs until cleaning is confirmed complete.
- Ordinary regression needs rows whose chosen target is known. A completed meal
  may have a valid dining target but an unknown ready target when cleaning runs
  beyond the observation window. Preserve censored rows for future survival models.
- IDs, final visit totals, censoring flags, outcome columns, and target columns
  are metadata/labels, not training features. Fit imputers/scalers only on train.

`parties.csv` records `target_wait_to_seat_seconds` only for seated parties.
Abandonment and censored waits are separate outcomes; evaluating only seated
parties omits customers who gave up. Later queue-policy evaluation should include
abandonment and excessive waits as well as mean seating wait.

Reusing the same configuration, dates, seed, generator, and runtime/timezone-data
versions reproduces the files. Per-day seeds depend on the root seed and date,
so extending the number of days preserves earlier runs. The manifest's checksums
allow exact file comparison. Different simulation settings should be kept as
separate experiments, and synthetic results must be reported separately from
public-video or future real-restaurant results.

## Verification

The focused tests need no video/ML packages:

```powershell
python -m unittest discover -s tests -p test_simulation.py -v
```

For an isolated Windows Python 3.12 environment:

```powershell
uv run --no-project --python 3.12 --with tzdata python -m unittest discover -s tests -p test_simulation.py -v
```

They cover event validation, seed reproducibility, table capacity/reuse, cleaning
resources, queue ordering and abandonment, service alerts, incomplete labels,
arrival interval boundaries, chronological groups, checksums, and refusing to
overwrite a dataset. The upstream video tests still need the video dependencies.
