# Restaurant intelligence: computer vision and ML

## Agreed direction

Four-person academic project in vision artificielle, based on
[Raafat-Nagy/Restaurant-Analysis](https://github.com/Raafat-Nagy/Restaurant-Analysis).

The system should observe the restaurant, predict demand and table readiness,
and help staff manage seating and service. The queue recommendation must change
with predicted table availability; eight people is an example, not a fixed limit.
The team will start with public data and a simulated restaurant.

The timestamped event contract, seeded simulator, two-camera CUDA vision runtime,
calibration UI, entrance queue analytics, and CSV training exports are implemented.
See [SIMULATION.md](SIMULATION.md) and [WINDOWS_GUIDE.md](WINDOWS_GUIDE.md).
Forecasting models and the dynamic queue recommendation are still planned.

## What the cloned project already provides

- YOLO26 pose inference and a heuristic to exclude standing people from seated counts.
- Configurable table polygons and capacities in `config/regions.json`.
- Stabilized guest counts and elapsed occupancy timers.
- SQLite tables and occupancy sessions.
- FastAPI serving an annotated MJPEG video and a simple viewing page.
- Existing tests using temporary databases and synthetic video.

Current boundaries:

- ByteTrack IDs are intentionally local to each camera; no cross-camera identity is attempted.
- The entrance queue exists, but party-size input, forecasting, and learned duration models do not.
- Table colors reflect guest counts, not a confirmed cleaning/service lifecycle.
- The dashboard and `/api/status` expose table and GPU statistics; `/api/queue`
  exposes observed queue state. The ML panel remains an honest placeholder.
- The database stores video-relative seconds. It needs run/source identifiers and
  an optional known recording date/time before calendar features can be meaningful.
- Sessions still occupied when a clip ends are closed at that boundary. Interrupted
  sessions are closed at their start on restart. These are not observed departures;
  a training export must identify censoring or exclude these records from ordinary
  completed-duration labels.

## Target behavior

1. Detect and track people; associate seated guests with configured table zones.
2. Track a separate entrance zone without applying the seated-person filter there.
3. Maintain table states: `READY`, `OCCUPIED`, `NEEDS_CLEANING`, `CLEANING`, and
   `UNKNOWN`. An empty table after a visit is not automatically ready.
4. Start an unserved timer when a party sits. Raise one alert after a configurable
   five-minute threshold unless a service event has been recorded.
5. Raise a cleaning alert after a confirmed departure. Close it after readiness is
   confirmed. Use manual/simulated service and cleaning confirmations for the MVP;
   automated waiter-interaction and cleaning recognition need separate labels.
6. Estimate each occupied table's remaining time to readiness, including cleaning.
7. Estimate waits for queued parties, and recommend how many people can be seated
   within a configurable horizon, initially proposed as 15 minutes.
8. Forecast arrivals in 15-minute intervals and display peak periods in a weekly report.

Show state labels as well as colors, and show estimates with an uncertainty range
when validation supports one. Alerts are timer/state rules; they do not need an ML
classifier. Phone push delivery is a later integration after in-app alerts work.

## Queue example and seating policy

At 12:30, suppose two four-seat tables are predicted to be clean and ready at
12:38 and 12:41. With a 15-minute horizon, queued parties of four and four could
both be seated: the near-term recommendation is eight people.

This is not a rule that every queue must contain at most eight people:

- A single party of eight cannot use those tables unless combining tables is allowed.
- Two parties of three would seat six people under a one-party-per-table policy.
- Cleaning delays, reservations, existing queue commitments, and uncertainty can
  reduce the capacity available to a newly arriving party.
- If the tables become ready outside the horizon, they do not count toward that
  horizon's immediate recommendation.

MVP assumptions: one party per table, no table combining, and first-come-first-served
for parties that can fit the configured tables. Oversized parties receive an
explicit unsupported-party result. Any queue bypass rule must be explicit.
Party sizes come from simulated/host input initially; a camera's people count
alone does not reliably identify which people belong to the same party.

Implement seating as a small event-based scheduler: assign queued parties to
compatible tables in readiness order, consume those table commitments once, and
calculate each party's predicted seating time. Distinguish total forecast seat
capacity from the number of actual queued people that can be seated.
The result is staff guidance, not an automatic door lock or guaranteed departure time.

## Data strategy

| Source | Useful for | Limitation |
| --- | --- | --- |
| Demo video linked by the upstream `Makefile` | Table calibration, visual pipeline checks, manually labeled occupancy/count examples | One camera/clip cannot establish multi-day demand patterns or dependable meal-duration statistics; availability is not a training license |
| [Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting/data) | Candidate public benchmark for daily visitor forecasting | Access follows competition rules; inspect files before adoption. Daily visitor targets cannot validate 12:00-14:00 peaks, queue waits, or individual dining durations |
| Seeded restaurant simulator (implemented) | Timestamped arrivals, parties, seating, meals, cleaning, service, queues, and staffing scenarios | Results establish performance in simulation, not accuracy at a real restaurant |

Do not assume that a public dataset contains synchronized video, table departures,
cleaning confirmations, and customer arrivals. No dataset has been downloaded or
adopted as the project's ground truth yet.

Proposed simulator settings: table layout/capacities, opening hours, variable
arrival rates, party-size distribution, meal-duration distribution, cleaner/server
availability, cleaning/service delays, and abandonment thresholds. Reservations
remain a later extension.
The lunch peak is an adjustable simulation assumption, not evidence learned about
an actual restaurant. Record seed and configuration for every generated run.
Generate independent days with stochastic variation and evaluate under changed
peaks, slower service, different meal durations, and demand surges.

Minimal event contract shared by simulation and future vision adapters:

- Identity: `restaurant_id`, `run_id`, `event_id`, `source`, `is_synthetic`.
- Time: `video_time_seconds` where relevant; timezone-aware `occurred_at` only when
  known, plus provenance for any assigned recording start time.
- Seating: `table_id`, `capacity`, `party_id`, `party_size`, `event_type`.
- Quality: detection confidence where applicable, observation/confirmation method,
  and `end_reason`/censoring status for incomplete visits.

Event types include party arrival, seating, service confirmation, departure,
cleaning start, ready confirmation, and abandonment. Keep observed and predicted
events distinct. Preserve an append-only event history to reconstruct labels.

## Model comparison

We will train multiple candidate models for each prediction task on the same
splits. Person detection, duration prediction, and arrival forecasting have
different inputs and targets; they should be evaluated separately.

| Task | Baseline | Initial candidates | Evaluation |
| --- | --- | --- | --- |
| Seated counting and occupancy | Current pretrained pose pipeline | Small versus larger pose models; tracking and threshold variants; fine-tuning only with enough annotated frames | Count MAE, occupancy precision/recall/F1, transition delay, end-to-end FPS |
| Arrival forecasting | Same weekday/time-slot historical average | Ridge, Random Forest, histogram gradient boosting with lag/calendar features | MAE/RMSE in arrivals per 15-minute interval; errors during peaks |
| Remaining meal duration | Conditional historical median among comparable visits that have lasted at least the current elapsed time | Ridge, Random Forest, histogram gradient boosting on elapsed-visit snapshots | MAE in minutes; error by elapsed duration and party size |
| Cleaning duration | Historical median cleaning time | Model only if enough confirmed cleaning records exist | MAE in minutes |
| Party wait and queue recommendation | Scheduler using baseline durations | Same scheduler using each learned duration model | Wait MAE, excessive-wait rate, abandonment, utilization, queue size |

Start with these comparisons before adding XGBoost/CatBoost, temporal neural
networks, or survival models. Larger model lists are useful only if there is
enough data and time to compare them fairly.

For arrival forecasting, use past counts and calendar information available at
prediction time. Forecast future arrivals separately from future occupancy:
occupancy also depends on departures and seating decisions.

For remaining duration, train on snapshots of ongoing visits. Features can include
elapsed seated time, party size, capacity, and currently observed load. The label
is time from the snapshot to the actual departure. Predict cleaning separately or
use a confirmed-ready label when available. Do not use eventual maximum party size,
future service events, actual future departures, or final visit duration as inputs.
Simply subtracting elapsed time from an unconditional average is a weak baseline
because the party is known to still be present.

## Evaluation rules

- Use chronological train/validation/test periods for forecasts; fit preprocessing
  and tune hyperparameters on training/validation only.
- Split by day/run/visit before making duration snapshots. Keep every snapshot of
  a visit in one split. For vision, keep neighboring frames and repeated copies
  of a clip together; use held-out camera scenes when available.
- Compute lag/rolling features using only data available before the forecast origin.
  Account for the forecast horizon when separating training labels from validation.
- Treat incomplete/cut-off visits as censored, not completed meals.
- Keep simulated, public-video, and any later real-restaurant results separate.
- Evaluate unseen simulation seeds and changed parameter distributions, rather than
  only reproducing the same generator settings.
- Compare every candidate with a simple baseline on identical held-out examples.
  Report model settings, seeds, dataset version, sample counts, and prediction latency.
- Calibrate prediction ranges on validation data and measure test coverage if ranges
  are shown. Do not present arbitrary ranges as measured confidence.

## Four-person responsibilities

| Person | Ownership | Concrete deliverables |
| --- | --- | --- |
| 1: Computer vision | Detection/tracking, polygons, entrance/table observations | Annotated video examples, occupancy/count benchmark, timestamped vision events |
| 2: Data and simulation | Event schema, labeling/export, simulator, data quality | Reproducible scenarios, documented labels, shared chronological dataset splits |
| 3: ML and evaluation | Demand/duration training, features, model comparisons | Repeatable training commands, saved models/metadata, comparison report |
| 4: Application and integration | API, seating scheduler, table states, alerts, dashboard | Usable restaurant demo, queue estimates, analytics views, integration checks |

All four contribute to the report and presentation. Use the same event contract,
dataset split definitions, and metric names across components. These are proposed
responsibilities; names and dates can be assigned once the academic deadline is known.

## Suggested implementation order

1. Reproduce the upstream demo and baseline tests; obtain video/weights and calibrate
   the table layout. Keep a record of baseline results.
2. Add the event contract, recording/run metadata, censoring information, and a seeded
   simulator. Produce a labeled dataset before building training scripts.
3. Build table/service states, in-app alerts, and the seating scheduler with simple
   duration estimates. Check mixed party sizes and cleaning delays.
4. Train and compare the initial forecasting and duration models with held-out data.
5. Connect model outputs to the API and dashboard; evaluate the complete system under
   quiet, lunch-rush, and delayed-cleaning scenarios.
6. Prepare a demonstration and report that distinguish existing upstream work,
   the team's contributions, simulation results, and measured vision results.

The first complete demo should show a party arriving, a table becoming occupied,
an unserved alert after five simulated minutes, departure/cleaning, updated waits
for queued parties, and a demand forecast. Use simulation time consistently so a
sped-up replay does not alter restaurant timers.

## Local setup on Windows

Run commands from the cloned `Restaurant-Analysis` directory:

```powershell
uv sync --frozen
Copy-Item .env.example .env
uv run python -m unittest discover -s tests -v
```

The project dependency environment has now been installed on the development
machine. Use the test command after each integration change.

To run actual video inference, also obtain the demo clip referenced in `Makefile`
and the configured pose weights (`models/yolo26n-pose.pt` by default), or point
`.env` to your own files. Neither is included in the Git repository.
Then launch with `uv run python src/main.py` and open `http://localhost:5000`.
The demonstration inference has been run locally. No forecasting or remaining-time
model has been trained yet.

## Technical references

- [Ultralytics tracking](https://docs.ultralytics.com/modes/track/): tracking IDs,
  tracker configuration, and integration with pretrained models.
- [scikit-learn time-series validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split): chronological evaluation.
- [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).
- [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
