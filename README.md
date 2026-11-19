# Restaurant Analysis

Analyze two recorded restaurant cameras on one NVIDIA GPU: a dining-room camera for seated guests
and table occupancy, and an entrance camera for staff, customers, passersby, and queue dwell time.
One shared YOLO26 Pose model performs FP16 batched inference while each camera keeps its own
ByteTrack state. No face recognition or cross-camera identity matching is used.

## Team project: synthetic training data

The first extension for our four-person project generates timestamped restaurant events,
queue outcomes, visit durations, and training tables. It runs separately from the video stack:

```powershell
python -m src.simulation --days 90 --seed 42 --output data/simulation/my-first-dataset
```

On Windows, an isolated Python 3.12 run can supply timezone data without installing the video dependencies:

```powershell
uv run --no-project --python 3.12 --with tzdata python -m src.simulation --days 90 --seed 42 --output data/simulation/my-first-dataset
```

Use one command and a new output directory for each dataset. See [SIMULATION.md](SIMULATION.md)
for configuration, file schemas, tests, and training guidance, [PROJECT_PLAN.md](PROJECT_PLAN.md)
for the technical scope, and [ROADMAP.md](ROADMAP.md) for the project phases. These datasets are
synthetic; forecasting models and the live queue recommendation are subsequent implementation steps.

## Demo Video

Watch the full demonstration on YouTube:

[![Watch the video](https://img.youtube.com/vi/iWFZjKku5C4/hqdefault.jpg)](https://youtu.be/iWFZjKku5C4)

## Features

- **Standing-person filtering** — `StandingFilter` excludes standing people from occupancy counts.
  The original detections remain visible in the processed video.
- **Table-based guest counting** — `config/regions.json` defines each table's counting polygon and
  capacity. A `display_polygon` can optionally provide a separate outline for drawing.
- **Count stabilization** — `LoggedCount` smooths noisy frame-by-frame counts using recent readings
  and configurable increase, decrease, and empty delays.
- **Occupancy sessions and timers** — SQLite records visits, their maximum guest count, and
  video-timeline start/end values. The on-video dashboard displays table occupancy, seated guests,
  and elapsed visit time.
- **Operations dashboard** — the page at `/` shows the processed video from `/stream` and live table
  statistics from `/api/status`, plus confirmed and uncertain queue counts from `/api/queue`.
- **Two independent camera pipelines** — CPU reader threads keep bounded latest-frame queues; one
  GPU model batches the two frames while dining and entrance trackers remain independent.
- **Staff and queue evidence** — a stable colored apron marker, trajectory, zones, and dwell time
  distinguish `staff`, `customer`, and `unknown`. A person becomes confirmed in the queue after
  three seconds; fast passersby and staff are excluded.
- **Semi-automatic calibration** — `/calibration` stores normalized per-camera zones in
  `config/cameras.json`; `src/calibration.py` can propose table polygons with YOLO Seg.
- **Strict CUDA execution** — production inference refuses to start when CUDA is unavailable. The
  runtime reports GPU, VRAM, current FPS, ETA, lag, dropped frames, and automatic 640→512 fallback.
- **Background processing** — browser refreshes never restart inference. At end-of-video, the last
  annotated frame remains visible and the status becomes `completed`.
- **Optional output and resizing** — save an annotated copy with `SAVE_OUTPUT_VIDEO` (the input is
  unchanged), or downscale frames with `RESIZE_VIDEO` and `VIDEO_WIDTH` (no upscaling).

## Tech Stack

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Ultralytics YOLO26 Pose
- OpenCV, NumPy, and Supervision
- FastAPI and Uvicorn
- Pydantic Settings and SQLite

## Project Structure

```
Restaurant_Analysis/
├── .env.example
├── .gitignore
├── .python-version
├── Makefile
├── README.md
├── config/
│   ├── cameras.json
│   └── regions.json
├── data/
│   └── restaurant.db
├── frontend/
│   ├── app.js
│   ├── calibration.html
│   ├── calibration.css
│   ├── calibration.js
│   ├── favicon.svg
│   ├── index.html
│   └── style.css
├── models/
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dashboard.py
│   ├── main.py
│   ├── cameras.py
│   ├── calibration.py
│   ├── multi_camera_runtime.py
│   ├── video_processing.py
│   ├── counting/
│   │   ├── __init__.py
│   │   ├── load_regions.py
│   │   └── logged_count.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   └── repository.py
│   ├── filters/
│   │   ├── __init__.py
│   │   └── standing_filter.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── entrance_processor.py
│   │   ├── occupancy_service.py
│   │   ├── person_roles.py
│   │   └── video_processor.py
│   └── utils/
│       ├── __init__.py
│       ├── video_utils.py
│       └── video_writer.py
├── tests/
│   ├── test_api.py
│   ├── test_logged_count.py
│   ├── test_occupancy_service.py
│   ├── test_occupancy_sessions.py
│   ├── test_regions.py
│   ├── test_standing_filter.py
│   └── test_video_processing.py
├── uv.lock
└── videos/
    ├── output/
    └── restaurant-demo.mp4
```

## Installation

Install [uv](https://docs.astral.sh/uv/) and Python 3.12. On Linux, `opencv-python` also requires
the OpenGL library (`libgl1` on Debian/Ubuntu).

```bash
git clone https://github.com/Raafat-Nagy/Restaurant-Analysis.git
cd Restaurant-Analysis
uv sync
make download-video
make download-model
cp .env.example .env
```

Place YOLO26 pose weights in `models/`. The default `MODEL_PATH` is `models/yolo26n-pose.pt`;
provide that file or set `MODEL_PATH` in `.env` to the weights you have.

On this Windows laptop, the user-level `GpuPython` PowerShell module provides
`Test-GpuPython`, `Install-GpuTorch`, `Enable-GpuProject`, and `New-GpuProject`. For example:

```powershell
Test-GpuPython -Python .\.venv\Scripts\python.exe -RequireTorch
```

## Configuration

Settings are listed in `.env.example`; values can also be supplied through environment variables.
Run commands from the project root so relative video and model paths resolve as expected.

- **Input files:** `DINING_VIDEO_PATH`, `ENTRANCE_VIDEO_PATH`, and `MODEL_PATH` (default
  `models/yolo26n-pose.pt`). Leave the entrance path empty until a second recording is ready.
- **Staff model:** leave `ROLE_MODEL_PATH` empty to use the calibrated apron marker only, or point
  it to validated `staff/customer` weights. See [TRAINING.md](TRAINING.md).
- **Inference:** `DEVICE=cuda:0`, `REQUIRE_CUDA=true`, `HALF_PRECISION=true`, `TARGET_FPS=10`,
  `INFERENCE_BATCH_SIZE=2`, `CONF`, `IOU`, and `IMGSZ`.
- **Queue/staff:** `QUEUE_DWELL_SECONDS` and the `STAFF_MARKER_*` HSV settings.
- **Count stabilization:** `COUNT_INCREASE_DELAY`, `COUNT_DECREASE_DELAY`, `COUNT_EMPTY_DELAY`, and
  `COUNT_HISTORY_SIZE`.
- **Output video:** `SAVE_OUTPUT_VIDEO` (off by default) and `OUTPUT_VIDEO_PATH`.
- **Resizing:** `RESIZE_VIDEO` (off by default) and `VIDEO_WIDTH`.
- **On-video dashboard:** `SHOW_DASHBOARD` (on by default) and `DASHBOARD_WIDTH`.

Edit the `zones` list in `config/regions.json` to set table names, capacities, and polygons. Each
zone requires `counting_polygon`; `display_polygon` is optional and, when present, is used to draw
the table outline. Configure the entrance, queue, sidewalk, and entry line at `/calibration`; those
coordinates are normalized in `config/cameras.json`.

## Usage

```bash
uv run python src/main.py
```

Alternatively, run `make run`. Open [http://localhost:5000/](http://localhost:5000/) to view the
dashboard. The streams are available at `/stream/dining` and `/stream/entrance`; JSON snapshots are
available at `/api/status`, `/api/queue`, and `/api/calibration`. Processing starts with the
application and replays the inputs from the beginning on the next application start. Occupancy and
queue timers use video time, not processing or browser time. The last run is written to
`data/benchmarks/last_run.json`; the last successfully completed run is kept in `latest.json`.

## Testing

```bash
uv run python -m pytest -q
```

The suite covers CUDA refusal, bounded
camera queues, automatic resolution fallback, normalized calibration, queue dwell/staff/passersby,
count stabilization, occupancy sessions, web routes, and background streaming. Unit tests do not
need a YOLO model; the Windows guide includes the real RTX smoke test.

## Database

SQLite stores configured table capacities in `tables` and visits in `occupancy_sessions`
(`started_at`, `ended_at`, and `max_people`). Each processing run records separate session rows;
history is preserved, only one session per table may be active, and sessions left by an interrupted
run are closed when the next run starts. Browser connections do not create sessions.

## Acknowledgments

Inspired by the [Restaurant Analysis using Computer Vision – Full Course](https://youtu.be/KqL5ymW3foI), which introduced key concepts in person detection, table-based counting, and occupancy monitoring. This project extends these concepts with count stabilization, standing-person filtering, and SQLite-based occupancy tracking.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
