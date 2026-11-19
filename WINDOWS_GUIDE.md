# View and run the project on Windows

This guide is for the copy of the project on this computer. Open **PowerShell**
and enter the following commands one block at a time.

## 1. Open the project folder

```powershell
cd 'C:\Users\elhou\OneDrive\Documents\ChatGPT\vision artificielle\Restaurant-Analysis'
code .
```

`code .` opens the folder in VS Code. The VS Code Explorer on the left shows
the code and files. You can also open File Explorer in that folder with:

```powershell
explorer .
```

## 2. Install the video application's dependencies

Python and `uv` are already installed on this computer. From the project folder:

```powershell
uv sync --frozen
```

The first installation downloads large packages including PyTorch and OpenCV.
Wait for it to finish before continuing. It creates a local `.venv` in this
project; there is no need to activate it when using `uv run`.

PyTorch is locked to the official CUDA 13.0 wheels (`torch 2.14.1+cu130` and
`torchvision 0.29.1+cu130`). Verify the actual environment, GPU, matrix operation,
and VRAM with the user-level command installed on this laptop:

```powershell
Test-GpuPython -Python .\.venv\Scripts\python.exe -RequireTorch
```

The same `GpuPython` module is auto-loadable in every new PowerShell window:

```powershell
Get-Command Test-GpuPython, Install-GpuTorch, Enable-GpuProject, New-GpuProject
New-GpuProject -Path .\my-next-ml-project
```

`uv` also has user-level automatic accelerator selection in
`$env:APPDATA\uv\uv.toml`. Every virtual environment still contains its own
PyTorch installation, but the CUDA index and checks no longer need to be rediscovered.

## 3. Download the demo files

The video and model are separate downloads; they are not in the Git repository.
These commands skip either file if it is already present:

```powershell
New-Item -ItemType Directory -Force -Path 'videos', 'models' | Out-Null

if (-not (Test-Path 'videos/restaurant-demo.mp4')) {
    curl.exe --fail --location --continue-at - --output 'videos/restaurant-demo.mp4.part' 'https://cvzone-storage.sfo3.cdn.digitaloceanspaces.com/courses/restaurat-analysis/videos/Cctv-Resturant-Demo.mp4'
    if ($LASTEXITCODE -ne 0 -or (Get-Item 'videos/restaurant-demo.mp4.part').Length -ne 43720551) { throw 'Video download is incomplete; retry it.' }
    Move-Item 'videos/restaurant-demo.mp4.part' 'videos/restaurant-demo.mp4'
}

if (-not (Test-Path 'models/yolo26n-pose.pt')) {
    curl.exe --fail --location --continue-at - --output 'models/yolo26n-pose.pt.part' 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt'
    if ($LASTEXITCODE -ne 0 -or (Get-Item 'models/yolo26n-pose.pt.part').Length -ne 7878574) { throw 'Model download is incomplete; retry it.' }
    Move-Item 'models/yolo26n-pose.pt.part' 'models/yolo26n-pose.pt'
}

if (-not (Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
}
```

The video is approximately 42 MiB and the nano pose model approximately 7.5 MiB. The `.part`
files retain any download progress so you can rerun the block after an interruption.
Wait for both commands to finish before starting the application.

## 4. Run and view the video demo

```powershell
uv run python src/main.py
```

Leave PowerShell open. When the server has started, visit
[http://localhost:5000](http://localhost:5000) in your browser. The page displays
the processed video with the upstream project's table and people annotations.
Processing speed depends on your computer. The video runs once per application
start; to replay it, stop the application and run the command again.

The page shows seated people, table occupancy, GPU/VRAM, and the entrance queue
when a second video is configured. The queue module reports confirmed and uncertain
people; forecasting and the dynamic queue recommendation remain later ML phases.

To connect your two recordings, edit `.env`:

```dotenv
DINING_VIDEO_PATH=videos/my-dining-room.mp4
ENTRANCE_VIDEO_PATH=videos/my-entrance.mp4
DEVICE=cuda:0
REQUIRE_CUDA=true
TARGET_FPS=10
INFERENCE_BATCH_SIZE=2
```

Restart the backend, open [http://localhost:5000/calibration](http://localhost:5000/calibration),
and draw the entrance queue, sidewalk/passerby corridor, interior, and entry line.
The values are stored as normalized coordinates in `config/cameras.json`.
Staff classification uses the configured apron color over several observations;
uncertain people remain `unknown`. There is no facial recognition.

Live endpoints:

- `/stream/dining` and `/stream/entrance` — annotated MJPEG streams;
- `/api/status` — GPU, VRAM, FPS, progress, ETA, lag, and skipped/dropped frames;
- `/api/queue` — confirmed/uncertain queue counts and anonymous track IDs;
- `/api/calibration` — current per-camera normalized regions.

The runtime begins at 640 px and automatically records a 512 px fallback in
`data/benchmarks/latest.json` if real-time throughput is insufficient.

To stop the application, return to PowerShell and press **Ctrl+C**. If port 5000
is already occupied, stop the other application using that port first.

## 5. View the simulated data

The first synthetic dataset has already been generated locally. To open its folder:

```powershell
explorer .\data\simulation\baseline-90d
```

- `arrivals_15min.csv` contains simulated arrivals for every 15-minute interval.
- `visits.csv` contains the observed duration and status of each simulated visit.
- `duration_snapshots.csv` contains examples for remaining-time prediction.
- `parties.csv` contains simulated queue outcomes.
- `manifest.json` describes the data and training/validation/test splits.

Open CSV files in Excel, VS Code, or a text editor. In Excel, use **Data → From
Text/CSV** and choose comma as the delimiter if columns are not separated
automatically. See [SIMULATION.md](SIMULATION.md) for field descriptions and
instructions to generate another dataset.

The vision runtime and simulator currently produce separate evidence streams.
Forecasting models, remaining-table-time prediction, and the dynamic queue-limit
scheduler are still to be trained and integrated.
