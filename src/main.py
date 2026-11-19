"""FastAPI app: the web page and the MJPEG video stream."""

import json
import logging
import csv
import io
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from cameras import load_camera_config, resolve_camera_sources, save_camera_config
from config import PROJECT_ROOT, REGIONS_CONFIG_PATH, settings
from counting.load_regions import load_zones
from database.connection import init_database
from database.repository import export_snapshot_rows, operations_history, sync_tables
from operations_config import load_operations_settings, save_operations_settings
from video_processing import (
    acknowledge_alert,
    apply_operations_settings,
    apply_table_action,
    get_calibration_frame,
    get_operations_status,
    get_processing_status,
    get_queue_status,
    sample_calibration_hsv,
    start_processing,
    stop_processing,
    stream_frames,
)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
CALIBRATION_DRAFT_PATH = PROJECT_ROOT / "config" / "calibration_draft.json"

logger = logging.getLogger(__name__)


def _prepare_database():
    """Create the schema and synchronize the tables before processing starts."""
    init_database()
    camera_config = load_camera_config()
    dining = next(camera for camera in camera_config["cameras"] if camera["role"] == "dining")
    tables = dining.get("regions", {}).get("tables", [])
    if tables:
        sync_tables([(table["id"], table["capacity"]) for table in tables])
    else:
        sync_tables([(zone["name"], zone["max"]) for zone in load_zones(REGIONS_CONFIG_PATH)])


class ConnectionCloseMiddleware:
    """Advertise single-use connections: every response ends its connection.

    ``uvicorn.run`` below sets ``timeout_keep_alive=0``, so the server closes
    each connection as soon as its response completes. Without this header a
    client or proxy assumes HTTP keep-alive and may write its next request
    into a connection that is already closing; that request is lost or
    answered with uvicorn's protocol-error 400. The page, its static assets
    and the repeated ``/stream`` connections all go through this path.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_close(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"connection", b"close"))
                message = dict(message, headers=headers)
            await send(message)

        await self.app(scope, receive, send_with_close)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One background processing thread for the whole server lifetime:
    # browsers only read the frames it publishes, they never start it.
    _prepare_database()
    try:
        start_processing()
    except Exception:
        # A missing model or unreadable video must not abort start-up: the
        # page still loads and reports that the stream is unavailable.
        logger.exception("Video processing could not start")
        stop_processing()
    try:
        yield
    finally:
        stop_processing()


app = FastAPI(lifespan=lifespan)

# Single-use connections: clients are told the connection ends with each
# response, so no request can be written into a closing connection.
app.add_middleware(ConnectionCloseMiddleware)

# Static assets for the dashboard (HTML/CSS/JS in frontend/).
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/calibration")
def calibration_page():
    return FileResponse(FRONTEND_DIR / "calibration.html")


@app.get("/stream")
def stream():
    return StreamingResponse(
        stream_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/{camera_id}")
def camera_stream(camera_id: str):
    if camera_id not in {"dining", "entrance"}:
        raise HTTPException(status_code=404, detail="Unknown camera")
    return StreamingResponse(
        stream_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/status")
def status():
    """Current observed table analytics for the browser dashboard."""
    return get_processing_status()


@app.get("/api/queue")
def queue_status():
    return get_queue_status()


@app.get("/api/operations")
def operations_status():
    return get_operations_status()


@app.get("/api/operations/history")
def history(hours: int = Query(24, ge=1, le=24 * 366)):
    return operations_history(hours)


@app.get("/api/operations/settings")
def operations_settings():
    return load_operations_settings().model_dump()


@app.put("/api/operations/settings")
def update_operations_settings(payload: dict = Body(...)):
    try:
        before = load_operations_settings()
        saved = save_operations_settings(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    apply_operations_settings(saved)
    restart_keys = {
        "camera_reconnect_attempts",
        "camera_reconnect_delay_seconds",
        "stream_jpeg_quality",
    }
    restart_required = any(
        getattr(before, key) != getattr(saved, key) for key in restart_keys
    )
    return {"settings": saved.model_dump(), "restart_required": restart_required}


@app.post("/api/tables/{table_id}/actions/{action}")
def table_action(table_id: str, action: str):
    try:
        return {"table": apply_table_action(table_id, action)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Table inconnue: {table_id}") from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/alerts/{alert_id}/acknowledge")
def alert_acknowledge(alert_id: str):
    try:
        return {"alert": acknowledge_alert(alert_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Alerte inconnue ou déjà résolue") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/operations/export.csv")
def export_operations(hours: int = Query(24 * 30, ge=1, le=24 * 366 * 5)):
    rows = export_snapshot_rows(hours)
    output = io.StringIO(newline="")
    fields = [
        "recorded_at",
        "video_time",
        "occupied_tables",
        "total_tables",
        "seated_guests",
        "queue_people",
        "queue_groups",
        "staff_visible",
        "average_wait_seconds",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=restaurant-vision-stats.csv"},
    )


@app.get("/api/calibration")
def calibration():
    config = load_camera_config()
    resolved = {camera["id"]: camera for camera in resolve_camera_sources(config)}
    for camera in config["cameras"]:
        camera["calibration_required"] = resolved[camera["id"]]["calibration_required"]
    return config


@app.get("/api/calibration/table-proposals")
def table_proposals():
    if not CALIBRATION_DRAFT_PATH.is_file():
        return {"tables": [], "requires_confirmation": True, "state": "not_generated"}
    try:
        return json.loads(CALIBRATION_DRAFT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Invalid calibration draft: {exc}") from exc


@app.get("/api/calibration/frame/{camera_id}")
def calibration_frame(camera_id: str):
    if camera_id not in {"dining", "entrance"}:
        raise HTTPException(status_code=404, detail="Unknown camera")
    frame = get_calibration_frame(camera_id)
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame is available for this camera")
    return Response(content=frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/calibration/color-sample/{camera_id}")
def calibration_color_sample(
    camera_id: str,
    x: float = Query(..., ge=0, le=1),
    y: float = Query(..., ge=0, le=1),
):
    if camera_id not in {"dining", "entrance"}:
        raise HTTPException(status_code=404, detail="Unknown camera")
    sample = sample_calibration_hsv(camera_id, x, y)
    if sample is None:
        raise HTTPException(status_code=503, detail="No frame is available for this camera")
    return sample


@app.put("/api/calibration")
def update_calibration(payload: dict = Body(...)):
    try:
        for camera in payload.get("cameras", []):
            camera.pop("calibration_required", None)
            setting_name = camera.get("source_setting", "")
            if setting_name:
                camera["calibration_source"] = getattr(settings, setting_name, "")
        saved = save_camera_config(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"config": saved, "restart_required": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, timeout_keep_alive=0)
