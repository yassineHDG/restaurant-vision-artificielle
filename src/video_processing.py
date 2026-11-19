"""Background video execution and MJPEG streaming.

This module runs the video pipeline in one background thread and streams the
latest processed frames to browsers. It owns thread lifecycle, the shared
frame mailbox, and synchronization only: per-frame detection, filtering, and
counting live in ``services.video_processor.VideoProcessor``, and occupancy
sessions are managed by ``services.occupancy_service.OccupancyService``.
Browsers only read the mailbox -- connecting, refreshing, or disconnecting
never starts a processing run and never touches processing state.
"""

import threading
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from config import settings
from dashboard import RestaurantDashboard
from services.video_processor import VideoProcessor
from multi_camera_runtime import MultiCameraRuntime
from utils.video_utils import encode_frame, resize_frame
from utils.video_writer import close_writer, open_writer, write_frame

VIDEO_PATH = settings.VIDEO_PATH

# --- shared frame state ---
#
# One background thread processes the video exactly once and publishes each
# encoded frame into this thread-safe mailbox; MJPEG clients only read it.

_frame_condition = threading.Condition()
_latest_chunk = None
_frame_seq = 0
_processing_done = False
_processing_state = "idle"
_processing_error = None
_latest_stats = {
    "state": "idle",
    "error": None,
    "processing_done": False,
    "frame_sequence": 0,
    "video_time_seconds": 0.0,
    "source": Path(VIDEO_PATH).name if "://" not in VIDEO_PATH else "Flux caméra",
    "seated_guests": 0,
    "occupied_tables": 0,
    "available_tables": 0,
    "total_tables": 0,
    "total_capacity": 0,
    "tables": [],
}

_processing_thread = None
_thread_lock = threading.Lock()
_stop_event = threading.Event()

_processor = None  # shared VideoProcessor (tests may pass a stand-in)
_runtime = None


def _get_processor() -> VideoProcessor:
    global _processor
    if _processor is None:
        _processor = VideoProcessor()
    return _processor


def _publish_chunk(chunk):
    """Store the newest encoded frame and wake every waiting streamer."""
    global _latest_chunk, _frame_seq
    with _frame_condition:
        _latest_chunk = chunk
        _frame_seq += 1
        _frame_condition.notify_all()


def _source_label() -> str:
    """Return a safe source label without exposing camera credentials."""
    return "Flux caméra" if "://" in VIDEO_PATH else Path(VIDEO_PATH).name


def _empty_stats(tables, state="starting", error=None):
    table_rows = [
        {
            "id": table["id"],
            "capacity": int(table.get("max", 0)),
            "guests": 0,
            "occupied": False,
            "status": "available",
            "occupancy_seconds": None,
        }
        for table in tables
    ]
    return {
        "state": state,
        "error": error,
        "processing_done": False,
        "frame_sequence": 0,
        "video_time_seconds": 0.0,
        "source": _source_label(),
        "seated_guests": 0,
        "occupied_tables": 0,
        "available_tables": len(table_rows),
        "total_tables": len(table_rows),
        "total_capacity": sum(row["capacity"] for row in table_rows),
        "tables": table_rows,
    }


def _publish_stats(stats, video_time, tables):
    """Publish one JSON-safe analytics snapshot for the web dashboard."""
    global _latest_stats
    table_rows = []
    for table in tables:
        table_id = table["id"]
        guests = int(stats["table_counts"].get(table_id, 0))
        elapsed = stats["occupancy_seconds"].get(table_id)
        table_rows.append(
            {
                "id": table_id,
                "capacity": int(table.get("max", 0)),
                "guests": guests,
                "occupied": guests > 0,
                "status": "occupied" if guests > 0 else "available",
                "occupancy_seconds": round(float(elapsed), 3) if elapsed is not None else None,
            }
        )

    occupied = sum(row["occupied"] for row in table_rows)
    with _frame_condition:
        _latest_stats = {
            "state": "running",
            "error": None,
            "processing_done": False,
            "frame_sequence": _frame_seq,
            "video_time_seconds": round(float(video_time), 3),
            "source": _source_label(),
            "seated_guests": int(stats["seated_guests"]),
            "occupied_tables": occupied,
            "available_tables": len(table_rows) - occupied,
            "total_tables": len(table_rows),
            "total_capacity": sum(row["capacity"] for row in table_rows),
            "tables": table_rows,
        }


def _set_processing_done(state="completed", error=None):
    global _processing_done, _processing_state, _processing_error, _latest_stats
    with _frame_condition:
        _processing_done = True
        _processing_state = state
        _processing_error = error
        _latest_stats = dict(
            _latest_stats,
            state=state,
            error=error,
            processing_done=True,
            frame_sequence=_frame_seq,
        )
        _frame_condition.notify_all()


def _get_legacy_status():
    """Return an isolated snapshot for the public status endpoint."""
    with _frame_condition:
        return deepcopy(_latest_stats)


def _process_video(processor):
    """Run the processing pipeline once and publish frames for the streamers."""
    cap = None
    writer = None
    writer_tried = False
    completed = False
    processing_error = None

    frame_index = 0
    use_pts = True
    last_pts_ms = None
    video_time = 0.0

    try:
        # Failures while initializing the run (drawing panel, session setup,
        # video file) still have to run the cleanup below.
        processor.begin_run()

        dashboard = RestaurantDashboard(
            tables=processor.tables,
            width=settings.DASHBOARD_WIDTH,
            title_scale=0.48,
            title_thickness=1,
            heading_scale=0.44,
            heading_thickness=1,
            stat_scale=0.72,
            stat_thickness=1,
            table_id_scale=0.45,
            table_id_thickness=1,
            table_count_scale=0.37,
            table_count_thickness=1,
            timer_scale=0.4,
            timer_thickness=1,
            text_scale=0.5,
            text_thickness=1,
        )

        cap = cv2.VideoCapture(VIDEO_PATH)

        if not cap.isOpened():
            raise RuntimeError(f"Error reading video file: {VIDEO_PATH}")

        fps = sv.VideoInfo.from_video_path(VIDEO_PATH).fps
        # Frame-index timeline fallback when presentation timestamps are unusable.
        timeline_fps = fps if fps and np.isfinite(fps) and fps > 0 else None

        while not _stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                break

            # Video timeline in seconds: the presentation timestamp when
            # reliable (finite, non-negative, strictly increasing),
            # otherwise frame index / the video's actual FPS. Never
            # wall-clock processing time, and never allowed to move
            # backwards. If both sources are unusable, the timeline holds,
            # so a timer can never increase incorrectly.
            if use_pts:
                pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if (
                    np.isfinite(pts_ms)
                    and pts_ms >= 0
                    and (last_pts_ms is None or pts_ms > last_pts_ms)
                ):
                    last_pts_ms = pts_ms
                    video_time = max(video_time, pts_ms / 1000.0)
                else:
                    use_pts = False
            if not use_pts and timeline_fps is not None:
                video_time = max(video_time, frame_index / timeline_fps)

            frame, stats = processor.process_frame(frame, video_time)
            _publish_stats(stats, video_time, processor.tables)

            # Resize the video frame only, before the dashboard is added,
            # so the dashboard width stays independent of VIDEO_WIDTH.
            if settings.RESIZE_VIDEO:
                frame = resize_frame(frame, width=settings.VIDEO_WIDTH)

            # Optional dashboard panel. Counting, stabilization, timers and
            # the occupancy database keep running either way; only the panel
            # drawing is skipped.
            if settings.SHOW_DASHBOARD:
                dashboard.update(
                    table_counts=stats["table_counts"],
                    seated_guests=stats["seated_guests"],
                    occupancy_seconds=stats["occupancy_seconds"],
                )
                panel = dashboard.render(height=frame.shape[0])
                frame = np.hstack([panel, frame])

            if not writer_tried:
                writer = open_writer(frame, fps)
                writer_tried = True

            write_frame(writer, frame)

            frame_data = encode_frame(frame)

            if frame_data is not None:
                _publish_chunk(frame_data)

            frame_index += 1
        completed = not _stop_event.is_set()
    except Exception as exc:
        processing_error = f"{type(exc).__name__}: {exc}"
    finally:
        if cap is not None:
            cap.release()
        close_writer(writer, completed)
        # Processing has ended: close sessions that are still active (people
        # may still be seated) at the last processed video timestamp. A
        # failure there must not leave the stream waiting forever, so the
        # completion flag is always set.
        try:
            processor.end_run(video_time)
        finally:
            if processing_error is not None:
                _set_processing_done("error", processing_error)
            elif completed:
                _set_processing_done("completed")
            else:
                _set_processing_done("stopped")


def _stream_legacy_frames():
    """Yield MJPEG chunks with the latest frames from the background thread.

    A client that connects mid-run immediately receives the latest available
    frame and then every new one. Clients never trigger processing, and a
    disconnect only ends this generator. The stream ends after the last frame
    once processing has finished.
    """
    last_seq = 0
    while True:
        with _frame_condition:
            while (
                _frame_seq == last_seq
                and not _processing_done
                and not _stop_event.is_set()
            ):
                _frame_condition.wait(timeout=1.0)
            if _frame_seq == last_seq:
                # Processing finished and there is no newer frame.
                return
            last_seq = _frame_seq
            chunk = _latest_chunk
        yield chunk


def _start_legacy_processing(processor=None):
    """Start the single background processing thread (idempotent).

    Calling this again while processing is running is a no-op, so only one
    thread can ever process the video. ``processor`` defaults to a shared
    ``VideoProcessor``; tests may pass a lightweight stand-in.
    """
    global _processing_thread, _latest_chunk, _frame_seq, _processing_done
    global _processing_state, _processing_error, _latest_stats

    with _thread_lock:
        if _processing_thread is not None and _processing_thread.is_alive():
            return

        if processor is None:
            processor = _get_processor()

        with _frame_condition:
            _latest_chunk = None
            _frame_seq = 0
            _processing_done = False
            _processing_state = "starting"
            _processing_error = None
            _latest_stats = _empty_stats(processor.tables)

        _stop_event.clear()
        thread = threading.Thread(
            target=_process_video,
            args=(processor,),
            name="video-processing",
            daemon=True,
        )
        _processing_thread = thread
        thread.start()


def _stop_legacy_processing():
    """Stop the background thread and release the video resources.

    The thread reference is cleared only after the thread has actually
    stopped. While it is still running the reference stays visible, so
    ``start_processing`` can never spawn a second processing thread
    alongside it.
    """
    global _processing_thread

    _stop_event.set()
    with _frame_condition:
        _frame_condition.notify_all()

    with _thread_lock:
        thread = _processing_thread
        if thread is not None:
            thread.join(timeout=10.0)
            if not thread.is_alive():
                _processing_thread = None


def start_processing(processor=None):
    """Start production multi-camera processing or the isolated legacy test worker."""
    global _runtime
    if processor is not None:
        return _start_legacy_processing(processor)
    with _thread_lock:
        if _runtime is not None:
            return
        runtime = MultiCameraRuntime()
        runtime.start()
        _runtime = runtime


def stop_processing():
    """Stop whichever processing mode is active."""
    global _runtime
    runtime = _runtime
    if runtime is not None:
        runtime.stop()
        _runtime = None
    _stop_legacy_processing()


def stream_frames(camera_id="dining"):
    runtime = _runtime
    if runtime is not None:
        return runtime.stream(camera_id)
    return _stream_legacy_frames()


def get_processing_status():
    runtime = _runtime
    if runtime is not None:
        return runtime.get_status()
    return _get_legacy_status()


def get_queue_status():
    runtime = _runtime
    if runtime is not None:
        return runtime.get_queue_status()
    return {
        "state": "not_running",
        "confirmed_queue_count": 0,
        "confirmed_group_count": 0,
        "uncertain_queue_count": 0,
        "staff_visible": 0,
        "average_wait_seconds": 0.0,
        "max_wait_seconds": 0.0,
        "groups": [],
        "tracks": [],
        "calibration_required": True,
    }


def get_operations_status():
    runtime = _runtime
    if runtime is not None:
        return runtime.get_operations_status()
    return {
        "state": "not_running",
        "occupancy_rate": 0.0,
        "staff_visible": 0,
        "average_wait_seconds": 0.0,
        "estimated_wait_seconds": 0.0,
        "queue_people": 0,
        "queue_groups": 0,
        "active_alerts": 0,
        "alerts": [],
        "recent_alerts": [],
        "tables": {},
        "cameras": {},
    }


def apply_table_action(table_id: str, action: str):
    if _runtime is None:
        raise RuntimeError("Processing runtime is not running")
    return _runtime.table_action(table_id, action)


def acknowledge_alert(alert_id: str):
    if _runtime is None:
        raise RuntimeError("Processing runtime is not running")
    return _runtime.acknowledge_alert(alert_id)


def apply_operations_settings(new_settings):
    if _runtime is not None:
        _runtime.update_operations_settings(new_settings)


def sample_calibration_hsv(camera_id: str, x: float, y: float):
    if _runtime is None:
        return None
    return _runtime.sample_raw_hsv(camera_id, x, y)


def get_calibration_frame(camera_id="dining"):
    runtime = _runtime
    if runtime is None:
        return None
    return runtime.get_raw_jpeg(camera_id)
