"""Batched two-camera runtime with one shared CUDA pose model."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "http_persistent;0|reconnect;1|reconnect_streamed;1|reconnect_delay_max;2",
)

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from cameras import load_camera_config, resolve_camera_sources
from config import settings
from dashboard import RestaurantDashboard
from operations_config import OperationsSettings, load_operations_settings
from services.entrance_processor import EntranceProcessor
from services.operations_service import OperationsService
from services.video_processor import VideoProcessor
from utils.video_utils import encode_frame, resize_frame
from youtube_stream import resolve_capture_source


@dataclass
class FramePacket:
    camera_id: str
    frame: np.ndarray
    frame_index: int
    total_frames: int
    video_time: float
    duration: float


class CameraReader:
    """Read one source at source speed and publish only target-FPS frames."""

    def __init__(
        self,
        camera: dict,
        target_fps: float,
        stop_event: threading.Event,
        policy: OperationsSettings | None = None,
    ):
        self.camera = camera
        self.target_fps = target_fps
        self.stop_event = stop_event
        self.policy = policy or load_operations_settings()
        self.queue: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self.done = False
        self.error: str | None = None
        self.skipped_frames = 0
        self.dropped_frames = 0
        self.fps = 0.0
        self.total_frames = 0
        self.duration = 0.0
        self.reconnect_count = 0
        self.last_frame_at: float | None = None
        self.state = "starting"
        self.thread = threading.Thread(
            target=self._run,
            name=f"camera-reader-{camera['id']}",
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def _publish(self, packet: FramePacket):
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
            self.queue.put_nowait(packet)

    def _run(self):
        source = self.camera["source"]
        is_live = "://" in str(source)
        capture = None
        started = time.monotonic()
        frame_index = 0
        reconnect_attempt = 0
        try:
            while not self.stop_event.is_set():
                try:
                    capture_source = resolve_capture_source(str(source))
                except Exception as exc:
                    if not is_live or reconnect_attempt >= self.policy.camera_reconnect_attempts:
                        raise RuntimeError(
                            f"Unable to resolve live source for {self.camera['id']}: {exc}"
                        ) from exc
                    reconnect_attempt += 1
                    self.reconnect_count += 1
                    self.state = "reconnecting"
                    if self.stop_event.wait(self.policy.camera_reconnect_delay_seconds):
                        break
                    continue
                capture = cv2.VideoCapture(capture_source)
                if is_live:
                    # Keep the live reader close to the newest HLS segment.
                    # Some FFmpeg backends ignore this hint, but it is safe for
                    # backends that support it and prevents buffer growth.
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not capture.isOpened():
                    capture.release()
                    capture = None
                    if not is_live or reconnect_attempt >= self.policy.camera_reconnect_attempts:
                        raise RuntimeError(f"Unable to open {source}")
                    reconnect_attempt += 1
                    self.reconnect_count += 1
                    self.state = "reconnecting"
                    if self.stop_event.wait(self.policy.camera_reconnect_delay_seconds):
                        break
                    continue

                self.fps = float(capture.get(cv2.CAP_PROP_FPS))
                if not np.isfinite(self.fps) or self.fps <= 0:
                    self.fps = self.target_fps
                self.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.duration = self.total_frames / self.fps if self.total_frames else 0.0
                sample_interval = max(1, int(round(self.fps / self.target_fps)))
                live_publish_interval = 1.0 / self.target_fps
                next_live_publish = time.monotonic()
                self.state = "running"
                received_frame = False
                while not self.stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        break
                    received_frame = True
                    reconnect_attempt = 0
                    self.last_frame_at = time.monotonic()
                    video_time = (
                        self.last_frame_at - started
                        if is_live
                        else frame_index / self.fps
                    )
                    if frame_index % sample_interval:
                        self.skipped_frames += 1
                        frame_index += 1
                        continue
                    if not is_live:
                        deadline = started + video_time
                        remaining = deadline - time.monotonic()
                        if remaining > 0 and self.stop_event.wait(remaining):
                            break
                    else:
                        # HLS/OpenCV can return a burst of already-buffered
                        # frames. Pace published frames in wall-clock time so
                        # the dashboard receives a steady live cadence.
                        remaining = next_live_publish - time.monotonic()
                        if remaining > 0 and self.stop_event.wait(remaining):
                            break
                        next_live_publish = max(
                            next_live_publish + live_publish_interval,
                            time.monotonic(),
                        )
                    self._publish(
                        FramePacket(
                            camera_id=self.camera["id"],
                            frame=frame,
                            frame_index=frame_index,
                            total_frames=self.total_frames,
                            video_time=video_time,
                            duration=self.duration,
                        )
                    )
                    frame_index += 1
                capture.release()
                capture = None
                if self.stop_event.is_set() or not is_live:
                    self.state = "stopped" if self.stop_event.is_set() else "completed"
                    break
                reconnect_attempt += 1
                self.reconnect_count += 1
                self.state = "reconnecting"
                if reconnect_attempt > self.policy.camera_reconnect_attempts:
                    raise RuntimeError(f"Camera stream lost after {reconnect_attempt} attempts")
                if not received_frame and self.stop_event.wait(
                    self.policy.camera_reconnect_delay_seconds
                ):
                    break
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "error"
        finally:
            if capture is not None:
                capture.release()
            self.done = True


class FrameMailbox:
    def __init__(self, jpeg_quality=95):
        self.condition = threading.Condition()
        self.jpeg_quality = jpeg_quality
        self.chunk = None
        self.sequence = 0
        self.done = False

    def publish(self, frame):
        chunk = encode_frame(frame, self.jpeg_quality)
        if chunk is None:
            return
        with self.condition:
            self.chunk = chunk
            self.sequence += 1
            self.condition.notify_all()

    def finish(self):
        with self.condition:
            self.done = True
            self.condition.notify_all()

    def stream(self, stop_event):
        last_sequence = -1
        while not stop_event.is_set():
            with self.condition:
                if self.sequence == last_sequence:
                    self.condition.wait(timeout=1.0)
                chunk = self.chunk
                sequence = self.sequence
                done = self.done
            if chunk is not None and (sequence != last_sequence or done):
                last_sequence = sequence
                yield chunk


class MultiCameraRuntime:
    def __init__(self):
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.thread: threading.Thread | None = None
        self.operations_settings = load_operations_settings()
        self.camera_config = load_camera_config()
        self.cameras = resolve_camera_sources(self.camera_config)
        self.readers: dict[str, CameraReader] = {}
        self.mailboxes = {
            camera["id"]: FrameMailbox(self.operations_settings.stream_jpeg_quality)
            for camera in self.cameras
        }
        self.raw_frames: dict[str, np.ndarray] = {}
        self.model = None
        self.role_model = None
        self.dining_processor = None
        self.entrance_processor = None
        self.operations: OperationsService | None = None
        self.started_at = None
        self.analyzed_frames = 0
        self.inference_seconds = 0.0
        self.analysis_times = deque()
        self.current_imgsz = settings.IMGSZ
        self.resolution_adjusted = False
        self.dashboard = None
        self.active_cameras = []
        self.queue_stats = self._empty_queue()
        self.status = self._empty_status()

    @staticmethod
    def _empty_queue():
        return {
            "state": "not_configured",
            "confirmed_queue_count": 0,
            "confirmed_group_count": 0,
            "uncertain_queue_count": 0,
            "staff_visible": 0,
            "average_wait_seconds": 0.0,
            "max_wait_seconds": 0.0,
            "groups": [],
            "completed_groups": [],
            "tracks": [],
            "calibration_required": True,
        }

    def _empty_status(self):
        dining = next(camera for camera in self.cameras if camera["role"] == "dining")
        dining_source = str(dining.get("source") or "")
        source_label = (
            dining.get("label", "Salle")
            if "://" in dining_source
            else Path(dining_source or "non configurée").name
        )
        return {
            "state": "idle",
            "error": None,
            "processing_done": False,
            "frame_sequence": 0,
            "video_time_seconds": 0.0,
            "source": source_label,
            "seated_guests": 0,
            "occupied_tables": 0,
            "available_tables": 0,
            "total_tables": 0,
            "total_capacity": 0,
            "tables": [],
            "occupancy_rate": 0.0,
            "average_wait_seconds": 0.0,
            "estimated_wait_seconds": 0.0,
            "active_alerts": 0,
            "gpu": {
                "required": settings.REQUIRE_CUDA,
                "device": settings.DEVICE,
                "name": None,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "memory_allocated_mib": 0.0,
                "memory_reserved_mib": 0.0,
            },
            "performance": {
                "target_fps_per_camera": settings.TARGET_FPS,
                "inference_fps": 0.0,
                "average_fps": 0.0,
                "frames_analyzed": 0,
                "eta_seconds": None,
                "configured_input_size": settings.IMGSZ,
                "input_size": settings.IMGSZ,
                "resolution_adjustment": None,
            },
            "cameras": {
                camera["id"]: {
                    "label": camera.get("label", camera["id"]),
                    "role": camera["role"],
                    "active": camera["active"],
                    "calibration_required": camera.get("calibration_required", False),
                    "state": "starting" if camera["active"] else "not_configured",
                    "video_time_seconds": 0.0,
                    "duration_seconds": 0.0,
                    "progress_percent": 0.0,
                    "frames_analyzed": 0,
                    "frames_skipped": 0,
                    "frames_dropped": 0,
                    "lag_seconds": 0.0,
                    "reconnects": 0,
                    "last_frame_age_seconds": None,
                    "error": None,
                }
                for camera in self.cameras
            },
        }

    def _validate_device(self):
        wants_cuda = settings.DEVICE.lower().startswith("cuda")
        if wants_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required but unavailable. Run Test-GpuPython and Install-GpuTorch."
            )
        if settings.REQUIRE_CUDA and not wants_cuda:
            raise RuntimeError("REQUIRE_CUDA=true requires DEVICE=cuda:0")
        if wants_cuda:
            torch.cuda.set_device(0)
            with self.lock:
                self.status["gpu"]["name"] = torch.cuda.get_device_name(0)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self._validate_device()
        active = [camera for camera in self.cameras if camera["active"]]
        if not active:
            raise RuntimeError("No camera source is configured")
        self.model = YOLO(settings.MODEL_PATH)
        if settings.ROLE_MODEL_PATH:
            self.role_model = YOLO(settings.ROLE_MODEL_PATH)
        dining = next(camera for camera in self.cameras if camera["role"] == "dining")
        self.dining_processor = VideoProcessor(
            model=self.model,
            frame_rate=settings.TARGET_FPS,
            camera_config=dining,
        )
        self.operations = OperationsService(
            self.dining_processor.tables, source=dining.get("source", "")
        )
        if settings.SHOW_DASHBOARD:
            self.dashboard = RestaurantDashboard(
                tables=self.dining_processor.tables,
                width=settings.DASHBOARD_WIDTH,
            )
        entrance = next(camera for camera in self.cameras if camera["role"] == "entrance")
        self.entrance_processor = EntranceProcessor(
            entrance,
            frame_rate=settings.TARGET_FPS,
            operations_settings=self.operations_settings,
        )
        self.dining_processor.begin_run()
        self.active_cameras = active
        with self.lock:
            self.status["state"] = "warming_up"
        self.thread = threading.Thread(target=self._run, name="multi-camera-gpu", daemon=True)
        self.thread.start()

    def _take_batch(self):
        packets = []
        for camera in self.cameras:
            reader = self.readers.get(camera["id"])
            if reader is None:
                continue
            try:
                packets.append(reader.queue.get_nowait())
            except queue.Empty:
                continue
        return packets[: settings.INFERENCE_BATCH_SIZE]

    def _all_finished(self):
        return bool(self.readers) and all(reader.done and reader.queue.empty() for reader in self.readers.values())

    def _run(self):
        error = None
        last_dining_time = 0.0
        try:
            warmup_started = time.perf_counter()
            self._warmup_model()
            warmup_seconds = time.perf_counter() - warmup_started
            if self.stop_event.is_set():
                return
            for camera in self.active_cameras:
                reader = CameraReader(
                    camera,
                    settings.TARGET_FPS,
                    self.stop_event,
                    policy=self.operations_settings,
                )
                self.readers[camera["id"]] = reader
                reader.start()
            self.started_at = time.monotonic()
            with self.lock:
                self.status["state"] = "starting"
                self.status["performance"]["warmup_seconds"] = round(warmup_seconds, 3)
            while not self.stop_event.is_set():
                packets = self._take_batch()
                if not packets:
                    if self._all_finished():
                        break
                    self.stop_event.wait(0.005)
                    continue
                started = time.perf_counter()
                results = self.model.predict(
                    [packet.frame for packet in packets],
                    conf=settings.CONF,
                    iou=settings.IOU,
                    imgsz=self.current_imgsz,
                    device=settings.DEVICE,
                    quantize=16 if settings.HALF_PRECISION else None,
                    verbose=False,
                )
                role_results = (
                    self.role_model.predict(
                        [packet.frame for packet in packets],
                        conf=settings.ROLE_MODEL_CONF,
                        imgsz=self.current_imgsz,
                        device=settings.DEVICE,
                        quantize=16 if settings.HALF_PRECISION else None,
                        verbose=False,
                    )
                    if self.role_model is not None
                    else [None] * len(packets)
                )
                elapsed = time.perf_counter() - started
                self.inference_seconds += elapsed
                self.analyzed_frames += len(packets)
                for packet, result, role_result in zip(packets, results, role_results):
                    with self.lock:
                        self.raw_frames[packet.camera_id] = packet.frame.copy()
                    camera = next(item for item in self.cameras if item["id"] == packet.camera_id)
                    if camera["role"] == "dining":
                        frame, stats = self.dining_processor.process_result(
                            packet.frame, result, packet.video_time, role_result=role_result
                        )
                        last_dining_time = packet.video_time
                        frame = self._add_dashboard(frame, stats)
                        self._update_dining_status(packet, stats)
                    else:
                        frame, queue_stats = self.entrance_processor.process_result(
                            packet.frame, result, packet.video_time, role_result=role_result
                        )
                        self.queue_stats = dict(queue_stats, state="running")
                        if self.operations is not None:
                            self.operations.update_queue(queue_stats, packet.video_time)
                    if settings.RESIZE_VIDEO:
                        frame = resize_frame(frame, settings.VIDEO_WIDTH)
                    self.mailboxes[packet.camera_id].publish(frame)
                    self._update_camera_status(packet)
                now = time.monotonic()
                self.analysis_times.extend([now] * len(packets))
                self._update_performance()
                self._adapt_resolution_if_needed()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if self.dining_processor is not None:
                    self.dining_processor.end_run(last_dining_time)
                if self.operations is not None:
                    self.operations.end_run(last_dining_time)
            finally:
                state = "error" if error else ("stopped" if self.stop_event.is_set() else "completed")
                with self.lock:
                    self.status["state"] = state
                    self.status["error"] = error
                    self.status["processing_done"] = True
                    for camera_id, reader in self.readers.items():
                        self.status["cameras"][camera_id]["state"] = (
                            "error" if reader.error else state
                        )
                for mailbox in self.mailboxes.values():
                    mailbox.finish()
                self._write_benchmark()

    def _warmup_model(self):
        """Compile and initialize CUDA kernels before any video reader advances."""
        # A single image is enough to compile CUDA kernels. Warming up with a
        # full two-camera batch can take minutes on a 4 GB laptop GPU, while
        # production inference still batches frames from both cameras.
        batch_size = 1
        dummy = np.zeros((settings.IMGSZ, settings.IMGSZ, 3), dtype=np.uint8)
        self.model.predict(
            [dummy for _ in range(max(1, batch_size))],
            conf=settings.CONF,
            iou=settings.IOU,
            imgsz=settings.IMGSZ,
            device=settings.DEVICE,
            quantize=16 if settings.HALF_PRECISION else None,
            verbose=False,
        )
        if self.role_model is not None:
            self.role_model.predict(
                [dummy for _ in range(max(1, batch_size))],
                conf=settings.ROLE_MODEL_CONF,
                imgsz=settings.IMGSZ,
                device=settings.DEVICE,
                quantize=16 if settings.HALF_PRECISION else None,
                verbose=False,
            )

    def _add_dashboard(self, frame, stats):
        if not settings.SHOW_DASHBOARD:
            return frame
        self.dashboard.update(
            table_counts=stats["table_counts"],
            seated_guests=stats["seated_guests"],
            occupancy_seconds=stats["occupancy_seconds"],
        )
        return np.hstack([self.dashboard.render(frame.shape[0]), frame])

    def _update_dining_status(self, packet, stats):
        tables = []
        for table in self.dining_processor.tables:
            table_id = table["id"]
            guests = int(stats["table_counts"].get(table_id, 0))
            elapsed = stats["occupancy_seconds"].get(table_id)
            tables.append(
                {
                    "id": table_id,
                    "capacity": int(table["max"]),
                    "guests": guests,
                    "occupied": guests > 0,
                    "status": "occupied" if guests else "available",
                    "occupancy_seconds": round(float(elapsed), 3) if elapsed is not None else None,
                }
            )
        operational_tables = (
            self.operations.update_tables(
                tables,
                packet.video_time,
                staff_visible=int(stats.get("staff_visible", 0)),
            )
            if self.operations is not None
            else {}
        )
        for table in tables:
            operational = operational_tables.get(table["id"], {})
            table.update(operational)
            table["status"] = operational.get("state", table["status"])
        occupied = sum(table["occupied"] for table in tables)
        available = sum(table["status"] in {"free", "ready", "available"} for table in tables)
        live = self.operations.live_status() if self.operations is not None else {}
        with self.lock:
            self.status.update(
                state="running",
                frame_sequence=self.status["frame_sequence"] + 1,
                video_time_seconds=round(packet.video_time, 3),
                seated_guests=int(stats["seated_guests"]),
                occupied_tables=occupied,
                available_tables=available,
                total_tables=len(tables),
                total_capacity=sum(table["capacity"] for table in tables),
                tables=tables,
                staff_visible=int(stats.get("staff_visible", 0)),
                occupancy_rate=live.get("occupancy_rate", 0.0),
                average_wait_seconds=live.get("average_wait_seconds", 0.0),
                estimated_wait_seconds=live.get("estimated_wait_seconds", 0.0),
                active_alerts=live.get("active_alerts", 0),
            )

    def _update_camera_status(self, packet):
        reader = self.readers[packet.camera_id]
        progress = (
            packet.frame_index / max(1, packet.total_frames - 1) * 100
            if packet.total_frames
            else 0.0
        )
        with self.lock:
            camera = self.status["cameras"][packet.camera_id]
            wall_time = max(0.0, time.monotonic() - self.started_at)
            camera.update(
                state="running",
                video_time_seconds=round(packet.video_time, 3),
                duration_seconds=round(packet.duration, 3),
                progress_percent=round(min(100.0, progress), 2),
                frames_analyzed=camera["frames_analyzed"] + 1,
                frames_skipped=reader.skipped_frames,
                frames_dropped=reader.dropped_frames,
                lag_seconds=round(max(0.0, wall_time - packet.video_time), 3),
                reconnects=reader.reconnect_count,
                last_frame_age_seconds=round(
                    max(0.0, time.monotonic() - reader.last_frame_at), 2
                ) if reader.last_frame_at is not None else None,
                error=reader.error,
            )

    def _update_performance(self):
        elapsed = max(1e-6, time.monotonic() - self.started_at)
        average_fps = self.analyzed_frames / elapsed
        window_start = time.monotonic() - 5.0
        while self.analysis_times and self.analysis_times[0] < window_start:
            self.analysis_times.popleft()
        if len(self.analysis_times) >= 2:
            window_span = max(1e-6, self.analysis_times[-1] - self.analysis_times[0])
            effective_fps = (len(self.analysis_times) - 1) / window_span
        else:
            effective_fps = average_fps
        remaining = []
        for camera in self.status["cameras"].values():
            if camera["active"] and camera["duration_seconds"]:
                remaining.append(max(0.0, camera["duration_seconds"] - camera["video_time_seconds"]))
        with self.lock:
            self.status["performance"].update(
                inference_fps=round(effective_fps, 2),
                average_fps=round(average_fps, 2),
                model_fps=round(
                    self.analyzed_frames / max(1e-6, self.inference_seconds), 2
                ),
                frames_analyzed=self.analyzed_frames,
                eta_seconds=round(max(remaining), 1) if remaining else None,
            )
            if torch.cuda.is_available():
                self.status["gpu"].update(
                    memory_allocated_mib=round(torch.cuda.memory_allocated() / 1024**2, 1),
                    memory_reserved_mib=round(torch.cuda.memory_reserved() / 1024**2, 1),
                )

    def _adapt_resolution_if_needed(self):
        """Drop from 640 to 512 once when real-time throughput is insufficient."""
        if self.resolution_adjusted or self.current_imgsz <= 512:
            return
        elapsed = max(1e-6, time.monotonic() - self.started_at)
        active_count = max(1, len(self.readers))
        if elapsed < 5.0 or self.analyzed_frames < active_count * 20:
            return
        required_total_fps = settings.TARGET_FPS * active_count
        effective_fps = self.analyzed_frames / elapsed
        if effective_fps >= required_total_fps * 0.95:
            return
        previous = self.current_imgsz
        self.current_imgsz = 512
        self.resolution_adjusted = True
        reason = (
            f"throughput {effective_fps:.2f} FPS below "
            f"{required_total_fps:.2f} FPS target"
        )
        with self.lock:
            self.status["performance"].update(
                input_size=512,
                resolution_adjustment={
                    "from": previous,
                    "to": 512,
                    "reason": reason,
                },
            )

    def _write_benchmark(self):
        """Persist the last run's hardware and throughput decision."""
        benchmark_dir = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "benchmarks"
        )
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            payload = {
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "state": self.status["state"],
                "error": self.status["error"],
                "gpu": deepcopy(self.status["gpu"]),
                "performance": deepcopy(self.status["performance"]),
                "cameras": deepcopy(self.status["cameras"]),
            }
        serialized = json.dumps(payload, indent=2)
        targets = [benchmark_dir / "last_run.json"]
        if payload["state"] == "completed" and payload["error"] is None:
            targets.append(benchmark_dir / "latest.json")
        for benchmark_path in targets:
            temporary = benchmark_path.with_suffix(".tmp")
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(benchmark_path)

    def get_status(self):
        self._refresh_camera_health()
        with self.lock:
            return deepcopy(self.status)

    def get_queue_status(self):
        with self.lock:
            return deepcopy(self.queue_stats)

    def get_operations_status(self):
        live = self.operations.live_status() if self.operations is not None else {
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
            "settings": self.operations_settings.model_dump(),
        }
        with self.lock:
            return dict(
                live,
                state=self.status["state"],
                cameras=deepcopy(self.status["cameras"]),
            )

    def table_action(self, table_id: str, action: str):
        if self.operations is None:
            raise RuntimeError("Operations service is not running")
        return self.operations.table_action(table_id, action)

    def acknowledge_alert(self, alert_id: str):
        if self.operations is None:
            raise RuntimeError("Operations service is not running")
        return self.operations.acknowledge_alert(alert_id)

    def update_operations_settings(self, new_settings: OperationsSettings):
        self.operations_settings = new_settings
        if self.operations is not None:
            self.operations.update_settings(new_settings)
        if self.entrance_processor is not None:
            self.entrance_processor.operations_settings = new_settings

    def sample_raw_hsv(self, camera_id: str, x: float, y: float):
        with self.lock:
            frame = self.raw_frames.get(camera_id)
            if frame is None:
                return None
            height, width = frame.shape[:2]
            px = max(0, min(width - 1, int(round(float(x) * (width - 1)))))
            py = max(0, min(height - 1, int(round(float(y) * (height - 1)))))
            bgr = frame[py, px].reshape(1, 1, 3)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
        return {
            "x": px,
            "y": py,
            "bgr": [int(value) for value in bgr[0, 0]],
            "hsv": [int(value) for value in hsv],
        }

    def _refresh_camera_health(self):
        now = time.monotonic()
        with self.lock:
            for camera_id, reader in self.readers.items():
                status = self.status["cameras"][camera_id]
                last_frame_at = getattr(reader, "last_frame_at", None)
                age = (
                    max(0.0, now - last_frame_at)
                    if last_frame_at is not None
                    else None
                )
                status.update(
                    reconnects=getattr(reader, "reconnect_count", 0),
                    last_frame_age_seconds=round(age, 2) if age is not None else None,
                    error=getattr(reader, "error", None),
                )
                if getattr(reader, "error", None):
                    status["state"] = "error"
                elif getattr(reader, "state", None) == "reconnecting":
                    status["state"] = "reconnecting"
                elif (
                    age is not None
                    and age > self.operations_settings.stream_stale_seconds
                    and not getattr(reader, "done", False)
                ):
                    status["state"] = "stale"

    def get_raw_jpeg(self, camera_id: str):
        with self.lock:
            frame = self.raw_frames.get(camera_id)
            if frame is None:
                return None
            frame = frame.copy()
        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.operations_settings.stream_jpeg_quality],
        )
        return buffer.tobytes() if ok else None

    def stream(self, camera_id: str):
        mailbox = self.mailboxes.get(camera_id)
        if mailbox is None:
            raise KeyError(camera_id)
        return mailbox.stream(self.stop_event)

    def stop(self):
        self.stop_event.set()
        for mailbox in self.mailboxes.values():
            with mailbox.condition:
                mailbox.condition.notify_all()
        for reader in self.readers.values():
            if reader.thread.is_alive():
                reader.thread.join(timeout=3)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
