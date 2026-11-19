"""GPU guard, bounded queues, and adaptive-resolution tests."""

import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from multi_camera_runtime import CameraReader, FramePacket, MultiCameraRuntime


class MultiCameraRuntimeTests(unittest.TestCase):
    def test_cuda_guard_never_silently_falls_back_to_cpu(self):
        runtime = MultiCameraRuntime()
        with patch("multi_camera_runtime.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is required"):
                runtime._validate_device()

    def test_camera_reader_keeps_only_the_latest_pending_frame(self):
        reader = CameraReader({"id": "dining"}, 10, threading.Event())
        first = FramePacket("dining", np.zeros((2, 2, 3)), 1, 2, 0.1, 0.2)
        second = FramePacket("dining", np.ones((2, 2, 3)), 2, 2, 0.2, 0.2)
        reader._publish(first)
        reader._publish(second)
        self.assertEqual(reader.queue.qsize(), 1)
        self.assertEqual(reader.queue.get_nowait().frame_index, 2)
        self.assertEqual(reader.dropped_frames, 1)

    def test_640_falls_back_to_512_when_total_target_is_missed(self):
        runtime = MultiCameraRuntime()
        runtime.started_at = time.monotonic() - 10
        runtime.readers = {"dining": SimpleNamespace(), "entrance": SimpleNamespace()}
        runtime.analyzed_frames = 100
        runtime._adapt_resolution_if_needed()
        self.assertEqual(runtime.current_imgsz, 512)
        decision = runtime.get_status()["performance"]["resolution_adjustment"]
        self.assertEqual((decision["from"], decision["to"]), (640, 512))

    def test_cuda_warmup_uses_the_configured_batch_before_readers_start(self):
        runtime = MultiCameraRuntime()
        runtime.active_cameras = [{"id": "dining"}, {"id": "entrance"}]
        runtime.model = SimpleNamespace(predict=lambda *args, **kwargs: None)
        with patch.object(runtime.model, "predict", wraps=runtime.model.predict) as predict:
            runtime._warmup_model()
        sources = predict.call_args.args[0]
        options = predict.call_args.kwargs
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].shape, (640, 640, 3))
        self.assertEqual(options["device"], "cuda:0")
        self.assertEqual(options["quantize"], 16)

    def test_camera_health_reports_reconnecting_then_stale(self):
        runtime = MultiCameraRuntime()
        reader = SimpleNamespace(
            last_frame_at=time.monotonic() - 20,
            reconnect_count=2,
            error=None,
            state="reconnecting",
            done=False,
        )
        runtime.readers = {"dining": reader}
        runtime._refresh_camera_health()
        self.assertEqual(runtime.status["cameras"]["dining"]["state"], "reconnecting")
        self.assertEqual(runtime.status["cameras"]["dining"]["reconnects"], 2)

        reader.state = "running"
        runtime._refresh_camera_health()
        self.assertEqual(runtime.status["cameras"]["dining"]["state"], "stale")


if __name__ == "__main__":
    unittest.main(verbosity=2)
