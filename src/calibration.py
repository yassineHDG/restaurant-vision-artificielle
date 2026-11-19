"""Offline table proposals for the calibration screen.

Usage:
    uv run python src/calibration.py propose-tables --video videos/example.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import PROJECT_ROOT, settings
from youtube_stream import resolve_capture_source

DINING_TABLE_CLASS = 60


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)


def propose_tables(
    video_path: str,
    samples: int = 12,
    confidence: float = 0.10,
    image_size: int = 960,
) -> dict:
    is_live = "://" in str(video_path)
    capture_source = resolve_capture_source(str(video_path))
    capture = cv2.VideoCapture(capture_source, cv2.CAP_FFMPEG if is_live else cv2.CAP_ANY)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    if is_live:
        positions = np.arange(max(1, samples), dtype=int)
    else:
        total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        positions = np.linspace(0, total - 1, num=max(1, samples), dtype=int)
    model = YOLO(settings.CALIBRATION_MODEL_PATH)
    best = []
    best_frame = 0
    try:
        for frame_index in positions:
            if not is_live:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                continue
            result = model.predict(
                frame,
                classes=[DINING_TABLE_CLASS],
                conf=confidence,
                imgsz=image_size,
                device=settings.DEVICE,
                quantize=16 if settings.HALF_PRECISION else None,
                verbose=False,
            )[0]
            proposals = []
            if result.masks is not None:
                for mask_tensor, detection_confidence in zip(
                    result.masks.data, result.boxes.conf
                ):
                    mask = mask_tensor.cpu().numpy()
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    contour = _largest_contour((mask > 0.5).astype(np.uint8))
                    if contour is None or len(contour) < 3:
                        continue
                    points = [
                        [round(float(x) / frame.shape[1], 6), round(float(y) / frame.shape[0], 6)]
                        for x, y in contour
                    ]
                    proposals.append(
                        {
                            "polygon": points,
                            "confidence": round(float(detection_confidence), 4),
                        }
                    )
            if len(proposals) > len(best):
                best = proposals
                best_frame = int(frame_index)
    finally:
        capture.release()
    return {
        "video": str(video_path),
        "live_source": is_live,
        "sampled_frames": [int(value) for value in positions],
        "selected_frame": best_frame,
        "confidence_threshold": confidence,
        "image_size": image_size,
        "tables": best,
        "requires_confirmation": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Restaurant Vision calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    propose = subparsers.add_parser("propose-tables")
    propose.add_argument("--video", default=settings.DINING_VIDEO_PATH)
    propose.add_argument("--samples", type=int, default=12)
    propose.add_argument("--conf", type=float, default=0.10)
    propose.add_argument("--imgsz", type=int, default=960)
    propose.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "config" / "calibration_draft.json"),
    )
    args = parser.parse_args()
    if args.command == "propose-tables":
        result = propose_tables(args.video, args.samples, args.conf, args.imgsz)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
