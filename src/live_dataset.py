"""Collect a small, reproducible frame sample from authorized live sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# YouTube HLS playlists may move between Google video hosts. Disabling HTTP
# connection reuse avoids FFmpeg trying to reuse one host's connection for a
# segment served by another host.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "http_persistent;0|reconnect;1|reconnect_streamed;1|reconnect_delay_max;2",
)

import cv2
import numpy as np
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from youtube_stream import stream_url_from_info, youtube_extractor_options


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "live_sources.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training" / "live_candidates"


def load_sources(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = payload.get("sources", [])
    if not sources:
        raise ValueError("No live sources are configured")
    ids = set()
    for source in sources:
        source_id = str(source.get("id", "")).strip()
        if not source_id or source_id in ids:
            raise ValueError("Every source requires a unique non-empty id")
        if not source.get("authorized"):
            raise ValueError(f"Source {source_id} is not marked as authorized")
        if not str(source.get("url", "")).startswith("https://"):
            raise ValueError(f"Source {source_id} requires an HTTPS URL")
        ids.add(source_id)
    return sources


def extractor_options(cookie_browser: str | None) -> dict:
    return youtube_extractor_options(cookie_browser)


def resolve_source(source: dict, cookie_browser: str | None = None) -> tuple[dict, str]:
    with YoutubeDL(extractor_options(cookie_browser)) as downloader:
        info = downloader.extract_info(source["url"], download=False)
    if info is None:
        raise RuntimeError(f"No metadata returned for {source['id']}")
    stream_url = stream_url_from_info(info)
    metadata = {
        "id": info.get("id", source["id"]),
        "title": info.get("title", source.get("label", source["id"])),
        "uploader": info.get("uploader"),
        "is_live": bool(info.get("is_live")),
        "live_status": info.get("live_status"),
        "webpage_url": info.get("webpage_url", source["url"]),
        "format_id": info.get("format_id"),
        "resolution": info.get("resolution"),
        "fps": info.get("fps"),
    }
    return metadata, stream_url


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contact_sheet(images: list[Path], target: Path, label: str) -> None:
    thumbnails = []
    tile_width, tile_height = 480, 270
    for index, path in enumerate(images, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        canvas = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
        scale = min(tile_width / frame.shape[1], tile_height / frame.shape[0])
        resized = cv2.resize(
            frame,
            (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))),
        )
        y = (tile_height - resized.shape[0]) // 2
        x = (tile_width - resized.shape[1]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.rectangle(canvas, (0, 0), (tile_width, 31), (10, 20, 20), -1)
        cv2.putText(
            canvas,
            f"{label} | frame {index}",
            (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 245, 243),
            1,
            cv2.LINE_AA,
        )
        thumbnails.append(canvas)
    if not thumbnails:
        return
    columns = min(3, len(thumbnails))
    rows = math.ceil(len(thumbnails) / columns)
    blank = np.zeros_like(thumbnails[0])
    thumbnails.extend([blank] * (rows * columns - len(thumbnails)))
    sheet = np.vstack(
        [np.hstack(thumbnails[row * columns : (row + 1) * columns]) for row in range(rows)]
    )
    cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def capture_source(
    source: dict,
    stream_url: str,
    output: Path,
    frame_count: int,
    interval_seconds: float,
    warmup_seconds: float,
) -> dict:
    source_dir = output / source["id"]
    source_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open source {source['id']}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    started = time.monotonic()
    next_sample = started + warmup_seconds
    deadline = next_sample + max(30.0, frame_count * interval_seconds * 3)
    last_read_at = started
    images = []
    try:
        while len(images) < frame_count and time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok:
                if time.monotonic() - last_read_at > 8.0:
                    capture.release()
                    capture = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    last_read_at = time.monotonic()
                    if not capture.isOpened():
                        time.sleep(0.5)
                        continue
                time.sleep(0.05)
                continue
            last_read_at = time.monotonic()
            now = time.monotonic()
            if now < next_sample:
                continue
            filename = f"{source['id']}-{len(images) + 1:04d}.jpg"
            path = source_dir / filename
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write {path}")
            images.append(path)
            next_sample = now + interval_seconds
    finally:
        capture.release()
    if not images:
        raise RuntimeError(
            f"Captured {len(images)}/{frame_count} frames from {source['id']} before timeout"
        )
    sheet = output / f"preview-{source['id']}.jpg"
    contact_sheet(images, sheet, source.get("label", source["id"]))
    return {
        "source_id": source["id"],
        "purpose": source.get("purpose"),
        "requested_frames": frame_count,
        "captured_frames": len(images),
        "complete": len(images) == frame_count,
        "frames": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
            }
            for path in images
        ],
        "preview": str(sheet.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }


def selected_sources(sources: list[dict], requested: list[str]) -> list[dict]:
    if not requested or requested == ["all"]:
        return sources
    wanted = set(requested)
    selected = [source for source in sources if source["id"] in wanted]
    missing = wanted - {source["id"] for source in selected}
    if missing:
        raise ValueError(f"Unknown source ids: {', '.join(sorted(missing))}")
    return selected


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Authorized live-camera dataset collector")
    root.add_argument("--config", default=str(DEFAULT_CONFIG))
    root.add_argument(
        "--cookies-from-browser",
        choices=("edge", "chrome", "firefox"),
        help="Use an existing local browser session only with the user's explicit approval.",
    )
    commands = root.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--source", action="append", default=[])
    capture = commands.add_parser("capture")
    capture.add_argument("--source", action="append", default=[])
    capture.add_argument("--frames", type=int, default=6)
    capture.add_argument("--interval", type=float, default=3.0)
    capture.add_argument("--warmup", type=float, default=2.0)
    capture.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        sources = selected_sources(load_sources(Path(args.config)), args.source)
        resolved = []
        for source in sources:
            metadata, stream_url = resolve_source(source, args.cookies_from_browser)
            resolved.append((source, metadata, stream_url))
        if args.command == "inspect":
            print(json.dumps([metadata for _, metadata, _ in resolved], indent=2))
            return 0
        if args.frames < 1 or args.frames > 500:
            raise ValueError("--frames must be between 1 and 500")
        if args.interval < 0.5:
            raise ValueError("--interval must be at least 0.5 seconds")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path(args.output).resolve() / timestamp
        output.mkdir(parents=True, exist_ok=False)
        captures = []
        errors = []
        for source, metadata, stream_url in resolved:
            try:
                result = capture_source(
                    source,
                    stream_url,
                    output,
                    args.frames,
                    args.interval,
                    args.warmup,
                )
                captures.append(dict(result, metadata=metadata))
            except RuntimeError as exc:
                errors.append({"source_id": source["id"], "error": str(exc)})
        manifest = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "frames_per_source": args.frames,
            "interval_seconds": args.interval,
            "captures": captures,
            "errors": errors,
        }
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(output),
                    "manifest": str(manifest_path),
                    "captured_sources": len(captures),
                    "failed_sources": len(errors),
                },
                indent=2,
            )
        )
        return 0 if captures else 2
    except (DownloadError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
