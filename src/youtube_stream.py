"""Resolve approved public YouTube pages to short-lived capture URLs."""

from __future__ import annotations

from urllib.parse import urlparse

from yt_dlp import YoutubeDL


YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def is_youtube_url(source: str) -> bool:
    """Return whether *source* is an HTTP(S) YouTube page URL."""
    try:
        parsed = urlparse(str(source))
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and parsed.hostname in YOUTUBE_HOSTS


def youtube_extractor_options(cookie_browser: str | None = None) -> dict:
    """Options shared by live runtime and authorized dataset collection."""
    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "format": "best[height<=720][vcodec!=none]/bestvideo[height<=720]/best",
        "socket_timeout": 20,
        "extractor_args": {
            "youtube": {
                "player_client": ["mweb", "ios", "android_vr"],
            }
        },
    }
    if cookie_browser:
        options["cookiesfrombrowser"] = (cookie_browser, None, None, None)
    return options


def stream_url_from_info(info: dict) -> str:
    """Select a video-bearing URL from one yt-dlp metadata response."""
    stream_url = info.get("url")
    if stream_url:
        return str(stream_url)
    requested = info.get("requested_formats") or []
    video = next(
        (item for item in requested if item.get("vcodec") not in {None, "none"}),
        None,
    )
    if video and video.get("url"):
        return str(video["url"])
    raise RuntimeError("YouTube did not return a readable video stream")


def resolve_youtube_stream(source: str) -> str:
    """Resolve a public YouTube page without persisting credentials or URLs."""
    with YoutubeDL(youtube_extractor_options()) as downloader:
        info = downloader.extract_info(source, download=False)
    if not info:
        raise RuntimeError("YouTube returned no stream metadata")
    return stream_url_from_info(info)


def resolve_capture_source(source: str) -> str:
    """Return local/RTSP sources unchanged and resolve YouTube page URLs."""
    return resolve_youtube_stream(source) if is_youtube_url(source) else source
