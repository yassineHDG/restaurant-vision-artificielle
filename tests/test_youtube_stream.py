"""Public YouTube source resolution tests."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from youtube_stream import (
    is_youtube_url,
    resolve_capture_source,
    stream_url_from_info,
    youtube_extractor_options,
)


class YouTubeStreamTests(unittest.TestCase):
    def test_detects_youtube_pages_without_matching_other_sources(self):
        self.assertTrue(is_youtube_url("https://www.youtube.com/watch?v=abc"))
        self.assertTrue(is_youtube_url("https://youtu.be/abc"))
        self.assertFalse(is_youtube_url("rtsp://camera.local/live"))
        self.assertFalse(is_youtube_url("videos/restaurant.mp4"))

    def test_local_and_rtsp_sources_are_not_resolved(self):
        for source in ("videos/demo.mp4", "rtsp://camera.local/live"):
            with patch("youtube_stream.resolve_youtube_stream") as resolver:
                self.assertEqual(resolve_capture_source(source), source)
                resolver.assert_not_called()

    def test_selects_video_url_from_combined_format_metadata(self):
        info = {
            "requested_formats": [
                {"vcodec": "none", "url": "https://audio.example/stream"},
                {"vcodec": "avc1", "url": "https://video.example/stream"},
            ]
        }
        self.assertEqual(stream_url_from_info(info), "https://video.example/stream")

    def test_public_options_do_not_read_browser_cookies(self):
        self.assertNotIn("cookiesfrombrowser", youtube_extractor_options())
        clients = youtube_extractor_options()["extractor_args"]["youtube"]["player_client"]
        self.assertIn("mweb", clients)


if __name__ == "__main__":
    unittest.main(verbosity=2)
