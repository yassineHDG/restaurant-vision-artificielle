"""Focused tests for the FastAPI app and operations dashboard."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from main import FRONTEND_DIR, app


class AppRoutesTests(unittest.TestCase):
    def setUp(self):
        self.paths = {route.path for route in app.routes if hasattr(route, "path")}

    def test_page_stream_and_status_routes_exist(self):
        self.assertIn("/", self.paths)
        self.assertIn("/calibration", self.paths)
        self.assertIn("/stream", self.paths)
        self.assertIn("/stream/{camera_id}", self.paths)
        self.assertIn("/api/status", self.paths)
        self.assertIn("/api/queue", self.paths)
        self.assertIn("/api/calibration", self.paths)
        self.assertIn("/api/calibration/table-proposals", self.paths)
        self.assertIn("/api/calibration/frame/{camera_id}", self.paths)
        self.assertIn("/api/operations", self.paths)
        self.assertIn("/api/operations/history", self.paths)
        self.assertIn("/api/operations/settings", self.paths)
        self.assertIn("/api/operations/export.csv", self.paths)
        self.assertIn("/api/tables/{table_id}/actions/{action}", self.paths)
        self.assertIn("/api/alerts/{alert_id}/acknowledge", self.paths)

    def test_frontend_files_served_by_the_page(self):
        index = FRONTEND_DIR / "index.html"
        self.assertTrue(index.is_file())

        html = index.read_text(encoding="utf-8")
        self.assertIn("/static/style.css", html)
        self.assertIn('rel="icon" type="image/svg+xml" href="/static/favicon.svg"', html)
        self.assertIn('src="/static/favicon.svg"', html)
        self.assertIn("/static/app.js", html)
        self.assertTrue((FRONTEND_DIR / "favicon.svg").is_file())
        self.assertTrue((FRONTEND_DIR / "style.css").is_file())
        self.assertTrue((FRONTEND_DIR / "app.js").is_file())
        self.assertTrue((FRONTEND_DIR / "calibration.html").is_file())
        self.assertTrue((FRONTEND_DIR / "calibration.css").is_file())
        self.assertTrue((FRONTEND_DIR / "calibration.js").is_file())
        calibration_html = (FRONTEND_DIR / "calibration.html").read_text(encoding="utf-8")
        self.assertIn('id="delete-table"', calibration_html)
        self.assertIn('id="table-zone-controls"', calibration_html)
        self.assertIn('id="expand-table-zone"', calibration_html)

    def test_page_has_live_operations_ui(self):
        html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

        self.assertIn("Restaurant Vision", html)
        self.assertIn("Supervision du restaurant", html)
        self.assertIn('id="table-list"', html)
        self.assertIn("File d’attente", html)
        self.assertIn("Prévisions d’affluence", html)
        self.assertIn('id="video-fullscreen"', html)
        self.assertIn('id="alerts-list"', html)
        self.assertIn('id="history-chart"', html)
        self.assertIn('id="settings-dialog"', html)
        self.assertIn("Attente moyenne / groupe", html)

    def test_frontend_requests_stream_and_live_status(self):
        script = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")

        self.assertIn("/stream", script)
        self.assertIn("/api/status", script)
        self.assertIn("/api/queue", script)
        self.assertIn("/stream/entrance", script)
        self.assertIn("fetch(", script)
        self.assertIn("Flux vidéo indisponible — nouvelle tentative…", script)
        self.assertIn("requestFullscreen", script)
        self.assertIn('event.key === "Escape"', script)
        self.assertIn("/api/operations/history", script)
        self.assertIn("/api/operations/settings", script)
        self.assertIn("/acknowledge", script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
