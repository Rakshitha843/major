"""
Phase 1 tests.  Run from the project root:

    python -m unittest discover -s tests -v
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from utils import config, env_check

FASTAPI_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None  # needed by FastAPI's TestClient
)


class ConfigTests(unittest.TestCase):
    def test_project_folders_exist(self):
        for folder in (config.DATA_DIR, config.MODELS_DIR, config.FRONTEND_DIR):
            self.assertTrue(folder.is_dir(), f"missing folder: {folder}")

    def test_server_is_local_only(self):
        # Privacy design: the dashboard must not be reachable from other computers.
        self.assertEqual(config.HOST, "127.0.0.1")

    def test_every_model_has_a_download_url(self):
        for path, url in config.MODEL_DOWNLOADS.items():
            self.assertTrue(url.startswith("https://"))
            self.assertEqual(path.suffix, ".task")


class EnvCheckTests(unittest.TestCase):
    def test_python_check_passes_on_this_interpreter(self):
        self.assertTrue(env_check.check_python()["ok"])

    def test_package_check_reports_every_required_package(self):
        results = env_check.check_packages()
        self.assertEqual(len(results), len(env_check.REQUIRED_PACKAGES))
        for result in results:
            self.assertIn("ok", result)

    def test_missing_model_is_reported_with_fix_instructions(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "does_not_exist.task"
            result = env_check.check_models([missing])[0]
            self.assertFalse(result["ok"])
            self.assertIn("download_models.py", result["detail"])

    def test_present_model_is_reported_ok(self):
        with tempfile.TemporaryDirectory() as folder:
            fake_model = Path(folder) / "fake.task"
            fake_model.write_bytes(b"x" * 100)
            self.assertTrue(env_check.check_models([fake_model])[0]["ok"])

    def test_bad_camera_index_fails_gracefully(self):
        # Must return a clear message, never raise an exception.
        result = env_check.check_camera(camera_index=99)
        self.assertFalse(result["ok"])
        self.assertIsInstance(result["detail"], str)


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_system_check_structure(self):
        body = self.client.get("/api/system-check").json()
        self.assertIn("all_ok", body)
        self.assertTrue(len(body["checks"]) > 0)

    def test_dashboard_page_is_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("no-store", response.headers["cache-control"])    # never show an old cached page


if __name__ == "__main__":
    unittest.main()
