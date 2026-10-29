"""Unit tests for utils/logger.py.

Covers the parts build telemetry depends on: optional structured fields, the
multi-line detail block, the shared JSON fields on every record, writing to
LOG_FILE, and redaction of secrets in both messages and fields.

The logger keeps its settings in module-level variables (LOG_FORMAT, LOG_FILE,
...), so each test patches them with mock.patch.object and captures what would
have been printed.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils import logger


class LoggerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log_file = Path(self._tmp.name) / "azhpc.log"

    def tearDown(self):
        self._tmp.cleanup()

    def _emit(self, log_format, call):
        """Run `call()` with the given format, returning (stdout, log file text)."""
        out = io.StringIO()
        with mock.patch.object(logger, "LOG_FORMAT", log_format), \
             mock.patch.object(logger, "LOG_LEVEL", "debug"), \
             mock.patch.object(logger, "LOG_FILE", str(self.log_file)), \
             mock.patch.object(logger, "RUN_ID", "run-123"), \
             contextlib.redirect_stdout(out):
            call()
        written = self.log_file.read_text(encoding="utf-8") if self.log_file.exists() else ""
        return out.getvalue(), written


class JsonFormatTests(LoggerTestCase):
    def test_fields_are_nested_under_data(self):
        out, _ = self._emit("json", lambda: logger.log_info(
            "install-pmix", "Installing PMIx", version="4.2.9", attempt=2))
        record = json.loads(out)
        self.assertEqual(record["msg"], "Installing PMIx")
        self.assertEqual(record["data"], {"version": "4.2.9", "attempt": 2})
        self.assertEqual(record["run_id"], "run-123")

    def test_no_fields_means_no_data_key(self):
        out, _ = self._emit("json", lambda: logger.log_info("op", "plain"))
        self.assertNotIn("data", json.loads(out))

    def test_non_json_values_become_strings(self):
        out, _ = self._emit("json", lambda: logger.log_info(
            "op", "wrote file", path=Path("/opt/azurehpc/x")))
        self.assertEqual(json.loads(out)["data"]["path"], "/opt/azurehpc/x")

    def test_error_detail_has_the_shared_fields_and_reaches_the_file(self):
        out, written = self._emit("json", lambda: logger.log_error_detail(
            "install-x", "Command failed", "line 1\nline 2", exit_code=2))
        record = json.loads(out)
        self.assertEqual(record["level"], "error")
        self.assertEqual(record["error_detail"], "line 1\nline 2")
        self.assertEqual(record["run_id"], "run-123")
        self.assertEqual(record["data"], {"exit_code": 2})
        self.assertEqual(json.loads(written), record)


class TextFormatTests(LoggerTestCase):
    def test_fields_are_appended_as_key_value(self):
        out, written = self._emit("text", lambda: logger.log_info(
            "install-pmix", "Installing PMIx", version="4.2.9", distro="ubuntu24.04"))
        self.assertIn("Installing PMIx  [version=4.2.9 distro=ubuntu24.04]", out)
        self.assertIn("[version=4.2.9 distro=ubuntu24.04]", written)

    def test_error_detail_lines_go_to_screen_and_file(self):
        out, written = self._emit("text", lambda: logger.log_error_detail(
            "install-x", "Command failed", "E: one\nE: two"))
        for text in (out, written):
            self.assertIn("Command failed", text)
            self.assertIn("    | E: one", text)
            self.assertIn("    | E: two", text)


class RedactionTests(LoggerTestCase):
    def test_secrets_are_redacted_in_message_fields_and_detail(self):
        out, _ = self._emit("json", lambda: logger.log_error_detail(
            "op", "GET https://x/?sig=abc123", "Authorization: Bearer tok.en",
            url="https://x/?token=zzz"))
        record = json.loads(out)
        self.assertNotIn("abc123", out)
        self.assertNotIn("tok.en", out)
        self.assertNotIn("zzz", out)
        self.assertIn("[REDACTED]", record["msg"])
        self.assertIn("[REDACTED]", record["data"]["url"])


class LevelTests(LoggerTestCase):
    def test_debug_is_dropped_at_info_level(self):
        out = io.StringIO()
        with mock.patch.object(logger, "LOG_FORMAT", "text"), \
             mock.patch.object(logger, "LOG_LEVEL", "info"), \
             mock.patch.object(logger, "LOG_FILE", ""), \
             contextlib.redirect_stdout(out):
            logger.log_debug("op", "hidden", key="value")
        self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
