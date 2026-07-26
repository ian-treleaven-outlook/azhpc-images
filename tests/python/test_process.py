"""Unit tests for utils/process.py.

Focus on run_capture, which returns both the exit code and the captured stdout
(used by callers that must parse a command's output, e.g. ofed_info).

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from utils import process
from utils.process import run_capture


# run_capture calls subprocess.run and reads .returncode/.stdout/.stderr off the result.
# This builds a fake result so we can drive those without launching a real process.
def _completed(returncode, stdout="", stderr=""):
    """Build a fake subprocess.CompletedProcess-like result."""
    result = mock.Mock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


# run_capture is the stdout-capturing sibling of exec_program (used for things like
# ofed_info where we need the output, not just the exit code). We patch subprocess.run
# so nothing actually executes.
class RunCaptureTests(unittest.TestCase):
    def test_returns_code_and_stdout_on_success(self):
        """A successful command yields (0, its stdout)."""
        with mock.patch.object(process.subprocess, "run",
                               return_value=_completed(0, stdout="Version: 4.1.5-1\n")):
            rc, out = run_capture(["apt-cache", "show", "openmpi"], "test")
        self.assertEqual(rc, 0)
        self.assertEqual(out, "Version: 4.1.5-1\n")

    def test_returns_nonzero_code_with_partial_stdout(self):
        """A failing command still returns its exit code and whatever stdout it produced."""
        with mock.patch.object(process.subprocess, "run",
                               return_value=_completed(1, stdout="partial")), \
             mock.patch.object(process, "log_error") as logged:
            rc, out = run_capture(["ofed_info"], "test")
        self.assertEqual(rc, 1)
        self.assertEqual(out, "partial")
        logged.assert_called_once()   # a nonzero exit is reported as an error

    def test_only_stdout_is_returned_not_stderr(self):
        """stderr is logged but must not be folded into the parsed output."""
        with mock.patch.object(process.subprocess, "run",
                               return_value=_completed(0, stdout="data", stderr="a warning")):
            rc, out = run_capture(["some", "cmd"], "test")
        self.assertEqual(rc, 0)
        self.assertEqual(out, "data")

    def test_returns_127_when_program_missing(self):
        """If the program can't be launched, return (127, "") like exec_program's 127."""
        with mock.patch.object(process.subprocess, "run",
                               side_effect=FileNotFoundError("no such file")), \
             mock.patch.object(process, "log_error") as logged:
            rc, out = run_capture(["does-not-exist"], "test")
        self.assertEqual(rc, 127)
        self.assertEqual(out, "")
        logged.assert_called_once()   # a launch failure is reported as an error


if __name__ == "__main__":
    unittest.main()
