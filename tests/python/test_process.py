"""Unit tests for utils/process.py.

Covers run_capture, which returns both the exit code and the captured stdout
(used by callers that must parse a command's output, e.g. ofed_info), and the
child-process teardown in exec_program.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

from utils import process
from utils.process import run_capture

_REPO_ROOT = Path(__file__).resolve().parents[2]


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


# A build step can run for half an hour (make, apt). If someone Ctrl-Cs the CLI, or
# the pipeline times out and SIGTERMs us, the child must not be left running - an
# orphaned apt keeps the dpkg lock and wedges every later step. Popen on its own
# does NOT do this: the child is simply reparented and keeps going. These tests
# spawn a real python that runs a real long-lived child, kill the parent, and check
# the grandchild actually went away.
def _marker_pids(marker):
    """PIDs of live processes whose command line contains `marker`."""
    out = subprocess.run(["ps", "-eo", "pid=,args="],
                         capture_output=True, text=True).stdout
    return [int(line.split(None, 1)[0])
            for line in out.splitlines() if marker in line]


def _wait_until(predicate, timeout=15.0):
    """Poll `predicate` until it's true or we run out of patience."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


@unittest.skipUnless(os.name == "posix", "process groups are POSIX-only")
class ChildTeardownTests(unittest.TestCase):
    def setUp(self):
        # Unique argv string so we can pick our child out of the process table
        # without matching anyone else's `sleep`.
        self.marker = f"azhpc-teardown-{uuid.uuid4().hex}"
        self._tmp = tempfile.TemporaryDirectory()
        self.runner = Path(self._tmp.name) / "runner.py"
        self.runner.write_text(
            "from utils.process import exec_program\n"
            # sleeps ~forever; the marker rides along in argv so `ps` can see it
            f"exec_program([{sys.executable!r}, '-c',"
            f" 'import time; time.sleep(300)', {self.marker!r}], 'teardown-test')\n",
            encoding="utf-8")
        self.parent = None

    def tearDown(self):
        # Belt and braces: never leave strays behind if an assertion blew up.
        if self.parent and self.parent.poll() is None:
            self.parent.kill()
            self.parent.wait(timeout=10)
        for pid in _marker_pids(self.marker):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._tmp.cleanup()

    def _start_parent(self):
        """Launch a python that's blocked inside exec_program, and wait for the child."""
        env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT),
               "LOG_LEVEL": "error", "LOG_FILE": os.devnull}
        self.parent = subprocess.Popen(
            [sys.executable, str(self.runner)],
            cwd=str(_REPO_ROOT), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.assertTrue(_wait_until(lambda: _marker_pids(self.marker)),
                        "child process never started")

    def _assert_child_reaped_after(self, sig):
        self._start_parent()
        os.kill(self.parent.pid, sig)
        self.parent.wait(timeout=15)
        self.assertTrue(
            _wait_until(lambda: not _marker_pids(self.marker)),
            f"child survived {sig.name} to the parent - it was orphaned, and on a "
            f"real build it would still be holding the package manager lock")

    def test_sigterm_to_parent_kills_the_child(self):
        # What a pipeline timeout or `kill <pid>` looks like.
        self._assert_child_reaped_after(signal.SIGTERM)

    def test_sigint_to_parent_kills_the_child(self):
        # What Ctrl-C looks like. Worth testing separately: exec_program puts the
        # child in its own process group, so it no longer gets the terminal's
        # SIGINT for free - the handler has to pass it on.
        self._assert_child_reaped_after(signal.SIGINT)

    def test_child_runs_in_its_own_process_group(self):
        # The teardown signals the group, not the pid, so `make -j` takes its
        # compilers with it. That's only safe if the group isn't ours.
        self._start_parent()
        (child_pid,) = _marker_pids(self.marker)
        self.assertNotEqual(os.getpgid(child_pid), os.getpgid(os.getpid()))


if __name__ == "__main__":
    unittest.main()
