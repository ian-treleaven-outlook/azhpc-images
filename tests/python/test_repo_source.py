"""Unit tests for utils/repo_source.py (write_repo_source_info).

The git calls are mocked; one test runs the real bash function against a real
temporary git repository and checks the Python version records the same values.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils import repo_source

_OP = "install-test"
_REPO = Path(__file__).resolve().parents[2]


class WriteRepoSourceInfoTests(unittest.TestCase):
    def _run(self, *, commit=None, ref=None, git=None):
        """Call the function with git and the versions file mocked.

        `git` maps a git subcommand ("rev-parse" / "symbolic-ref") to the
        (rc, stdout) run_capture should return for it.
        """
        git = git or {}

        def fake_run_capture(command, op, **kwargs):
            return git[command[3]]

        with mock.patch.object(repo_source, "run_capture",
                               side_effect=fake_run_capture) as run, \
             mock.patch.object(repo_source, "write_component_version") as write, \
             mock.patch.object(repo_source, "log_info"):
            ok = repo_source.write_repo_source_info("NVLOOM", "/src/nvloom", _OP,
                                                    commit=commit, ref=ref)
        return ok, run, write

    def test_given_commit_and_ref_are_recorded_without_git(self):
        ok, run, write = self._run(commit="abc123", ref="v1.2")
        self.assertTrue(ok)
        run.assert_not_called()
        self.assertEqual(write.call_args_list, [
            mock.call("NVLOOM_REPO_COMMIT", "abc123"),
            mock.call("NVLOOM_REPO_REF", "v1.2")])

    def test_missing_values_are_read_from_the_checkout(self):
        ok, run, write = self._run(git={"rev-parse": (0, "deadbeef\n"),
                                        "symbolic-ref": (0, "main\n")})
        self.assertTrue(ok)
        self.assertEqual(run.call_args_list[0].args[0],
                         ["git", "-C", "/src/nvloom", "rev-parse", "HEAD"])
        self.assertEqual(run.call_args_list[1].args[0],
                         ["git", "-C", "/src/nvloom", "symbolic-ref", "--quiet",
                          "--short", "HEAD"])
        # A detached HEAD is normal, so that probe must not log an ERROR.
        self.assertIs(run.call_args_list[1].kwargs["failure_is_error"], False)
        self.assertEqual(write.call_args_list, [
            mock.call("NVLOOM_REPO_COMMIT", "deadbeef"),
            mock.call("NVLOOM_REPO_REF", "main")])

    def test_jq_null_counts_as_not_given(self):
        ok, _, write = self._run(commit="null", ref="null",
                                 git={"rev-parse": (0, "deadbeef\n"),
                                      "symbolic-ref": (0, "main\n")})
        self.assertTrue(ok)
        self.assertEqual(write.call_args_list[0], mock.call("NVLOOM_REPO_COMMIT", "deadbeef"))

    def test_detached_head_is_recorded_as_detached(self):
        ok, _, write = self._run(commit="abc123",
                                 git={"symbolic-ref": (1, "")})
        self.assertTrue(ok)
        self.assertEqual(write.call_args_list[1], mock.call("NVLOOM_REPO_REF", "DETACHED"))

    def test_git_failure_returns_false_and_records_nothing(self):
        ok, _, write = self._run(git={"rev-parse": (128, "")})
        self.assertFalse(ok)
        write.assert_not_called()


@unittest.skipUnless(shutil.which("bash") and shutil.which("git") and shutil.which("jq"),
                     "needs bash, git and jq")
class MatchesBashTests(unittest.TestCase):
    """Run the real bash write_repo_source_info and compare the result."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.repo = self.tmp / "repo"
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
               "-C", str(self.repo)]
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main", str(self.repo)], check=True)
        subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], check=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _bash(self, args: str) -> dict:
        # Source the real utilities.sh, then replace write_component_version
        # (which writes to /opt/azurehpc) with one writing to a temp file. The
        # real write_repo_source_info logic is what's being compared.
        versions = self.tmp / "bash.json"
        script = (f"source {_REPO}/utils/utilities.sh; "
                  f"echo '{{}}' > {versions}; "
                  "write_component_version(){ "
                  f"jq --arg k \"$1\" --arg v \"$2\" '. + {{($k): $v}}' {versions} > {versions}.new"
                  f" && mv {versions}.new {versions}; }}; "
                  f"write_repo_source_info {args}")
        subprocess.run(["bash", "-c", script], check=True, capture_output=True,
                       env={**os.environ, "UTILS_DIR": str(_REPO / "utils")})
        return json.loads(versions.read_text(encoding="utf-8"))

    def _python(self, **kwargs) -> dict:
        versions = self.tmp / "python.json"
        real_write = repo_source.write_component_version
        with mock.patch.object(repo_source, "write_component_version",
                               side_effect=lambda c, v: real_write(c, v, path=versions)), \
             mock.patch("utils.process.log_info"), mock.patch.object(repo_source, "log_info"):
            self.assertTrue(repo_source.write_repo_source_info(
                "X", str(self.repo), _OP, **kwargs))
        return json.loads(versions.read_text(encoding="utf-8"))

    def test_read_from_checkout_matches_bash(self):
        self.assertEqual(self._python(), self._bash(f"X {self.repo}"))

    def test_given_values_match_bash(self):
        self.assertEqual(self._python(commit="c0ffee", ref="v9"),
                         self._bash(f"X {self.repo} --commit c0ffee --ref v9"))

    def test_detached_head_matches_bash(self):
        subprocess.run(["git", "-C", str(self.repo), "checkout", "-q", "--detach"], check=True)
        self.assertEqual(self._python(), self._bash(f"X {self.repo}"))


if __name__ == "__main__":
    unittest.main()
