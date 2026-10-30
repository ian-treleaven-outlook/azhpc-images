"""Unit tests for components/python/trivy_scan.py.

  * Pure helpers (release JSON -> version, file names, the skip options and the
    exact scan command lines) are tested directly.
  * retry_command() is tested with utils.process.exec_program and time.sleep
    mocked, so the attempt and sleep counts can be checked without waiting.
  * install() is tested with run_capture (the GitHub API call) and
    utils.process.exec_program (wget, tar, the scans) mocked. The fake wget
    writes a small real .tar.gz and the fake tar unpacks it, so the
    chmod/cleanup code genuinely runs, inside a temporary directory that
    stands in for the work dir and /opt/azurehpc. Nothing touches the network.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import io
import json
import os
import stat
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import trivy_scan
from utils import files, process
from utils.build_config import BuildConfig, build_plan

_OP = "trivy-scan"
_DB = ("mcr.microsoft.com/mirror/ghcr/aquasecurity/trivy-db:2,"
       "ghcr.io/aquasecurity/trivy-db:2,public.ecr.aws/aquasecurity/trivy-db")
_RELEASE = json.dumps({"name": "v0.58.1", "tag_name": "v0.58.1"})


def _expected_scans(work_dir, report_dir):
    """The scan argument lists, written out by hand from the bash (checked with
    real bash: SKIP_FILES expands to nothing, each SKIP_DIRS entry to two words)."""
    common = ["--db-repository", _DB,
              "--skip-dirs", "/var/lib/waagent", "--skip-dirs", "/snap",
              "--skip-dirs", "/mnt", "--skip-dirs", f"{work_dir}/trivy",
              "--ignore-unfixed"]
    return [
        ["./trivy/trivy", "--scanners", "vuln", "rootfs", "-f", "json", *common,
         "--list-all-pkgs=false", "-o", f"{report_dir}/trivy-report-rootfs.json", "/"],
        ["./trivy/trivy", "--scanners", "vuln", "rootfs", "-f", "cyclonedx", *common,
         "-o", f"{report_dir}/trivy-cyclonedx-rootfs.json", "/"],
    ]


def _fake_tarball(path):
    """Write a .tar.gz shaped like the Trivy release: a non-executable trivy
    binary (so the chmod is visible) plus a LICENSE file."""
    with tarfile.open(path, "w:gz") as archive:
        for name, mode in (("trivy", 0o644), ("LICENSE", 0o644)):
            data = b"content"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = mode
            archive.addfile(info, io.BytesIO(data))


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
class VersionFromReleaseTests(unittest.TestCase):
    def test_drops_the_leading_v(self):
        self.assertEqual(trivy_scan.version_from_release(_RELEASE), "0.58.1")

    def test_drops_the_first_character_whatever_it_is(self):
        # Mirrors ${TRIVY_VERSION:1}, which doesn't check for a "v".
        self.assertEqual(trivy_scan.version_from_release('{"name": "0.58.1"}'), ".58.1")

    def test_invalid_json_returns_none(self):
        self.assertIsNone(trivy_scan.version_from_release(""))
        self.assertIsNone(trivy_scan.version_from_release("<html>rate limited</html>"))

    def test_missing_or_non_string_name_returns_none(self):
        self.assertIsNone(trivy_scan.version_from_release('{"tag_name": "v1"}'))
        self.assertIsNone(trivy_scan.version_from_release('{"name": null}'))
        self.assertIsNone(trivy_scan.version_from_release('["v1"]'))

    def test_name_with_nothing_after_the_first_character_returns_none(self):
        self.assertIsNone(trivy_scan.version_from_release('{"name": "v"}'))
        self.assertIsNone(trivy_scan.version_from_release('{"name": ""}'))


class NameAndArgsTests(unittest.TestCase):
    def test_trivy_arch(self):
        self.assertEqual(trivy_scan.trivy_arch("aarch64"), "Linux-ARM64")
        self.assertEqual(trivy_scan.trivy_arch("x86_64"), "Linux-64bit")
        self.assertEqual(trivy_scan.trivy_arch(""), "Linux-64bit")

    def test_tarball_name(self):
        self.assertEqual(trivy_scan.tarball_name("0.58.1", "x86_64"),
                         "trivy_0.58.1_Linux-64bit.tar.gz")
        self.assertEqual(trivy_scan.tarball_name("0.58.1", "aarch64"),
                         "trivy_0.58.1_Linux-ARM64.tar.gz")

    def test_skip_args_matches_bash_word_splitting(self):
        self.assertEqual(trivy_scan.skip_args(["/snap", "/w/trivy"]),
                         ["--skip-dirs", "/snap", "--skip-dirs", "/w/trivy"])
        self.assertEqual(trivy_scan.skip_args([]), [])

    def test_scan_commands_are_exact(self):
        self.assertEqual(trivy_scan.scan_commands("/w/trivy", "/opt/azurehpc"),
                         _expected_scans("/w", "/opt/azurehpc"))


# --------------------------------------------------------------------------- #
# retry_command(): exec_program and time.sleep mocked.
# --------------------------------------------------------------------------- #
class RetryCommandTests(unittest.TestCase):
    def _retry(self, results):
        env = {"A": "1"}
        with mock.patch.object(process, "exec_program", side_effect=results) as ex, \
             mock.patch.object(trivy_scan.time, "sleep") as sleep, \
             mock.patch.object(trivy_scan, "log_error"):
            ok = trivy_scan.retry_command(["./trivy/trivy", "x"], env, cwd="/w")
        return ok, ex, sleep, env

    def test_success_first_time_does_not_sleep(self):
        ok, ex, sleep, env = self._retry([0])

        self.assertTrue(ok)
        ex.assert_called_once_with(["timeout", "600", "./trivy/trivy", "x"], _OP,
                                   cwd="/w", env=env)
        sleep.assert_not_called()

    def test_success_on_third_attempt_sleeps_30s_between_failures(self):
        ok, ex, sleep, _ = self._retry([1, 124, 0])

        self.assertTrue(ok)
        self.assertEqual(ex.call_count, 3)
        self.assertEqual(sleep.call_args_list, [mock.call(30), mock.call(30)])

    def test_success_on_the_tenth_attempt(self):
        ok, ex, sleep, _ = self._retry([1] * 9 + [0])

        self.assertTrue(ok)
        self.assertEqual(ex.call_count, 10)
        self.assertEqual(sleep.call_count, 9)

    def test_gives_up_after_ten_attempts_without_a_final_sleep(self):
        ok, ex, sleep, _ = self._retry([1] * 10)

        self.assertFalse(ok)
        self.assertEqual(ex.call_count, 10)
        self.assertEqual(sleep.call_count, 9)


# --------------------------------------------------------------------------- #
# install(): network and commands mocked, paths redirected to a temp dir.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.work = os.path.join(self.root, "work")
        self.reports = os.path.join(self.root, "opt", "azurehpc")
        for name, value in (("_WORK_DIR", self.work), ("_REPORT_DIR", self.reports)):
            patcher = mock.patch.object(trivy_scan, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _install(self, env, *, release=(0, _RELEASE), scan_results=None,
                 wget_rc=0, tarball_writer=_fake_tarball):
        """Run install(env) with run_capture, exec_program and time.sleep mocked.

        The fake `tar -xf <file> -C trivy` unpacks the tarball with tarfile
        and exits 2 if it isn't one, like tar.

        scan_results: return codes for successive scan attempts (default all 0).
        Returns (rc, commands, exec_mock, capture_mock, sleep_mock, binary_modes);
        binary_modes holds the trivy binary's mode at each scan attempt.
        """
        commands = []
        binary_modes = []
        scans = iter(scan_results or [])

        def fake_exec(command, op, *, cwd=None, env=None):
            commands.append(command)
            if command[0] == "wget":
                if wget_rc == 0:
                    tarball_writer(os.path.join(cwd, command[-1]))
                return wget_rc
            if command[0] == "tar":
                try:
                    with tarfile.open(os.path.join(cwd, command[2])) as archive:
                        archive.extractall(os.path.join(cwd, command[4]), filter="data")
                except tarfile.TarError:
                    return 2
                return 0
            binary_modes.append(os.stat(os.path.join(cwd, "trivy", "trivy")).st_mode)
            return next(scans, 0)

        with mock.patch.object(trivy_scan, "run_capture", return_value=release) as cap, \
             mock.patch.object(process, "exec_program", side_effect=fake_exec) as ex, \
             mock.patch.object(files, "log_error"), \
             mock.patch.object(trivy_scan.time, "sleep") as sleep, \
             mock.patch.object(trivy_scan, "log_error"):
            rc = trivy_scan.install(env)
        return rc, commands, ex, cap, sleep, binary_modes

    def test_success_downloads_unpacks_scans_and_cleans_up(self):
        env = {"ARCHITECTURE": "x86_64"}
        rc, commands, ex, cap, sleep, modes = self._install(env)

        self.assertEqual(rc, 0)
        cap.assert_called_once_with([
            "curl", "-sfL", "-H", "Accept: application/vnd.github+json",
            "-H", "X-GitHub-Api-Version: 2022-11-28",
            "https://api.github.com/repos/aquasecurity/trivy/releases/latest",
        ], _OP, env=env)
        scans = _expected_scans(self.work, self.reports)
        self.assertEqual(commands, [
            ["wget", "https://github.com/aquasecurity/trivy/releases/download/"
                     "v0.58.1/trivy_0.58.1_Linux-64bit.tar.gz",
             "-O", "trivy_0.58.1_Linux-64bit.tar.gz"],
            ["tar", "-xf", "trivy_0.58.1_Linux-64bit.tar.gz", "-C", "trivy"],
            ["timeout", "600", *scans[0]],
            ["timeout", "600", *scans[1]],
        ])
        # Every command runs from the work dir (so ./trivy/trivy resolves).
        for call in ex.call_args_list:
            self.assertEqual(call.kwargs, {"cwd": self.work, "env": env})
        # chmod a+x happened before the scans: 0o644 became 0o755.
        self.assertEqual([stat.S_IMODE(m) for m in modes], [0o755, 0o755])
        self.assertTrue(os.path.isdir(self.reports))       # mkdir -p
        self.assertFalse(os.path.exists(self.work))        # rm -rf
        sleep.assert_not_called()

    def test_aarch64_downloads_the_arm64_tarball(self):
        rc, commands, *_ = self._install({"ARCHITECTURE": "aarch64"})

        self.assertEqual(rc, 0)
        self.assertEqual(commands[0][1],
                         "https://github.com/aquasecurity/trivy/releases/download/"
                         "v0.58.1/trivy_0.58.1_Linux-ARM64.tar.gz")

    def test_flaky_scan_is_retried_then_succeeds(self):
        rc, commands, _, _, sleep, _ = self._install({"ARCHITECTURE": "x86_64"},
                                                     scan_results=[124, 1, 0, 0])

        self.assertEqual(rc, 0)
        self.assertEqual(len(commands), 2 + 3 + 1)      # wget, tar, 3 json, 1 cyclonedx
        self.assertEqual(sleep.call_count, 2)

    def test_github_query_failure_returns_3_before_downloading(self):
        rc, commands, *_ = self._install({}, release=(22, ""))

        self.assertEqual(rc, 3)
        self.assertEqual(commands, [])
        self.assertFalse(os.path.exists(self.reports))

    def test_bad_release_json_returns_3(self):
        rc, commands, *_ = self._install({}, release=(0, '{"message": "rate limit"}'))

        self.assertEqual(rc, 3)
        self.assertEqual(commands, [])

    def test_wget_failure_returns_3_and_cleans_up(self):
        rc, commands, *_ = self._install({}, wget_rc=8)

        self.assertEqual(rc, 3)
        self.assertEqual([c[0] for c in commands], ["wget"])
        self.assertFalse(os.path.exists(self.work))

    def test_corrupt_tarball_returns_3(self):
        def garbage(path):
            Path(path).write_bytes(b"not a tarball")

        rc, commands, *_ = self._install({}, tarball_writer=garbage)

        self.assertEqual(rc, 3)
        self.assertEqual([c[0] for c in commands], ["wget", "tar"])
        self.assertFalse(os.path.exists(self.work))

    def test_first_scan_exhausting_retries_returns_3_and_skips_the_second(self):
        rc, commands, _, _, sleep, _ = self._install({}, scan_results=[1] * 10)

        self.assertEqual(rc, 3)
        self.assertEqual(len(commands), 2 + 10)
        # Only the JSON scan was attempted.
        self.assertTrue(all("json" in c for c in commands[2:]))
        self.assertEqual(sleep.call_count, 9)
        self.assertFalse(os.path.exists(self.work))

    def test_second_scan_exhausting_retries_returns_3(self):
        rc, commands, *_ = self._install({}, scan_results=[0] + [1] * 10)

        self.assertEqual(rc, 3)
        self.assertEqual(len(commands), 2 + 1 + 10)

    def test_unwritable_report_dir_returns_3(self):
        # A regular file where a parent directory should be: makedirs raises.
        blocker = os.path.join(self.root, "opt")
        Path(blocker).write_text("x", encoding="utf-8")
        rc, commands, *_ = self._install({})

        self.assertEqual(rc, 3)
        self.assertEqual(commands, [])


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "trivy-scan"]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, trivy_scan.install)
        self.assertEqual(steps[0].script, "trivy_scan.sh")


if __name__ == "__main__":
    unittest.main()
