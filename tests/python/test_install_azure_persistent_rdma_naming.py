"""Unit tests for components/python/install_azure_persistent_rdma_naming.py.

Two kinds of test:

  * The generated files are compared with fixtures in
    tests/python/fixtures/install_azure_persistent_rdma_naming/. Those fixtures
    were written by running the real bash heredocs (RDMA_CORE_VERSION=56.1) in
    bash, so they are byte for byte what the bash step writes.
  * install() runs with the download, tar, build.sh and systemctl faked by
    FakeCommands below (patched where the toolkit runs them:
    utils.download.download_and_verify and utils.process.exec_program).
    /tmp, /usr/sbin and /etc/systemd/system are pointed at a temporary folder,
    so the tests never touch the real system or use the network.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_azure_persistent_rdma_naming as rdma
from utils import component_config, download, files, process
from utils.build_config import BuildConfig, build_plan
from utils.download import ChecksumError

_FIXTURES = Path(__file__).parent / "fixtures" / "install_azure_persistent_rdma_naming"
_VERSION = "59.0"
_TARBALL = "v59.0.tar.gz"
_URL = "https://github.com/linux-rdma/rdma-core/archive/refs/tags/v59.0.tar.gz"
_OP = "install-persistent-rdma-naming"


def _fixture(name: str) -> str:
    # newline="" keeps the text exactly as stored (no newline translation).
    with open(_FIXTURES / name, encoding="utf-8", newline="") as handle:
        return handle.read()


def _mode(path: str) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


# --------------------------------------------------------------------------- #
# Generated text, compared with what real bash writes.
# --------------------------------------------------------------------------- #
class GeneratedTextTests(unittest.TestCase):
    def test_naming_script_matches_bash_output(self):
        self.assertEqual(rdma.naming_script("56.1"),
                         _fixture("azure_persistent_rdma_naming.sh"))

    def test_only_the_version_changes_with_the_version(self):
        expected = _fixture("azure_persistent_rdma_naming.sh").replace(
            "rdma_rename=/usr/sbin/rdma_rename_56.1\n",
            "rdma_rename=/usr/sbin/rdma_rename_59.0\n")
        self.assertEqual(rdma.naming_script("59.0"), expected)

    def test_escaped_dollars_stay_literal(self):
        script = rdma.naming_script("59.0")
        # The bash source wrote these as \$...; they must reach the file as $.
        self.assertIn("mapfile -t all_devices < <(ibdev2netdev -v | sort -n | "
                      "awk '{print $2}')", script)
        self.assertIn('for old_device in "${all_devices[@]}"; do', script)
        self.assertIn("\t\tsuffix=${dev#$prefix}\n", script)
        self.assertIn("ib_index=$(next_index mlx5_ib)", script)
        self.assertIn(r"\([a-zA-Z]*\)$/\1/p'", script)
        self.assertNotIn("${RDMA_CORE_VERSION}", script)

    def test_naming_script_skips_devices_that_are_already_named(self):
        # #614: the script can run again (the timer reruns it), so devices
        # already called mlx5_ib<N>/mlx5_an<N> are left alone.
        script = rdma.naming_script("59.0")
        self.assertIn("\t\tmlx5_ib*|mlx5_an*) continue ;;\n", script)
        self.assertIn('echo "Unknown device type for $old_device."\n', script)

    def test_service_unit_matches_bash_output(self):
        self.assertEqual(rdma._SERVICE_UNIT,
                         _fixture("azure_persistent_rdma_naming.service"))

    def test_timer_unit_matches_bash_output(self):
        self.assertEqual(rdma._TIMER_UNIT, _fixture("azure_persistent_rdma_naming.timer"))

    def test_the_monitor_script_is_gone(self):
        # Upstream deleted it (#614); the port no longer runs it.
        components = Path(__file__).resolve().parents[2] / "components"
        self.assertFalse((components /
                          "install_azure_persistent_rdma_naming_monitor.sh").exists())
        self.assertFalse(hasattr(rdma, "_MONITOR_SCRIPT"))

    def test_rdma_rename_path(self):
        self.assertEqual(rdma.rdma_rename_path("59.0"), "/usr/sbin/rdma_rename_59.0")


# --------------------------------------------------------------------------- #
# install(): external effects faked, file system in a temporary folder.
# --------------------------------------------------------------------------- #
class FakeCommands:
    """Stands in for exec_program and download_and_verify (via fetch_verified).

    Records every (command, cwd). The fake `bash build.sh` creates
    build/bin/rdma_rename in its cwd, as the real build would. fail_at=N makes
    the Nth command (counting from 0) return exit code 1.
    """

    def __init__(self, fail_at=None):
        self.commands = []
        self.downloads = []
        self.fail_at = fail_at

    def download(self, url, sha256, dest_dir="."):
        self.downloads.append((url, sha256, dest_dir))
        return Path(dest_dir) / os.path.basename(url)

    def run(self, command, op, *, cwd=None, env=None, **_):
        self.commands.append((command, cwd))
        if len(self.commands) - 1 == self.fail_at:
            return 1
        if command == ["bash", "build.sh"]:
            binary = os.path.join(cwd, "build", "bin", "rdma_rename")
            os.makedirs(os.path.dirname(binary))
            with open(binary, "w", encoding="utf-8") as handle:
                handle.write("ELF\n")
            os.chmod(binary, 0o755)
        return 0


class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = self._tmp.name
        self.work = os.path.join(root, "tmp")
        self.sbin = os.path.join(root, "usr", "sbin")
        self.systemd = os.path.join(root, "etc", "systemd", "system")
        for folder in (self.work, self.sbin, self.systemd):
            os.makedirs(folder)
        self.src = os.path.join(self.work, "rdma-core")
        self.naming = os.path.join(self.sbin, "azure_persistent_rdma_naming.sh")
        self.service = os.path.join(self.systemd, "azure_persistent_rdma_naming.service")
        self.timer = os.path.join(self.systemd, "azure_persistent_rdma_naming.timer")
        self.components = os.path.join(root, "components")
        versions = {"rdma_core": {"common": {"version": _VERSION, "sha256": "sha"}}}
        self.env = {"COMPONENT_VERSIONS": json.dumps(versions),
                    "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "x86_64",
                    "COMPONENT_DIR": self.components}

    def tearDown(self):
        self._tmp.cleanup()

    def _install(self, env=None, fake=None, download_error=None, naming_path=None):
        """Run install() with everything external faked.

        Returns (rc, fake, logs), where logs holds the module's fake log_info,
        log_warn and log_error, plus the toolkit's ("files_info",
        "files_error", "download_error", "config_error"), so tests can check what was reported.
        """
        fake = fake or FakeCommands()
        download_fake = mock.Mock(side_effect=download_error or fake.download)
        logs = {}
        with mock.patch.object(rdma, "_WORK_DIR", self.work), \
             mock.patch.object(rdma, "_SBIN_DIR", self.sbin), \
             mock.patch.object(rdma, "_NAMING_SCRIPT", naming_path or self.naming), \
             mock.patch.object(rdma, "_SERVICE_FILE", self.service), \
             mock.patch.object(rdma, "_TIMER_FILE", self.timer), \
             mock.patch.object(process, "exec_program", side_effect=fake.run), \
             mock.patch.object(download, "download_and_verify", download_fake), \
             mock.patch.object(component_config, "log_error") as logs["config_error"], \
             mock.patch.object(download, "log_error") as logs["download_error"], \
             mock.patch.object(files, "log_info") as logs["files_info"], \
             mock.patch.object(files, "log_error") as logs["files_error"], \
             mock.patch.object(rdma, "log_info") as logs["info"], \
             mock.patch.object(rdma, "log_warn") as logs["warn"], \
             mock.patch.object(rdma, "log_error") as logs["error"]:
            rc = rdma.install(env or self.env)
        return rc, fake, logs

    def _expected_commands(self):
        return [
            (["tar", "-xf", _TARBALL, "-C", "rdma-core", "--strip-components=1"],
             self.work),
            (["bash", "build.sh"], self.src),
            (["systemctl", "daemon-reload"], None),
            (["systemctl", "enable", "azure_persistent_rdma_naming.service"], None),
            (["systemctl", "enable", "azure_persistent_rdma_naming.timer"], None),
            (["systemctl", "start", "azure_persistent_rdma_naming.service"], None),
            (["systemctl", "start", "azure_persistent_rdma_naming.timer"], None),
        ]

    @staticmethod
    def _messages(log):
        return [c.args[1] for c in log.call_args_list]

    # ---- success -----------------------------------------------------------
    def test_builds_installs_and_enables_the_service(self):
        rc, fake, logs = self._install()

        self.assertEqual(rc, 0)
        self.assertEqual(fake.downloads, [(_URL, "sha", self.work)])
        self.assertEqual(fake.commands, self._expected_commands())

        binary = os.path.join(self.sbin, "rdma_rename_59.0")
        self.assertEqual(Path(binary).read_text(encoding="utf-8"), "ELF\n")
        self.assertEqual(_mode(binary), 0o755)
        self.assertFalse(os.path.exists(self.src))           # rm -rf rdma-core

        with open(self.naming, encoding="utf-8", newline="") as handle:
            self.assertEqual(handle.read(), rdma.naming_script(_VERSION))
        self.assertEqual(_mode(self.naming), 0o755)
        with open(self.service, encoding="utf-8", newline="") as handle:
            self.assertEqual(handle.read(), _fixture("azure_persistent_rdma_naming.service"))
        with open(self.timer, encoding="utf-8", newline="") as handle:
            self.assertEqual(handle.read(), _fixture("azure_persistent_rdma_naming.timer"))
        logs["error"].assert_not_called()
        logs["warn"].assert_not_called()

    def test_existing_files_are_overwritten(self):
        for path in (self.naming, self.service, self.timer):
            Path(path).write_text("old content that is longer than the new one\n" * 50)
        rc, _, logs = self._install()

        self.assertEqual(rc, 0)
        self.assertEqual(Path(self.naming).read_text(), rdma.naming_script(_VERSION))
        self.assertEqual(Path(self.service).read_text(), rdma._SERVICE_UNIT)
        self.assertEqual(Path(self.timer).read_text(), rdma._TIMER_UNIT)
        wrote = [c for c in logs["files_info"].call_args_list
                 if c.kwargs["path"] == self.timer]
        self.assertEqual(wrote[0].kwargs["action"], "replaced")

    def test_existing_source_folder_is_reused_without_extracting(self):
        # bash: `mkdir rdma-core && tar ...` - a failing mkdir skips tar but
        # set -e does not stop the script, so it builds in the existing folder.
        os.makedirs(self.src)
        rc, fake, logs = self._install()

        self.assertEqual(rc, 0)
        self.assertEqual(fake.commands, self._expected_commands()[1:])
        logs["warn"].assert_called_once()
        self.assertIn("already exists", logs["warn"].call_args.args[1])

    # ---- logging -----------------------------------------------------------
    def test_logs_start_files_service_and_summary(self):
        _, _, logs = self._install()
        info = logs["info"]

        start = info.call_args_list[0]
        self.assertEqual(start.args, (_OP, "Installing persistent RDMA naming"))
        self.assertEqual(start.kwargs["rdma_core_version"], _VERSION)
        self.assertEqual(start.kwargs["source"], _URL)

        wrote = [(c.kwargs["path"], c.kwargs["action"])
                 for c in logs["files_info"].call_args_list]
        self.assertEqual(wrote, [(os.path.join(self.sbin, f"rdma_rename_{_VERSION}"), "created"),
                                 (self.naming, "created"), (self.service, "created"),
                                 (self.timer, "created")])

        enabled = [c for c in info.call_args_list if c.args[1] ==
                   "Enabled and started azure_persistent_rdma_naming.service and "
                   "azure_persistent_rdma_naming.timer"]
        self.assertEqual(enabled[0].kwargs, {"service": "azure_persistent_rdma_naming.service",
                                             "timer": "azure_persistent_rdma_naming.timer"})
        summary = info.call_args_list[-1]
        self.assertEqual(summary.args[1], "Persistent RDMA naming installed")
        self.assertEqual(summary.kwargs["rdma_core_version"], _VERSION)
        self.assertEqual(summary.kwargs["timer"], "azure_persistent_rdma_naming.timer")

    # ---- configuration -----------------------------------------------------
    def test_missing_config_returns_3_before_downloading(self):
        envs = [dict(self.env, COMPONENT_VERSIONS="{}"),
                dict(self.env, COMPONENT_VERSIONS=json.dumps(
                    {"rdma_core": {"common": {"version": _VERSION}}}))]   # no sha256
        for env in envs:
            with self.subTest(env=env):
                rc, fake, logs = self._install(env=env)
                self.assertEqual(rc, 3)
                self.assertEqual(fake.downloads, [])
                self.assertEqual(fake.commands, [])
                logs["config_error"].assert_called_once()

    def test_component_dir_is_not_needed(self):
        # Only the removed monitor installer needed it.
        env = {k: v for k, v in self.env.items() if k != "COMPONENT_DIR"}
        rc, fake, _ = self._install(env=env)
        self.assertEqual(rc, 0)
        self.assertEqual(fake.commands, self._expected_commands())

    # ---- failures: each returns 3 and stops --------------------------------
    def test_failed_download_returns_3(self):
        for error in (ChecksumError("bad"), OSError("404")):
            with self.subTest(error=error):
                rc, fake, logs = self._install(download_error=error)
                self.assertEqual(rc, 3)
                self.assertEqual(fake.commands, [])
                self.assertIn(_URL, logs["download_error"].call_args.args[1])

    def test_each_failing_command_returns_3_and_stops(self):
        for index in range(len(self._expected_commands())):
            # A fresh temporary folder for each case (setUp makes a new one).
            self.tearDown()
            self.setUp()
            expected = self._expected_commands()
            with self.subTest(failing=expected[index][0]):
                rc, fake, logs = self._install(fake=FakeCommands(fail_at=index))
                self.assertEqual(rc, 3)
                self.assertEqual(fake.commands, expected[:index + 1])
                # exec_program logs the failed command; nothing repeats it.
                logs["error"].assert_not_called()

    def test_missing_built_binary_returns_3(self):
        fake = FakeCommands()
        original = fake.run

        def run(command, op, **kwargs):
            if command == ["bash", "build.sh"]:          # a "build" that makes nothing
                fake.commands.append((command, kwargs.get("cwd")))
                return 0
            return original(command, op, **kwargs)
        fake.run = run
        rc, fake, logs = self._install(fake=fake)

        self.assertEqual(rc, 3)
        self.assertEqual(fake.commands, self._expected_commands()[:2])
        self.assertIn("could not copy", logs["files_error"].call_args.args[1])

    def test_unwritable_timer_returns_3_before_systemctl(self):
        os.makedirs(self.timer)                  # a folder can't be opened for writing
        rc, fake, logs = self._install()

        self.assertEqual(rc, 3)
        self.assertEqual(fake.commands, self._expected_commands()[:2])
        self.assertIn(f"could not write {self.timer}",
                      logs["files_error"].call_args.args[1])

    def test_unwritable_naming_script_returns_3_before_systemctl(self):
        # write_file creates missing folders, so make the "folder" a file.
        not_a_dir = os.path.join(self._tmp.name, "a-file")
        Path(not_a_dir).write_text("")
        rc, fake, logs = self._install(naming_path=os.path.join(not_a_dir, "naming.sh"))

        self.assertEqual(rc, 3)
        self.assertEqual(fake.commands, self._expected_commands()[:2])
        self.assertIn("could not write", logs["files_error"].call_args.args[1])


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == _OP]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, rdma.install)
        self.assertEqual(steps[0].script, "install_azure_persistent_rdma_naming.sh")


if __name__ == "__main__":
    unittest.main()
