"""Unit tests for components/python/install_health_checks.py.

Two kinds of test:

  * The pure helper sed_substitute() is tested directly with real lines from
    the upstream AzNHC conf files (v0.4.5). Its results were also compared
    against the real `sed -i` commands from the bash script. chmod_plus_x()
    runs on a real temporary file.
  * install() runs with the download, tar, dos2unix and the image-pull script
    faked by FakeCommands below. /opt/azurehpc/test is pointed at a temporary
    folder, so the tests never touch /opt, run docker or use the network.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_health_checks
from utils import download as download_module
from utils import process
from utils.build_config import BuildConfig, build_plan
from utils.download import ChecksumError

_VERSION = "0.4.5"
_URL = "https://github.com/Azure/azurehpc-health-checks/archive/refs/tags/v0.4.5.tar.gz"

# Real lines from the upstream conf files (before our edits).
_ND40 = ("# NVIDIA\n"
         " * || check_gpu_bw 10\n"
         " * || check_nccl_allreduce 110.0 1 8G $AZ_NHC_ROOT/topofiles/ndv2-topo.xml\n")
_H200 = (" * || check_gpu_bw 48 335\n"
         " * || check_nccl_allreduce 431.0 1 16G $AZ_NHC_ROOT/topofiles/ndv5-topo.xml\n")
_H100 = (" * || check_gpu_bw 48 335\n"
         " * || check_nccl_allreduce 460.0 1 16G $AZ_NHC_ROOT/topofiles/ndv5-topo.xml\n")
_CONFS = {"nd40rs_v2.conf": _ND40, "nd96isr_h200_v5.conf": _H200,
          "nd96isr_h100_v5.conf": _H100}


def _edit(index, text):
    """Apply the index-th NVIDIA conf edit to `text`."""
    _conf, pattern, replacement = install_health_checks._NVIDIA_CONF_EDITS[index]
    return install_health_checks.sed_substitute(text, pattern, replacement)


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
class SedSubstituteTests(unittest.TestCase):
    def test_nd40rs_v2_gpu_bandwidth_threshold(self):
        self.assertEqual(_edit(0, _ND40), _ND40.replace("check_gpu_bw 10", "check_gpu_bw 9"))

    def test_h200_and_h100_drop_the_topology_file(self):
        self.assertEqual(_edit(1, _H200), " * || check_gpu_bw 48 335\n"
                                          " * || check_nccl_allreduce 431.0 1 16G\n")
        self.assertEqual(_edit(2, _H100), " * || check_gpu_bw 48 335\n"
                                          " * || check_nccl_allreduce 460.0 1 16G\n")

    def test_edits_only_touch_their_own_numbers(self):
        # The 431.0 edit leaves a 460.0 line alone, and vice versa.
        self.assertEqual(_edit(1, _H100), _H100)
        self.assertEqual(_edit(2, _H200), _H200)

    def test_mirrors_sed_quirks(self):
        # "10" also matches the start of "100", exactly as sed does.
        self.assertEqual(_edit(0, "* || check_gpu_bw 100\n"), "* || check_gpu_bw 90\n")
        # Only the first match on each line is replaced (no `g` flag).
        self.assertEqual(_edit(0, "* || check_gpu_bw 10 * || check_gpu_bw 10\n"),
                         "* || check_gpu_bw 9 * || check_gpu_bw 10\n")
        # "." matches any character.
        self.assertEqual(
            _edit(1, "* || check_nccl_allreduce 431x0 1 16G $AZ_NHC_ROOT/topofiles/ndv5-topoXxml\n"),
            "* || check_nccl_allreduce 431.0 1 16G\n")
        # A missing final newline stays missing.
        self.assertEqual(_edit(0, "* || check_gpu_bw 10"), "* || check_gpu_bw 9")


class ChmodPlusXTests(unittest.TestCase):
    def test_adds_execute_bits_allowed_by_the_umask(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "script.sh")
            Path(path).touch()
            os.chmod(path, 0o644)
            old = os.umask(0o022)
            try:
                install_health_checks.chmod_plus_x(path)
            finally:
                os.umask(old)
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o755)


# --------------------------------------------------------------------------- #
# install(): external effects faked, file system in a temporary folder.
# --------------------------------------------------------------------------- #
class FakeCommands:
    """Stands in for utils.process.exec_program and utils.download.download_and_verify.

    Records every command. The fake `tar` creates the parts of the upstream
    repository that install() works on. fail_at=N makes the Nth command
    (counting from 0) return exit code 1.
    """

    def __init__(self, fail_at=None, layout=True):
        self.commands = []
        self.downloads = []
        self.fail_at = fail_at
        self.layout = layout

    def download(self, url, sha256, dest_dir="."):
        self.downloads.append((url, sha256, dest_dir))
        return Path(dest_dir) / os.path.basename(url)

    def run(self, command, op, *, cwd=None, env=None):
        self.commands.append((command, cwd))
        if len(self.commands) - 1 == self.fail_at:
            return 1
        if command[0] == "tar" and self.layout:
            repo = os.path.join(cwd, command[command.index("-C") + 1])
            files = {"triggerGHR/triggerGHR.sh": "upstream\n",
                     "triggerGHR/config/b.conf": "b\r\n",
                     "triggerGHR/config/a.conf": "a\r\n",
                     "dockerfile/pull-image-mcr.sh": "#!/bin/bash\n"}
            files.update({f"conf/{name}": text for name, text in _CONFS.items()})
            for name, text in files.items():
                path = os.path.join(repo, name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8", newline="") as handle:
                    handle.write(text)
                os.chmod(path, 0o644)
        return 0


class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = self._tmp.name
        self.dest = os.path.join(root, "opt", "azurehpc", "test")
        os.makedirs(self.dest)
        self.repo = os.path.join(self.dest, "azurehpc-health-checks")
        components = os.path.join(root, "components")
        os.makedirs(components)
        with open(os.path.join(components, "trigger_aznhc_GHR.sh"), "w",
                  encoding="utf-8") as handle:
            handle.write("ours\n")
        versions = {"aznhc": {"common": {"version": _VERSION, "sha256": "sha"}}}
        self.env = {"COMPONENT_VERSIONS": json.dumps(versions),
                    "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "x86_64",
                    "COMPONENT_DIR": components, "GPU": "NVIDIA"}

    def tearDown(self):
        self._tmp.cleanup()

    def _install(self, env=None, fake=None, download_error=None, write_error=None):
        """Run install() with everything external faked; return (rc, fake, record)."""
        fake = fake or FakeCommands()
        download = mock.Mock(side_effect=download_error or fake.download)
        with mock.patch.object(install_health_checks, "_DEST_TEST_DIR", self.dest), \
             mock.patch.object(process, "exec_program", side_effect=fake.run), \
             mock.patch.object(download_module, "download_and_verify", download), \
             mock.patch.object(install_health_checks, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(install_health_checks, "log_error"), \
             mock.patch.object(install_health_checks, "log_warn"):
            rc = install_health_checks.install(env or self.env)
        return rc, fake, record

    def _read(self, *parts):
        with open(os.path.join(self.repo, *parts), encoding="utf-8", newline="") as handle:
            return handle.read()

    def _expected_commands(self, image):
        configs = os.path.join(self.repo, "triggerGHR", "config")
        return [
            (["tar", "-xf", "v0.4.5.tar.gz", "-C", "azurehpc-health-checks",
              "--strip-components=1"], self.dest),
            (["dos2unix", f"{configs}/a.conf", f"{configs}/b.conf"], self.repo),
            (["./dockerfile/pull-image-mcr.sh", image], self.repo),
        ]

    def _assert_prepared(self):
        self.assertEqual(self._read("triggerGHR", "triggerGHR.sh"), "ours\n")
        for script in (("triggerGHR", "triggerGHR.sh"),
                       ("dockerfile", "pull-image-mcr.sh")):
            mode = os.stat(os.path.join(self.repo, *script)).st_mode
            self.assertTrue(mode & stat.S_IXUSR, script)

    # ---- success -----------------------------------------------------------
    def test_nvidia_edits_confs_and_pulls_cuda(self):
        rc, fake, record = self._install()

        self.assertEqual(rc, 0)
        self.assertEqual(fake.downloads, [(_URL, "sha", self.dest)])
        self.assertEqual(fake.commands, self._expected_commands("cuda"))
        self._assert_prepared()
        self.assertEqual(self._read("conf", "nd40rs_v2.conf"),
                         "# NVIDIA\n"
                         " * || check_gpu_bw 9\n"
                         " * || check_nccl_allreduce 110.0 1 8G $AZ_NHC_ROOT/topofiles/ndv2-topo.xml\n")
        self.assertEqual(self._read("conf", "nd96isr_h200_v5.conf"),
                         " * || check_gpu_bw 48 335\n"
                         " * || check_nccl_allreduce 431.0 1 16G\n")
        self.assertEqual(self._read("conf", "nd96isr_h100_v5.conf"),
                         " * || check_gpu_bw 48 335\n"
                         " * || check_nccl_allreduce 460.0 1 16G\n")
        record.assert_called_once_with("AZ_HEALTH_CHECKS", _VERSION)

    def test_amd_pulls_rocm_and_leaves_confs_alone(self):
        rc, fake, record = self._install(env=dict(self.env, GPU="AMD"))

        self.assertEqual(rc, 0)
        self.assertEqual(fake.commands, self._expected_commands("rocm"))
        self._assert_prepared()
        for name, text in _CONFS.items():
            self.assertEqual(self._read("conf", name), text)
        record.assert_called_once_with("AZ_HEALTH_CHECKS", _VERSION)

    def test_existing_folder_is_reused_without_extracting(self):
        # bash: `mkdir X && tar ...` - a failing mkdir skips tar but set -e
        # does not stop the script, so it carries on with the existing folder.
        self.assertEqual(self._install()[0], 0)
        rc, fake, record = self._install()

        self.assertEqual(rc, 0)
        self.assertEqual(fake.commands, self._expected_commands("cuda")[1:])
        record.assert_called_once_with("AZ_HEALTH_CHECKS", _VERSION)

    # ---- configuration -----------------------------------------------------
    def test_missing_config_or_gpu_returns_3_before_downloading(self):
        envs = [dict(self.env, COMPONENT_VERSIONS="{}"),
                dict(self.env, COMPONENT_VERSIONS=json.dumps(
                    {"aznhc": {"common": {"version": _VERSION}}})),       # no sha256
                {k: v for k, v in self.env.items() if k != "GPU"}]
        for env in envs:
            with self.subTest(env=env):
                rc, fake, record = self._install(env=env)
                self.assertEqual(rc, 3)
                self.assertEqual(fake.downloads, [])
                self.assertEqual(fake.commands, [])
                record.assert_not_called()

    # ---- failures: each returns 3 and stops --------------------------------
    def test_failed_download_returns_3(self):
        for error in (ChecksumError("bad"), OSError("404")):
            with self.subTest(error=error):
                rc, fake, record = self._install(download_error=error)
                self.assertEqual(rc, 3)
                self.assertEqual(fake.commands, [])
                self.assertFalse(os.path.exists(self.repo))
                record.assert_not_called()

    def test_each_failing_command_returns_3_and_stops(self):
        for gpu, image in (("NVIDIA", "cuda"), ("AMD", "rocm")):
            expected = self._expected_commands(image)
            for index in range(len(expected)):
                with self.subTest(gpu=gpu, failing=expected[index][0]):
                    shutil.rmtree(self.repo, ignore_errors=True)
                    rc, fake, record = self._install(env=dict(self.env, GPU=gpu),
                                                     fake=FakeCommands(fail_at=index))
                    self.assertEqual(rc, 3)
                    self.assertEqual(fake.commands, expected[:index + 1])
                    record.assert_not_called()

    def test_missing_trigger_script_returns_3(self):
        # bash's `rm` (no -f) fails when the file isn't there.
        rc, fake, record = self._install(fake=FakeCommands(layout=False))
        self.assertEqual(rc, 3)
        self.assertEqual(len(fake.commands), 1)          # dos2unix never ran
        record.assert_not_called()

    def test_missing_conf_file_returns_3_before_pulling(self):
        fake = FakeCommands()
        original = fake.run

        def run(command, op, **kwargs):
            rc = original(command, op, **kwargs)
            if command[0] == "tar":
                os.remove(os.path.join(self.repo, "conf", "nd96isr_h100_v5.conf"))
            return rc
        fake.run = run
        rc, fake, record = self._install(fake=fake)

        self.assertEqual(rc, 3)
        self.assertEqual(fake.commands, self._expected_commands("cuda")[:2])
        record.assert_not_called()

    def test_unwritable_versions_file_returns_3(self):
        rc, fake, _ = self._install(write_error=PermissionError("denied"))
        self.assertEqual(rc, 3)
        self.assertEqual(fake.commands, self._expected_commands("cuda"))


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_script_and_args(self):
        for vendor, gpu, arg in (("NVidia", "A100", "NVIDIA"), ("AMD", "MI300", "AMD")):
            with self.subTest(vendor=vendor):
                cfg = BuildConfig(vendor=vendor, gpu=gpu, os="Ubuntu24",
                                  fips=False, spec_path=None)
                steps = [s for s in build_plan(cfg) if s.op == "install-health-checks"]

                self.assertEqual(len(steps), 1)
                self.assertIs(steps[0].action, install_health_checks.install)
                self.assertEqual(steps[0].script, "install_health_checks.sh")
                # The bash fallback still receives the vendor as $1.
                self.assertEqual(steps[0].args, (arg,))


if __name__ == "__main__":
    unittest.main()
