"""Unit tests for components/python/install_rocm.py.

  * Pure helpers (repo text, dnf command lists) are tested directly.
  * install() runs with every external effect mocked: config lookup, download,
    exec_program (apt/dnf/udevadm/systemctl...), the kernel version and the
    versions file. The files it writes go to a temporary directory, by pointing
    the module's path constants there with mock.patch.object. Nothing touches
    /etc, /opt or the network.

Run from the repo root:
    python3 -m unittest tests.python.test_install_rocm -v
"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_rocm
from utils.build_config import BuildConfig, build_plan

_KERNEL = "6.8.0-1015-azure"
_DEB_URL = ("https://repo.radeon.com/amdgpu-install/6.4.4/ubuntu/noble/"
            "amdgpu-install_6.4.60404-1_all.deb")
_UBUNTU_CFG = {"version": "6.4.4", "url": _DEB_URL, "sha256": "abc123"}
_AZL_CFG = {"version": "7.0.3"}      # versions.json has no url for Azure Linux

_UBUNTU_COMMANDS = [
    ["apt", "install", "-y", "./amdgpu-install_6.4.60404-1_all.deb"],
    ["amdgpu-install", "-y", "--usecase=graphics,rocm"],
    ["apt", "install", "-y", "rocm-bandwidth-test"],
    ["udevadm", "control", "--reload-rules"],
    ["sudo", "udevadm", "trigger"],
    ["update-initramfs", "-c", "-k", _KERNEL],
    ["systemctl", "start", "rocmstartup"],
    ["systemctl", "enable", "rocmstartup"],
]

_AZL_COMMANDS = [
    ["dnf", "install", "-y", "azurelinux-repos-amd"],
    ["dnf", "-y", "install", f"kernel-drivers-gpu-{_KERNEL}"],
    ["dnf", "-y", "install", "amdgpu", "amdgpu-firmware", "amdgpu-headers"],
    ["dnf", "repolist", "--refresh"],
    ["dnf", "install", "-y", "rocm-dev", "rocm-validation-suite", "rocm-bandwidth-test"],
    ["dnf", "install", "-y", "rocm-smi-lib", "rocm-core", "rocm-device-libs",
     "rocm-llvm", "rocm-validation-suite"],
    ["dnf", "install", "-y", "rocm-bandwidth-test"],
    ["udevadm", "control", "--reload-rules"],
    ["sudo", "udevadm", "trigger"],
    ["systemctl", "start", "rocmstartup"],
    ["systemctl", "enable", "rocmstartup"],
]

# Exactly what the three `echo -e ... | tee [-a]` lines produce (checked
# against real bash when the port was written).
_EXPECTED_UNIT = ("[Unit]\n\nDescription=Runs /etc/init.d/initamdgpu.sh\n\n\n"
                  "[Service]\n\nExecStart=/etc/init.d/initamdgpu.sh\n\n\n"
                  "[Install]\n\nWantedBy=multi-user.target\n")


class PureHelperTests(unittest.TestCase):
    def test_repo_text_expands_the_version(self):
        self.assertEqual(install_rocm.amd_rocm_repo_text("7.0.3"), (
            "[amd_rocm]\n"
            'name="AMD ROCM packages repo for Azure Linux 3.0"\n'
            "baseurl=https://repo.radeon.com/rocm/azurelinux3/7.0.3/main/\n"
            "enabled=1\nrepo_gpgcheck=0\ngpgcheck=0\nsslverify=0\n"))

    def test_azure_linux_commands_use_the_kernel(self):
        before, after = install_rocm.azure_linux_commands(_KERNEL)
        self.assertEqual(before + after, _AZL_COMMANDS[:7])

    def test_service_unit_text(self):
        self.assertEqual(install_rocm._SERVICE_UNIT, _EXPECTED_UNIT)

    def test_init_script_is_literal(self):
        # Quoted heredoc: shell syntax kept, not expanded.
        text = install_rocm._INIT_SCRIPT_TEXT
        self.assertTrue(text.startswith("#!/bin/sh\nat_count=0\n"))
        self.assertIn("at_count=$(($at_count + 1))", text)
        self.assertIn("sudo modprobe amdgpu ip_block_mask=0x7f", text)
        self.assertTrue(text.endswith("done\n\nexit 0\n"))


class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.paths = {
            "_WORK_DIR": str(root / "work"),
            "_AMD_ROCM_REPO_FILE": str(root / "amd_rocm.repo"),
            "_UDEV_RULES_FILE": str(root / "99-amdgpu-permissive.rules"),
            "_BLACKLIST_FILE": str(root / "blacklist.conf"),
            "_INIT_SCRIPT": str(root / "initamdgpu.sh"),
            "_SERVICE_FILE": str(root / "rocmstartup.service"),
        }
        for name, value in self.paths.items():
            patcher = mock.patch.object(install_rocm, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _download(self, url, sha256, dest_dir="."):
        # Like the real helper: the file lands in dest_dir under the URL's name.
        path = Path(dest_dir) / Path(url).name
        path.write_bytes(b"deb")
        self.downloaded = path
        return path

    def _install(self, distro, cfg, *, fail_on=None, rc=1, kernel=_KERNEL,
                 write_error=None):
        """Run install() with mocks; fail_on = a command (list) that returns rc."""
        def fake_exec(command, op, *, cwd=None, env=None):
            return rc if command == fail_on else 0

        with mock.patch.object(install_rocm, "config_for", return_value=cfg), \
             mock.patch.object(install_rocm, "download_and_verify",
                               side_effect=self._download) as download, \
             mock.patch.object(install_rocm, "exec_program",
                               side_effect=fake_exec) as run, \
             mock.patch.object(install_rocm.platform, "release", return_value=kernel), \
             mock.patch.object(install_rocm, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(install_rocm, "log_info") as info, \
             mock.patch.object(install_rocm, "log_warn") as warn, \
             mock.patch.object(install_rocm, "log_error") as error:
            result = install_rocm.install({"DISTRIBUTION": distro})
        self.run_mock, self.download, self.record = run, download, record
        self.info, self.warn, self.error = info, warn, error
        return result

    def _commands(self):
        return [c.args[0] for c in self.run_mock.call_args_list]

    def _read(self, key):
        return Path(self.paths[key]).read_text(encoding="utf-8")

    def _info_messages(self):
        return [c.args[1] for c in self.info.call_args_list]

    # ---- success paths ------------------------------------------------------
    def test_ubuntu_success(self):
        self.assertEqual(self._install("ubuntu24.04", _UBUNTU_CFG), 0)
        self.assertEqual(self._commands(), _UBUNTU_COMMANDS)
        self.download.assert_called_once_with(_DEB_URL, "abc123",
                                              dest_dir=self.paths["_WORK_DIR"])
        # apt install of ./<deb> runs inside the work dir.
        self.assertEqual(self.run_mock.call_args_list[0].kwargs["cwd"],
                         self.paths["_WORK_DIR"])
        self.assertFalse(os.path.exists(self.paths["_WORK_DIR"]))   # deb + dir removed
        self.record.assert_called_once_with("ROCM", "6.4.4")

        self.assertEqual(self._read("_UDEV_RULES_FILE"),
                         'KERNEL=="kfd", MODE="0666"\n'
                         'SUBSYSTEM=="drm", KERNEL=="renderD*", MODE="0666"\n')
        self.assertEqual(self._read("_BLACKLIST_FILE"), "blacklist amdgpu\n")
        self.assertEqual(self._read("_INIT_SCRIPT"), install_rocm._INIT_SCRIPT_TEXT)
        self.assertEqual(self._read("_SERVICE_FILE"), _EXPECTED_UNIT)
        self.assertTrue(os.stat(self.paths["_INIT_SCRIPT"]).st_mode & 0o111)
        self.assertEqual(stat.S_IMODE(os.stat(self.paths["_SERVICE_FILE"]).st_mode), 0o644)
        self.assertFalse(os.path.exists(self.paths["_AMD_ROCM_REPO_FILE"]))
        self.error.assert_not_called()

    def test_ubuntu2204_runs_the_same_amdgpu_install(self):
        self.assertEqual(self._install("ubuntu22.04", _UBUNTU_CFG), 0)
        self.assertEqual(self._commands(), _UBUNTU_COMMANDS)

    def test_azure_linux_success(self):
        self.assertEqual(self._install("azurelinux3.0", _AZL_CFG), 0)
        self.assertEqual(self._commands(), _AZL_COMMANDS)
        self.download.assert_not_called()
        self.assertEqual(self._read("_AMD_ROCM_REPO_FILE"),
                         install_rocm.amd_rocm_repo_text("7.0.3"))
        # No blacklist / initramfs on Azure Linux.
        self.assertFalse(os.path.exists(self.paths["_BLACKLIST_FILE"]))
        self.assertEqual(self._read("_SERVICE_FILE"), _EXPECTED_UNIT)
        self.record.assert_called_once_with("ROCM", "7.0.3")

    def test_appends_and_overwrites_like_bash(self):
        for key in ("_AMD_ROCM_REPO_FILE", "_UDEV_RULES_FILE", "_SERVICE_FILE"):
            Path(self.paths[key]).write_text("old\n", encoding="utf-8")
        self.assertEqual(self._install("azurelinux3.0", _AZL_CFG), 0)
        # >> appends to the repo file; > replaces the others.
        self.assertEqual(self._read("_AMD_ROCM_REPO_FILE"),
                         "old\n" + install_rocm.amd_rocm_repo_text("7.0.3"))
        self.assertFalse(self._read("_UDEV_RULES_FILE").startswith("old"))
        self.assertEqual(self._read("_SERVICE_FILE"), _EXPECTED_UNIT)
        self.warn.assert_called_once()          # rerun warning for the repo append

        Path(self.paths["_BLACKLIST_FILE"]).write_text("blacklist foo\n", encoding="utf-8")
        self.assertEqual(self._install("ubuntu24.04", _UBUNTU_CFG), 0)
        self.assertEqual(self._read("_BLACKLIST_FILE"), "blacklist foo\nblacklist amdgpu\n")

    def test_udev_reload_failure_skips_trigger_and_continues(self):
        # bash: `udevadm control --reload-rules && sudo udevadm trigger` under
        # set -e - a failure before && only skips the trigger.
        rc = self._install("ubuntu24.04", _UBUNTU_CFG,
                           fail_on=["udevadm", "control", "--reload-rules"])
        self.assertEqual(rc, 0)
        self.assertNotIn(["sudo", "udevadm", "trigger"], self._commands())
        self.assertIn(["systemctl", "enable", "rocmstartup"], self._commands())
        self.warn.assert_called_once()

    # ---- logging ------------------------------------------------------------
    def test_key_log_lines(self):
        self._install("ubuntu24.04", _UBUNTU_CFG)
        start = self.info.call_args_list[0]
        self.assertIn("Installing ROCm 6.4.4 on ubuntu24.04", start.args[1])
        self.assertEqual(start.kwargs["source"], _DEB_URL)
        self.assertEqual(start.kwargs["kernel"], _KERNEL)
        messages = self._info_messages()
        self.assertIn("Ubuntu: installing ROCm with amdgpu-install", messages)
        self.assertIn(f"Wrote {self.paths['_UDEV_RULES_FILE']}", messages)
        self.assertIn(f"Appended to {self.paths['_BLACKLIST_FILE']}", messages)
        self.assertIn("Recorded ROCM=6.4.4", messages)
        self.assertIn("Started and enabled rocmstartup.service", messages)
        self.assertEqual(messages[-1], "ROCm 6.4.4 installed and configured")

    # ---- failures -----------------------------------------------------------
    def test_unsupported_distro_fails_before_any_change(self):
        rc = self._install("rocky9.8", _UBUNTU_CFG)
        self.assertEqual(rc, 3)
        self.run_mock.assert_not_called()
        self.record.assert_not_called()
        self.assertFalse(os.path.exists(self.paths["_UDEV_RULES_FILE"]))
        self.assertIn("not supported", self.error.call_args.args[1])

    def test_missing_config_returns_3(self):
        for cfg in (None, {}, {"url": _DEB_URL}):
            with self.subTest(cfg=cfg):
                self.assertEqual(self._install("ubuntu24.04", cfg), 3)
                self.run_mock.assert_not_called()
                self.error.assert_called_once()

    def test_ubuntu_missing_url_or_sha_returns_3(self):
        self.assertEqual(self._install("ubuntu24.04", {"version": "6.4.4"}), 3)
        self.download.assert_not_called()

    def test_empty_kernel_returns_3(self):
        self.assertEqual(self._install("azurelinux3.0", _AZL_CFG, kernel=""), 3)
        self.run_mock.assert_not_called()

    def test_download_failure_returns_3(self):
        with mock.patch.object(self, "_download", side_effect=RuntimeError("checksum")):
            self.assertEqual(self._install("ubuntu24.04", _UBUNTU_CFG), 3)
        self.run_mock.assert_not_called()
        self.assertFalse(os.path.exists(self.paths["_WORK_DIR"]))

    def test_each_failing_command_stops_the_step(self):
        cases = [("ubuntu24.04", _UBUNTU_CFG, _UBUNTU_COMMANDS),
                 ("azurelinux3.0", _AZL_CFG, _AZL_COMMANDS)]
        for distro, cfg, commands in cases:
            for index, command in enumerate(commands):
                if command[:2] == ["udevadm", "control"]:
                    continue        # tolerated, see the dedicated test
                with self.subTest(distro=distro, command=command):
                    self.assertEqual(self._install(distro, cfg, fail_on=command), 3)
                    self.assertEqual(self._commands(), commands[:index + 1])
                    self.error.assert_called()
                    # ROCM is recorded only after the udev trigger, like bash.
                    trigger = commands.index(["sudo", "udevadm", "trigger"])
                    self.assertEqual(self.record.called, index > trigger)

    def test_file_write_error_returns_3(self):
        os.makedirs(self.paths["_UDEV_RULES_FILE"])      # a directory can't be opened
        self.assertEqual(self._install("ubuntu24.04", _UBUNTU_CFG), 3)
        self.assertNotIn(["udevadm", "control", "--reload-rules"], self._commands())
        self.assertIn("file operation failed", self.error.call_args.args[1])

    def test_unwritable_versions_file_returns_3(self):
        rc = self._install("azurelinux3.0", _AZL_CFG,
                           write_error=PermissionError("denied"))
        self.assertEqual(rc, 3)
        self.assertNotIn(["systemctl", "start", "rocmstartup"], self._commands())


class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="AMD", gpu="MI300X", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "install-rocm"]
        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, install_rocm.install)
        self.assertEqual(steps[0].script, "install_rocm.sh")


if __name__ == "__main__":
    unittest.main()
