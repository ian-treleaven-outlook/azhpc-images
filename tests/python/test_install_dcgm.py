"""Unit tests for components/python/install_dcgm.py.

  * Pure helpers are tested directly with realistic inputs. dcgmi_version()
    was also compared against the real `awk '{print $3}'` + bash word
    splitting in WSL, and the C probe source was compiled and run there.
  * A parity test checks the C probe source against the heredoc in
    components/install_dcgm.sh, byte for byte.
  * install() is tested with exec_program, run_capture, config_for and
    write_component_version mocked. The probe's temporary directory is created
    inside a test-owned temporary directory, and must be gone afterwards.

Run from the repo root:
    python3 -m unittest tests.python.test_install_dcgm -v
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_dcgm
from utils.build_config import BuildConfig, build_plan

_OP = "install-dcgm"
_REPO = Path(__file__).resolve().parents[2]

_UBUNTU_DCGM = "1:4.6.0-1"
_AZL_DCGM = "4.6.0"
_DCGMI_OUT = "dcgmi  version: 4.4.1\n"

_CUDA_13 = {"driver": {"version": "13.0", "distribution": "ubuntu2404"}}
_CUDA_12 = {"driver": {"version": "12.9", "distribution": "ubuntu2404"}}

_SERVICE = [
    ["systemctl", "--now", "enable", "nvidia-dcgm"],
    ["systemctl", "start", "nvidia-dcgm"],
    ["systemctl", "is-active", "--quiet", "nvidia-dcgm"],
]
_RHEL_INSTALL = ["dnf", "install", "--assumeyes", "--setopt=install_weak_deps=True"]


def _env(distribution="ubuntu24.04", **extra):
    env = {"DISTRIBUTION": distribution, "GPU": "NVIDIA", "SKU": "H100",
           "ARCHITECTURE": "x86_64", "TARGET_NODE_TYPE": "azure_vm_regular"}
    env.update(extra)
    return env


# --------------------------------------------------------------------------- #
# Pure helpers and parity of the C source.
# --------------------------------------------------------------------------- #
class HelperTests(unittest.TestCase):
    def test_parse_cuda_driver_version(self):
        self.assertEqual(install_dcgm.parse_cuda_driver_version("13000\n"), 13000)
        self.assertEqual(install_dcgm.parse_cuda_driver_version("12090"), 12090)
        for bad in ["", "\n", "0\n", "-1", "13.0", "abc", "13000 extra"]:
            with self.subTest(bad=bad):
                self.assertIsNone(install_dcgm.parse_cuda_driver_version(bad))

    def test_cuda_major(self):
        self.assertEqual(install_dcgm.cuda_major(13000), 13)
        self.assertEqual(install_dcgm.cuda_major(13020), 13)
        self.assertEqual(install_dcgm.cuda_major(12090), 12)
        self.assertEqual(install_dcgm.cuda_major(9020), 9)

    def test_sku_cuda_major(self):
        self.assertEqual(install_dcgm.sku_cuda_major(_CUDA_13), 13)
        self.assertEqual(install_dcgm.sku_cuda_major(_CUDA_12), 12)
        self.assertEqual(install_dcgm.sku_cuda_major({"driver": {"version": "13"}}), 13)
        for bad in [None, {}, {"driver": {}}, {"driver": "13.0"},
                    {"driver": {"version": "null"}}, {"driver": {"version": ""}}]:
            with self.subTest(bad=bad):
                self.assertIsNone(install_dcgm.sku_cuda_major(bad))

    def test_dcgmi_version(self):
        self.assertEqual(install_dcgm.dcgmi_version(_DCGMI_OUT), "4.4.1")
        self.assertEqual(install_dcgm.dcgmi_version("dcgmi  version: 4.4.1"), "4.4.1")
        # A shorter line first prints "" in awk, which word splitting drops.
        self.assertEqual(install_dcgm.dcgmi_version("banner line\n" + _DCGMI_OUT),
                         "4.4.1")
        self.assertIsNone(install_dcgm.dcgmi_version(""))
        self.assertIsNone(install_dcgm.dcgmi_version("a b\n"))

    def test_probe_source_matches_the_bash_heredoc(self):
        script = (_REPO / "components" / "install_dcgm.sh").read_text(encoding="utf-8")
        # The text between the `cat > ... <<'EOF'` line and the line "EOF".
        match = re.search(r"<<'EOF'\n(.*?\n)EOF\n", script, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertEqual(install_dcgm._CUDA_PROBE_SOURCE, match.group(1))


# --------------------------------------------------------------------------- #
# install(): external effects mocked.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(install_dcgm, "_PROBE_PARENT_DIR", self._tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _install(self, env, *, probe=(0, "13000\n"), dcgmi=(0, _DCGMI_OUT),
                 cuda_cfg=_CUDA_13, dcgm_cfg=None, fail_on=None, write_error=None):
        """Run install(env) with everything external mocked.

        probe: (rc, stdout) of the compiled probe; dcgmi: of `dcgmi --version`.
        fail_on: a command (list) for which exec_program returns 1.
        Returns (rc, mocks); mocks["probe_source"] is the C file's text as it
        was on disk when gcc ran.
        """
        if dcgm_cfg is None:
            dcgm_cfg = {"version": _UBUNTU_DCGM}
        seen = {}

        def fake_exec(command, op, **kwargs):
            if command[0] == "gcc":
                seen["probe_source"] = Path(command[1]).read_text(encoding="utf-8")
                seen["probe_dir"] = os.path.dirname(command[1])
            return 1 if command == fail_on else 0

        def fake_capture(command, op, **kwargs):
            return dcgmi if command[0] == "dcgmi" else probe

        def fake_config(component, env):
            return {"cuda": cuda_cfg, "dcgm": dcgm_cfg}[component]

        with mock.patch.object(install_dcgm, "exec_program",
                               side_effect=fake_exec) as exec_program, \
             mock.patch.object(install_dcgm, "run_capture",
                               side_effect=fake_capture) as run_capture, \
             mock.patch.object(install_dcgm, "config_for",
                               side_effect=fake_config) as config_for, \
             mock.patch.object(install_dcgm, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(install_dcgm, "log_info") as log_info, \
             mock.patch.object(install_dcgm, "log_error") as log_error:
            rc = install_dcgm.install(env)
        return rc, {"exec": exec_program, "capture": run_capture,
                    "config_for": config_for, "record": record,
                    "info": log_info, "error": log_error, **seen}

    @staticmethod
    def _commands(mocks):
        """exec_program commands, minus the gcc compile of the probe."""
        return [c.args[0] for c in mocks["exec"].call_args_list
                if c.args[0][0] != "gcc"]

    def _assert_probe_ran_and_was_cleaned_up(self, m):
        gcc = m["exec"].call_args_list[0].args[0]
        probe_dir = m["probe_dir"]
        self.assertEqual(gcc, ["gcc", f"{probe_dir}/cuda_driver_api_version.c",
                               "-ldl", "-o", f"{probe_dir}/cuda_driver_api_version"])
        self.assertEqual(m["capture"].call_args_list[0].args[0],
                         [f"{probe_dir}/cuda_driver_api_version"])
        self.assertEqual(m["probe_source"], install_dcgm._CUDA_PROBE_SOURCE)
        self.assertTrue(probe_dir.startswith(self._tmp.name))
        self.assertFalse(os.path.exists(probe_dir))
        self.assertEqual(os.listdir(self._tmp.name), [])

    # ---- success paths -----------------------------------------------------
    def test_ubuntu_same_cuda_major(self):
        env = _env()
        rc, m = self._install(env)

        self.assertEqual(rc, 0)
        self._assert_probe_ran_and_was_cleaned_up(m)
        self.assertEqual(self._commands(m), [
            ["apt-get", "install", "-y",
             "datacenter-gpu-manager-4-cuda13=1:4.6.0-1",
             "datacenter-gpu-manager-4-core=1:4.6.0-1",
             "datacenter-gpu-manager-4-proprietary=1:4.6.0-1",
             "datacenter-gpu-manager-4-proprietary-cuda13=1:4.6.0-1"],
            *_SERVICE,
        ])
        for call in m["exec"].call_args_list:
            self.assertEqual(call.args[1], _OP)
            self.assertIs(call.kwargs["env"], env)
        m["record"].assert_called_once_with("DCGM", _UBUNTU_DCGM)
        m["error"].assert_not_called()

    def test_ubuntu_lower_sku_cuda_adds_packages(self):
        rc, m = self._install(_env(SKU="V100"), cuda_cfg=_CUDA_12)

        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m)[1], [
            "apt-get", "install", "-y",
            "datacenter-gpu-manager-4-cuda12=1:4.6.0-1",
            "datacenter-gpu-manager-4-proprietary-cuda12=1:4.6.0-1"])
        self.assertEqual(self._commands(m)[2:], _SERVICE)

    def test_ubuntu_compares_numerically(self):
        # 9 < 13 numerically (a text comparison would say "9" > "13").
        rc, m = self._install(_env(), cuda_cfg={"driver": {"version": "9.2"}})
        self.assertEqual(rc, 0)
        self.assertIn("datacenter-gpu-manager-4-cuda9=1:4.6.0-1", self._commands(m)[1])

    def test_ubuntu_higher_sku_cuda_adds_nothing(self):
        rc, m = self._install(_env(), probe=(0, "12090\n"))
        self.assertEqual(rc, 0)
        self.assertEqual(len(self._commands(m)), 1 + len(_SERVICE))
        self.assertIn("datacenter-gpu-manager-4-cuda12=1:4.6.0-1", self._commands(m)[0])

    def test_azure_linux_always_installs_cuda13(self):
        # Mirrors the bash bug: even a V100 (CUDA 12 SKU) gets cuda13.
        rc, m = self._install(_env("azurelinux3.0", SKU="V100"), cuda_cfg=_CUDA_12,
                              probe=(0, "12090\n"), dcgm_cfg={"version": _AZL_DCGM})

        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m), [
            ["dnf", "install", "-y", "datacenter-gpu-manager-4-cuda13-4.6.0"],
            *_SERVICE,
        ])
        m["record"].assert_called_once_with("DCGM", _AZL_DCGM)

    def test_azure_linux_does_not_need_the_sku_cuda_version(self):
        rc, m = self._install(_env("azurelinux3.0"), cuda_cfg=None,
                              dcgm_cfg={"version": _AZL_DCGM})
        self.assertEqual(rc, 0)

    def test_rhel_family_records_the_dcgmi_version(self):
        rc, m = self._install(_env("almalinux9.8", SKU="V100"), cuda_cfg=_CUDA_12)

        self.assertEqual(rc, 0)
        self._assert_probe_ran_and_was_cleaned_up(m)
        self.assertEqual(self._commands(m), [
            ["dnf", "clean", "expire-cache"],
            [*_RHEL_INSTALL, "datacenter-gpu-manager-4-cuda13"],
            [*_RHEL_INSTALL, "datacenter-gpu-manager-4-cuda12"],
            *_SERVICE,
        ])
        self.assertEqual(m["capture"].call_args_list[1].args[0], ["dcgmi", "--version"])
        m["record"].assert_called_once_with("DCGM", "4.4.1")
        # The RHEL family never reads the dcgm config.
        self.assertEqual([c.args[0] for c in m["config_for"].call_args_list], ["cuda"])

    def test_rhel_family_same_cuda_major(self):
        rc, m = self._install(_env("rocky8.10"))
        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m)[:2], [
            ["dnf", "clean", "expire-cache"],
            [*_RHEL_INSTALL, "datacenter-gpu-manager-4-cuda13"],
        ])
        self.assertEqual(self._commands(m)[2:], _SERVICE)

    # ---- logging -----------------------------------------------------------
    def test_key_log_lines(self):
        rc, m = self._install(_env(SKU="V100"), cuda_cfg=_CUDA_12)
        self.assertEqual(rc, 0)

        calls = m["info"].call_args_list
        self.assertEqual(calls[0].args, (_OP, "Installing DCGM"))
        self.assertEqual(calls[0].kwargs["distro"], "ubuntu24.04")
        by_message = {c.args[1]: c.kwargs for c in calls}
        self.assertEqual(by_message["Probed CUDA driver API version 13000 (CUDA 13)"],
                         {"cuda_driver_api_version": 13000, "cuda_major": 13})
        decision = [c for c in calls if "also installing the cuda12" in c.args[1]]
        self.assertEqual(decision[0].kwargs, {"cuda_major": 13, "sku_cuda_major": 12})
        ubuntu = [c for c in calls if c.args[1].startswith("Ubuntu:")]
        self.assertEqual(ubuntu[0].kwargs["version_source"], "versions.json")
        self.assertEqual(by_message["Recorded DCGM version 1:4.6.0-1"],
                         {"component": "DCGM", "version": _UBUNTU_DCGM})
        self.assertIn("nvidia-dcgm service is active", by_message)
        self.assertEqual(calls[-1].args[1], "DCGM installed")

    # ---- failure paths -----------------------------------------------------
    def test_probe_failures_return_3_before_installing(self):
        for probe in [(1, ""), (0, ""), (0, "garbage\n")]:
            with self.subTest(probe=probe):
                rc, m = self._install(_env(), probe=probe)
                self.assertEqual(rc, 3)
                self.assertEqual(self._commands(m), [])
                m["record"].assert_not_called()
                m["error"].assert_called()
                self.assertEqual(os.listdir(self._tmp.name), [])

    def test_compile_failure_returns_3_and_cleans_up(self):
        def fail_gcc(command, op, **kwargs):
            return 1
        with mock.patch.object(install_dcgm, "exec_program", side_effect=fail_gcc), \
             mock.patch.object(install_dcgm, "run_capture") as run_capture, \
             mock.patch.object(install_dcgm, "log_error") as log_error:
            rc = install_dcgm.install(_env())
        self.assertEqual(rc, 3)
        run_capture.assert_not_called()
        self.assertIn("compile", log_error.call_args_list[0].args[1])
        self.assertEqual(os.listdir(self._tmp.name), [])

    def test_probe_dir_os_error_returns_3(self):
        with mock.patch.object(install_dcgm.tempfile, "mkdtemp",
                               side_effect=PermissionError("no tmp")), \
             mock.patch.object(install_dcgm, "exec_program") as exec_program, \
             mock.patch.object(install_dcgm, "log_error") as log_error:
            rc = install_dcgm.install(_env())
        self.assertEqual(rc, 3)
        exec_program.assert_not_called()
        self.assertIn("no tmp", log_error.call_args.args[1])

    def test_missing_dcgm_config_returns_3(self):
        for distribution in ["ubuntu24.04", "azurelinux3.0"]:
            with self.subTest(distribution=distribution):
                rc, m = self._install(_env(distribution), dcgm_cfg={})
                self.assertEqual(rc, 3)
                self.assertEqual(self._commands(m), [])
                self.assertIn("dcgm version", m["error"].call_args_list[0].args[1])

    def test_missing_sku_cuda_version_returns_3_on_ubuntu_and_rhel(self):
        for distribution in ["ubuntu24.04", "almalinux9.8"]:
            with self.subTest(distribution=distribution):
                rc, m = self._install(_env(distribution), cuda_cfg=None)
                self.assertEqual(rc, 3)
                self.assertEqual(self._commands(m), [])
                self.assertIn("cuda driver version",
                              m["error"].call_args_list[0].args[1])

    def test_each_failing_command_returns_3_and_stops(self):
        sequences = {
            "ubuntu24.04": [
                ["apt-get", "install", "-y",
                 *install_dcgm.ubuntu_packages(13, _UBUNTU_DCGM)],
                ["apt-get", "install", "-y",
                 *install_dcgm.ubuntu_extra_packages(12, _UBUNTU_DCGM)],
            ],
            "almalinux9.8": [
                ["dnf", "clean", "expire-cache"],
                [*_RHEL_INSTALL, "datacenter-gpu-manager-4-cuda13"],
                [*_RHEL_INSTALL, "datacenter-gpu-manager-4-cuda12"],
            ],
        }
        for distribution, installs in sequences.items():
            sequence = installs + _SERVICE
            for index, failing in enumerate(sequence):
                with self.subTest(distribution=distribution, failing=failing):
                    rc, m = self._install(_env(distribution), cuda_cfg=_CUDA_12,
                                          fail_on=failing)
                    self.assertEqual(rc, 3)
                    self.assertEqual(self._commands(m), sequence[:index + 1])
                    m["error"].assert_called()
                    # The version is recorded only after a successful install.
                    if index < len(installs):
                        m["record"].assert_not_called()
                    else:
                        m["record"].assert_called_once()

    def test_inactive_service_logs_dcgm_inactive(self):
        rc, m = self._install(_env(), fail_on=_SERVICE[2])
        self.assertEqual(rc, 3)
        self.assertEqual(m["error"].call_args.args[1], "DCGM is inactive!")

    def test_dcgmi_failure_returns_3(self):
        for dcgmi in [(1, _DCGMI_OUT), (0, ""), (127, "")]:
            with self.subTest(dcgmi=dcgmi):
                rc, m = self._install(_env("almalinux9.8"), dcgmi=dcgmi)
                self.assertEqual(rc, 3)
                m["record"].assert_not_called()
                self.assertEqual(self._commands(m)[-1],
                                 [*_RHEL_INSTALL, "datacenter-gpu-manager-4-cuda13"])

    def test_unwritable_versions_file_returns_3(self):
        rc, m = self._install(_env(), write_error=PermissionError("denied"))
        self.assertEqual(rc, 3)
        self.assertEqual(len(self._commands(m)), 1)    # no systemctl after it
        self.assertIn("denied", m["error"].call_args.args[1])


# --------------------------------------------------------------------------- #
# Wiring.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "install-dcgm"]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, install_dcgm.install)
        self.assertEqual(steps[0].script, "install_dcgm.sh")


if __name__ == "__main__":
    unittest.main()
