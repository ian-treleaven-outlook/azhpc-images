"""Unit tests for components/python/install_nvidiagriddriver.py.

  * The pure helpers (gridd.conf edits, nvcc version parse, names/URLs) are
    tested directly. Their results were also compared against real
    grep/sed/echo and grep/awk/cut in bash (WSL), and CUDA_PROFILE_CONTENT
    against the file the two `echo | tee` lines render.
  * install() is tested with its external effects mocked: config_for,
    download_and_verify, exec_program (the fake performs `cp` so the licensing
    edits run on real temporary files), run_capture (nvcc),
    write_component_version and install_cuda_samples.install. Every path
    constant points into a temporary directory.

Run from the repo root:
    python3 -m unittest tests.python.test_install_nvidiagriddriver -v
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_nvidiagriddriver as grid
from utils.build_config import BuildConfig, build_plan

_REPO = Path(__file__).resolve().parents[2]
_COMPONENT_DIR = "/repo/components"
_PERSISTENCE = f"{_COMPONENT_DIR}/configure_nvidia_persistence.sh"

_GRID_VERSION = "595.91.07"
_GRID_URL = ("https://download.microsoft.com/download/x/"
             f"NVIDIA-Linux-x86_64-{_GRID_VERSION}-grid-azure.run")
_GRID_CFG = {"url": _GRID_URL, "sha256": "abc123", "version": _GRID_VERSION}
_CUDA_CFG = {"driver": {"version": "13.2", "distribution": "ubuntu2404"},
             "samples": {"version": "13.2", "sha256": "def"}}
_INSTALLER = f"NVIDIA-Linux-x86_64-{_GRID_VERSION}-grid-azure.run"

_NVCC_OUT = ("nvcc: NVIDIA (R) Cuda compiler driver\n"
             "Copyright (c) 2005-2025 NVIDIA Corporation\n"
             "Built on Wed_Aug_20_01:58:59_PM_PDT_2025\n"
             "Cuda compilation tools, release 13.2, V13.2.51\n"
             "Build cuda_13.2.r13.2/compiler.36424714_0\n")

_TEMPLATE = ("ServerAddress=\n"
             "FeatureType=0\n"
             "# IgnoreSP=TRUE\n"
             "EnableUI=TRUE\n")
_EXPECTED_CONF = ("ServerAddress=\n"
                  "# IgnoreSP=TRUE\n"
                  "EnableUI=FALSE\n"
                  "IgnoreSP=FALSE\n")


def _env(distribution="ubuntu24.04", **extra):
    env = {"DISTRIBUTION": distribution, "COMPONENT_DIR": _COMPONENT_DIR,
           "GPU": "NVIDIA", "SKU": "NCv6", "ARCHITECTURE": "x86_64",
           "TARGET_NODE_TYPE": "azure_vm_regular"}
    env.update(extra)
    return env


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
class NameTests(unittest.TestCase):
    def test_installer_name(self):
        self.assertEqual(grid.driver_installer_name("595.91.07"), _INSTALLER)

    def test_toolkit_package_replaces_every_dot(self):
        self.assertEqual(grid.cuda_toolkit_package("13.2"), "cuda-toolkit-13-2")
        self.assertEqual(grid.cuda_toolkit_package("12.9.1"), "cuda-toolkit-12-9-1")


class GriddEditTests(unittest.TestCase):
    def test_split_lines_only_on_newline(self):
        self.assertEqual(grid.split_lines("a\r\nb\x0bc\nd"), ["a\r\n", "b\x0bc\n", "d"])
        self.assertEqual(grid.split_lines(""), [])
        self.assertEqual(grid.split_lines("\n\n"), ["\n", "\n"])

    def test_has_key_is_anchored_at_line_start(self):
        self.assertTrue(grid.has_key("x\nIgnoreSP=TRUE\n", "IgnoreSP"))
        self.assertFalse(grid.has_key("# IgnoreSP=TRUE\n IgnoreSP=1\n", "IgnoreSP"))
        self.assertFalse(grid.has_key("IgnoreSPX=1\n", "IgnoreSP"))

    def test_set_key_rewrites_every_matching_line(self):
        text, changed = grid.set_key("IgnoreSP=TRUE\nx\nIgnoreSP=\n", "IgnoreSP", "FALSE")
        self.assertEqual(text, "IgnoreSP=FALSE\nx\nIgnoreSP=FALSE\n")
        self.assertEqual(changed, 2)

    def test_set_key_keeps_missing_final_newline_and_drops_cr_like_sed(self):
        self.assertEqual(grid.set_key("a\nEnableUI=1", "EnableUI", "FALSE"),
                         ("a\nEnableUI=FALSE", 1))
        self.assertEqual(grid.set_key("EnableUI=1\r\n", "EnableUI", "FALSE"),
                         ("EnableUI=FALSE\n", 1))

    def test_delete_lines_matches_prefix_like_sed(self):
        text = "FeatureType=0\nFeatureType=01\nFeatureType=1\n FeatureType=0\n"
        self.assertEqual(grid.delete_lines(text, "FeatureType=0"),
                         ("FeatureType=1\n FeatureType=0\n", 2))
        self.assertEqual(grid.delete_lines("a\nFeatureType=0", "FeatureType=0"),
                         ("a\n", 1))


class NvccVersionTests(unittest.TestCase):
    def test_release_line(self):
        self.assertEqual(grid.nvcc_release_version(_NVCC_OUT), "13.2.51")

    def test_first_word_wins_like_bash(self):
        out = "x release 1 2 3 Vy\n" + _NVCC_OUT
        self.assertEqual(grid.nvcc_release_version(out), "y")

    def test_short_matching_line_is_dropped_like_word_splitting(self):
        self.assertEqual(grid.nvcc_release_version("short release\n" + _NVCC_OUT),
                         "13.2.51")

    def test_no_match_is_none(self):
        self.assertIsNone(grid.nvcc_release_version(""))
        self.assertIsNone(grid.nvcc_release_version("nvcc: command not found\n"))


class BashScriptTests(unittest.TestCase):
    def test_cuda_repo_setup_moved_to_the_bootstrap(self):
        # Upstream #643: the keyring/apt-get update lines left this script for
        # install_cuda_repo_and_dkms.sh, which the bootstrap runs.
        script = (_REPO / "components" / "install_nvidiagriddriver.sh").read_text()
        self.assertIn("apt install -y cuda-toolkit-${CUDA_DRIVER_VERSION//./-}", script)
        for gone in ("cuda-keyring", "dpkg -i", "apt-get update", "CUDA_DRIVER_DISTRIBUTION"):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, script)
        bootstrap = (_REPO / "distros" / "ubuntu24.04" / "install_utils.sh").read_text()
        self.assertIn("$COMPONENT_DIR/install_cuda_repo_and_dkms.sh", bootstrap)


class CudaProfileContentTests(unittest.TestCase):
    def test_matches_the_echo_lines_in_the_bash_script(self):
        # Each `echo '<text>' | tee [-a] /etc/profile.d/cuda.sh` writes <text>
        # plus a newline; read the texts from the .sh so the two can't drift.
        script = (_REPO / "components" / "install_nvidiagriddriver.sh").read_text()
        texts = re.findall(r"^echo '([^']*)' \| tee (?:-a )?/etc/profile\.d/cuda\.sh",
                           script, flags=re.M)
        self.assertEqual(len(texts), 2)
        self.assertEqual(grid.CUDA_PROFILE_CONTENT, "".join(t + "\n" for t in texts))


# --------------------------------------------------------------------------- #
# install(): external effects mocked.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = self._tmp.name
        self.work = os.path.join(root, "work")
        self.template = os.path.join(root, "gridd.conf.template")
        self.conf = os.path.join(root, "gridd.conf")
        self.profile = os.path.join(root, "cuda.sh")
        self.sources = os.path.join(root, "sources.list.d")
        os.makedirs(self.sources)
        # What the bootstrap's cuda-keyring install leaves behind.
        self.cuda_source = os.path.join(self.sources, "cuda-ubuntu2404-x86_64.list")
        open(self.cuda_source, "w").close()
        with open(self.template, "w", encoding="utf-8") as handle:
            handle.write(_TEMPLATE)
        for name, value in (("_WORK_DIR", self.work), ("_GRIDD_TEMPLATE", self.template),
                            ("_GRIDD_CONF", self.conf), ("_CUDA_PROFILE", self.profile),
                            ("_APT_SOURCES_DIR", self.sources)):
            patcher = mock.patch.object(grid, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _install(self, env=None, *, configs=None, fail_on=None, nvcc=(0, _NVCC_OUT),
                 download=None, samples_rc=0, samples_error=None, write_error=None):
        """Run install(env) with everything external mocked.

        configs: {component: config} for config_for (default: both present).
        fail_on: first word of the command exec_program reports as failing.
        download: side_effect for download_and_verify (default: returns a path
            named after the URL).
        Returns (rc, mocks).
        """
        env = env or _env()
        configs = configs if configs is not None else {"nvidia_grid": _GRID_CFG,
                                                       "cuda": _CUDA_CFG}
        seen_work_dir = []

        def fake_exec(command, op, **kwargs):
            seen_work_dir.append(os.path.isdir(self.work))
            if command[0] == fail_on:
                return 1
            if command[0] == "cp":
                shutil.copyfile(command[1], command[2])
            return 0

        def fake_download(url, sha256, dest_dir="."):
            return Path(dest_dir) / os.path.basename(url)

        with mock.patch.object(grid, "config_for",
                               side_effect=lambda c, e: configs.get(c)) as config_for, \
             mock.patch.object(grid, "download_and_verify",
                               side_effect=download or fake_download) as dl, \
             mock.patch.object(grid, "exec_program", side_effect=fake_exec) as exec_program, \
             mock.patch.object(grid, "run_capture", return_value=nvcc) as run_capture, \
             mock.patch.object(grid, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(grid.install_cuda_samples, "install",
                               return_value=samples_rc,
                               side_effect=samples_error) as samples, \
             mock.patch.object(grid, "log_info") as log_info, \
             mock.patch.object(grid, "log_warn") as log_warn, \
             mock.patch.object(grid, "log_error") as log_error:
            rc = grid.install(env)
        return rc, {"config_for": config_for, "download": dl, "exec": exec_program,
                    "capture": run_capture, "record": record, "samples": samples,
                    "info": log_info, "warn": log_warn, "error": log_error,
                    "work_dir_existed": seen_work_dir}

    @staticmethod
    def _commands(mocks):
        return [c.args[0] for c in mocks["exec"].call_args_list]

    @staticmethod
    def _messages(log_mock):
        return [c.args[1] for c in log_mock.call_args_list]

    def _all_commands(self):
        return [
            ["bash", _INSTALLER, "--silent", "--dkms", "--kernel-module-type=open"],
            ["cp", self.template, self.conf],
            ["apt", "install", "-y", "cuda-toolkit-13-2"],
            [_PERSISTENCE],
        ]

    # ---- success -------------------------------------------------------------
    def test_ubuntu_happy_path(self):
        env = _env()
        rc, m = self._install(env)

        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m), self._all_commands())
        for call in m["exec"].call_args_list:
            self.assertEqual(call.kwargs, {"cwd": self.work, "env": env})
        self.assertTrue(all(m["work_dir_existed"]))
        m["download"].assert_called_once_with(_GRID_URL, "abc123", dest_dir=self.work)
        m["capture"].assert_called_once_with(
            ["bash", "-c", "source /etc/profile; nvcc --version"], "install-nv-grid",
            cwd=self.work, env=env)
        self.assertEqual(m["record"].call_args_list,
                         [mock.call("NVIDIA_GRID", _GRID_VERSION),
                          mock.call("CUDA", "13.2.51")])
        m["samples"].assert_called_once_with(env)
        self.assertEqual([c.args[0] for c in m["config_for"].call_args_list],
                         ["nvidia_grid", "cuda"])

        with open(self.conf, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), _EXPECTED_CONF)
        with open(self.profile, "rb") as handle:
            self.assertEqual(handle.read(), grid.CUDA_PROFILE_CONTENT.encode())
        self.assertEqual(os.stat(self.profile).st_mode & 0o777, 0o644)
        self.assertFalse(os.path.exists(self.work))          # cleaned up
        m["error"].assert_not_called()

        # Only the cleanup-difference warning on a normal Ubuntu run.
        warnings = self._messages(m["warn"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("rm -rf -- */", warnings[0])

        info = self._messages(m["info"])
        self.assertTrue(any("Installing NVIDIA GRID driver 595.91.07 and "
                            "cuda-toolkit-13-2" in msg for msg in info))
        self.assertTrue(any(msg.startswith("Set EnableUI=FALSE") for msg in info))
        self.assertTrue(any(msg.startswith("Appended IgnoreSP=FALSE") for msg in info))
        self.assertTrue(any(msg.startswith("Removed 1 line(s) starting with FeatureType=0")
                            for msg in info))
        self.assertTrue(any(msg.startswith("Created ") and self.profile in msg
                            for msg in info))
        self.assertTrue(any("Recorded CUDA version 13.2.51" == msg for msg in info))
        self.assertTrue(any(msg.startswith("NVIDIA CUDA apt source present")
                            for msg in info))
        source_log = [c for c in m["info"].call_args_list
                      if c.args[1].startswith("NVIDIA CUDA apt source present")]
        self.assertEqual(source_log[0].kwargs["cuda_apt_sources"], self.cuda_source)
        self.assertTrue(any("CUDA samples" in msg for msg in info))
        self.assertTrue(any("nvidia-persistenced" in msg for msg in info))
        self.assertEqual(info[-1], "NVIDIA GRID driver installed")
        start = m["info"].call_args_list[0].kwargs
        self.assertEqual(start["grid_version"], _GRID_VERSION)
        self.assertEqual(start["source"], _GRID_URL)
        self.assertEqual(start["cuda_toolkit_version"], "13.2")
        self.assertEqual(start["distro"], "ubuntu24.04")

    def test_existing_profile_is_replaced(self):
        with open(self.profile, "w", encoding="utf-8") as handle:
            handle.write("old\n")
        os.chmod(self.profile, 0o600)
        rc, m = self._install()
        self.assertEqual(rc, 0)
        with open(self.profile, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), grid.CUDA_PROFILE_CONTENT)
        self.assertEqual(os.stat(self.profile).st_mode & 0o777, 0o644)
        self.assertTrue(any(msg.startswith("Replaced ") for msg in self._messages(m["info"])))

    def test_stale_work_dir_is_emptied_first(self):
        os.makedirs(os.path.join(self.work, "old"))
        seen = []

        def fake_download(url, sha256, dest_dir="."):
            seen.append(os.listdir(dest_dir))
            return Path(dest_dir) / os.path.basename(url)

        rc, _ = self._install(download=fake_download)
        self.assertEqual(rc, 0)
        self.assertEqual(seen, [[]])

    def test_template_without_final_newline_glues_like_bash(self):
        with open(self.template, "w", encoding="utf-8") as handle:
            handle.write("EnableUI=TRUE")
        rc, m = self._install()
        self.assertEqual(rc, 0)
        with open(self.conf, encoding="utf-8") as handle:
            # IgnoreSP=FALSE is glued onto "EnableUI=TRUE", then the EnableUI
            # rewrite replaces that whole line, so IgnoreSP is lost. Real
            # grep/sed/echo give exactly this too.
            self.assertEqual(handle.read(), "EnableUI=FALSE\n")
        self.assertTrue(any("glued" in msg for msg in self._messages(m["warn"])))

    # ---- warnings --------------------------------------------------------------
    def test_non_ubuntu_warns_and_fails_at_apt_like_bash(self):
        rc, m = self._install(_env("almalinux9.8"), fail_on="apt")
        self.assertEqual(rc, 3)
        self.assertTrue(any("assumes Ubuntu" in msg for msg in self._messages(m["warn"])))
        self.assertEqual(self._commands(m)[-1], ["apt", "install", "-y", "cuda-toolkit-13-2"])
        self.assertFalse(os.path.exists(self.work))

    def test_missing_cuda_apt_source_warns_and_continues(self):
        os.remove(self.cuda_source)
        rc, m = self._install()
        self.assertEqual(rc, 0)          # the (mocked) apt decides, as in bash
        self.assertIn(["apt", "install", "-y", "cuda-toolkit-13-2"], self._commands(m))
        self.assertTrue(any("no CUDA apt source" in msg and "install_cuda_repo_and_dkms.sh"
                            in msg for msg in self._messages(m["warn"])))

    def test_installer_name_mismatch_warns(self):
        cfg = dict(_GRID_CFG, url="https://example.com/grid-driver.run")
        rc, m = self._install(configs={"nvidia_grid": cfg, "cuda": _CUDA_CFG})
        self.assertEqual(rc, 0)          # the mocked installer "succeeds"
        self.assertEqual(self._commands(m)[0][1], _INSTALLER)
        self.assertTrue(any("grid-driver.run" in msg for msg in self._messages(m["warn"])))

    def test_failed_rewrite_appends_like_bash_or_operator(self):
        # sed fails -> `|| echo` runs: the old line stays and the key is appended.
        real_open = open
        calls = {"n": 0}

        def flaky_open(path, mode="r", *args, **kwargs):
            if path == self.conf and mode == "w" and calls["n"] == 0:
                calls["n"] += 1
                raise PermissionError("read-only")
            return real_open(path, mode, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=flaky_open):
            rc, m = self._install()
        self.assertEqual(rc, 0)
        with open(self.conf, encoding="utf-8") as handle:
            self.assertEqual(handle.read(),
                             "ServerAddress=\n# IgnoreSP=TRUE\nEnableUI=TRUE\n"
                             "IgnoreSP=FALSE\nEnableUI=FALSE\n")
        self.assertTrue(any("grep && sed || echo" in msg
                            for msg in self._messages(m["warn"])))

    # ---- missing config ----------------------------------------------------------
    def test_cuda_distribution_is_no_longer_needed(self):
        # Only install_cuda_repo_and_dkms.sh reads driver.distribution now.
        cuda = {"driver": {"version": "13.2"}}
        rc, m = self._install(configs={"nvidia_grid": _GRID_CFG, "cuda": cuda})
        self.assertEqual(rc, 0)
        m["error"].assert_not_called()

    def test_missing_config_fails_before_anything_runs(self):
        cases = {
            "no grid": {"cuda": _CUDA_CFG},
            "no grid version": {"nvidia_grid": dict(_GRID_CFG, version=""),
                                "cuda": _CUDA_CFG},
            "no grid sha256": {"nvidia_grid": {"url": _GRID_URL, "version": "1"},
                               "cuda": _CUDA_CFG},
            "no cuda": {"nvidia_grid": _GRID_CFG},
            "no cuda version": {"nvidia_grid": _GRID_CFG,
                                "cuda": {"driver": {"distribution": "ubuntu2404"}}},
        }
        for name, configs in cases.items():
            with self.subTest(name):
                rc, m = self._install(configs=configs)
                self.assertEqual(rc, 3)
                m["exec"].assert_not_called()
                m["download"].assert_not_called()
                self.assertIn("versions.json", m["error"].call_args.args[1])

    # ---- failures ------------------------------------------------------------------
    def test_download_failure(self):
        rc, m = self._install(download=OSError("404"))
        self.assertEqual(rc, 3)
        m["exec"].assert_not_called()
        self.assertIn("404", m["error"].call_args.args[1])
        self.assertFalse(os.path.exists(self.work))

    def test_each_command_failure_stops_the_step(self):
        expected = self._all_commands()
        for index, command in enumerate(expected):
            with self.subTest(command=command[0]):
                rc, m = self._install(fail_on=command[0])
                self.assertEqual(rc, 3)
                # The fake fails every command with that first word; "bash" is
                # first, the persistence script last.
                self.assertEqual(self._commands(m), expected[:index + 1])
                m["error"].assert_called()
                self.assertFalse(os.path.exists(self.work))

    def test_nvcc_failure_or_empty_version(self):
        for nvcc in [(1, ""), (0, "bash: nvcc: command not found\n")]:
            with self.subTest(nvcc=nvcc):
                rc, m = self._install(nvcc=nvcc)
                self.assertEqual(rc, 3)
                self.assertEqual(m["record"].call_args_list,
                                 [mock.call("NVIDIA_GRID", _GRID_VERSION)])
                m["samples"].assert_not_called()

    def test_samples_failure(self):
        rc, m = self._install(samples_rc=3)
        self.assertEqual(rc, 3)
        self.assertNotIn([_PERSISTENCE], self._commands(m))
        self.assertIn("samples", m["error"].call_args.args[1])

    def test_samples_exception_is_caught(self):
        rc, m = self._install(samples_error=RuntimeError("boom"))
        self.assertEqual(rc, 3)
        self.assertIn("boom", m["error"].call_args.args[1])
        self.assertNotIn([_PERSISTENCE], self._commands(m))

    def test_unwritable_versions_file(self):
        rc, m = self._install(write_error=PermissionError("denied"))
        self.assertEqual(rc, 3)
        self.assertEqual(len(self._commands(m)), 1)          # stopped after the driver
        self.assertIn("denied", m["error"].call_args.args[1])

    def test_cp_failure_is_named(self):
        rc, m = self._install(fail_on="cp")
        self.assertEqual(rc, 3)
        self.assertIn("could not copy", m["error"].call_args.args[1])

    def test_unwritable_profile(self):
        os.makedirs(self.profile)               # a directory: open() fails
        rc, m = self._install()
        self.assertEqual(rc, 3)
        self.assertIn(self.profile, m["error"].call_args.args[1])
        m["capture"].assert_not_called()

    def test_unreadable_conf_when_deleting_feature_type(self):
        real_open = open

        def flaky_open(path, mode="r", *args, **kwargs):
            # The third read of gridd.conf is the FeatureType=0 pass.
            if path == self.conf and mode == "r":
                flaky_open.reads += 1
                if flaky_open.reads == 3:
                    raise PermissionError("nope")
            return real_open(path, mode, *args, **kwargs)
        flaky_open.reads = 0

        with mock.patch("builtins.open", side_effect=flaky_open):
            rc, m = self._install()
        self.assertEqual(rc, 3)
        self.assertIn("FeatureType=0", m["error"].call_args.args[1])


# --------------------------------------------------------------------------- #
# Wiring.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="NCv6", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "install-nv-grid"]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, grid.install)
        self.assertEqual(steps[0].script, "install_nvidiagriddriver.sh")
        self.assertTrue(steps[0].when(cfg))
        other = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                            fips=False, spec_path=None)
        self.assertFalse(steps[0].when(other))


if __name__ == "__main__":
    unittest.main()
