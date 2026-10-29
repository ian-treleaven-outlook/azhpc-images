"""Unit tests for components/python/install_rccl.py.

  * Pure helpers (folder naming, sed edits, memory maths, rpm parsing, the
    rccl-tests script) are tested directly with real inputs. The expected
    values for the sed edits and memory maths were produced with real GNU
    sed/awk/bash when the port was written.
  * install() runs with every external effect mocked: config lookup, download,
    exec_program, run_capture (nproc, rpm) and the versions file. A fake
    exec_program creates the directories that `git clone` / `tar` would, so
    the real Python file handling (CMakeLists.txt edit, sysctl.conf append,
    cleanup) runs against a temporary directory. Nothing touches /etc, /opt,
    /proc or the network.

Run from the repo root:
    python3 -m unittest tests.python.test_install_rccl -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_rccl
from utils.build_config import BuildConfig, build_plan

_NPROC = 96
_MEM_KB = 1584000000            # ~1510 GB -> 16 linker jobs (the cap)

_CLONE_CFG = {   # ubuntu24.04 in versions.json
    "version": "2.22.3", "branch": "release/rocm-rel-6.4",
    "commit": "2f7ac66cd64c68d4af8bb4562ce193778a7e470e",
    "url": "https://github.com/ROCm/rccl/archive/refs/tags/rocm-6.4.4.tar.gz",
    "sha256": "d16a"}
_TARBALL_CFG = {   # ubuntu22.04 in versions.json
    "version": "2.20.5",
    "url": "https://github.com/ROCm/rccl/archive/refs/tags/rocm-6.2.4.tar.gz",
    "sha256": "12a0"}

_CMAKELISTS = (
    "project(rccl)\n"
    "  target_compile_options(rccl PRIVATE -fgpu-rdc -parallel-jobs=12)\n"
    "  target_link_options(rccl PRIVATE -parallel-jobs=16)\n")

def _rccl_tests(prefix="/opt/rccl"):
    # Called inside tests, so the script names the patched rccl-tests dir.
    return ["bash", "-c", install_rccl.rccl_tests_script(prefix)]


_SYSCTL = [["sysctl", "kernel.numa_balancing=0"], ["sysctl", "vm.max_map_count=1048576"]]


def _perftest(work):
    return [
        ["git", "clone", "https://github.com/linux-rdma/perftest.git"],
        ["./autogen.sh"],
        ["./configure", "--enable-rocm", "--with-rocm=/opt/rocm",
         f"--prefix={install_rccl._PERFTEST_PREFIX_DIR}/"],
        ["make", f"-j{_NPROC}"],
        ["make", "install"],
    ]


class PureHelperTests(unittest.TestCase):
    def test_folder_name_like_basename(self):
        tarball = install_rccl.tarball_name(_CLONE_CFG["url"])
        self.assertEqual(tarball, "rocm-6.4.4.tar.gz")
        self.assertEqual(install_rccl.source_folder_name(tarball), "rccl-rocm-6.4.4")
        # basename keeps a suffix that is the whole name, and other suffixes.
        self.assertEqual(install_rccl.source_folder_name(".tar.gz"), "rccl-.tar.gz")
        self.assertEqual(install_rccl.source_folder_name("x.tgz"), "rccl-x.tgz")

    def test_uses_git_clone(self):
        self.assertTrue(install_rccl.uses_git_clone("release/rocm-rel-6.4"))
        for branch in (None, "", "null"):
            self.assertFalse(install_rccl.uses_git_clone(branch))

    def test_sed_edits_match_gnu_sed(self):
        # Input and expected output generated with the bash script's own
        # `sed -i -E` commands (nproc=96, linker jobs=7).
        original = "\n".join([
            "target_compile_options(rccl PRIVATE -parallel-jobs=12)",
            "  target_compile_options( rccl   PRIVATE -fgpu-rdc -parallel-jobs=12 -O3)",
            "target_compile_options(rccl\tPRIVATE -parallel-jobs=12 -parallel-jobs=12)",
            "target_compile_options(rccl PRIVATE -parallel-jobs=123)",
            "target_compile_options(rccl PRIVATE -parallel-jobs=8)",
            "target_compile_options(rccl PUBLIC -parallel-jobs=12)",
            "target_compile_options(rcclx PRIVATE -parallel-jobs=12)",
            "target_compile_options(rccl PRIVATE (x) -parallel-jobs=12)",
            "target_link_options(rccl PRIVATE -parallel-jobs=16)",
            "  target_link_options(rccl PRIVATE -fgpu-rdc -parallel-jobs=16 --hip-link)",
            "target_link_options(rccl PRIVATE -parallel-jobs=16 -parallel-jobs=8)",
            "target_link_options(rccl PRIVATE -parallel-jobs=)",
            "target_link_options( rccl PRIVATE -parallel-jobs=999)",
            "target_link_options(rccl PRIVATE -parallel-jobs=16)\r",
            "no newline target_compile_options(rccl PRIVATE -parallel-jobs=12)"])
        expected = "\n".join([
            "target_compile_options(rccl PRIVATE -parallel-jobs=96)",
            "  target_compile_options( rccl   PRIVATE -fgpu-rdc -parallel-jobs=96 -O3)",
            "target_compile_options(rccl\tPRIVATE -parallel-jobs=12 -parallel-jobs=96)",
            "target_compile_options(rccl PRIVATE -parallel-jobs=963)",
            "target_compile_options(rccl PRIVATE -parallel-jobs=8)",
            "target_compile_options(rccl PUBLIC -parallel-jobs=12)",
            "target_compile_options(rcclx PRIVATE -parallel-jobs=12)",
            "target_compile_options(rccl PRIVATE (x) -parallel-jobs=12)",
            "target_link_options(rccl PRIVATE -parallel-jobs=7)",
            "  target_link_options(rccl PRIVATE -fgpu-rdc -parallel-jobs=7 --hip-link)",
            "target_link_options(rccl PRIVATE -parallel-jobs=16 -parallel-jobs=7)",
            "target_link_options(rccl PRIVATE -parallel-jobs=)",
            "target_link_options( rccl PRIVATE -parallel-jobs=7)",
            "target_link_options(rccl PRIVATE -parallel-jobs=7)\r",
            "no newline target_compile_options(rccl PRIVATE -parallel-jobs=96)"])
        text, compile_changes = install_rccl.set_compile_jobs(original, 96)
        text, link_changes = install_rccl.set_link_jobs(text, 7)
        self.assertEqual(text, expected)
        self.assertEqual((compile_changes, link_changes), (5, 5))

    def test_memory_and_linker_jobs_match_awk_and_bash(self):
        # (MemTotal kB, awk mem_gb, bash num_linker_jobs) from the real tools.
        cases = [(0, 0, 1), (1048575, 0, 1), (1048576, 1, 1), (16777215, 15, 1),
                 (16777216, 16, 1), (17825792, 17, 2), (268435456, 256, 16),
                 (269484032, 257, 16), (2113929216, 2016, 16)]
        for kb, gb, jobs in cases:
            with self.subTest(kb=kb):
                meminfo = f"MemFree:  5 kB\nMemTotal:       {kb} kB\n"
                self.assertEqual(install_rccl.mem_gb_from_meminfo(meminfo), gb)
                self.assertEqual(install_rccl.linker_jobs(gb), jobs)

    def test_linker_job_boundaries_in_gb(self):
        for gb, jobs in ((0, 1), (1, 1), (16, 1), (17, 2), (256, 16), (257, 16)):
            self.assertEqual(install_rccl.linker_jobs(gb), jobs)

    def test_no_memtotal(self):
        self.assertIsNone(install_rccl.mem_gb_from_meminfo("MemFree: 5 kB\n"))

    def test_rpm_version(self):
        self.assertEqual(install_rccl.azure_linux_rccl_version("2.22.3-1.azl3\n"),
                         "2.22.3-1.azl3")
        self.assertIsNone(install_rccl.azure_linux_rccl_version(" \n"))

    def test_rccl_tests_script(self):
        script = install_rccl.rccl_tests_script("/opt/rocm")
        self.assertTrue(script.startswith(
            "set -e\nsource /etc/profile.d/modules.sh\nmodule load mpi/hpcx\n"))
        self.assertIn('-DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm;$HPCX"', script)
        self.assertIn("git sparse-checkout set projects/rccl-tests\n", script)
        self.assertIn("cp ./*_perf /opt/rccl-tests/\n", script)
        self.assertTrue(script.endswith("rm -rf rocm-systems\nmodule unload mpi/hpcx\n"))


class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.root = root
        self.work = str(root / "work")
        self.meminfo = root / "meminfo"
        self.meminfo.write_text(f"MemTotal:       {_MEM_KB} kB\n", encoding="utf-8")
        self.sysctl_conf = root / "sysctl.conf"
        self.tests_dir = root / "rccl-tests"
        for name, value in (("_WORK_DIR", self.work), ("_MEMINFO", str(self.meminfo)),
                            ("_SYSCTL_CONF", str(self.sysctl_conf)),
                            ("_RCCL_TESTS_DIR", str(self.tests_dir)),
                            ("_PERFTEST_PREFIX_DIR", str(root / "rocm-perftest"))):
            patcher = mock.patch.object(install_rccl, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.cmakelists_after = None

    # A fake exec_program that also does what the real command would do to the
    # work directory, so the Python file handling around it is exercised.
    def _fake_exec(self, command, op, *, cwd=None, env=None):
        self.calls.append((command, cwd, env))
        if command == self.fail_on:
            return self.fail_rc
        if command[:2] == ["git", "clone"] and "--branch" in command:
            self._make_source(os.path.join(cwd, command[-1]))
        elif command[0] == "tar":
            self._make_source(os.path.join(cwd, "rccl-" + command[2][:-len(".tar.gz")]))
        elif command[:2] == ["git", "clone"]:
            os.makedirs(os.path.join(cwd, "perftest"))
        elif command[0] == "bash":
            Path(self.tests_dir, "all_reduce_perf").touch()
        elif command[0] == "cmake":
            self.cmakelists_after = Path(cwd, "..", "CMakeLists.txt").read_text()
        return 0

    @staticmethod
    def _make_source(path):
        os.makedirs(path)
        Path(path, "CMakeLists.txt").write_text(_CMAKELISTS, encoding="utf-8")

    def _install(self, distro, cfg, *, fail_on=None, fail_rc=2, nproc=(0, f"{_NPROC}\n"),
                 rpm=(0, "2.22.3-1.azl3"), write_error=None):
        self.calls, self.fail_on, self.fail_rc = [], fail_on, fail_rc

        def fake_capture(command, op, *, cwd=None, env=None):
            return nproc if command == ["nproc"] else rpm

        def fake_download(url, sha256, dest_dir="."):
            path = Path(dest_dir) / Path(url).name
            path.write_bytes(b"tgz")
            return path

        env = {"DISTRIBUTION": distro, "PATH": "/usr/bin"}
        with mock.patch.object(install_rccl, "config_for", return_value=cfg) as config, \
             mock.patch.object(install_rccl, "download_and_verify",
                               side_effect=fake_download) as download, \
             mock.patch.object(install_rccl, "exec_program", side_effect=self._fake_exec), \
             mock.patch.object(install_rccl, "run_capture", side_effect=fake_capture) as cap, \
             mock.patch.object(install_rccl, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(install_rccl, "log_info") as info, \
             mock.patch.object(install_rccl, "log_warn") as warn, \
             mock.patch.object(install_rccl, "log_error") as error:
            result = install_rccl.install(env)
        self.config, self.download, self.capture, self.record = config, download, cap, record
        self.info, self.warn, self.error = info, warn, error
        return result

    def _commands(self):
        return [c[0] for c in self.calls]

    def _source_commands(self, cfg):
        folder = "rccl-" + install_rccl.tarball_name(cfg["url"])[:-len(".tar.gz")]
        if "branch" in cfg:
            fetch = [["git", "clone", "--branch", cfg["branch"],
                      "https://github.com/ROCm/rccl.git", folder],
                     ["git", "checkout", cfg["commit"]]]
        else:
            fetch = [["tar", "-xzf", install_rccl.tarball_name(cfg["url"])]]
        return fetch + [
            ["cmake", "-DCMAKE_PREFIX_PATH=/opt/rocm/", "-DCMAKE_INSTALL_PREFIX=/opt/rccl", ".."],
            ["make", f"-j{_NPROC}"],
            ["make", "install"],
        ]

    # ---- success paths ------------------------------------------------------
    def test_ubuntu_git_clone_build(self):
        self.assertEqual(self._install("ubuntu24.04", _CLONE_CFG), 0)
        self.assertEqual(self._commands(), self._source_commands(_CLONE_CFG) + _SYSCTL
                         + [_rccl_tests(), ["apt", "install", "-y", "libpci-dev"]]
                         + _perftest(self.work))
        self.download.assert_not_called()
        # cwd of each command.
        source = os.path.join(self.work, "rccl-rocm-6.4.4")
        self.assertEqual(self.calls[0][1], self.work)
        self.assertEqual(self.calls[1][1], source)
        self.assertEqual(self.calls[2][1], os.path.join(source, "build"))
        self.assertEqual(self.calls[7][1], self.work)                  # rccl-tests script
        self.assertEqual(self.calls[-4][1], os.path.join(self.work, "perftest"))
        # cmake gets CXX + CMAKE_POLICY_VERSION_MINIMUM on top of the build env.
        cmake_env = self.calls[2][2]
        self.assertEqual((cmake_env["CXX"], cmake_env["CMAKE_POLICY_VERSION_MINIMUM"],
                          cmake_env["PATH"]), ("/opt/rocm/bin/hipcc", "3.5", "/usr/bin"))
        self.assertEqual(self.cmakelists_after, (
            "project(rccl)\n"
            f"  target_compile_options(rccl PRIVATE -fgpu-rdc -parallel-jobs={_NPROC})\n"
            "  target_link_options(rccl PRIVATE -parallel-jobs=16)\n"))
        self.record.assert_called_once_with("RCCL", "2.22.3")
        self.assertEqual(self.sysctl_conf.read_text(),
                         "kernel.numa_balancing=0\nvm.max_map_count=1048576\n")
        self.assertTrue(self.tests_dir.is_dir())
        self.assertFalse(os.path.exists(self.work))       # all scratch removed
        self.error.assert_not_called()
        self.warn.assert_not_called()

    def test_ubuntu_tarball_build_and_small_memory(self):
        self.meminfo.write_text("MemTotal:       32768000 kB\n")   # 31 GB -> 2 jobs
        self.assertEqual(self._install("ubuntu22.04", _TARBALL_CFG), 0)
        self.assertEqual(self._commands()[:4], self._source_commands(_TARBALL_CFG))
        self.download.assert_called_once_with(_TARBALL_CFG["url"], "12a0",
                                              dest_dir=self.work)
        self.assertIn("target_link_options(rccl PRIVATE -parallel-jobs=2)",
                      self.cmakelists_after)
        self.record.assert_called_once_with("RCCL", "2.20.5")

    def test_rhel_family_builds_from_source_without_ubuntu_extras(self):
        self.assertEqual(self._install("rocky9.8", _CLONE_CFG), 0)
        commands = self._commands()
        self.assertNotIn(_SYSCTL[0], commands)
        self.assertNotIn(["apt", "install", "-y", "libpci-dev"], commands)
        # Still persisted in sysctl.conf on every distro.
        self.assertEqual(self.sysctl_conf.read_text(),
                         "kernel.numa_balancing=0\nvm.max_map_count=1048576\n")

    def test_azure_linux_uses_packages(self):
        self.sysctl_conf.write_text("existing=1\n")
        self.assertEqual(self._install("azurelinux3.0", None), 0)
        self.config.assert_not_called()      # bash ignores the config here
        self.assertEqual(self._commands(), [
            ["dnf", "install", "-y", "rccl", "rccl-devel", "rccl-unittests"],
            _rccl_tests("/opt/rocm"),
        ] + _perftest(self.work))
        self.capture.assert_any_call(
            ["rpm", "-q", "--queryformat", "%{VERSION}-%{RELEASE}", "rccl"],
            "install-rccl", env=mock.ANY)
        self.record.assert_called_once_with("RCCL", "2.22.3-1.azl3")
        self.assertEqual(self.sysctl_conf.read_text(),     # appended, not replaced
                         "existing=1\nkernel.numa_balancing=0\nvm.max_map_count=1048576\n")
        self.warn.assert_not_called()       # "existing=1" is not one of ours

    def test_rerun_warns_that_sysctl_lines_are_already_there(self):
        self.sysctl_conf.write_text("vm.max_map_count=1048576\n")
        self.assertEqual(self._install("azurelinux3.0", None), 0)
        self.warn.assert_called_once()
        self.assertEqual(self.warn.call_args.kwargs["settings"],
                         ["vm.max_map_count=1048576"])
        # Still appended, like bash's `tee -a`.
        self.assertEqual(self.sysctl_conf.read_text(),
                         "vm.max_map_count=1048576\nkernel.numa_balancing=0\n"
                         "vm.max_map_count=1048576\n")

    def test_no_matching_cmake_line_warns_and_continues(self):
        def fake(path):
            os.makedirs(path)
            Path(path, "CMakeLists.txt").write_text("project(rccl)\n")
        with mock.patch.object(self, "_make_source", side_effect=fake):
            self.assertEqual(self._install("ubuntu24.04", _CLONE_CFG), 0)
        self.assertEqual(self.warn.call_count, 2)

    def test_missing_memtotal_uses_one_linker_job(self):
        self.meminfo.write_text("MemFree: 1 kB\n")
        self.assertEqual(self._install("ubuntu24.04", _CLONE_CFG), 0)
        self.assertIn("-parallel-jobs=1)", self.cmakelists_after.splitlines()[2])
        self.warn.assert_called_once()

    # ---- logging ------------------------------------------------------------
    def test_key_log_lines(self):
        self._install("ubuntu24.04", _CLONE_CFG)
        start = self.info.call_args_list[0]
        self.assertIn("Installing RCCL 2.22.3 on ubuntu24.04 from source", start.args[1])
        self.assertEqual(start.kwargs["branch"], "release/rocm-rel-6.4")
        by_message = {c.args[1]: c.kwargs for c in self.info.call_args_list}
        self.assertIn("RCCL source: git clone of a pinned branch/commit "
                      "(works around ROCm/rccl#1877)", by_message)
        jobs = by_message["RCCL link parallelism: detected 1510 GB RAM -> -parallel-jobs=16"]
        self.assertEqual((jobs["mem_gb"], jobs["linker_jobs"], jobs["nproc"]), (1510, 16, 96))
        self.assertIn("Recorded RCCL=2.22.3", by_message)
        self.assertEqual(by_message[f"Installed 1 rccl-tests binaries in {self.tests_dir}"]
                         ["binaries"], ["all_reduce_perf"])
        self.assertEqual(self.info.call_args_list[-1].args[1],
                         "RCCL 2.22.3 installed with rccl-tests and perftest")

    # ---- failures -----------------------------------------------------------
    def test_bad_source_config_returns_3(self):
        cases = [None, {}, {"version": "1"},
                 {**_CLONE_CFG, "commit": None},
                 {**_TARBALL_CFG, "sha256": ""}]
        for cfg in cases:
            with self.subTest(cfg=cfg):
                self.assertEqual(self._install("ubuntu24.04", cfg), 3)
                self.assertEqual(self.calls, [])
                self.capture.assert_not_called()
                self.error.assert_called_once()

    def test_branch_null_uses_tarball(self):
        cfg = {**_TARBALL_CFG, "branch": "null"}
        self.assertEqual(self._install("ubuntu22.04", cfg), 0)
        self.download.assert_called_once()

    def test_bad_nproc_returns_3(self):
        for result in ((1, ""), (0, "abc\n"), (0, "0\n")):
            with self.subTest(result=result):
                self.assertEqual(self._install("ubuntu24.04", _CLONE_CFG, nproc=result), 3)
                self.assertEqual(self.calls, [])

    def test_rpm_failure_returns_3(self):
        for rpm in ((1, "package rccl is not installed\n"), (0, "")):
            with self.subTest(rpm=rpm):
                self.assertEqual(self._install("azurelinux3.0", None, rpm=rpm), 3)
                self.record.assert_not_called()

    def test_download_failure_returns_3(self):
        with mock.patch.object(install_rccl, "download_and_verify",
                               side_effect=RuntimeError("checksum")):
            self.calls, self.fail_on = [], None
            with mock.patch.object(install_rccl, "config_for", return_value=_TARBALL_CFG), \
                 mock.patch.object(install_rccl, "exec_program", side_effect=self._fake_exec), \
                 mock.patch.object(install_rccl, "run_capture", return_value=(0, "4\n")), \
                 mock.patch.object(install_rccl, "log_error") as error:
                self.assertEqual(install_rccl.install({"DISTRIBUTION": "ubuntu22.04"}), 3)
        self.assertEqual(self.calls, [])
        self.assertIn("download/verify", error.call_args.args[1])

    def test_each_failing_command_stops_the_step(self):
        cases = [
            ("ubuntu24.04", _CLONE_CFG, self._source_commands(_CLONE_CFG) + _SYSCTL
             + [_rccl_tests(), ["apt", "install", "-y", "libpci-dev"]] + _perftest(self.work)),
            ("azurelinux3.0", None,
             [["dnf", "install", "-y", "rccl", "rccl-devel", "rccl-unittests"],
              _rccl_tests("/opt/rocm")]
             + _perftest(self.work)),
        ]
        for distro, cfg, commands in cases:
            record_after = 4 if cfg else 0     # RCCL recorded after make install / dnf
            for command in commands:
                # The first matching command fails (perftest's `make -j96` /
                # `make install` repeat RCCL's, so those stop at RCCL's build).
                index = commands.index(command)
                with self.subTest(distro=distro, command=command):
                    self.assertEqual(self._install(distro, cfg, fail_on=command), 3)
                    self.assertEqual(self._commands(), commands[:index + 1])
                    self.assertEqual(self.record.called, index > record_after)
                    self.error.assert_called()
                    self.assertFalse(os.path.exists(self.work))

    def test_missing_source_folder_returns_3(self):
        with mock.patch.object(self, "_make_source"):     # clone creates nothing
            self.assertEqual(self._install("ubuntu24.04", _CLONE_CFG), 3)
        self.assertNotIn("cmake", [c[0] for c in self._commands()])
        self.assertIn("was not created", self.error.call_args.args[1])

    def test_sysctl_conf_write_error_returns_3(self):
        os.makedirs(self.sysctl_conf)          # a directory can't be opened for append
        self.assertEqual(self._install("azurelinux3.0", None), 3)
        self.assertNotIn("bash", [c[0] for c in self._commands()])
        self.assertIn("file operation failed", self.error.call_args.args[1])

    def test_unwritable_versions_file_returns_3(self):
        rc = self._install("azurelinux3.0", None, write_error=PermissionError("denied"))
        self.assertEqual(rc, 3)
        self.assertFalse(self.sysctl_conf.exists())


class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="AMD", gpu="MI300X", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "install-rccl"]
        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, install_rccl.install)
        self.assertEqual(steps[0].script, "install_rccl.sh")


if __name__ == "__main__":
    unittest.main()
