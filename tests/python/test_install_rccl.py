"""Unit tests for components/python/install_rccl.py.

  * Pure helpers (folder naming, ref choice, sed edits, memory maths, rpm and
    CMake version parsing, the rccl-tests script and cmake arguments) are
    tested directly with real inputs. The expected values for the sed edits,
    memory maths, jq ref choice and awk version parsing were produced with
    real GNU sed/awk/jq/bash. Key lines are also compared with the bash script
    itself (components/install_rccl.sh), so an upstream change shows up here.
  * install() runs with every external effect mocked: config lookup, download,
    exec_program, run_capture (nproc, rpm, git rev-parse), write_repo_source_info
    and the versions file. A fake exec_program creates the directories that
    `git clone` / `git init` / `tar` would, so the real Python file handling
    (CMakeLists.txt edit, sysctl.conf append, cleanup) runs against a temporary
    directory. Nothing touches /etc, /opt, /proc or the network.

Run from the repo root:
    python3 -m unittest tests.python.test_install_rccl -v
"""

from __future__ import annotations

import os
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_rccl
from utils.build_config import BuildConfig, build_plan

_BASH = Path(__file__).resolve().parents[2] / "components" / "install_rccl.sh"

_NPROC = 96
_MEM_KB = 1584000000            # ~1510 GB -> 16 linker jobs (the cap)
_THEROCK_PIN = "0123456789abcdef0123456789abcdef01234567"

_CLONE_CFG = {   # ubuntu24.04 in versions.json
    "version": "2.22.3", "branch": "release/rocm-rel-6.4",
    "commit": "2f7ac66cd64c68d4af8bb4562ce193778a7e470e",
    "url": "https://github.com/ROCm/rccl/archive/refs/tags/rocm-6.4.4.tar.gz",
    "sha256": "d16a"}
_REF_CFG = {**_CLONE_CFG, "ref": "rocm-6.4.4"}      # new-style entry: ref wins
_TARBALL_CFG = {   # ubuntu22.04 in versions.json
    "version": "2.20.5",
    "url": "https://github.com/ROCm/rccl/archive/refs/tags/rocm-6.2.4.tar.gz",
    "sha256": "12a0"}
_ROCM_2604_CFG = {"version": "10.0"}

_CMAKELISTS = (
    "project(rccl)\n"
    "  target_compile_options(rccl PRIVATE -fgpu-rdc -parallel-jobs=12)\n"
    "  target_link_options(rccl PRIVATE -parallel-jobs=16)\n")

_U2604_CMAKE_ARGS = ["-DROCM_PATH=/opt/rocm/core", "-DGPU_TARGETS=gfx90a;gfx942;gfx1250",
                     "-DCMAKE_INSTALL_RPATH=/opt/rocm/lib",
                     "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"]


def _rccl_tests(prefix="/opt/rccl", args=("-DGPU_TARGETS=gfx90a;gfx942",)):
    # Called inside tests, so the script names the patched rccl-tests dir.
    return ["bash", "-c", install_rccl.rccl_tests_script(prefix, list(args))]


_SYSCTL = [["sysctl", "kernel.numa_balancing=0"], ["sysctl", "vm.max_map_count=1048576"]]
_ROCM_SYSTEMS_CLONE = ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse",
                       "https://github.com/ROCm/rocm-systems.git"]
_THEROCK_COMMANDS = [
    ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse",
     "--branch", "therock-10.0", "https://github.com/ROCm/TheRock.git"],
    ["git", "init", "rocm-systems"],
    ["git", "-C", "rocm-systems", "remote", "add", "origin",
     "https://github.com/ROCm/rocm-systems.git"],
    ["git", "-C", "rocm-systems", "sparse-checkout", "set", "projects/rccl-tests"],
    ["git", "-C", "rocm-systems", "fetch", "--depth=1", "--filter=blob:none",
     "origin", _THEROCK_PIN],
    ["git", "-C", "rocm-systems", "checkout", "--detach", "FETCH_HEAD"],
]
_LIBPCI = ["apt", "install", "-y", "libpci-dev"]
_LIBIBUMAD = ["apt", "install", "-y", "libibumad-dev"]


def _perftest():
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

    def test_rccl_ref_matches_jq(self):
        # (config, what `jq -r '.ref // .branch // empty'` prints), from real jq.
        cases = [({"ref": "v1", "branch": "b"}, "v1"), ({"branch": "b"}, "b"),
                 ({"ref": None, "branch": "b"}, "b"), ({"ref": "", "branch": "b"}, ""),
                 ({"ref": False, "branch": "b"}, "b"), ({"ref": "null"}, "null"),
                 ({}, ""), (None, ""), ({"branch": False}, ""), ({"ref": 1}, "1")]
        for config, expected in cases:
            with self.subTest(config=config):
                self.assertEqual(install_rccl.rccl_ref(config), expected)

    def test_uses_git_clone(self):
        self.assertTrue(install_rccl.uses_git_clone("release/rocm-rel-6.4"))
        for ref in (None, "", "null"):
            self.assertFalse(install_rccl.uses_git_clone(ref))

    def test_rccl_version_from_cmake_matches_awk(self):
        # (file text, what the bash awk prints), from real awk.
        cases = [
            ('x\nset(PACKAGE_VERSION "2.27.7")\nset(PACKAGE_VERSION "9.9.9")\n', "2.27.7"),
            (' set(PACKAGE_VERSION "1.2.3")\nset(PACKAGE_VERSION "")\n', ""),
            ('set(PACKAGE_VERSION "1.2.3\nset(PACKAGE_VERSION "4.5.6")\n', "1.2.3"),
            ('set(PACKAGE_VERSION "4.5.6" x "7")\r\n', "4.5.6"),
            ("set(PACKAGE_VERSION 1.2.3)\n", ""),
            ("", ""),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(install_rccl.rccl_version_from_cmake(text), expected)

    def test_is_release_version_matches_bash_regex(self):
        for version, ok in (("2.27.7", True), ("10.0.0", True), ("2.27", False),
                            ("2.27.7-1", False), ("", False), (" 1.2.3", False),
                            ("1.2.3\n", False)):
            with self.subTest(version=version):
                self.assertEqual(install_rccl.is_release_version(version), ok)

    def test_cmake_args_per_distro(self):
        self.assertEqual(install_rccl.rccl_tests_cmake_args("ubuntu24.04"),
                         ["-DGPU_TARGETS=gfx90a;gfx942"])
        self.assertEqual(install_rccl.rccl_tests_cmake_args("azurelinux3.0"),
                         ["-DGPU_TARGETS=gfx90a;gfx942"])
        self.assertEqual(install_rccl.rccl_tests_cmake_args("ubuntu26.04"), _U2604_CMAKE_ARGS)
        self.assertEqual(install_rccl.therock_branch("10.0"), "therock-10.0")
        self.assertEqual(install_rccl.rccl_source_cmake_command(), [
            "cmake", "-DCMAKE_PREFIX_PATH=/opt/rocm/", "-DCMAKE_INSTALL_PREFIX=/opt/rccl",
            "-DGPU_TARGETS=gfx90a;gfx942", "-DAMDGPU_TARGETS=gfx90a;gfx942", ".."])

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
        script = install_rccl.rccl_tests_script("/opt/rocm", _U2604_CMAKE_ARGS)
        self.assertTrue(script.startswith(
            "set -e\nsource /etc/profile.d/modules.sh\nmodule load mpi/hpcx\n"
            "pushd ./rocm-systems\n"))
        self.assertNotIn("git clone", script)          # cloned before, in Python
        self.assertIn('-DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm;$HPCX_MPI_DIR"', script)
        # Each extra argument stays one word; the ';' can't end the command.
        self.assertIn("    -DUSE_MPI=ON \\\n    -DROCM_PATH=/opt/rocm/core "
                      "'-DGPU_TARGETS=gfx90a;gfx942;gfx1250' "
                      "-DCMAKE_INSTALL_RPATH=/opt/rocm/lib "
                      "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \\\n    ..\n", script)
        self.assertIn("git sparse-checkout set projects/rccl-tests\n", script)
        self.assertIn("cp ./*_perf /opt/rccl-tests/\n", script)
        self.assertTrue(script.endswith("rm -rf rocm-systems\nmodule unload mpi/hpcx\n"))


class BashTextTests(unittest.TestCase):
    """Key values of the port, against the new bash."""

    def setUp(self):
        self.bash = _BASH.read_text(encoding="utf-8")

    def test_gpu_targets_and_cmake_args(self):
        self.assertIn(f'AMD_GPU_TARGETS="{install_rccl._AMD_GPU_TARGETS}"', self.bash)
        lines = [l.strip() for l in self.bash.split("\n")
                 if l.strip().startswith("RCCL_TEST_CMAKE_ARGS=(")]
        self.assertEqual(len(lines), 2)
        default = shlex.split(lines[0][len("RCCL_TEST_CMAKE_ARGS=("):-1].replace(
            "${AMD_GPU_TARGETS}", install_rccl._AMD_GPU_TARGETS))
        u2604 = shlex.split(lines[1][len("RCCL_TEST_CMAKE_ARGS=("):-1])
        self.assertEqual(install_rccl.rccl_tests_cmake_args("ubuntu24.04"), default)
        self.assertEqual(install_rccl.rccl_tests_cmake_args("ubuntu26.04"), u2604)
        self.assertIn('RCCL_TEST_GIT_ARGS=(--branch "therock-${rocm_version}")', self.bash)

    def test_key_lines(self):
        for line in (
                "awk -F '\"' '/^set\\(PACKAGE_VERSION \"/ {print $2; exit}' "
                + install_rccl._RCCL_CMAKE_VERSION_FILE,
                '[[ "$rccl_version" =~ ^[0-9]+\\.[0-9]+\\.[0-9]+$ ]]',
                "rccl_ref=$(jq -r '.ref // .branch // empty' <<< $rccl_metadata)",
                '-DGPU_TARGETS="${AMD_GPU_TARGETS}" -DAMDGPU_TARGETS="${AMD_GPU_TARGETS}" ..',
                'write_repo_source_info "RCCL" ./${rccl_folder}',
                'write_repo_source_info "RCCL_TESTS" rocm-systems',
                'write_repo_source_info "PERFTEST" perftest',
                'if [[ $DISTRIBUTION == "azurelinux3.0" || $DISTRIBUTION == "ubuntu26.04" ]]; then',
                "rccl_tests_commit=$(git -C TheRock rev-parse HEAD:rocm-systems)",
                '-DCMAKE_PREFIX_PATH="$RCCL_PREFIX;/opt/rocm;$HPCX_MPI_DIR"',
                "apt install -y libibumad-dev",
                install_rccl._THEROCK_GIT_URL):
            with self.subTest(line=line):
                self.assertIn(line, self.bash)


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
        self.cmake_version = root / "rccl-config-version.cmake"
        self.cmake_version.write_text(
            'set(PACKAGE_VERSION "2.27.7")\n\nif (PACKAGE_FIND_VERSION ...)\n',
            encoding="utf-8")
        for name, value in (("_WORK_DIR", self.work), ("_MEMINFO", str(self.meminfo)),
                            ("_SYSCTL_CONF", str(self.sysctl_conf)),
                            ("_RCCL_TESTS_DIR", str(self.tests_dir)),
                            ("_PERFTEST_PREFIX_DIR", str(root / "rocm-perftest")),
                            ("_RCCL_CMAKE_VERSION_FILE", str(self.cmake_version))):
            patcher = mock.patch.object(install_rccl, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.cmakelists_after = None

    # A fake exec_program that also does what the real command would do to the
    # work directory, so the Python file handling around it is exercised.
    def _fake_exec(self, command, op, *, cwd=None, env=None):
        self.calls.append((command, cwd, env))
        self.events.append(("exec", command[0]))
        if command == self.fail_on:
            return self.fail_rc
        if command[:2] == ["git", "clone"]:
            url = command[-1] if command[-1].startswith("https://") else command[-2]
            if url == install_rccl._RCCL_GIT_URL:
                self._make_source(os.path.join(cwd, command[-1]))
            elif url == install_rccl._THEROCK_GIT_URL:
                os.makedirs(os.path.join(cwd, "TheRock"))
            elif url == install_rccl._ROCM_SYSTEMS_GIT_URL:
                os.makedirs(os.path.join(cwd, "rocm-systems"))
            else:
                os.makedirs(os.path.join(cwd, "perftest"))
        elif command[:2] == ["git", "init"]:
            os.makedirs(os.path.join(cwd, command[2]))
        elif command[0] == "tar":
            self._make_source(os.path.join(cwd, "rccl-" + command[2][:-len(".tar.gz")]))
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
                 rpm=(0, "2.22.3-1.azl3"), rev_parse=(0, f"{_THEROCK_PIN}\n"),
                 write_error=None, repo_ok=True, rocm_cfg=_ROCM_2604_CFG):
        self.calls, self.events, self.fail_on, self.fail_rc = [], [], fail_on, fail_rc
        self.repo_dirs_existed = {}

        def fake_capture(command, op, *, cwd=None, env=None):
            if command == ["nproc"]:
                return nproc
            if command[:2] == ["git", "-C"]:
                self.events.append(("capture", "rev-parse"))
                return rev_parse
            return rpm

        def fake_download(url, sha256, dest_dir="."):
            path = Path(dest_dir) / Path(url).name
            path.write_bytes(b"tgz")
            return path

        def fake_config(component, env):
            return {"rccl": cfg, "rocm": rocm_cfg}[component]

        def fake_repo(component, repo_dir, op, *, commit=None, ref=None, env=None):
            self.events.append(("repo", component))
            self.repo_dirs_existed[component] = os.path.isdir(repo_dir)
            return repo_ok

        def fake_record(component, version):
            self.events.append(("version", component))
            if write_error:
                raise write_error

        env = {"DISTRIBUTION": distro, "PATH": "/usr/bin"}
        with mock.patch.object(install_rccl, "config_for", side_effect=fake_config) as config, \
             mock.patch.object(install_rccl, "download_and_verify",
                               side_effect=fake_download) as download, \
             mock.patch.object(install_rccl, "exec_program", side_effect=self._fake_exec), \
             mock.patch.object(install_rccl, "run_capture", side_effect=fake_capture) as cap, \
             mock.patch.object(install_rccl, "write_repo_source_info",
                               side_effect=fake_repo) as repo, \
             mock.patch.object(install_rccl, "write_component_version",
                               side_effect=fake_record) as record, \
             mock.patch.object(install_rccl, "log_info") as info, \
             mock.patch.object(install_rccl, "log_warn") as warn, \
             mock.patch.object(install_rccl, "log_error") as error:
            result = install_rccl.install(env)
        self.config, self.download, self.capture, self.record = config, download, cap, record
        self.repo = repo
        self.info, self.warn, self.error = info, warn, error
        return result

    def _commands(self):
        return [c[0] for c in self.calls]

    def _by_message(self):
        return {c.args[1]: c.kwargs for c in self.info.call_args_list}

    def _source_commands(self, cfg):
        folder = "rccl-" + install_rccl.tarball_name(cfg["url"])[:-len(".tar.gz")]
        ref = install_rccl.rccl_ref(cfg)
        if install_rccl.uses_git_clone(ref):
            fetch = [["git", "clone", "--branch", ref,
                      "https://github.com/ROCm/rccl.git", folder],
                     ["git", "checkout", cfg["commit"]]]
        else:
            fetch = [["tar", "-xzf", install_rccl.tarball_name(cfg["url"])]]
        return fetch + [
            ["cmake", "-DCMAKE_PREFIX_PATH=/opt/rocm/", "-DCMAKE_INSTALL_PREFIX=/opt/rccl",
             "-DGPU_TARGETS=gfx90a;gfx942", "-DAMDGPU_TARGETS=gfx90a;gfx942", ".."],
            ["make", f"-j{_NPROC}"],
            ["make", "install"],
        ]

    def _ubuntu_source_all(self, cfg):
        return (self._source_commands(cfg) + _SYSCTL
                + [_ROCM_SYSTEMS_CLONE, _rccl_tests(), _LIBPCI] + _perftest())

    def _u2604_all(self):
        return (_SYSCTL + _THEROCK_COMMANDS
                + [_rccl_tests("/opt/rocm", _U2604_CMAKE_ARGS), _LIBPCI, _LIBIBUMAD]
                + _perftest())

    def _azl_all(self):
        return ([["dnf", "install", "-y", "rccl", "rccl-devel", "rccl-unittests"],
                 _ROCM_SYSTEMS_CLONE, _rccl_tests("/opt/rocm")] + _perftest())

    # ---- success paths ------------------------------------------------------
    def test_ubuntu_git_clone_build(self):
        self.assertEqual(self._install("ubuntu24.04", _CLONE_CFG), 0)
        self.assertEqual(self._commands(), self._ubuntu_source_all(_CLONE_CFG))
        self.download.assert_not_called()
        # cwd of each command.
        source = os.path.join(self.work, "rccl-rocm-6.4.4")
        self.assertEqual(self.calls[0][1], self.work)
        self.assertEqual(self.calls[1][1], source)
        self.assertEqual(self.calls[2][1], os.path.join(source, "build"))
        self.assertEqual(self.calls[7][1], self.work)                  # rocm-systems clone
        self.assertEqual(self.calls[8][1], self.work)                  # rccl-tests script
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

    def test_repo_source_info_like_bash(self):
        self.assertEqual(self._install("ubuntu24.04", _CLONE_CFG), 0)
        self.assertEqual(self.repo.call_args_list, [
            mock.call("RCCL", os.path.join(self.work, "rccl-rocm-6.4.4"), "install-rccl",
                      commit=_CLONE_CFG["commit"], ref="release/rocm-rel-6.4", env=mock.ANY),
            mock.call("RCCL_TESTS", os.path.join(self.work, "rocm-systems"), "install-rccl",
                      commit=None, ref=None, env=mock.ANY),
            mock.call("PERFTEST", os.path.join(self.work, "perftest"), "install-rccl",
                      commit=None, ref=None, env=mock.ANY)])
        # Each checkout still exists when it is recorded.
        self.assertEqual(self.repo_dirs_existed,
                         {"RCCL": True, "RCCL_TESTS": True, "PERFTEST": True})
        # Order: RCCL repo after make install, before RCCL's version; RCCL_TESTS
        # right after the clone (before the build); PERFTEST right after its clone.
        events = [e for e in self.events if e[0] in ("repo", "version")
                  or e[1] in ("make", "bash", "./autogen.sh")]
        self.assertEqual(events, [
            ("exec", "make"), ("exec", "make"), ("repo", "RCCL"), ("version", "RCCL"),
            ("repo", "RCCL_TESTS"), ("exec", "bash"), ("repo", "PERFTEST"),
            ("exec", "./autogen.sh"), ("exec", "make"), ("exec", "make")])

    def test_ref_takes_precedence_over_branch(self):
        self.assertEqual(self._install("ubuntu24.04", _REF_CFG), 0)
        self.assertEqual(self._commands()[0][:4], ["git", "clone", "--branch", "rocm-6.4.4"])
        self.assertEqual(self.repo.call_args_list[0].kwargs["ref"], "rocm-6.4.4")
        self.assertEqual(self.info.call_args_list[0].kwargs["ref"], "rocm-6.4.4")

    def test_ubuntu_tarball_build_and_small_memory(self):
        self.meminfo.write_text("MemTotal:       32768000 kB\n")   # 31 GB -> 2 jobs
        self.assertEqual(self._install("ubuntu22.04", _TARBALL_CFG), 0)
        self.assertEqual(self._commands(), self._ubuntu_source_all(_TARBALL_CFG))
        self.download.assert_called_once_with(_TARBALL_CFG["url"], "12a0",
                                              dest_dir=self.work)
        self.assertIn("target_link_options(rccl PRIVATE -parallel-jobs=2)",
                      self.cmakelists_after)
        self.record.assert_called_once_with("RCCL", "2.20.5")
        # No RCCL repo info for a tarball, as in bash.
        self.assertEqual([c.args[0] for c in self.repo.call_args_list],
                         ["RCCL_TESTS", "PERFTEST"])

    def test_rhel_family_builds_from_source_without_ubuntu_extras(self):
        self.assertEqual(self._install("rocky9.8", _CLONE_CFG), 0)
        commands = self._commands()
        self.assertNotIn(_SYSCTL[0], commands)
        self.assertNotIn(_LIBPCI, commands)
        self.assertNotIn(_LIBIBUMAD, commands)
        self.assertIn(_rccl_tests(), commands)
        # Still persisted in sysctl.conf on every distro.
        self.assertEqual(self.sysctl_conf.read_text(),
                         "kernel.numa_balancing=0\nvm.max_map_count=1048576\n")

    def test_azure_linux_uses_packages(self):
        self.sysctl_conf.write_text("existing=1\n")
        self.assertEqual(self._install("azurelinux3.0", None), 0)
        self.config.assert_not_called()      # bash ignores the config here
        self.assertEqual(self._commands(), self._azl_all())
        self.capture.assert_any_call(
            ["rpm", "-q", "--queryformat", "%{VERSION}-%{RELEASE}", "rccl"],
            "install-rccl", env=mock.ANY)
        self.record.assert_called_once_with("RCCL", "2.22.3-1.azl3")
        self.assertEqual(self.sysctl_conf.read_text(),     # appended, not replaced
                         "existing=1\nkernel.numa_balancing=0\nvm.max_map_count=1048576\n")
        self.assertEqual([c.args[0] for c in self.repo.call_args_list],
                         ["RCCL_TESTS", "PERFTEST"])
        self.warn.assert_not_called()       # "existing=1" is not one of ours

    def test_ubuntu2604_uses_the_rocm_sdk_rccl_and_therock_pin(self):
        self.assertEqual(self._install("ubuntu26.04", None), 0)
        self.assertEqual(self._commands(), self._u2604_all())
        self.assertEqual([c.args[0] for c in self.config.call_args_list], ["rocm"])
        self.download.assert_not_called()
        self.record.assert_called_once_with("RCCL", "2.27.7")
        self.capture.assert_any_call(["git", "-C", "TheRock", "rev-parse", "HEAD:rocm-systems"],
                                     "install-rccl", cwd=self.work, env=mock.ANY)
        # TheRock clone and the rocm-systems commands run in the work dir.
        self.assertTrue(all(cwd == self.work for command, cwd, _ in self.calls
                            if command[0] in ("git", "bash") and "perftest" not in command[-1]))
        self.assertEqual([c.args[0] for c in self.repo.call_args_list],
                         ["RCCL_TESTS", "PERFTEST"])
        # sysctl applied now (Ubuntu) and persisted.
        self.assertEqual(self.sysctl_conf.read_text(),
                         "kernel.numa_balancing=0\nvm.max_map_count=1048576\n")
        self.assertFalse(os.path.exists(self.work))
        self.error.assert_not_called()

    def test_ubuntu2604_logs(self):
        self._install("ubuntu26.04", None)
        start = self.info.call_args_list[0]
        self.assertIn("Using the RCCL from the ROCm 10.0 SDK packages on ubuntu26.04",
                      start.args[1])
        self.assertEqual((start.kwargs["rccl_prefix"], start.kwargs["rccl_tests_cmake_args"]),
                         ("/opt/rocm", _U2604_CMAKE_ARGS))
        by_message = self._by_message()
        self.assertEqual(by_message["Packaged RCCL version is 2.27.7"]["version"], "2.27.7")
        pin = by_message[f"TheRock therock-10.0 pins rocm-systems at {_THEROCK_PIN}"]
        self.assertEqual(pin["rccl_tests_commit"], _THEROCK_PIN)
        self.assertIn("Ubuntu 26.04: perftest also needs libibumad-dev", by_message)
        self.assertEqual(by_message["Building rccl-tests from ROCm/rocm-systems "
                                    "(sparse clone) with HPC-X"]["cmake_args"],
                         _U2604_CMAKE_ARGS)

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
        self.assertEqual((start.kwargs["ref"], start.kwargs["gpu_targets"]),
                         ("release/rocm-rel-6.4", "gfx90a;gfx942"))
        by_message = self._by_message()
        self.assertIn("RCCL source: git clone of a pinned ref/commit "
                      "(works around ROCm/rccl#1877)", by_message)
        jobs = by_message["RCCL link parallelism: detected 1510 GB RAM -> -parallel-jobs=16"]
        self.assertEqual((jobs["mem_gb"], jobs["linker_jobs"], jobs["nproc"]), (1510, 16, 96))
        self.assertEqual(by_message["Building RCCL 2.22.3 into /opt/rccl for gfx90a;gfx942"]
                         ["gpu_targets"], "gfx90a;gfx942")
        self.assertIn("Recorded RCCL=2.22.3", by_message)
        self.assertEqual(by_message[f"Installed 1 rccl-tests binaries in {self.tests_dir}"]
                         ["binaries"], ["all_reduce_perf"])
        self.assertEqual(self.info.call_args_list[-1].args[1],
                         "RCCL 2.22.3 installed with rccl-tests and perftest")

    def test_tarball_logs_that_there_is_no_repo_info(self):
        self._install("ubuntu22.04", _TARBALL_CFG)
        self.assertIn("RCCL built from the release tarball; no repo commit/ref to "
                      "record, as in bash", self._by_message())

    # ---- failures -----------------------------------------------------------
    def test_bad_source_config_returns_3(self):
        cases = [None, {}, {"version": "1"},
                 {**_CLONE_CFG, "commit": None},
                 {**_REF_CFG, "commit": ""},
                 {**_TARBALL_CFG, "sha256": ""}]
        for cfg in cases:
            with self.subTest(cfg=cfg):
                self.assertEqual(self._install("ubuntu24.04", cfg), 3)
                self.assertEqual(self.calls, [])
                self.capture.assert_not_called()
                self.error.assert_called_once()

    def test_ubuntu2604_missing_rocm_version_returns_3(self):
        for rocm_cfg in (None, {}, {"version": ""}):
            with self.subTest(rocm_cfg=rocm_cfg):
                self.assertEqual(self._install("ubuntu26.04", None, rocm_cfg=rocm_cfg), 3)
                self.assertEqual(self.calls, [])
                self.capture.assert_not_called()
                self.assertIn("TheRock branch", self.error.call_args.args[1])

    def test_ubuntu2604_bad_rccl_version_returns_3(self):
        for content in (None, 'set(PACKAGE_VERSION "2.27")\n', "nothing\n"):
            with self.subTest(content=content):
                if content is None:
                    self.cmake_version.unlink(missing_ok=True)
                else:
                    self.cmake_version.write_text(content, encoding="utf-8")
                self.assertEqual(self._install("ubuntu26.04", None), 3)
                self.assertEqual(self.calls, [])
                self.record.assert_not_called()
                self.error.assert_called_once()

    def test_ubuntu2604_empty_or_failed_rev_parse_returns_3(self):
        for rev_parse in ((128, ""), (0, "\n")):
            with self.subTest(rev_parse=rev_parse):
                self.assertEqual(self._install("ubuntu26.04", None, rev_parse=rev_parse), 3)
                self.assertEqual(self._commands()[-1], _THEROCK_COMMANDS[0])
                self.assertIn("pinned by TheRock", self.error.call_args.args[1])
                self.assertFalse(os.path.exists(self.work))

    def test_repo_source_info_failure_returns_3(self):
        cases = [("ubuntu24.04", _CLONE_CFG, ["make", "install"]),
                 ("azurelinux3.0", None, _ROCM_SYSTEMS_CLONE)]
        for distro, cfg, last in cases:
            with self.subTest(distro=distro):
                self.assertEqual(self._install(distro, cfg, repo_ok=False), 3)
                self.assertEqual(self._commands()[-1], last)
                self.assertIn("could not record", self.error.call_args.args[1])

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
            self.calls, self.events, self.fail_on = [], [], None
            with mock.patch.object(install_rccl, "config_for", return_value=_TARBALL_CFG), \
                 mock.patch.object(install_rccl, "exec_program", side_effect=self._fake_exec), \
                 mock.patch.object(install_rccl, "run_capture", return_value=(0, "4\n")), \
                 mock.patch.object(install_rccl, "log_error") as error:
                self.assertEqual(install_rccl.install({"DISTRIBUTION": "ubuntu22.04"}), 3)
        self.assertEqual(self.calls, [])
        self.assertIn("download/verify", error.call_args.args[1])

    def test_each_failing_command_stops_the_step(self):
        cases = [
            # (distro, cfg, commands, index after which RCCL is recorded)
            ("ubuntu24.04", _CLONE_CFG, self._ubuntu_source_all(_CLONE_CFG), 4),
            ("azurelinux3.0", None, self._azl_all(), 0),
            ("ubuntu26.04", None, self._u2604_all(), -1),
        ]
        for distro, cfg, commands, record_after in cases:
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
