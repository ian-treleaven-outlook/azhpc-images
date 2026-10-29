"""Unit tests for components/python/install_nccl.py.

  * Pure helpers (names, paths, globs, job count, the two bash -c scripts) are
    tested directly. The `make -j 0` behaviour and the script syntax were
    checked with real GNU make/bash when the port was written.
  * install() runs with every external effect mocked: config lookup,
    exec_program, run_capture (nproc), dnf_pin_packages and the versions
    file. A fake exec_program creates what wget/tar/make pkg.redhat.build
    would, so the real Python file handling (glob expansion, cleanup) runs
    against a temporary directory. Nothing touches /tmp, /etc, /opt or the
    network.
  * The dependency commands themselves are tested in test_packaging.py
    (NcclDepsTests); here we check install() runs them exactly once.

Run from the repo root:
    python3 -m unittest tests.python.test_install_nccl -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_nccl
from utils.build_config import BuildConfig, build_plan

_V = "2.30.4-1"
_CUDA = "13.0"
_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_NCCL_CFG = {"version": _V, "rdmasharpplugins": {"commit": _COMMIT}}
_CUDA_CFG = {"driver": {"version": _CUDA}}
_NPROC = 96
_TARBALL = f"v{_V}.tar.gz"
_URL = f"https://github.com/NVIDIA/nccl/archive/refs/tags/{_TARBALL}"

_UBUNTU_DEPS = [
    ["apt", "install", "-y", "build-essential", "devscripts", "debhelper", "fakeroot",
     "patchelf"],
    ["apt", "install", "-y", "zlib1g-dev"]]
_AZL_DEPS = [["dnf", "install", "-y", "rpm-build", "rpmdevtools", "autoconf", "automake",
              "git", "libtool", "patchelf"]]
_RHEL_DEPS = [["dnf", "install", "-y", "rpm-build", "rpmdevtools", "patchelf"]]

_BUILD = [["wget", _URL], ["tar", "-xvf", _TARBALL], ["make", "-j", "95", "src.build"]]

_EXPECTED_SHARP_SCRIPT = """set -e
source /etc/profile.d/modules.sh
module load mpi/hpcx

if [[ -z "${HPCX_SHARP_DIR:-}" || ! -d "${HPCX_SHARP_DIR}" ]]; then
    echo "HPC-X module did not provide a valid HPCX_SHARP_DIR; cannot build NCCL RDMA SHARP plugin."
    exit 1
fi
if [[ -z "${HPCX_UCX_DIR:-}" || ! -d "${HPCX_UCX_DIR}" ]]; then
    echo "HPC-X module did not provide a valid HPCX_UCX_DIR; cannot build NCCL RDMA SHARP plugin."
    exit 1
fi
mkdir -p /usr/local/nccl-rdma-sharp-plugins
git clone https://github.com/Mellanox/nccl-rdma-sharp-plugins.git
pushd nccl-rdma-sharp-plugins
git checkout master
LIBTOOLIZE./autogen.sh
./configure --prefix=/usr/local/nccl-rdma-sharp-plugins --with-cuda=/usr/local/cuda
make
make install
cat > /etc/ld.so.conf.d/nccl-rdma-sharp-plugins.conf <<EOF
/usr/local/nccl-rdma-sharp-plugins/lib
EOF
ldconfig
popd
module unload mpi/hpcx
"""

_EXPECTED_TESTS_SCRIPT = """set -e
source /etc/profile.d/modules.sh
module load mpi/hpcx
git clone https://github.com/NVIDIA/nccl-tests.git
pushd nccl-tests
make MPI=1 MPI_HOME=${HPCX_MPI_DIR} CUDA_HOME=/usr/local/cuda
popd
mv nccl-tests /opt/.
module unload mpi/hpcx
"""


def _sharp(distro="ubuntu24.04", commit=_COMMIT):
    return ["bash", "-c", install_nccl.sharp_plugin_script(commit, distro)]


def _tests():
    # Called inside tests, so the script names the patched /opt dir.
    return ["bash", "-c", install_nccl.nccl_tests_script()]


class PureHelperTests(unittest.TestCase):
    def test_names_and_url(self):
        self.assertEqual(install_nccl.tarball_name(_V), _TARBALL)
        self.assertEqual(install_nccl.download_url(_V), _URL)

    def test_distro_family(self):
        for distro, family in (("ubuntu22.04", "ubuntu"), ("ubuntu24.04", "ubuntu"),
                               ("azurelinux3.0", "azurelinux"), ("almalinux9.8", "rhel"),
                               ("rocky8.10", "rhel"), ("", "rhel")):
            self.assertEqual(install_nccl.distro_family(distro), family)

    def test_make_jobs_is_nproc_minus_one_unclamped(self):
        self.assertEqual(install_nccl.make_jobs(96), 95)
        self.assertEqual(install_nccl.make_jobs(1), 0)        # bash: make -j 0

    def test_deb_packages(self):
        self.assertEqual(install_nccl.deb_packages(_V, _CUDA, "arm64"), [
            (f"libnccl2_{_V}+cuda{_CUDA}_arm64.deb", "libnccl2"),
            (f"libnccl-dev_{_V}+cuda{_CUDA}_arm64.deb", "libnccl-dev")])

    def test_azure_linux_rpm_patterns(self):
        for arch, rpm_arch in (("aarch64", "aarch64"), ("x86_64", "x86_64"),
                               ("", "x86_64")):
            with self.subTest(arch=arch):
                self.assertEqual(install_nccl.azure_linux_rpm_patterns(_V, arch), [
                    f"./build/pkg/rpm/{rpm_arch}/libnccl-{_V}+cuda*.{rpm_arch}.rpm",
                    f"./build/pkg/rpm/{rpm_arch}/libnccl-devel-{_V}+cuda*.{rpm_arch}.rpm",
                    f"./build/pkg/rpm/{rpm_arch}/libnccl-static-{_V}+cuda*.{rpm_arch}.rpm"])

    def test_rhel_rpm_paths(self):
        self.assertEqual(install_nccl.rhel_rpm_paths(_V, _CUDA), [
            f"./build/pkg/rpm/x86_64/libnccl-{_V}+cuda{_CUDA}.x86_64.rpm",
            f"./build/pkg/rpm/x86_64/libnccl-devel-{_V}+cuda{_CUDA}.x86_64.rpm",
            f"./build/pkg/rpm/x86_64/libnccl-static-{_V}+cuda{_CUDA}.x86_64.rpm"])

    def test_expand_glob_like_bash(self):
        with tempfile.TemporaryDirectory() as root:
            rpm_dir = Path(root, "build", "pkg", "rpm", "x86_64")
            rpm_dir.mkdir(parents=True)
            for name in ("libnccl-1+cuda13.1.x86_64.rpm", "libnccl-1+cuda12.9.x86_64.rpm",
                         "libnccl-devel-1+cuda13.1.x86_64.rpm"):
                (rpm_dir / name).touch()
            pattern = "./build/pkg/rpm/x86_64/libnccl-1+cuda*.x86_64.rpm"
            self.assertEqual(install_nccl.expand_glob(pattern, root), [
                "./build/pkg/rpm/x86_64/libnccl-1+cuda12.9.x86_64.rpm",
                "./build/pkg/rpm/x86_64/libnccl-1+cuda13.1.x86_64.rpm"])
            missing = "./build/pkg/rpm/x86_64/libnccl-2+cuda*.x86_64.rpm"
            self.assertEqual(install_nccl.expand_glob(missing, root), [missing])

    def test_is_commit_hash(self):
        self.assertTrue(install_nccl.is_commit_hash(_COMMIT))
        self.assertTrue(install_nccl.is_commit_hash("abc1234"))
        for ref in ("master", "v2.7.x", "abc12", "ABC1234"):
            self.assertFalse(install_nccl.is_commit_hash(ref))

    def test_bash_leftovers(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(install_nccl.bash_leftovers(root, _V), [])
            Path(root, "nccl-rdma-sharp-plugins").mkdir()
            Path(root, _TARBALL).touch()
            Path(root, "unrelated").mkdir()
            self.assertEqual(install_nccl.bash_leftovers(root, _V), [
                os.path.join(root, _TARBALL),
                os.path.join(root, "nccl-rdma-sharp-plugins")])

    def test_sharp_script_matches_bash(self):
        self.assertEqual(install_nccl.sharp_plugin_script("master", "ubuntu24.04"),
                         _EXPECTED_SHARP_SCRIPT.replace("LIBTOOLIZE", ""))
        self.assertEqual(install_nccl.sharp_plugin_script("master", "azurelinux3.0"),
                         _EXPECTED_SHARP_SCRIPT.replace("LIBTOOLIZE",
                                                        "libtoolize --verbose\n"))
        # Unusual refs are shell-quoted.
        self.assertIn("git checkout 'a b'\n",
                      install_nccl.sharp_plugin_script("a b", "ubuntu24.04"))

    def test_nccl_tests_script_matches_bash(self):
        self.assertEqual(install_nccl.nccl_tests_script(), _EXPECTED_TESTS_SCRIPT)


class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.work = str(root / "work")
        self.bash_tmp = root / "tmp"
        self.bash_tmp.mkdir()
        self.opt = root / "opt"
        self.ld_conf = root / "nccl-rdma-sharp-plugins.conf"
        for name, value in (("_WORK_DIR", self.work), ("_BASH_TMP_DIR", str(self.bash_tmp)),
                            ("_OPT_DIR", str(self.opt)), ("_LD_SO_CONF", str(self.ld_conf))):
            patcher = mock.patch.object(install_nccl, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.source = os.path.join(self.work, f"nccl-{_V}")

    # A fake exec_program that also does what the real command would do to the
    # work directory, so the Python file handling around it is exercised.
    def _fake_exec(self, command, op, *, cwd=None, env=None):
        self.calls.append((command, cwd, env))
        if command[0] == "wget":
            self.wget_in_empty_dir = os.listdir(cwd) == []
        if command == self.fail_on:
            return self.fail_rc
        if command[0] == "wget":
            Path(cwd, _TARBALL).write_bytes(b"tgz")
        elif command[0] == "tar" and self.tar_creates_source:
            os.makedirs(os.path.join(cwd, f"nccl-{_V}"))
        elif command == ["make", "pkg.redhat.build"]:
            for arch in ("x86_64", "aarch64"):
                rpm_dir = Path(cwd, "build", "pkg", "rpm", arch)
                rpm_dir.mkdir(parents=True)
                for name in self.rpm_names:
                    (rpm_dir / f"{name}-{_V}+cuda{_CUDA}.{arch}.rpm").touch()
        return 0

    def _install(self, distro="ubuntu24.04", arch="x86_64", arch_distro="amd64",
                 sku="A100", node_type="", nccl=_NCCL_CFG, cuda=_CUDA_CFG, *,
                 fail_on=None, fail_rc=2, nproc=(0, f"{_NPROC}\n"), pin_ok=True,
                 write_error=None, tar_creates_source=True,
                 rpm_names=("libnccl", "libnccl-devel", "libnccl-static")):
        self.calls, self.fail_on, self.fail_rc = [], fail_on, fail_rc
        self.tar_creates_source, self.rpm_names = tar_creates_source, rpm_names
        self.wget_in_empty_dir = None
        configs = {"nccl": nccl, "cuda": cuda}
        env = {"DISTRIBUTION": distro, "ARCHITECTURE": arch, "ARCHITECTURE_DISTRO": arch_distro,
               "SKU": sku, "TARGET_NODE_TYPE": node_type, "PATH": "/usr/bin"}
        self.env = env
        with mock.patch.object(install_nccl, "config_for",
                               side_effect=lambda c, e: configs[c]), \
             mock.patch.object(install_nccl, "exec_program", side_effect=self._fake_exec), \
             mock.patch.object(install_nccl, "run_capture", return_value=nproc) as cap, \
             mock.patch.object(install_nccl, "dnf_pin_packages", return_value=pin_ok) as pin, \
             mock.patch.object(install_nccl, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(install_nccl, "log_info") as info, \
             mock.patch.object(install_nccl, "log_warn") as warn, \
             mock.patch.object(install_nccl, "log_error") as error:
            result = install_nccl.install(env)
        self.capture, self.pin, self.record = cap, pin, record
        self.info, self.warn, self.error = info, warn, error
        return result

    def _commands(self):
        return [c[0] for c in self.calls]

    def _messages(self, logger):
        return {c.args[1]: c.kwargs for c in logger.call_args_list}

    @staticmethod
    def _ubuntu_commands(arch_distro="amd64", sharp=True):
        suffix = f"_{_V}+cuda{_CUDA}_{arch_distro}.deb"
        return (_UBUNTU_DEPS + _BUILD + [
            ["make", "pkg.debian.build"],
            ["dpkg", "-i", f"libnccl2{suffix}"], ["apt-mark", "hold", "libnccl2"],
            ["dpkg", "-i", f"libnccl-dev{suffix}"], ["apt-mark", "hold", "libnccl-dev"],
        ] + ([_sharp()] if sharp else []) + [_tests()])

    @staticmethod
    def _azl_commands(arch="aarch64"):
        return _AZL_DEPS + _BUILD + [["make", "pkg.redhat.build"]] + [
            ["dnf", "install", "-y", f"./build/pkg/rpm/{arch}/{n}-{_V}+cuda{_CUDA}.{arch}.rpm"]
            for n in ("libnccl", "libnccl-devel", "libnccl-static")
        ] + [_sharp("azurelinux3.0"), _tests()]

    @staticmethod
    def _rhel_commands():
        return _RHEL_DEPS + _BUILD + [["make", "pkg.redhat.build"]] + [
            ["rpm", "-i", f"./build/pkg/rpm/x86_64/{n}-{_V}+cuda{_CUDA}.x86_64.rpm"]
            for n in ("libnccl", "libnccl-devel", "libnccl-static")
        ] + [_sharp("almalinux9.8"), _tests()]

    # ---- success paths ------------------------------------------------------
    def test_ubuntu_build(self):
        self.assertEqual(self._install(), 0)
        self.assertEqual(self._commands(), self._ubuntu_commands())
        cwds = [c[1] for c in self.calls]
        deb_dir = os.path.join(self.source, "build", "pkg", "deb")
        self.assertEqual(cwds, [None, None, self.work, self.work, self.source, self.source]
                         + [deb_dir] * 4 + [self.work, self.work])
        self.assertTrue(all(c[2] is self.env for c in self.calls))
        self.capture.assert_called_once_with(["nproc"], "install-nccl", env=self.env)
        self.pin.assert_not_called()                 # Ubuntu holds with apt-mark
        self.assertEqual(self.record.call_args_list, [
            mock.call("NCCL-RDMA_SHARP_PLUGIN", _COMMIT), mock.call("NCCL", _V)])
        self.assertFalse(os.path.exists(self.work))  # all scratch removed
        self.error.assert_not_called()
        self.warn.assert_not_called()

    def test_dependencies_are_installed_exactly_once(self):
        for distro, deps in (("ubuntu22.04", _UBUNTU_DEPS), ("azurelinux3.0", _AZL_DEPS),
                             ("rocky8.10", _RHEL_DEPS)):
            with self.subTest(distro=distro):
                self.assertEqual(self._install(distro), 0)
                for command in deps:
                    self.assertEqual(self._commands().count(command), 1)
                self.assertEqual(self._commands()[:len(deps)], deps)

    def test_ubuntu_arm64_deb_names(self):
        self.assertEqual(self._install(arch="aarch64", arch_distro="arm64", sku="GB200"), 0)
        self.assertEqual(self._commands(), self._ubuntu_commands("arm64"))

    def test_azure_linux_aarch64(self):
        self.assertEqual(self._install("azurelinux3.0", arch="aarch64", cuda=None), 0)
        self.assertEqual(self._commands(), self._azl_commands("aarch64"))
        self.assertTrue(all(c[1] == self.source for c in self.calls[5:8]))
        self.pin.assert_called_once_with(["libnccl*"], "install-nccl", self.env)
        self.warn.assert_not_called()

    def test_azure_linux_x86_64(self):
        self.assertEqual(self._install("azurelinux3.0"), 0)
        self.assertEqual(self._commands(), self._azl_commands("x86_64"))

    def test_azure_linux_no_matching_rpm_passes_pattern_literally(self):
        rc = self._install("azurelinux3.0", fail_on=None, rpm_names=("libnccl", "libnccl-devel"))
        self.assertEqual(rc, 0)       # the mocked dnf "accepts" it; real dnf would fail
        pattern = f"./build/pkg/rpm/x86_64/libnccl-static-{_V}+cuda*.x86_64.rpm"
        self.assertIn(["dnf", "install", "-y", pattern], self._commands())
        self.warn.assert_called_once()
        self.assertEqual(self.warn.call_args.kwargs["pattern"], pattern)

    def test_rhel_family(self):
        self.assertEqual(self._install("almalinux9.8"), 0)
        self.assertEqual(self._commands(), self._rhel_commands())
        self.pin.assert_called_once_with(["libnccl*"], "install-nccl", self.env)
        self.warn.assert_not_called()

    def test_rhel_family_aarch64_warns_about_x86_64_paths(self):
        self.assertEqual(self._install("almalinux9.8", arch="aarch64"), 0)
        self.assertEqual(self._commands(), self._rhel_commands())   # still x86_64
        self.warn.assert_called_once()
        self.assertEqual(self.warn.call_args.kwargs["architecture"], "aarch64")

    # ---- SHARP plugin gating ------------------------------------------------
    def test_sharp_plugin_skipped_off_standard_ib(self):
        no_commit = {"version": _V}          # commit not needed when skipping
        for sku, node_type, mode in (("NCv6", "", "no_rdma"),
                                     ("GB200", "baremetal_1p", "mrc")):
            with self.subTest(sku=sku, node_type=node_type):
                self.assertEqual(self._install(sku=sku, node_type=node_type,
                                               nccl=no_commit), 0)
                self.assertEqual(self._commands(), self._ubuntu_commands(sharp=False))
                self.record.assert_called_once_with("NCCL", _V)
                decision = self._messages(self.info)[
                    f"Network mode {mode}: skipping the NCCL RDMA SHARP plugin "
                    "(no DOCA-OFED/SHARP/GPUDirect RDMA on this target)"]
                self.assertEqual(decision, {"network_mode": mode, "sharp": False})

    def test_sharp_plugin_built_on_baremetal_3p(self):
        self.assertEqual(self._install(node_type="baremetal_3p"), 0)
        self.assertIn(_sharp(), self._commands())
        self.assertIn({"network_mode": "standard_ib", "sharp": True},
                      list(self._messages(self.info).values()))

    def test_branch_name_commit_warns(self):
        cfg = {"version": _V, "rdmasharpplugins": {"commit": "master"}}
        self.assertEqual(self._install(nccl=cfg), 0)
        self.assertIn(_sharp(commit="master"), self._commands())
        self.warn.assert_called_once()
        self.assertEqual(self.warn.call_args.kwargs["commit"], "master")
        self.record.assert_any_call("NCCL-RDMA_SHARP_PLUGIN", "master")

    def test_ld_so_conf_created_or_replaced_logged(self):
        self._install()
        self.assertEqual(self._messages(self.info)[
            f"Created {self.ld_conf} (/usr/local/nccl-rdma-sharp-plugins/lib) and ran ldconfig"],
            {"path": str(self.ld_conf), "action": "created"})
        self.ld_conf.write_text("old\n")
        self._install()
        self.assertEqual(self._messages(self.info)[
            f"Replaced {self.ld_conf} (/usr/local/nccl-rdma-sharp-plugins/lib) and ran ldconfig"]
            ["action"], "replaced")

    # ---- bash quirks that warn ------------------------------------------------
    def test_one_cpu_runs_make_j_0_with_warning(self):
        self.assertEqual(self._install(nproc=(0, "1\n")), 0)
        self.assertIn(["make", "-j", "0", "src.build"], self._commands())
        self.warn.assert_called_once()
        self.assertEqual(self.warn.call_args.kwargs, {"nproc": 1, "jobs": 0})

    def test_bash_leftovers_in_tmp_warn_and_are_left_alone(self):
        (self.bash_tmp / "nccl-rdma-sharp-plugins").mkdir()
        self.assertEqual(self._install(), 0)
        self.warn.assert_called_once()
        self.assertEqual(self.warn.call_args.kwargs["paths"],
                         [str(self.bash_tmp / "nccl-rdma-sharp-plugins")])
        self.assertTrue((self.bash_tmp / "nccl-rdma-sharp-plugins").is_dir())

    def test_stale_work_dir_is_emptied_with_warning(self):
        os.makedirs(os.path.join(self.work, f"nccl-{_V}"))
        Path(self.work, _TARBALL).touch()
        self.assertEqual(self._install(), 0)
        self.assertTrue(self.wget_in_empty_dir)   # wget ran in an empty dir
        self.warn.assert_called_once()
        self.assertEqual(self.warn.call_args.kwargs, {"path": self.work})

    def test_existing_nccl_tests_dest_warns(self):
        (self.opt / "nccl-tests").mkdir(parents=True)
        self.assertEqual(self._install(), 0)
        self.warn.assert_called_once()
        self.assertEqual(self.warn.call_args.kwargs, {"path": str(self.opt / "nccl-tests")})

    # ---- logging ------------------------------------------------------------
    def test_key_log_lines(self):
        self._install()
        start = self.info.call_args_list[0]
        self.assertEqual(start.args[1], f"Installing NCCL {_V} (CUDA {_CUDA}) on ubuntu24.04 x86_64")
        for key, value in (("version", _V), ("cuda", _CUDA), ("sharp_commit", _COMMIT),
                           ("network_mode", "standard_ib"), ("url", _URL), ("sku", "A100")):
            self.assertEqual(start.kwargs[key], value)
        messages = self._messages(self.info)
        deps = messages["Installing NCCL build dependencies (ubuntu branch of install_nccl.sh)"]
        self.assertEqual(deps["packages"], ["build-essential", "devscripts", "debhelper",
                                            "fakeroot", "patchelf", "zlib1g-dev"])
        self.assertEqual(messages[f"Building NCCL {_V} (make -j 95 src.build)"]["jobs"], 95)
        self.assertIn(f"Installed libnccl2_{_V}+cuda{_CUDA}_amd64.deb and held libnccl2 "
                      "with apt-mark", messages)
        self.assertIn(f"Building the NCCL RDMA SHARP plugin at {_COMMIT} with HPC-X into "
                      "/usr/local/nccl-rdma-sharp-plugins", messages)
        self.assertIn(f"Recorded NCCL-RDMA_SHARP_PLUGIN={_COMMIT}", messages)
        self.assertIn(f"nccl-tests built and moved to {self.opt / 'nccl-tests'}", messages)
        self.assertIn(f"Recorded NCCL={_V}", messages)
        self.assertIn("Removed the NCCL tarball and source tree", messages)
        end = self.info.call_args_list[-1]
        self.assertEqual(end.args[1], f"NCCL {_V} installed (packages held), SHARP plugin "
                                      f"built, nccl-tests in {self.opt / 'nccl-tests'}")
        self.assertEqual(end.kwargs["sharp_plugin"], _COMMIT)

    # ---- failures -----------------------------------------------------------
    def test_missing_config_returns_3(self):
        cases = [
            dict(nccl=None),
            dict(nccl={"rdmasharpplugins": {"commit": _COMMIT}}),
            dict(nccl={"version": "null", "rdmasharpplugins": {"commit": _COMMIT}}),
            dict(cuda=None),                                   # ubuntu needs CUDA
            dict(distro="rocky8.10", cuda={"driver": {}}),     # so does RHEL
            dict(arch_distro=""),                              # .deb names
            dict(nccl={"version": _V}),                        # IB needs the commit
            dict(nccl={"version": _V, "rdmasharpplugins": {"commit": None}}),
        ]
        for kwargs in cases:
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                self.assertEqual(self._install(**kwargs), 3)
                self.assertEqual(self.calls, [])
                self.error.assert_called_once()

    def test_each_failing_command_stops_the_step(self):
        cases = [("ubuntu24.04", "x86_64", self._ubuntu_commands()),
                 ("azurelinux3.0", "aarch64", self._azl_commands()),
                 ("almalinux9.8", "x86_64", self._rhel_commands())]
        for distro, arch, commands in cases:
            sharp_index = commands.index(_sharp(distro))
            for index, command in enumerate(commands):
                with self.subTest(distro=distro, command=command[:3]):
                    self.assertEqual(self._install(distro, arch=arch, fail_on=command), 3)
                    self.assertEqual(self._commands(), commands[:index + 1])
                    # SHARP recorded only once its script succeeded; NCCL never.
                    self.assertEqual(self.record.called, index > sharp_index)
                    self.error.assert_called()
                    self.assertFalse(os.path.exists(self.work))

    def test_pin_failure_returns_3(self):
        self.assertEqual(self._install("almalinux9.8", pin_ok=False), 3)
        self.assertNotIn("bash", [c[0] for c in self._commands()])
        self.assertIn("libnccl*", self.error.call_args.args[1])

    def test_bad_nproc_returns_3(self):
        for result in ((1, ""), (0, "abc\n"), (0, "0\n")):
            with self.subTest(result=result):
                self.assertEqual(self._install(nproc=result), 3)
                self.assertEqual(self._commands(), _UBUNTU_DEPS + _BUILD[:2])

    def test_missing_source_folder_returns_3(self):
        self.assertEqual(self._install(tar_creates_source=False), 3)
        self.assertEqual(self._commands(), _UBUNTU_DEPS + _BUILD[:2])
        self.assertIn("was not created", self.error.call_args.args[1])

    def test_unwritable_versions_file_returns_3(self):
        rc = self._install(write_error=PermissionError("denied"))
        self.assertEqual(rc, 3)
        self.assertNotIn(_tests(), self._commands())     # failed recording SHARP
        self.assertIn("file operation failed", self.error.call_args.args[1])
        self.assertFalse(os.path.exists(self.work))


class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "install-nccl"]
        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, install_nccl.install)
        self.assertEqual(steps[0].script, "install_nccl.sh")


if __name__ == "__main__":
    unittest.main()
