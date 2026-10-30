"""Unit tests for components/python/install_nccl.py.

  * Pure helpers (names, paths, globs, job count, dpkg-query parsing, the
    bash -c scripts) are tested directly. make_jobs, apt_nccl_version and the
    rdmasharpplugins ref lookup were compared with real bash/sed/jq in WSL,
    and the scripts were syntax-checked with `bash -n`.
  * install() runs with every external effect mocked: config lookup,
    exec_program (where utils.process and install_packages() run commands,
    recorded in one ordered list), cpu_count, run_capture (dpkg-query), the
    download, dnf_pin_packages, write_repo_source_info and the versions file.
    A fake exec_program/download creates what the real commands would
    (tarball, source dir, rpms), so the real Python file handling (glob
    expansion, cleanup) runs against a temporary directory. Nothing touches
    /tmp, /etc, /opt or the network.
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
from utils import component_config, download, files, package_installer, process, repo_source
from utils.distro import Distro
from utils.download import ChecksumError
from utils.build_config import BuildConfig, build_plan

_V = "2.30.4-1"
_SHA = "fe12cb24"
_CUDA = "13.0"
_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_REF = "master"
_NCCL_CFG = {"version": _V, "sha256": _SHA,
             "rdmasharpplugins": {"branch": _REF, "commit": _COMMIT}}
_CUDA_CFG = {"driver": {"version": _CUDA}}
_NPROC = 96
_TARBALL = f"v{_V}.tar.gz"
_URL = f"https://github.com/NVIDIA/nccl/archive/refs/tags/{_TARBALL}"

# What install_packages() runs on Ubuntu / the RPM distros.
_APT = ["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0"]
_DNF = ["dnf", "install", "-y"]

_UBUNTU_DEPS = [
    [*_APT, "build-essential", "devscripts", "debhelper", "fakeroot", "patchelf"],
    [*_APT, "zlib1g-dev"]]
_UBUNTU2604_DEPS = [[*_APT, "build-essential", "patchelf", "zlib1g-dev"]]
_AZL_DEPS = [[*_DNF, "rpm-build", "rpmdevtools", "autoconf", "automake", "git", "libtool",
              "patchelf"]]
_RHEL_DEPS = [[*_DNF, "rpm-build", "rpmdevtools", "patchelf"]]

_BUILD = [["tar", "-xf", _TARBALL], ["make", "-j", "95", "src.build"]]
_APT_NCCL = [[*_APT, "libnccl2", "libnccl-dev"],
             ["apt-mark", "hold", "libnccl2"], ["apt-mark", "hold", "libnccl-dev"]]
_TESTS_CLONE = ["git", "clone", "https://github.com/NVIDIA/nccl-tests.git"]
_PURGE = ["apt-get", "purge", "-y", "patchelf"]

_EXPECTED_SHARP_CLONE = """set -e
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
git clone --branch v2.7.x https://github.com/Mellanox/nccl-rdma-sharp-plugins.git
pushd nccl-rdma-sharp-plugins
git checkout b246b19c
"""

_EXPECTED_SHARP_BUILD = """set -e
source /etc/profile.d/modules.sh
module load mpi/hpcx
pushd nccl-rdma-sharp-plugins
LIBTOOLIZE./autogen.sh
./configure --prefix=/usr/local/nccl-rdma-sharp-plugins --with-cuda=/usr/local/cuda
make
make install
cat > /etc/ld.so.conf.d/nccl-rdma-sharp-plugins.conf <<EOF
/usr/local/nccl-rdma-sharp-plugins/lib
EOF
ldconfig
popd
"""

_EXPECTED_TESTS_SCRIPT = """set -e
source /etc/profile.d/modules.sh
module load mpi/hpcx
pushd nccl-tests
make -j 7 MPI=1 MPI_HOME=${HPCX_MPI_DIR} CUDA_HOME=/usr/local/cuda
popd
mv nccl-tests /opt/.
"""

_HPCX_HEADER = "set -e\nsource /etc/profile.d/modules.sh\nmodule load mpi/hpcx\n"


def _script(lines):
    """The script run_script(lines, modules=["mpi/hpcx"]) runs with bash -c."""
    return _HPCX_HEADER + "\n".join(lines) + "\n"


def _sharp(distro="ubuntu24.04", ref=_REF, commit=_COMMIT):
    libtoolize = distro.startswith("azurelinux")
    return [["bash", "-c", _script(install_nccl.sharp_clone_lines(ref, commit))],
            ["bash", "-c", _script(install_nccl.sharp_build_lines(libtoolize))]]


def _tests(jobs=95):
    # Called inside tests, so the script names the patched /opt dir.
    return [_TESTS_CLONE, ["bash", "-c", _script(install_nccl.nccl_tests_lines(jobs))]]


class PureHelperTests(unittest.TestCase):
    def test_names_and_url(self):
        self.assertEqual(install_nccl.tarball_name(_V), _TARBALL)
        self.assertEqual(install_nccl.download_url(_V), _URL)

    def test_bash_branch(self):
        # Only Ubuntu 26.04 takes the apt branch; anything unrecognised takes
        # bash's `else` (the RHEL family).
        for distro, branch in (("ubuntu22.04", "ubuntu"), ("ubuntu24.04", "ubuntu"),
                               ("ubuntu26.04", "ubuntu26.04"),
                               ("azurelinux3.0", "azurelinux"), ("almalinux9.8", "rhel"),
                               ("rocky8.10", "rhel"), ("rhel9.8", "rhel"), ("", "rhel")):
            with self.subTest(distro=distro):
                self.assertEqual(install_nccl.bash_branch(Distro(distro)), branch)

    def test_make_jobs_leaves_one_core_but_at_least_one(self):
        self.assertEqual(install_nccl.make_jobs(96), 95)
        self.assertEqual(install_nccl.make_jobs(2), 1)
        self.assertEqual(install_nccl.make_jobs(1), 1)        # never make -j 0

    def test_apt_nccl_version_like_sed(self):
        self.assertEqual(install_nccl.apt_nccl_version("2.28.9-1+cuda13.0"), "2.28.9-1")
        self.assertEqual(install_nccl.apt_nccl_version("2.28.9-1+cuda13.0\n"), "2.28.9-1")
        self.assertEqual(install_nccl.apt_nccl_version("2.28.9-1"), "2.28.9-1")
        self.assertEqual(install_nccl.apt_nccl_version(""), "")
        self.assertEqual(install_nccl.apt_nccl_version("a+cudab\nc+cuda+cudad\n\n"), "a\nc")

    def test_jq_alternative(self):
        alt = install_nccl._jq_alternative
        self.assertEqual(alt("r", "b"), "r")
        self.assertEqual(alt(None, "b"), "b")
        self.assertEqual(alt(False, "b"), "b")
        self.assertEqual(alt("", "b"), "")          # "" is not null in jq
        self.assertIsNone(alt(None, None))

    def test_sharp_ref(self):
        self.assertEqual(install_nccl.sharp_ref({"ref": "r", "branch": "b"}), "r")
        self.assertEqual(install_nccl.sharp_ref({"branch": "b"}), "b")
        self.assertIsNone(install_nccl.sharp_ref({"ref": "", "branch": "b"}))  # "" wins
        self.assertIsNone(install_nccl.sharp_ref({}))

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

    def test_sharp_scripts_match_bash(self):
        self.assertEqual(_script(install_nccl.sharp_clone_lines("v2.7.x", "b246b19c")),
                         _EXPECTED_SHARP_CLONE)
        self.assertEqual(_script(install_nccl.sharp_build_lines(False)),
                         _EXPECTED_SHARP_BUILD.replace("LIBTOOLIZE", ""))
        self.assertEqual(_script(install_nccl.sharp_build_lines(True)),
                         _EXPECTED_SHARP_BUILD.replace("LIBTOOLIZE", "libtoolize --verbose\n"))
        # Unusual refs are shell-quoted.
        lines = install_nccl.sharp_clone_lines("a b", "c d")
        self.assertIn("git clone --branch 'a b' https://github.com/Mellanox/"
                      "nccl-rdma-sharp-plugins.git", lines)
        self.assertIn("git checkout 'c d'", lines)

    def test_nccl_tests_script_matches_bash(self):
        self.assertEqual(_script(install_nccl.nccl_tests_lines(7)), _EXPECTED_TESTS_SCRIPT)


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
        self.events.append(command)
        if command == self.fail_on:
            return self.fail_rc
        if command[0] == "tar" and self.tar_creates_source:
            os.makedirs(os.path.join(cwd, f"nccl-{_V}"))
        elif command == _TESTS_CLONE:
            os.makedirs(os.path.join(cwd, "nccl-tests"))
        elif command == ["make", "pkg.redhat.build"]:
            for arch in ("x86_64", "aarch64"):
                rpm_dir = Path(cwd, "build", "pkg", "rpm", arch)
                rpm_dir.mkdir(parents=True)
                for name in self.rpm_names:
                    (rpm_dir / f"{name}-{_V}+cuda{_CUDA}.{arch}.rpm").touch()
        return 0

    def _fake_download(self, url, sha256, dest_dir="."):
        self.events.append(("download", url, sha256))
        self.download_in_empty_dir = os.listdir(dest_dir) == []
        if self.download_error:
            raise self.download_error     # fetch_verified logs it and returns None
        path = Path(dest_dir, os.path.basename(url))
        path.write_bytes(b"tgz")
        return path

    def _fake_capture(self, command, op, *, env=None, failure_is_error=True):
        self.captures.append(command)
        return self.dpkg_query

    def _fake_source(self, component, repo_dir, op, *, commit=None, ref=None, env=None):
        self.events.append(("source", component, repo_dir, commit, ref))
        return self.source_ok

    def _fake_record(self, component, version):
        self.events.append(("version", component, version))
        if self.write_error:
            raise self.write_error

    def _install(self, distro="ubuntu24.04", arch="x86_64", arch_distro="amd64",
                 sku="A100", node_type="", nccl=_NCCL_CFG, cuda=_CUDA_CFG, *,
                 fail_on=None, fail_rc=2, nproc=_NPROC, pin_ok=True,
                 write_error=None, tar_creates_source=True, download_error=None,
                 dpkg_query=(0, "2.28.9-1+cuda13.0"), source_ok=True,
                 rpm_names=("libnccl", "libnccl-devel", "libnccl-static")):
        self.calls, self.events, self.captures = [], [], []
        self.fail_on, self.fail_rc = fail_on, fail_rc
        self.tar_creates_source, self.rpm_names = tar_creates_source, rpm_names
        self.nproc, self.dpkg_query, self.source_ok = nproc, dpkg_query, source_ok
        self.download_error, self.write_error = download_error, write_error
        self.download_in_empty_dir = None
        configs = {"nccl": nccl, "cuda": cuda}
        env = {"DISTRIBUTION": distro, "ARCHITECTURE": arch, "ARCHITECTURE_DISTRO": arch_distro,
               "SKU": sku, "TARGET_NODE_TYPE": node_type, "PATH": "/usr/bin"}
        self.env = env
        fake_config = lambda c, e: configs[c]     # noqa: E731
        with mock.patch.object(install_nccl, "config_for", side_effect=fake_config), \
             mock.patch.object(component_config, "config_for", side_effect=fake_config), \
             mock.patch.object(component_config, "log_error") as config_error, \
             mock.patch.object(process, "exec_program", side_effect=self._fake_exec), \
             mock.patch.object(package_installer, "exec_program", side_effect=self._fake_exec), \
             mock.patch.object(package_installer, "log_info"), \
             mock.patch.object(package_installer, "log_error") as pkg_error, \
             mock.patch.object(package_installer, "run_capture",
                               side_effect=self._fake_capture), \
             mock.patch.object(install_nccl, "cpu_count", return_value=nproc), \
             mock.patch.object(download, "download_and_verify",
                               side_effect=self._fake_download), \
             mock.patch.object(download, "log_error") as download_error, \
             mock.patch.object(install_nccl, "write_repo_source_info",
                               side_effect=self._fake_source) as source, \
             mock.patch.object(install_nccl, "dnf_pin_packages", return_value=pin_ok) as pin, \
             mock.patch.object(component_config, "write_component_version",
                               side_effect=self._fake_record) as record, \
             mock.patch.object(component_config, "log_info") as config_info, \
             mock.patch.object(files, "log_info"), \
             mock.patch.object(repo_source, "log_info"), \
             mock.patch.object(install_nccl, "log_info") as info, \
             mock.patch.object(install_nccl, "log_warn") as warn, \
             mock.patch.object(install_nccl, "log_error") as error:
            result = install_nccl.install(env)
        self.pin, self.record, self.source_mock = pin, record, source
        self.info, self.warn, self.error = info, warn, error
        self.config_error, self.download_error_log = config_error, download_error
        self.pkg_error, self.config_info = pkg_error, config_info
        return result

    def _commands(self):
        return [c[0] for c in self.calls]

    def _messages(self, logger):
        return {c.args[1]: c.kwargs for c in logger.call_args_list}

    @staticmethod
    def _ubuntu_commands(arch_distro="amd64", sharp=True, distro="ubuntu24.04"):
        suffix = f"_{_V}+cuda{_CUDA}_{arch_distro}.deb"
        return (_UBUNTU_DEPS + _BUILD + [
            ["make", "pkg.debian.build"],
            [*_APT, f"./libnccl2{suffix}"], ["apt-mark", "hold", "libnccl2"],
            [*_APT, f"./libnccl-dev{suffix}"],
            ["apt-mark", "hold", "libnccl-dev"],
        ] + (_sharp(distro) if sharp else []) + _tests()
            + ([_PURGE] if distro == "ubuntu22.04" else []))

    @staticmethod
    def _azl_commands(arch="aarch64"):
        return _AZL_DEPS + _BUILD + [["make", "pkg.redhat.build"]] + [
            [*_DNF, f"./build/pkg/rpm/{arch}/{n}-{_V}+cuda{_CUDA}.{arch}.rpm"]
            for n in ("libnccl", "libnccl-devel", "libnccl-static")
        ] + _sharp("azurelinux3.0") + _tests()

    @staticmethod
    def _rhel_commands(distro="almalinux9.8"):
        return _RHEL_DEPS + _BUILD + [["make", "pkg.redhat.build"]] + [
            ["rpm", "-i", f"./build/pkg/rpm/x86_64/{n}-{_V}+cuda{_CUDA}.x86_64.rpm"]
            for n in ("libnccl", "libnccl-devel", "libnccl-static")
        ] + _sharp(distro) + _tests()

    # ---- success paths ------------------------------------------------------
    def test_ubuntu_build(self):
        self.assertEqual(self._install(), 0)
        self.assertEqual(self._commands(), self._ubuntu_commands())
        cwds = [c[1] for c in self.calls]
        deb_dir = os.path.join(self.source, "build", "pkg", "deb")
        self.assertEqual(cwds, [None, None, self.work, self.source, self.source]
                         + [deb_dir] * 4 + [self.work] * 4)
        self.assertTrue(all(c[2] is self.env for c in self.calls))
        self.assertEqual(self.captures, [])          # nproc is cpu_count()
        self.pin.assert_not_called()                 # Ubuntu holds with apt-mark
        self.assertTrue(self.download_in_empty_dir)
        self.assertFalse(os.path.exists(self.work))  # all scratch removed
        self.error.assert_not_called()
        self.warn.assert_not_called()

    def test_download_is_checksummed(self):
        self.assertEqual(self._install(), 0)
        self.assertIn(("download", _URL, _SHA), self.events)
        self.assertLess(self.events.index(("download", _URL, _SHA)),
                        self.events.index(_BUILD[0]))

    def test_sources_and_versions_recorded_where_bash_records_them(self):
        self.assertEqual(self._install(), 0)
        sharp_dir = os.path.join(self.work, "nccl-rdma-sharp-plugins")
        tests_dir = os.path.join(self.work, "nccl-tests")
        clone, build = _sharp()
        tests_clone, tests_build = _tests()
        records = [e for e in self.events if isinstance(e, tuple) and e[0] != "download"]
        self.assertEqual(records, [
            ("source", "NCCL-RDMA_SHARP_PLUGIN", sharp_dir, _COMMIT, _REF),
            ("version", "NCCL-RDMA_SHARP_PLUGIN", _COMMIT),
            ("source", "NCCL_TESTS", tests_dir, None, None),
            ("version", "NCCL", _V)])
        # bash: checkout, record source, build; clone nccl-tests, record, make.
        order = [self.events.index(x) for x in (
            clone, records[0], build, records[1], tests_clone, records[2], tests_build,
            records[3])]
        self.assertEqual(order, sorted(order))

    def test_ref_from_ref_key_wins_over_branch(self):
        cfg = {"version": _V, "sha256": _SHA,
               "rdmasharpplugins": {"ref": "v2.7.x", "branch": "master", "commit": _COMMIT}}
        self.assertEqual(self._install(nccl=cfg), 0)
        self.assertIn(_sharp(ref="v2.7.x")[0], self._commands())

    def test_dependencies_are_installed_exactly_once(self):
        for distro, deps in (("ubuntu22.04", _UBUNTU_DEPS), ("azurelinux3.0", _AZL_DEPS),
                             ("rocky8.10", _RHEL_DEPS), ("ubuntu26.04", _UBUNTU2604_DEPS)):
            with self.subTest(distro=distro):
                self.assertEqual(self._install(distro), 0)
                for command in deps:
                    self.assertEqual(self._commands().count(command), 1)
                self.assertEqual(self._commands()[:len(deps)], deps)

    def test_ubuntu_arm64_deb_names(self):
        self.assertEqual(self._install(arch="aarch64", arch_distro="arm64", sku="GB200"), 0)
        self.assertEqual(self._commands(), self._ubuntu_commands("arm64"))

    def test_ubuntu22_04_purges_patchelf_after_recording(self):
        self.assertEqual(self._install("ubuntu22.04"), 0)
        self.assertEqual(self._commands(), self._ubuntu_commands(distro="ubuntu22.04"))
        self.assertLess(self.events.index(("version", "NCCL", _V)), self.events.index(_PURGE))
        self.assertIn("patchelf", self._messages(self.info)[
            "ubuntu22.04: purging patchelf (build-time only; USN-6036-1 is unfixed "
            "without Ubuntu Pro)"]["package"])

    def test_ubuntu26_04_installs_nccl_from_apt(self):
        self.assertEqual(self._install("ubuntu26.04", cuda=None, arch_distro=""), 0)
        self.assertEqual(self._commands(),
                         _UBUNTU2604_DEPS + _APT_NCCL + _sharp("ubuntu26.04") + _tests())
        self.assertNotIn("download", [e[0] for e in self.events if isinstance(e, tuple)])
        self.assertEqual(self.captures,
                         [["dpkg-query", "-W", "-f", "${Version}", "libnccl2"]])
        self.assertEqual(self.events[-1], ("version", "NCCL", "2.28.9-1"))
        self.pin.assert_not_called()
        self.warn.assert_not_called()
        self.assertEqual(self._messages(self.info)["Installed NCCL version is 2.28.9-1"]
                         ["version_source"], "dpkg-query")

    def test_ubuntu26_04_does_not_need_version_or_sha(self):
        cfg = {"rdmasharpplugins": {"branch": _REF, "commit": _COMMIT}}
        self.assertEqual(self._install("ubuntu26.04", nccl=cfg, cuda=None), 0)

    def test_ubuntu26_04_empty_dpkg_query_returns_3(self):
        for result in ((0, ""), (1, "")):
            with self.subTest(result=result):
                self.assertEqual(self._install("ubuntu26.04", dpkg_query=result), 3)
                self.assertEqual(self._commands(), _UBUNTU2604_DEPS + _APT_NCCL)
                self.assertIn("dpkg-query", self.pkg_error.call_args.args[1])
                self.record.assert_not_called()

    def test_azure_linux_aarch64(self):
        self.assertEqual(self._install("azurelinux3.0", arch="aarch64", cuda=None), 0)
        self.assertEqual(self._commands(), self._azl_commands("aarch64"))
        self.assertTrue(all(c[1] == self.source for c in self.calls[2:7]))
        self.pin.assert_called_once_with(["libnccl*"], "install-nccl", self.env)
        self.warn.assert_not_called()

    def test_azure_linux_x86_64(self):
        self.assertEqual(self._install("azurelinux3.0"), 0)
        self.assertEqual(self._commands(), self._azl_commands("x86_64"))

    def test_azure_linux_no_matching_rpm_passes_pattern_literally(self):
        rc = self._install("azurelinux3.0", rpm_names=("libnccl", "libnccl-devel"))
        self.assertEqual(rc, 0)       # the mocked dnf "accepts" it; real dnf would fail
        pattern = f"./build/pkg/rpm/x86_64/libnccl-static-{_V}+cuda*.x86_64.rpm"
        self.assertIn([*_DNF, pattern], self._commands())
        self.warn.assert_called_once()
        self.assertEqual(self.warn.call_args.kwargs["pattern"], pattern)

    def test_rhel_family(self):
        for distro in ("almalinux9.8", "rocky8.10", "rhel9.8", "rhel8.10"):
            with self.subTest(distro=distro):
                self.assertEqual(self._install(distro), 0)
                self.assertEqual(self._commands(), self._rhel_commands(distro))
                self.pin.assert_called_once_with(["libnccl*"], "install-nccl", self.env)
                self.warn.assert_not_called()

    def test_rhel_family_aarch64_warns_about_x86_64_paths(self):
        self.assertEqual(self._install("almalinux9.8", arch="aarch64"), 0)
        self.assertEqual(self._commands(), self._rhel_commands())   # still x86_64
        self.warn.assert_called_once()
        self.assertEqual(self.warn.call_args.kwargs["architecture"], "aarch64")

    # ---- SHARP plugin gating ------------------------------------------------
    def test_sharp_plugin_skipped_off_standard_ib(self):
        no_sharp = {"version": _V, "sha256": _SHA}    # ref/commit not needed when skipping
        for sku, node_type, mode in (("NCv6", "", "no_rdma"),
                                     ("GB200", "baremetal_1p", "mrc")):
            with self.subTest(sku=sku, node_type=node_type):
                self.assertEqual(self._install(sku=sku, node_type=node_type,
                                               nccl=no_sharp), 0)
                self.assertEqual(self._commands(), self._ubuntu_commands(sharp=False))
                self.assertEqual([e[1] for e in self.events
                                  if isinstance(e, tuple) and e[0] != "download"],
                                 ["NCCL_TESTS", "NCCL"])
                decision = self._messages(self.info)[
                    f"Network mode {mode}: skipping the NCCL RDMA SHARP plugin "
                    "(no DOCA-OFED/SHARP/GPUDirect RDMA on this target)"]
                self.assertEqual(decision, {"network_mode": mode, "sharp": False})

    def test_sharp_plugin_built_on_baremetal_3p(self):
        cfg = {"version": _V, "sha256": _SHA,
               "rdmasharpplugins": {"branch": "v2.7.x", "commit": "b246b19c"}}
        self.assertEqual(self._install(node_type="baremetal_3p", arch="aarch64",
                                       arch_distro="arm64", sku="GB200", nccl=cfg), 0)
        self.assertIn(_sharp(ref="v2.7.x", commit="b246b19c")[0], self._commands())
        self.assertIn({"network_mode": "standard_ib", "sharp": True},
                      list(self._messages(self.info).values()))

    def test_branch_name_commit_warns(self):
        cfg = {"version": _V, "sha256": _SHA,
               "rdmasharpplugins": {"branch": "master", "commit": "master"}}
        self.assertEqual(self._install(nccl=cfg), 0)
        self.assertIn(_sharp(commit="master")[0], self._commands())
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

    # ---- job count ------------------------------------------------------------
    def test_one_cpu_builds_with_make_j_1(self):
        self.assertEqual(self._install(nproc=1), 0)
        self.assertIn(["make", "-j", "1", "src.build"], self._commands())
        self.assertIn(_tests(jobs=1)[1], self._commands())
        self.warn.assert_not_called()

    # ---- bash quirks that warn ------------------------------------------------
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
        self.assertTrue(self.download_in_empty_dir)
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
        self.assertEqual(start.args[1], f"Installing NCCL {_V} on ubuntu24.04 x86_64")
        for key, value in (("version", _V), ("cuda", _CUDA), ("sharp_commit", _COMMIT),
                           ("sharp_ref", _REF), ("sha256", _SHA), ("source", "source"),
                           ("network_mode", "standard_ib"), ("url", _URL), ("sku", "A100")):
            self.assertEqual(start.kwargs[key], value)
        messages = self._messages(self.info)
        self.assertEqual(messages["Using make -j 95 (96 CPUs, one left free)"],
                         {"nproc": 96, "jobs": 95})
        deps = messages["Installing NCCL build dependencies (ubuntu branch of install_nccl.sh)"]
        self.assertEqual(deps["packages"], ["build-essential", "devscripts", "debhelper",
                                            "fakeroot", "patchelf", "zlib1g-dev"])
        self.assertEqual(messages[f"Building NCCL {_V} (make -j 95 src.build)"]["jobs"], 95)
        self.assertIn(f"Installed libnccl2_{_V}+cuda{_CUDA}_amd64.deb and held libnccl2 "
                      "with apt-mark", messages)
        self.assertIn(f"Building the NCCL RDMA SHARP plugin from {_REF} at {_COMMIT} with "
                      "HPC-X into /usr/local/nccl-rdma-sharp-plugins", messages)
        recorded = self._messages(self.config_info)
        self.assertIn(f"Recorded NCCL-RDMA_SHARP_PLUGIN {_COMMIT}", recorded)
        self.assertIn(f"nccl-tests built and moved to {self.opt / 'nccl-tests'}", messages)
        self.assertIn(f"Recorded NCCL {_V}", recorded)

        end = self.info.call_args_list[-1]
        self.assertEqual(end.args[1], f"NCCL {_V} installed (packages held), SHARP plugin "
                                      f"built, nccl-tests in {self.opt / 'nccl-tests'}")
        self.assertEqual(end.kwargs["sharp_plugin"], _COMMIT)

    # ---- failures -----------------------------------------------------------
    def test_missing_config_returns_3(self):
        cases = [
            dict(nccl=None),
            dict(nccl={"sha256": _SHA, "rdmasharpplugins": {"branch": _REF, "commit": _COMMIT}}),
            dict(nccl={"version": "null", "sha256": _SHA,
                       "rdmasharpplugins": {"branch": _REF, "commit": _COMMIT}}),
            dict(nccl={"version": _V, "rdmasharpplugins": {"branch": _REF, "commit": _COMMIT}}),
            dict(cuda=None),                                   # ubuntu needs CUDA
            dict(distro="rocky8.10", cuda={"driver": {}}),     # so does RHEL
            dict(arch_distro=""),                              # .deb names
            dict(nccl={"version": _V, "sha256": _SHA}),         # IB needs the commit
            dict(nccl={"version": _V, "sha256": _SHA,
                       "rdmasharpplugins": {"branch": _REF, "commit": None}}),
            dict(nccl={"version": _V, "sha256": _SHA,         # ... and the ref
                       "rdmasharpplugins": {"commit": _COMMIT}}),
            dict(distro="ubuntu26.04", nccl={"rdmasharpplugins": {"commit": _COMMIT}}),
        ]
        for kwargs in cases:
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                self.assertEqual(self._install(**kwargs), 3)
                self.assertEqual(self.calls, [])
                self.assertEqual(self.captures, [])
                # Logged once, by require_config or by install_nccl.
                self.assertEqual(self.error.call_count + self.config_error.call_count, 1)

    def test_each_failing_command_stops_the_step(self):
        cases = [("ubuntu24.04", "x86_64", self._ubuntu_commands()),
                 ("ubuntu22.04", "x86_64", self._ubuntu_commands(distro="ubuntu22.04")),
                 ("ubuntu26.04", "x86_64",
                  _UBUNTU2604_DEPS + _APT_NCCL + _sharp("ubuntu26.04") + _tests()),
                 ("azurelinux3.0", "aarch64", self._azl_commands()),
                 ("almalinux9.8", "x86_64", self._rhel_commands())]
        for distro, arch, commands in cases:
            sharp_build = commands.index(_sharp(distro)[1])
            for index, command in enumerate(commands):
                with self.subTest(distro=distro, command=command[:3]):
                    self.assertEqual(self._install(distro, arch=arch, fail_on=command), 3)
                    self.assertEqual(self._commands(), commands[:index + 1])
                    # SHARP recorded only once its build succeeded; NCCL only
                    # when everything but the purge succeeded.
                    versions = [e[1] for e in self.events
                                if isinstance(e, tuple) and e[0] == "version"]
                    self.assertEqual("NCCL-RDMA_SHARP_PLUGIN" in versions,
                                     index > sharp_build)
                    self.assertEqual("NCCL" in versions, command == _PURGE)
                    # (exec_program, mocked here, logs the failed command.)
                    self.assertFalse(os.path.exists(self.work))

    def test_download_failure_returns_3(self):
        self.assertEqual(self._install(download_error=ChecksumError("checksum mismatch")), 3)
        self.assertEqual(self._commands(), _UBUNTU_DEPS)
        self.assertIn("checksum mismatch", self.download_error_log.call_args.args[1])

    def test_repo_source_failure_returns_3(self):
        self.assertEqual(self._install(source_ok=False), 3)
        self.assertEqual(self._commands(), self._ubuntu_commands()[:10])   # SHARP clone
        self.source_mock.assert_called_once()     # (it logs its own git failure)
        self.record.assert_not_called()

    def test_pin_failure_returns_3(self):
        self.assertEqual(self._install("almalinux9.8", pin_ok=False), 3)
        self.assertNotIn("bash", [c[0] for c in self._commands()])
        self.pin.assert_called_once_with(["libnccl*"], "install-nccl", self.env)
        self.record.assert_not_called()           # (dnf_pin_packages logs the failure)

    def test_missing_source_folder_returns_3(self):
        self.assertEqual(self._install(tar_creates_source=False), 3)
        self.assertEqual(self._commands(), _UBUNTU_DEPS + _BUILD[:1])
        self.assertIn("was not created", self.error.call_args.args[1])

    def test_unwritable_versions_file_returns_3(self):
        rc = self._install(write_error=PermissionError("denied"))
        self.assertEqual(rc, 3)
        self.assertNotIn(_TESTS_CLONE, self._commands())     # failed recording SHARP
        self.assertIn("denied", self.config_error.call_args.args[1])   # record_version
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
