"""Unit tests for the Python packaging layer.

Everything that hits the real world here is mocked — downloads, exec_program
(apt/dnf/make/git), file writes — so the whole file runs in a couple seconds with
no network and no GPU. Most tests follow the same shape: fake the external calls,
run the component, then check the return code AND that it made the right calls
(for the fail-fast tests, that it stopped at the right point).

The negative tests patch the module's log_error/log_warn. That does two things:
it keeps the intentional error/warning out of the test output, and it lets us
assert the failure path actually ran (the error really got reported).

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import json
import os
import tarfile
import unittest
from unittest import mock

from utils import package_manager
from utils import package_installer
from utils.package_manager import PackageManager, detect_package_manager
from utils.package_installer import PackageInstaller
from components.python import install_mpifileutils, install_nccl, install_doca, install_cmake, install_libfabric, install_intel_libs, install_hpcdiag, install_monitoring_tools, install_cuda_samples, install_nvbandwidth_tool, install_aznfs


# detect_package_manager() finds a manager by calling shutil.which on each candidate.
# Faking which lets us pretend a box only has, say, tdnf — without touching the real PATH.
def _which(*available: str):
    """Build a shutil.which side-effect that only finds `available` names."""
    avail = set(available)
    return lambda name: f"/usr/bin/{name}" if name in avail else None


# Detection walks a fixed preference order (apt-get before apt, etc.) and returns the
# first one that's actually on PATH. Faking `which` is how we control what's "installed".
class DetectPackageManagerTests(unittest.TestCase):
    def test_prefers_apt_get_over_apt(self):
        with mock.patch.object(package_manager.shutil, "which",
                               side_effect=_which("apt-get", "apt")):
            pm = detect_package_manager()
        self.assertIsNotNone(pm)
        self.assertEqual(pm.name, "apt-get")

    def test_falls_back_to_apt_when_apt_get_absent(self):
        with mock.patch.object(package_manager.shutil, "which",
                               side_effect=_which("apt")):
            pm = detect_package_manager()
        self.assertEqual(pm.name, "apt")

    def test_picks_tdnf_on_azure_linux(self):
        with mock.patch.object(package_manager.shutil, "which",
                               side_effect=_which("tdnf")):
            pm = detect_package_manager()
        self.assertEqual(pm.name, "tdnf")

    def test_returns_none_when_nothing_found(self):
        with mock.patch.object(package_manager.shutil, "which",
                               side_effect=_which()), \
             mock.patch.object(package_manager, "log_error") as logged:
            pm = detect_package_manager()
        self.assertIsNone(pm)
        logged.assert_called_once()   # "no package manager found" is reported as an error


class InstallCommandTests(unittest.TestCase):
    def test_apt_get_disables_pty(self):
        pm = PackageManager(name="apt-get", path="/usr/bin/apt-get")
        self.assertEqual(
            pm.install_command("tree"),
            ["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0", "tree"],
        )

    def test_tdnf_command(self):
        pm = PackageManager(name="tdnf", path="/usr/bin/tdnf")
        self.assertEqual(pm.install_command("foo"),
                         ["tdnf", "install", "-y", "foo"])


# install_package loops over a list of packages, running each install via exec_program.
# We mock exec_program (return codes only), so nothing really installs and we can just
# check how the loop reacts: all ok, partial failure, single string, no manager.
class PackageInstallerTests(unittest.TestCase):
    def _installer_with(self, manager_name="apt-get"):
        # hand it a manager up front so it doesn't go detecting one off the real PATH
        pm = PackageManager(name=manager_name, path=f"/usr/bin/{manager_name}")
        return PackageInstaller(manager=pm)

    def test_all_succeed(self):
        installer = self._installer_with()
        with mock.patch("utils.package_installer.exec_program",
                        return_value=0) as exec_mock:
            ok = installer.install_package(["net-tools", "tree"])
        self.assertTrue(ok)
        self.assertEqual(exec_mock.call_count, 2)

    def test_partial_failure_returns_false_but_continues(self):
        installer = self._installer_with()
        # First package fails (rc=100), second succeeds (rc=0).
        with mock.patch("utils.package_installer.exec_program",
                        side_effect=[100, 0]) as exec_mock, \
             mock.patch.object(package_installer, "log_warn") as warned:
            ok = installer.install_package(["bad-pkg", "tree"])
        self.assertFalse(ok)
        self.assertEqual(exec_mock.call_count, 2)  # did not stop early
        warned.assert_called_once()   # the failed package is reported as a warning

    def test_single_string_is_accepted(self):
        installer = self._installer_with()
        with mock.patch("utils.package_installer.exec_program",
                        return_value=0) as exec_mock:
            ok = installer.install_package("tree")
        self.assertTrue(ok)
        self.assertEqual(exec_mock.call_count, 1)

    def test_no_manager_returns_false(self):
        installer = PackageInstaller(manager=None)
        with mock.patch("utils.package_installer.detect_package_manager",
                        return_value=None), \
             mock.patch.object(package_installer, "log_error") as logged:
            with mock.patch("utils.package_installer.exec_program") as exec_mock:
                ok = installer.install_package(["tree"])
        self.assertFalse(ok)
        exec_mock.assert_not_called()
        logged.assert_called_once()   # "no package manager detected" is reported as an error


# install_deps picks a different dependency list depending on the package manager
# (apt names vs rpm names). We give it a fake installer so we can assert *which* list
# it chose without doing any real installs.
class MpifileutilsDepsTests(unittest.TestCase):
    def _fake_installer(self, manager_name, succeeds=True):
        # stand-in PackageInstaller: .manager selects the branch, .install_package just
        # reports success/failure so we can inspect the chosen deps.
        fake = mock.Mock()
        fake.manager = (None if manager_name is None
                        else PackageManager(name=manager_name,
                                            path=f"/usr/bin/{manager_name}"))
        fake.install_package.return_value = succeeds
        return fake

    def test_selects_apt_dep_list(self):
        fake = self._fake_installer("apt-get")
        with mock.patch.object(install_mpifileutils, "PackageInstaller", return_value=fake):
            rc = install_mpifileutils.install_deps(env={})
        self.assertEqual(rc, 0)
        fake.install_package.assert_called_once_with(
            ["libbz2-dev", "libattr1-dev", "libarchive-dev", "libssl-dev", "libcap-dev"]
        )

    def test_selects_rpm_dep_list(self):
        fake = self._fake_installer("tdnf")
        with mock.patch.object(install_mpifileutils, "PackageInstaller", return_value=fake):
            rc = install_mpifileutils.install_deps(env={})
        self.assertEqual(rc, 0)
        fake.install_package.assert_called_once_with(
            ["bzip2-devel", "libattr-devel", "libarchive-devel"]
        )

    def test_returns_3_when_install_fails(self):
        fake = self._fake_installer("apt-get", succeeds=False)
        with mock.patch.object(install_mpifileutils, "PackageInstaller", return_value=fake):
            rc = install_mpifileutils.install_deps(env={})
        self.assertEqual(rc, 3)

    def test_returns_3_when_no_manager(self):
        fake = self._fake_installer(None)
        with mock.patch.object(install_mpifileutils, "PackageInstaller", return_value=fake):
            rc = install_mpifileutils.install_deps(env={})
        self.assertEqual(rc, 3)


class NcclDepsTests(unittest.TestCase):
    def _fake_installer(self, manager_name, succeeds=True):
        fake = mock.Mock()
        fake.manager = (None if manager_name is None
                        else PackageManager(name=manager_name,
                                            path=f"/usr/bin/{manager_name}"))
        fake.install_package.return_value = succeeds
        return fake

    def test_selects_apt_dep_list(self):
        fake = self._fake_installer("apt-get")
        with mock.patch.object(install_nccl, "PackageInstaller", return_value=fake):
            rc = install_nccl.install_deps(env={})
        self.assertEqual(rc, 0)
        fake.install_package.assert_called_once_with(
            ["build-essential", "devscripts", "debhelper", "fakeroot",
             "zlib1g-dev", "libibverbs-dev"]
        )

    def test_selects_rpm_dep_list(self):
        fake = self._fake_installer("tdnf")
        with mock.patch.object(install_nccl, "PackageInstaller", return_value=fake):
            rc = install_nccl.install_deps(env={})
        self.assertEqual(rc, 0)
        fake.install_package.assert_called_once_with(
            ["rpm-build", "rpmdevtools", "autoconf", "automake", "git", "libtool"]
        )

    def test_returns_3_when_no_manager(self):
        fake = self._fake_installer(None)
        with mock.patch.object(install_nccl, "PackageInstaller", return_value=fake):
            rc = install_nccl.install_deps(env={})
        self.assertEqual(rc, 3)


# Pure string parsing, no mocks needed — parse_openmpi_version digs the version out of
# `apt-cache show openmpi` output (mirrors the old awk one-liner).
class DocaVersionParseTests(unittest.TestCase):
    def test_parses_openmpi_version(self):
        sample = "Package: openmpi\nVersion: 4.1.5-1\nArchitecture: amd64\n"
        self.assertEqual(install_doca.parse_openmpi_version(sample), "4.1.5-1")

    def test_takes_first_version(self):
        # Mirrors awk '{...; exit}' — only the first Version line is used.
        sample = "Version: 4.1.5-1\nVersion: 5.0.0-2\n"
        self.assertEqual(install_doca.parse_openmpi_version(sample), "4.1.5-1")

    def test_missing_version_returns_none(self):
        self.assertIsNone(install_doca.parse_openmpi_version("Package: openmpi\n"))

    def test_blank_version_returns_none(self):
        self.assertIsNone(install_doca.parse_openmpi_version("Version: \n"))


class MpifileutilsConfigTests(unittest.TestCase):
    # The env dict is exactly what a component sees at runtime: COMPONENT_VERSIONS is the
    # parsed versions.json, and the rest (distro/arch/gpu/sku/node_type) is what
    # get_component_config uses to pick the right entry out of it.
    def _env(self, versions):
        return {
            "COMPONENT_VERSIONS": json.dumps(versions),
            "DISTRIBUTION": "ubuntu24.04",
            "ARCHITECTURE": "x86_64",
            "GPU": "NVIDIA",
            "SKU": "A100",
            "NODE_TYPE": "azure-vm",
        }

    def test_get_config_resolves_common_version(self):
        env = self._env({
            "mpifileutils": {"common": {"version": "0.12", "url": "u", "sha256": "s"}}
        })
        cfg = install_mpifileutils.config_for("mpifileutils", env)
        self.assertEqual(cfg["version"], "0.12")
        self.assertEqual(cfg["url"], "u")

    def test_get_config_missing_component_returns_none(self):
        env = self._env({})
        self.assertIsNone(install_mpifileutils.config_for("mpifileutils", env))


class MpifileutilsInstallTests(unittest.TestCase):
    def _env(self):
        return {
            "COMPONENT_VERSIONS": json.dumps({
                "mpifileutils": {"common": {
                    "version": "0.12",
                    "url": "https://example/mpifileutils-v0.12.tgz",
                    "sha256": "s",
                }}
            }),
            "DISTRIBUTION": "ubuntu24.04",
            "ARCHITECTURE": "x86_64",
            "GPU": "NVIDIA",
            "SKU": "A100",
            "NODE_TYPE": "azure-vm",
        }

    def test_install_orchestrates_all_steps(self):
        # Happy path. Stub out every real step — deps, download, the build subprocess,
        # the version write, and the tar/mkdir/cleanup calls — so nothing actually runs,
        # then confirm install() drove the key steps once each and returned 0.
        env = self._env()
        with mock.patch.object(install_mpifileutils, "install_deps", return_value=0) as deps, \
             mock.patch.object(install_mpifileutils, "download_and_verify",
                               return_value="/tmp/mpifileutils-src/mpifileutils-v0.12.tgz") as dl, \
             mock.patch.object(install_mpifileutils, "exec_program", return_value=0) as build, \
             mock.patch.object(install_mpifileutils, "write_component_version"), \
             mock.patch("tarfile.open"), \
             mock.patch("os.makedirs"), \
             mock.patch("shutil.rmtree"):
            rc = install_mpifileutils.install(env)
        self.assertEqual(rc, 0)
        deps.assert_called_once()
        dl.assert_called_once()
        build.assert_called_once()

    def test_install_fails_when_no_version(self):
        env = {"COMPONENT_VERSIONS": json.dumps({})}
        with mock.patch.object(install_mpifileutils, "log_error") as logged:
            self.assertEqual(install_mpifileutils.install(env), 3)
        logged.assert_called_once()   # missing version is reported as an error

    def test_install_fails_when_deps_fail(self):
        env = self._env()
        with mock.patch.object(install_mpifileutils, "install_deps", return_value=3), \
             mock.patch.object(install_mpifileutils, "download_and_verify") as dl, \
             mock.patch("os.makedirs"):
            rc = install_mpifileutils.install(env)
        self.assertEqual(rc, 3)
        dl.assert_not_called()  # stopped before downloading


class CmakeInstallTests(unittest.TestCase):
    def _env(self):
        return {
            "COMPONENT_VERSIONS": json.dumps({
                "cmake": {"common": {
                    "version": "4.3.1",
                    "url": "https://example/cmake-4.3.1-linux-x86_64.tar.gz",
                    "sha256": "s",
                }}
            }),
            "DISTRIBUTION": "ubuntu24.04",
            "ARCHITECTURE": "x86_64",
            "GPU": "NVIDIA",
            "SKU": "A100",
            "NODE_TYPE": "azure-vm",
        }

    def test_get_config_resolves_version(self):
        self.assertEqual(install_cmake.config_for("cmake", self._env())["version"], "4.3.1")

    def test_install_orchestrates_all_steps(self):
        env = self._env()
        with mock.patch.object(install_cmake, "download_and_verify",
                               return_value="/tmp/cmake-4.3.1-linux-x86_64.tar.gz") as dl, \
             mock.patch.object(install_cmake, "write_component_version") as wcv, \
             mock.patch("tarfile.open"), \
             mock.patch("shutil.copy") as cp, \
             mock.patch("glob.glob", return_value=[]), \
             mock.patch("shutil.rmtree"):
            rc = install_cmake.install(env)
        self.assertEqual(rc, 0)
        dl.assert_called_once()
        self.assertEqual(cp.call_count, 4)  # ccmake, cmake, cpack, ctest
        wcv.assert_called_once()

    def test_install_fails_when_no_version(self):
        with mock.patch.object(install_cmake, "log_error") as logged:
            self.assertEqual(
                install_cmake.install({"COMPONENT_VERSIONS": json.dumps({})}), 3)
        logged.assert_called_once()   # missing version is reported as an error


class LibfabricInstallTests(unittest.TestCase):
    def _env(self):
        return {
            "COMPONENT_VERSIONS": json.dumps({
                "libfabric": {"common": {
                    "version": "2.5.0",
                    "url": "https://example/libfabric-2.5.0.tar.bz2",
                    "sha256": "s",
                }}
            }),
            "DISTRIBUTION": "ubuntu24.04",
            "ARCHITECTURE": "x86_64",
            "GPU": "NVIDIA",
            "SKU": "NCv6",
            "NODE_TYPE": "azure-vm",
        }

    def test_get_config_resolves_version(self):
        self.assertEqual(install_libfabric.config_for("libfabric", self._env())["version"], "2.5.0")

    def test_install_runs_all_build_steps(self):
        env = self._env()
        with mock.patch.object(install_libfabric, "download_and_verify",
                               return_value="/tmp/libfabric-2.5.0.tar.bz2") as dl, \
             mock.patch.object(install_libfabric, "write_component_version") as wcv, \
             mock.patch.object(install_libfabric, "exec_program", return_value=0) as ex, \
             mock.patch("tarfile.open"), \
             mock.patch("shutil.rmtree"):
            rc = install_libfabric.install(env)
        self.assertEqual(rc, 0)
        dl.assert_called_once()
        self.assertEqual(ex.call_count, 3)  # configure, make, make install
        wcv.assert_called_once()

    def test_install_stops_on_build_failure(self):
        env = self._env()
        with mock.patch.object(install_libfabric, "download_and_verify",
                               return_value="/tmp/libfabric-2.5.0.tar.bz2"), \
             mock.patch.object(install_libfabric, "exec_program",
                               side_effect=[0, 1]) as ex, \
             mock.patch.object(install_libfabric, "write_component_version") as wcv, \
             mock.patch.object(install_libfabric, "log_error") as logged, \
             mock.patch("tarfile.open"), \
             mock.patch("shutil.rmtree"):
            rc = install_libfabric.install(env)
        self.assertEqual(rc, 3)
        self.assertEqual(ex.call_count, 2)  # stopped after make failed
        wcv.assert_not_called()
        logged.assert_called()   # build failure is reported as an error

    def test_install_fails_when_no_version(self):
        with mock.patch.object(install_libfabric, "log_error") as logged:
            self.assertEqual(
                install_libfabric.install({"COMPONENT_VERSIONS": json.dumps({})}), 3)
        logged.assert_called_once()   # missing version is reported as an error


class IntelLibsInstallTests(unittest.TestCase):
    def _env(self):
        return {
            "COMPONENT_VERSIONS": json.dumps({
                "intel_one_mkl": {"common": {
                    "version": "2025.3.1.11",
                    "url": "https://example/intel-onemkl-2025.3.1.11_offline.sh",
                    "sha256": "s",
                }}
            }),
            "DISTRIBUTION": "ubuntu24.04",
            "ARCHITECTURE": "x86_64",
            "GPU": "NVIDIA",
            "SKU": "A100",
            "NODE_TYPE": "azure-vm",
        }

    def test_get_config_resolves_version(self):
        self.assertEqual(
            install_intel_libs.config_for("intel_one_mkl", self._env())["version"], "2025.3.1.11")

    def test_install_runs_installer(self):
        env = self._env()
        with mock.patch.object(install_intel_libs, "download_and_verify",
                               return_value="/tmp/intel-onemkl-2025.3.1.11_offline.sh") as dl, \
             mock.patch.object(install_intel_libs, "exec_program", return_value=0) as ex, \
             mock.patch.object(install_intel_libs, "write_component_version") as wcv:
            rc = install_intel_libs.install(env)
        self.assertEqual(rc, 0)
        dl.assert_called_once()
        ex.assert_called_once()
        wcv.assert_called_once()

    def test_install_fails_when_installer_fails(self):
        env = self._env()
        with mock.patch.object(install_intel_libs, "download_and_verify",
                               return_value="/tmp/x.sh"), \
             mock.patch.object(install_intel_libs, "exec_program", return_value=1), \
             mock.patch.object(install_intel_libs, "write_component_version") as wcv, \
             mock.patch.object(install_intel_libs, "log_error") as logged:
            rc = install_intel_libs.install(env)
        self.assertEqual(rc, 3)
        wcv.assert_not_called()
        logged.assert_called()   # installer failure is reported as an error

    def test_install_fails_when_no_version(self):
        with mock.patch.object(install_intel_libs, "log_error") as logged:
            self.assertEqual(
                install_intel_libs.install({"COMPONENT_VERSIONS": json.dumps({})}), 3)
        logged.assert_called_once()   # missing version is reported as an error


class HpcdiagTests(unittest.TestCase):
    def test_latest_tarball_url_parses(self):
        api = json.dumps({"name": "r1",
                          "tarball_url": "https://api.github.com/x/tarball/hpcdiag-1"})
        self.assertEqual(install_hpcdiag.latest_tarball_url(api),
                         "https://api.github.com/x/tarball/hpcdiag-1")

    def test_latest_tarball_url_missing_returns_none(self):
        self.assertIsNone(install_hpcdiag.latest_tarball_url(json.dumps({"name": "r1"})))

    def test_install_orchestrates(self):
        api_json = json.dumps({"tarball_url": "https://x/tarball/hpcdiag-1"}).encode()
        resp_cm = mock.MagicMock()
        resp_cm.__enter__.return_value.read.return_value = api_json
        with mock.patch.object(install_hpcdiag.urllib.request, "urlopen", return_value=resp_cm), \
             mock.patch.object(install_hpcdiag, "download") as dl, \
             mock.patch("tarfile.open") as taropen, \
             mock.patch("shutil.copy") as cp, \
             mock.patch("shutil.rmtree"), \
             mock.patch("pathlib.Path.mkdir"), \
             mock.patch("pathlib.Path.unlink"):
            taropen.return_value.__enter__.return_value.getnames.return_value = ["topdir/README"]
            rc = install_hpcdiag.install(env={})
        self.assertEqual(rc, 0)
        dl.assert_called_once()
        cp.assert_called_once()

    def test_install_fails_when_no_tarball_url(self):
        api_json = json.dumps({"name": "r1"}).encode()
        resp_cm = mock.MagicMock()
        resp_cm.__enter__.return_value.read.return_value = api_json
        with mock.patch.object(install_hpcdiag.urllib.request, "urlopen", return_value=resp_cm), \
             mock.patch.object(install_hpcdiag, "download") as dl, \
             mock.patch.object(install_hpcdiag, "log_error") as logged:
            rc = install_hpcdiag.install(env={})
        self.assertEqual(rc, 3)
        dl.assert_not_called()
        logged.assert_called()   # missing tarball_url is reported as an error


class MonitoringToolsTests(unittest.TestCase):
    def _env(self):
        return {
            "COMPONENT_VERSIONS": json.dumps({
                "moneo": {"common": {"version": "1.2.3", "sha256": "s"}}
            }),
            "DISTRIBUTION": "ubuntu24.04",
            "ARCHITECTURE": "x86_64",
            "GPU": "NVIDIA",
            "SKU": "A100",
            "NODE_TYPE": "azure-vm",
        }

    def test_get_config_resolves_version(self):
        self.assertEqual(
            install_monitoring_tools.config_for("moneo", self._env())["version"], "1.2.3")

    def test_extract_stripped_drops_top_level_dir(self):
        import io
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tar_path = os.path.join(tmp, "moneo.tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                data = b"hi"
                info = tarfile.TarInfo("Moneo-1.2.3/moneo.py")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            dest = os.path.join(tmp, "Moneo")
            os.makedirs(dest)
            install_monitoring_tools.extract_stripped(tar_path, dest)
            self.assertTrue(os.path.exists(os.path.join(dest, "moneo.py")))
            self.assertFalse(os.path.exists(os.path.join(dest, "Moneo-1.2.3")))

    def test_ensure_line_appends_when_missing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bashrc")
            self.assertTrue(install_monitoring_tools.ensure_line(path, "alias x='y'"))
            with open(path) as handle:
                self.assertEqual(handle.read(), "alias x='y'\n")

    def test_ensure_line_skips_when_present(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bashrc")
            with open(path, "w") as handle:
                handle.write("alias x='y'\n")
            self.assertFalse(install_monitoring_tools.ensure_line(path, "alias x='y'"))

    def test_install_orchestrates(self):
        env = self._env()
        with mock.patch.object(install_monitoring_tools, "install_deps", return_value=0), \
             mock.patch.object(install_monitoring_tools, "download_and_verify",
                               return_value="/opt/azurehpc/tools/v1.2.3.tar.gz") as dl, \
             mock.patch.object(install_monitoring_tools, "extract_stripped") as ex, \
             mock.patch.object(install_monitoring_tools, "exec_program", return_value=0), \
             mock.patch.object(install_monitoring_tools, "ensure_line") as el, \
             mock.patch.object(install_monitoring_tools, "write_component_version") as wcv, \
             mock.patch("os.makedirs"), \
             mock.patch("os.chmod"), \
             mock.patch("os.remove"):
            rc = install_monitoring_tools.install(env)
        self.assertEqual(rc, 0)
        dl.assert_called_once()
        ex.assert_called_once()
        el.assert_called_once()
        wcv.assert_called_once()

    def test_install_fails_when_configure_fails(self):
        env = self._env()
        with mock.patch.object(install_monitoring_tools, "install_deps", return_value=0), \
             mock.patch.object(install_monitoring_tools, "download_and_verify",
                               return_value="/opt/azurehpc/tools/v1.2.3.tar.gz"), \
             mock.patch.object(install_monitoring_tools, "extract_stripped"), \
             mock.patch.object(install_monitoring_tools, "exec_program", return_value=1), \
             mock.patch.object(install_monitoring_tools, "write_component_version") as wcv, \
             mock.patch.object(install_monitoring_tools, "log_error") as logged, \
             mock.patch("os.makedirs"), \
             mock.patch("os.chmod"):
            rc = install_monitoring_tools.install(env)
        self.assertEqual(rc, 3)
        wcv.assert_not_called()
        logged.assert_called()   # configure failure is reported as an error

    def test_install_fails_when_no_version(self):
        with mock.patch.object(install_monitoring_tools, "log_error") as logged:
            self.assertEqual(
                install_monitoring_tools.install({"COMPONENT_VERSIONS": json.dumps({})}), 3)
        logged.assert_called_once()   # missing version is reported as an error


class CudaSamplesTests(unittest.TestCase):
    def _env(self):
        return {
            "COMPONENT_VERSIONS": json.dumps({
                "cuda": {"common": {
                    "driver": {"version": "12.4"},
                    "samples": {"version": "12.4.1", "sha256": "s"},
                }}
            }),
            "DISTRIBUTION": "ubuntu24.04",
            "ARCHITECTURE": "x86_64",
            "GPU": "NVIDIA",
            "SKU": "A100",
            "NODE_TYPE": "azure-vm",
        }

    def test_get_config_resolves_samples_version(self):
        self.assertEqual(
            install_cuda_samples.config_for("cuda", self._env())["samples"]["version"], "12.4.1")

    def test_install_runs_all_build_steps(self):
        env = self._env()
        with mock.patch.object(install_cuda_samples, "download_and_verify",
                               return_value="/tmp/cuda-samples-build/v12.4.1.tar.gz") as dl, \
             mock.patch.object(install_cuda_samples, "exec_program", return_value=0) as ex, \
             mock.patch("tarfile.open"), \
             mock.patch("os.makedirs"), \
             mock.patch("os.path.exists", return_value=False), \
             mock.patch("shutil.move") as mv, \
             mock.patch("shutil.rmtree"):
            rc = install_cuda_samples.install(env)
        self.assertEqual(rc, 0)
        dl.assert_called_once()
        self.assertEqual(ex.call_count, 2)  # cmake, make
        mv.assert_called_once()

    def test_install_stops_on_build_failure(self):
        env = self._env()
        with mock.patch.object(install_cuda_samples, "download_and_verify",
                               return_value="/tmp/cuda-samples-build/v12.4.1.tar.gz"), \
             mock.patch.object(install_cuda_samples, "exec_program",
                               side_effect=[0, 1]) as ex, \
             mock.patch.object(install_cuda_samples, "log_error") as logged, \
             mock.patch("tarfile.open"), \
             mock.patch("os.makedirs"), \
             mock.patch("shutil.move") as mv, \
             mock.patch("shutil.rmtree"):
            rc = install_cuda_samples.install(env)
        self.assertEqual(rc, 3)
        self.assertEqual(ex.call_count, 2)  # stopped after make failed
        mv.assert_not_called()
        logged.assert_called()   # build failure is reported as an error

    def test_install_fails_when_no_version(self):
        with mock.patch.object(install_cuda_samples, "log_error") as logged:
            self.assertEqual(
                install_cuda_samples.install({"COMPONENT_VERSIONS": json.dumps({})}), 3)
        logged.assert_called_once()   # missing version is reported as an error


class NvbandwidthTests(unittest.TestCase):
    def _env(self):
        return {
            "COMPONENT_VERSIONS": json.dumps({
                "nvbandwidth": {"common": {
                    "version": "0.9",
                    "url": "https://github.com/NVIDIA/nvbandwidth.git",
                }}
            }),
            "DISTRIBUTION": "ubuntu24.04",
            "ARCHITECTURE": "x86_64",
            "GPU": "NVIDIA",
            "SKU": "GB200",
            "NODE_TYPE": "azure-vm",
        }

    def test_get_config_resolves_version(self):
        self.assertEqual(
            install_nvbandwidth_tool.config_for("nvbandwidth", self._env())["version"], "0.9")

    def test_install_runs_clone_and_build(self):
        env = self._env()
        with mock.patch.object(install_nvbandwidth_tool, "install_deps", return_value=0), \
             mock.patch.object(install_nvbandwidth_tool, "exec_program", return_value=0) as ex, \
             mock.patch.object(install_nvbandwidth_tool, "write_component_version") as wcv, \
             mock.patch("os.makedirs"), \
             mock.patch("os.path.exists", return_value=False), \
             mock.patch("shutil.move") as mv, \
             mock.patch("shutil.rmtree"):
            rc = install_nvbandwidth_tool.install(env)
        self.assertEqual(rc, 0)
        self.assertEqual(ex.call_count, 3)  # git clone, cmake, make
        mv.assert_called_once()
        wcv.assert_called_once()

    def test_install_stops_on_clone_failure(self):
        env = self._env()
        with mock.patch.object(install_nvbandwidth_tool, "install_deps", return_value=0), \
             mock.patch.object(install_nvbandwidth_tool, "exec_program",
                               side_effect=[1]) as ex, \
             mock.patch.object(install_nvbandwidth_tool, "write_component_version") as wcv, \
             mock.patch.object(install_nvbandwidth_tool, "log_error") as logged, \
             mock.patch("os.makedirs"), \
             mock.patch("shutil.rmtree"):
            rc = install_nvbandwidth_tool.install(env)
        self.assertEqual(rc, 3)
        self.assertEqual(ex.call_count, 1)  # stopped after clone failed
        wcv.assert_not_called()
        logged.assert_called()   # clone failure is reported as an error

    def test_install_fails_when_no_version(self):
        with mock.patch.object(install_nvbandwidth_tool, "log_error") as logged:
            self.assertEqual(
                install_nvbandwidth_tool.install({"COMPONENT_VERSIONS": json.dumps({})}), 3)
        logged.assert_called_once()   # missing version is reported as an error


class AznfsTests(unittest.TestCase):
    def test_install_ubuntu_uses_package_manager(self):
        with mock.patch.object(install_aznfs, "PackageInstaller") as PI:
            PI.return_value.manager = object()
            PI.return_value.install_package.return_value = True
            rc = install_aznfs.install({"DISTRIBUTION": "ubuntu24.04"})
        self.assertEqual(rc, 0)
        PI.return_value.install_package.assert_called_once_with(["aznfs"])

    def test_install_fails_without_package_manager(self):
        with mock.patch.object(install_aznfs, "PackageInstaller") as PI:
            PI.return_value.manager = None
            rc = install_aznfs.install({"DISTRIBUTION": "ubuntu24.04"})
        self.assertEqual(rc, 3)

    def test_install_azurelinux_runs_patched_installer(self):
        env = {"DISTRIBUTION": "azurelinux3.0",
               "COMPONENT_VERSIONS": json.dumps(
                   {"aznfs": {"common": {"version": "2.0", "sha256": "s"}}})}
        with mock.patch.object(install_aznfs, "download_and_verify",
                               return_value="/tmp/aznfs_install.sh") as dl, \
             mock.patch("pathlib.Path.read_text", return_value="yum install x"), \
             mock.patch("pathlib.Path.write_text") as wt, \
             mock.patch.object(install_aznfs, "exec_program", return_value=0) as ex:
            rc = install_aznfs.install(env)
        self.assertEqual(rc, 0)
        dl.assert_called_once()
        ex.assert_called_once()
        self.assertIn("tdnf", wt.call_args[0][0])  # yum -> tdnf patch applied

    def test_install_azurelinux_normalizes_installer_exit_code(self):
        # exec_program passes through whatever the vendor script exited with
        # (127 here = bash not found). We only ever hand back 0 or 3.
        env = {"DISTRIBUTION": "azurelinux3.0",
               "COMPONENT_VERSIONS": json.dumps(
                   {"aznfs": {"common": {"version": "2.0", "sha256": "s"}}})}
        with mock.patch.object(install_aznfs, "download_and_verify",
                               return_value="/tmp/aznfs_install.sh"), \
             mock.patch("pathlib.Path.read_text", return_value="yum install x"), \
             mock.patch("pathlib.Path.write_text"), \
             mock.patch.object(install_aznfs, "exec_program", return_value=127):
            rc = install_aznfs.install(env)
        self.assertEqual(rc, 3)

    def test_install_unsupported_distro(self):
        with mock.patch.object(install_aznfs, "log_error") as logged:
            self.assertEqual(install_aznfs.install({"DISTRIBUTION": "gentoo"}), 3)
        logged.assert_called_once()   # unsupported distro is reported as an error


if __name__ == "__main__":
    unittest.main()
