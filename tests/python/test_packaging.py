"""Unit tests for the Python packaging layer.

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
from utils.package_manager import PackageManager, detect_package_manager
from utils.package_installer import PackageInstaller
from components.python import install_mpifileutils, install_nccl, install_doca, install_cmake, install_libfabric, install_intel_libs, install_hpcdiag, install_monitoring_tools, install_cuda_samples, install_nvbandwidth_tool


def _which(*available: str):
    """Build a shutil.which side-effect that only finds `available` names."""
    avail = set(available)
    return lambda name: f"/usr/bin/{name}" if name in avail else None


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
                               side_effect=_which()):
            pm = detect_package_manager()
        self.assertIsNone(pm)


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


class PackageInstallerTests(unittest.TestCase):
    def _installer_with(self, manager_name="apt-get"):
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
                        side_effect=[100, 0]) as exec_mock:
            ok = installer.install_package(["bad-pkg", "tree"])
        self.assertFalse(ok)
        self.assertEqual(exec_mock.call_count, 2)  # did not stop early

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
                        return_value=None):
            with mock.patch("utils.package_installer.exec_program") as exec_mock:
                ok = installer.install_package(["tree"])
        self.assertFalse(ok)
        exec_mock.assert_not_called()


class MpifileutilsDepsTests(unittest.TestCase):
    def _fake_installer(self, manager_name, succeeds=True):
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
        cfg = install_mpifileutils.get_config(env)
        self.assertEqual(cfg["version"], "0.12")
        self.assertEqual(cfg["url"], "u")

    def test_get_config_missing_component_returns_none(self):
        env = self._env({})
        self.assertIsNone(install_mpifileutils.get_config(env))


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
        self.assertEqual(install_mpifileutils.install(env), 3)

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
        self.assertEqual(install_cmake.get_config(self._env())["version"], "4.3.1")

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
        self.assertEqual(
            install_cmake.install({"COMPONENT_VERSIONS": json.dumps({})}), 3)


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
        self.assertEqual(install_libfabric.get_config(self._env())["version"], "2.5.0")

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
             mock.patch("tarfile.open"), \
             mock.patch("shutil.rmtree"):
            rc = install_libfabric.install(env)
        self.assertEqual(rc, 3)
        self.assertEqual(ex.call_count, 2)  # stopped after make failed
        wcv.assert_not_called()

    def test_install_fails_when_no_version(self):
        self.assertEqual(
            install_libfabric.install({"COMPONENT_VERSIONS": json.dumps({})}), 3)


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
            install_intel_libs.get_config(self._env())["version"], "2025.3.1.11")

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
             mock.patch.object(install_intel_libs, "write_component_version") as wcv:
            rc = install_intel_libs.install(env)
        self.assertEqual(rc, 3)
        wcv.assert_not_called()

    def test_install_fails_when_no_version(self):
        self.assertEqual(
            install_intel_libs.install({"COMPONENT_VERSIONS": json.dumps({})}), 3)


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
             mock.patch.object(install_hpcdiag, "download") as dl:
            rc = install_hpcdiag.install(env={})
        self.assertEqual(rc, 3)
        dl.assert_not_called()


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
            install_monitoring_tools.get_config(self._env())["version"], "1.2.3")

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
             mock.patch("os.makedirs"), \
             mock.patch("os.chmod"):
            rc = install_monitoring_tools.install(env)
        self.assertEqual(rc, 3)
        wcv.assert_not_called()

    def test_install_fails_when_no_version(self):
        self.assertEqual(
            install_monitoring_tools.install({"COMPONENT_VERSIONS": json.dumps({})}), 3)


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
            install_cuda_samples.get_config(self._env())["samples"]["version"], "12.4.1")

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
             mock.patch("tarfile.open"), \
             mock.patch("os.makedirs"), \
             mock.patch("shutil.move") as mv, \
             mock.patch("shutil.rmtree"):
            rc = install_cuda_samples.install(env)
        self.assertEqual(rc, 3)
        self.assertEqual(ex.call_count, 2)  # stopped after make failed
        mv.assert_not_called()

    def test_install_fails_when_no_version(self):
        self.assertEqual(
            install_cuda_samples.install({"COMPONENT_VERSIONS": json.dumps({})}), 3)


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
            install_nvbandwidth_tool.get_config(self._env())["version"], "0.9")

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
             mock.patch("os.makedirs"), \
             mock.patch("shutil.rmtree"):
            rc = install_nvbandwidth_tool.install(env)
        self.assertEqual(rc, 3)
        self.assertEqual(ex.call_count, 1)  # stopped after clone failed
        wcv.assert_not_called()

    def test_install_fails_when_no_version(self):
        self.assertEqual(
            install_nvbandwidth_tool.install({"COMPONENT_VERSIONS": json.dumps({})}), 3)


if __name__ == "__main__":
    unittest.main()
