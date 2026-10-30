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
from pathlib import Path
from unittest import mock

from utils import component_config
from utils import package_manager
from utils import package_installer
from utils import component_config, process
from utils import download as download_module
from utils.package_manager import PackageManager, detect_package_manager
from utils.package_installer import PackageInstaller
from components.python import install_mpifileutils, install_nccl, install_doca, install_libfabric, install_intel_libs, install_hpcdiag, install_monitoring_tools, install_aznfs


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


# install_mpifileutils.install_deps installs install_mpifileutils.sh's package
# list (upstream 0c2b8f9), chosen by DISTRIBUTION like bash's if/elif/else, in
# one install_packages() call.
class MpifileutilsDepsTests(unittest.TestCase):
    def _deps(self, distro, rc=0):
        with mock.patch.object(package_installer, "exec_program", return_value=rc) as run, \
             mock.patch.object(package_installer, "log_info"), \
             mock.patch.object(package_installer, "log_error") as error, \
             mock.patch.object(install_mpifileutils, "log_info") as info:
            result = install_mpifileutils.install_deps(env={"DISTRIBUTION": distro})
        self.info, self.error = info, error
        return result, [c.args[0] for c in run.call_args_list]

    def test_ubuntu_command(self):
        for distro in ("ubuntu22.04", "ubuntu24.04", "ubuntu26.04"):
            with self.subTest(distro=distro):
                rc, commands = self._deps(distro)
                self.assertEqual(rc, 0)
                self.assertEqual(commands, [
                    ["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0", "libbz2-dev",
                     "libattr1-dev", "libarchive-dev", "libssl-dev", "libcap-dev"]])
                self.assertEqual(self.info.call_args.kwargs["branch"], "ubuntu")

    def test_azure_linux_and_rhel_family_command(self):
        for distro, branch in (("azurelinux3.0", "azurelinux"), ("almalinux9.8", "rhel-family"),
                               ("rocky8.10", "rhel-family"), ("rhel10.0", "rhel-family")):
            with self.subTest(distro=distro):
                rc, commands = self._deps(distro)
                self.assertEqual(rc, 0)
                self.assertEqual(commands, [
                    ["dnf", "install", "-y", "bzip2-devel", "libattr-devel",
                     "libarchive-devel"]])
                self.assertEqual(self.info.call_args.kwargs["branch"], branch)

    def test_unknown_distro_returns_3_without_running_anything(self):
        # bash's else branch would run dnf; install_packages() knows no manager.
        rc, commands = self._deps("")
        self.assertEqual(rc, 3)
        self.assertEqual(commands, [])
        self.assertEqual(self.info.call_args.kwargs["branch"], "rhel-family")
        self.error.assert_called_once()

    def test_returns_3_when_install_fails(self):
        rc, commands = self._deps("ubuntu24.04", rc=100)
        self.assertEqual(rc, 3)
        self.assertEqual(len(commands), 1)
        self.error.assert_called_once()


# install_nccl.install_deps installs install_nccl.sh's package lists with
# install_packages() (one command per bash line, stopping at the first
# failure), chosen by DISTRIBUTION.
class NcclDepsTests(unittest.TestCase):
    _APT = ["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0"]
    _DNF = ["dnf", "install", "-y"]

    def _deps(self, distro, rc=0):
        with mock.patch.object(package_installer, "exec_program", return_value=rc) as run, \
             mock.patch.object(package_installer, "log_info"), \
             mock.patch.object(package_installer, "log_error"), \
             mock.patch.object(install_nccl, "log_info"):
            result = install_nccl.install_deps(env={"DISTRIBUTION": distro})
        return result, [c.args[0] for c in run.call_args_list]

    def test_ubuntu_commands(self):
        # libibverbs-dev is deliberately not installed (commented out in bash).
        for distro in ("ubuntu22.04", "ubuntu24.04"):
            with self.subTest(distro=distro):
                rc, commands = self._deps(distro)
                self.assertEqual(rc, 0)
                self.assertEqual(commands, [
                    [*self._APT, "build-essential", "devscripts", "debhelper", "fakeroot",
                     "patchelf"],
                    [*self._APT, "zlib1g-dev"]])

    def test_ubuntu26_04_commands(self):
        rc, commands = self._deps("ubuntu26.04")
        self.assertEqual(rc, 0)
        self.assertEqual(commands, [
            [*self._APT, "build-essential", "patchelf", "zlib1g-dev"]])

    def test_azure_linux_commands(self):
        rc, commands = self._deps("azurelinux3.0")
        self.assertEqual(rc, 0)
        self.assertEqual(commands, [
            [*self._DNF, "rpm-build", "rpmdevtools", "autoconf", "automake", "git",
             "libtool", "patchelf"]])

    def test_rhel_family_commands(self):
        for distro in ("almalinux9.8", "rocky8.10", "rhel9.8"):
            with self.subTest(distro=distro):
                rc, commands = self._deps(distro)
                self.assertEqual(rc, 0)
                self.assertEqual(commands, [
                    [*self._DNF, "rpm-build", "rpmdevtools", "patchelf"]])

    def test_failure_returns_3_and_stops(self):
        rc, commands = self._deps("ubuntu24.04", rc=100)
        self.assertEqual(rc, 3)
        self.assertEqual(len(commands), 1)        # zlib1g-dev not attempted


# Pure string parsing, no mocks needed — parse_apt_cache_version digs the version out of
# `apt-cache show <pkg>` output (mirrors awk '/^Version:/ {print $2; exit}').
class DocaVersionParseTests(unittest.TestCase):
    def test_parses_openmpi_version(self):
        sample = "Package: openmpi\nVersion: 4.1.5-1\nArchitecture: amd64\n"
        self.assertEqual(install_doca.parse_apt_cache_version(sample), "4.1.5-1")

    def test_takes_first_version(self):
        # Mirrors awk '{...; exit}' — only the first Version line is used.
        sample = "Version: 4.1.5-1\nVersion: 5.0.0-2\n"
        self.assertEqual(install_doca.parse_apt_cache_version(sample), "4.1.5-1")

    def test_missing_version_returns_none(self):
        self.assertIsNone(install_doca.parse_apt_cache_version("Package: openmpi\n"))

    def test_blank_version_returns_none(self):
        self.assertIsNone(install_doca.parse_apt_cache_version("Version: \n"))


class MpifileutilsConfigTests(unittest.TestCase):
    # The env dict is exactly what a component sees at runtime: COMPONENT_VERSIONS is the
    # parsed versions.json, and the rest (distro/arch/gpu/sku/target node type) is what
    # get_component_config uses to pick the right entry out of it.
    def _env(self, versions):
        return {
            "COMPONENT_VERSIONS": json.dumps(versions),
            "DISTRIBUTION": "ubuntu24.04",
            "ARCHITECTURE": "x86_64",
            "GPU": "NVIDIA",
            "SKU": "A100",
            "TARGET_NODE_TYPE": "azure_vm_regular",
        }

    def test_get_config_resolves_common_version(self):
        env = self._env({
            "mpifileutils": {"common": {"version": "0.12", "url": "u", "sha256": "s"}}
        })
        cfg = component_config.config_for("mpifileutils", env)
        self.assertEqual(cfg["version"], "0.12")
        self.assertEqual(cfg["url"], "u")

    def test_get_config_missing_component_returns_none(self):
        env = self._env({})
        self.assertIsNone(component_config.config_for("mpifileutils", env))


class MpifileutilsInstallTests(unittest.TestCase):
    # The full command-list tests (tar, build, rm -rf, ls -la) are in
    # test_install_mpifileutils.py; these cover the early exits.
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
            "TARGET_NODE_TYPE": "azure_vm_regular",
        }

    def test_install_fails_when_no_version(self):
        env = {"COMPONENT_VERSIONS": json.dumps({})}
        # require_config reports the missing version.
        with mock.patch.object(component_config, "log_error") as logged:
            self.assertEqual(install_mpifileutils.install(env), 3)
        logged.assert_called_once()   # missing version is reported as an error

    def test_install_fails_when_deps_fail(self):
        env = self._env()
        with mock.patch.object(install_mpifileutils, "install_deps", return_value=3), \
             mock.patch.object(install_mpifileutils, "fetch_verified") as dl, \
             mock.patch("os.makedirs"):
            rc = install_mpifileutils.install(env)
        self.assertEqual(rc, 3)
        dl.assert_not_called()  # stopped before downloading


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
            "TARGET_NODE_TYPE": "azure_vm_regular",
        }

    def test_get_config_resolves_version(self):
        cfg = install_libfabric.require_config("libfabric", self._env(), "install-libfabric",
                                               "version", "url", "sha256")
        self.assertEqual(cfg["version"], "2.5.0")

    _TARBALL = "/tmp/libfabric-2.5.0.tar.bz2"
    _EXPECTED = [
        (["tar", "-xf", "/tmp/libfabric-2.5.0.tar.bz2"], "/tmp"),
        (["/tmp/libfabric-2.5.0/configure", "--prefix=/opt/libfabric", "--disable-psm3"],
         "/tmp/libfabric-2.5.0"),
        (["make", "-j", "8"], "/tmp/libfabric-2.5.0"),
        (["make", "install"], "/tmp/libfabric-2.5.0"),
    ]

    def _install(self, rcs=None):
        """Run install() with every command recorded (via exec_program) in order."""
        commands = []
        rcs = iter(rcs or [])

        def record(command, op, *, cwd=None, env=None, **_):
            commands.append((command, cwd))
            return next(rcs, 0)

        with mock.patch.object(install_libfabric, "fetch_verified",
                               return_value=self._TARBALL) as dl, \
             mock.patch.object(install_libfabric, "write_component_version") as wcv, \
             mock.patch.object(install_libfabric, "remove_paths") as rm, \
             mock.patch.object(install_libfabric.os, "cpu_count", return_value=8), \
             mock.patch("utils.process.exec_program", side_effect=record):
            rc = install_libfabric.install(self._env())
        return rc, commands, dl, wcv, rm

    def test_install_runs_all_build_steps(self):
        rc, commands, dl, wcv, rm = self._install()
        self.assertEqual(rc, 0)
        dl.assert_called_once_with("https://example/libfabric-2.5.0.tar.bz2", "s",
                                   "install-libfabric", dest_dir="/tmp")
        self.assertEqual(commands, self._EXPECTED)
        wcv.assert_called_once_with("LIBFABRIC", "2.5.0")
        rm.assert_called_once_with(Path("/tmp/libfabric-2.5.0"), self._TARBALL,
                                   op="install-libfabric")

    def test_install_stops_on_build_failure(self):
        rc, commands, _, wcv, rm = self._install(rcs=[0, 0, 1])
        self.assertEqual(rc, 3)
        self.assertEqual(commands, self._EXPECTED[:3])  # stopped after make failed
        wcv.assert_not_called()
        rm.assert_not_called()

    def test_install_stops_on_download_failure(self):
        with mock.patch.object(install_libfabric, "fetch_verified", return_value=None), \
             mock.patch("utils.process.exec_program") as ex, \
             mock.patch.object(install_libfabric, "write_component_version") as wcv:
            self.assertEqual(install_libfabric.install(self._env()), 3)
        ex.assert_not_called()
        wcv.assert_not_called()

    def test_install_fails_when_no_version(self):
        with mock.patch("utils.component_config.log_error") as logged:
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
            "TARGET_NODE_TYPE": "azure_vm_regular",
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
    """install_monitoring_tools.py, compared command by command with
    install_monitoring_tools.sh. /opt/azurehpc/tools and /etc/bash.bashrc are
    pointed at a temporary directory; commands and the download are faked."""

    _ALIAS = "alias moneo='python3 /opt/azurehpc/tools/Moneo/moneo.py'"

    def setUp(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tools = os.path.join(tmp.name, "opt", "azurehpc", "tools")
        os.makedirs(self.tools)
        self.moneo = os.path.join(self.tools, "Moneo")
        self.service_dir = os.path.join(self.moneo, "linux_service")
        self.bashrc = os.path.join(tmp.name, "bash.bashrc")
        for name, value in (("_MONITOR_DIR", self.tools), ("_BASHRC", self.bashrc)):
            patcher = mock.patch.object(install_monitoring_tools, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _env(self, distribution="ubuntu24.04", versions=None):
        return {
            "COMPONENT_VERSIONS": versions or json.dumps({
                "moneo": {"common": {"version": "0.3.5", "sha256": "s"}}
            }),
            "DISTRIBUTION": distribution,
            "ARCHITECTURE": "x86_64",
            "GPU": "NVIDIA",
            "SKU": "A100",
            "TARGET_NODE_TYPE": "azure_vm_regular",
        }

    # What install_packages() runs for bash's python3-pip installs.
    _APT_PIP = ["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0", "--only-upgrade",
                "python3-pip"]
    _DNF_PIP = ["dnf", "install", "-y", "python3-pip"]
    _PIP_UPGRADE = ["python3", "-m", "pip", "install", "--upgrade", "pip"]

    def _install(self, env, *, fail=None, script_mode=0o755, real_configure=False,
                 download_error=None):
        """Run install(env) with commands faked where the toolkit runs them
        (utils.process and utils.package_installer exec_program, one record)
        and the download faked in utils.download. The fake tar creates
        Moneo/linux_service/configure_service.sh (mode `script_mode`).
        fail: {first word of command: exit code}. Returns (rc, calls, mocks),
        calls being [(command, cwd)]."""
        from utils import component_config, download, files, process
        fail = fail or {}
        calls = []
        real_exec = process.exec_program

        def fake_download(url, sha256, dest_dir="."):
            if download_error:
                raise download_error
            path = os.path.join(dest_dir, os.path.basename(url))
            with open(path, "wb") as handle:
                handle.write(b"tarball")
            return path

        def fake_exec(command, op, *, cwd=None, env=None):
            self.assertEqual(op, "install-monitoring-tools")
            self.assertIs(env, self.env)
            calls.append((command, cwd))
            if command[0] == "tar" and "tar" not in fail:
                dest = command[command.index("-C") + 1]
                script = os.path.join(cwd, dest, "linux_service",
                                      "configure_service.sh")
                os.makedirs(os.path.dirname(script))
                with open(script, "w") as handle:
                    handle.write("#!/bin/sh\necho configured\n")
                os.chmod(script, script_mode)
            if command == ["./configure_service.sh"] and real_configure:
                return real_exec(command, op, cwd=cwd, env=env)
            return fail.get(command[0], 0)

        self.env = env
        with mock.patch.object(download, "download_and_verify",
                               side_effect=fake_download) as download_mock, \
             mock.patch.object(download, "log_error"), \
             mock.patch.object(component_config, "log_error") as config_error, \
             mock.patch.object(process, "exec_program", side_effect=fake_exec), \
             mock.patch.object(package_installer, "exec_program", side_effect=fake_exec), \
             mock.patch.object(package_installer, "log_info"), \
             mock.patch.object(package_installer, "log_error"), \
             mock.patch.object(files, "log_info") as files_info, \
             mock.patch.object(install_monitoring_tools, "write_component_version") as wcv, \
             mock.patch.object(install_monitoring_tools, "log_info") as info, \
             mock.patch.object(install_monitoring_tools, "log_warn") as warn, \
             mock.patch.object(install_monitoring_tools, "log_error") as error:
            rc = install_monitoring_tools.install(env)
        return rc, calls, {"download": download_mock, "record": wcv, "info": info,
                           "warn": warn, "error": error, "files_info": files_info,
                           "config_error": config_error}

    def _expected(self, pip):
        return [(pip, None),
                (["tar", "-xf", "v0.3.5.tar.gz", "-C", "Moneo", "--strip-components=1"],
                 self.tools),
                (["./configure_service.sh"], self.service_dir)]

    # ---- pure helpers ----------------------------------------------------------
    def test_pip_package_options_per_distro(self):
        # None: bash's else branch, `python3 -m pip install --upgrade pip`.
        cases = {
            "azurelinux3.0": [],
            "ubuntu24.04": ["--only-upgrade"],
            "ubuntu26.04": ["--only-upgrade"],
            "ubuntu22.04": None,
            "almalinux9.8": None,
            "rhel8.10": None,
        }
        for distro, options in cases.items():
            with self.subTest(distro=distro):
                self.assertEqual(install_monitoring_tools.pip_package_options(distro),
                                 options)

    def test_add_line_once_matches_grep_qxF(self):
        line = "alias x='y'"
        cases = [
            # (existing content, appended?, final content)
            (None, True, b"alias x='y'\n"),
            (b"alias x='y'\n", False, b"alias x='y'\n"),
            (b"a\nalias x='y'", False, b"a\nalias x='y'"),              # no final newline
            (b"alias x='y'\r\n", True, b"alias x='y'\r\nalias x='y'\n"),   # grep: \r is part of the line
            (b"  alias x='y'\n", True, b"  alias x='y'\nalias x='y'\n"),   # -x: whole line
            (b"last", True, b"lastalias x='y'\n"),                       # echo >> joins, like bash
        ]
        for before, appended, after in cases:
            with self.subTest(before=before):
                if os.path.exists(self.bashrc):
                    os.remove(self.bashrc)
                if before is not None:
                    with open(self.bashrc, "wb") as handle:
                        handle.write(before)
                self.assertIs(install_monitoring_tools.has_line(self.bashrc, line),
                              not appended)
                with mock.patch("utils.files.log_info"), \
                     mock.patch.object(install_monitoring_tools, "log_info"):
                    self.assertTrue(install_monitoring_tools.add_line_once(self.bashrc, line))
                with open(self.bashrc, "rb") as handle:
                    self.assertEqual(handle.read(), after)

    # ---- install() -------------------------------------------------------------
    def test_install_commands_per_distro(self):
        cases = {
            "ubuntu24.04": self._APT_PIP,
            "ubuntu26.04": self._APT_PIP,
            "azurelinux3.0": self._DNF_PIP,
            "ubuntu22.04": self._PIP_UPGRADE,
            "almalinux9.8": self._PIP_UPGRADE,
        }
        for distro, pip in cases.items():
            with self.subTest(distro=distro):
                self.tearDown_moneo()
                rc, calls, m = self._install(self._env(distro))
                self.assertEqual(rc, 0)
                self.assertEqual(calls, self._expected(pip))
                m["download"].assert_called_once_with(
                    "https://github.com/Azure/Moneo/archive/refs/tags/v0.3.5.tar.gz",
                    "s", dest_dir=self.tools)
                m["record"].assert_called_once_with("MONEO", "0.3.5")
                m["warn"].assert_not_called()
                m["error"].assert_not_called()

    def tearDown_moneo(self):
        import shutil
        shutil.rmtree(self.moneo, ignore_errors=True)

    def test_install_files_and_logs(self):
        rc, _, m = self._install(self._env())
        self.assertEqual(rc, 0)
        self.assertEqual(os.stat(self.moneo).st_mode & 0o777, 0o777)        # chmod 777 Moneo
        self.assertTrue(os.path.exists(os.path.join(self.tools, "v0.3.5.tar.gz")))  # left, as bash
        with open(self.bashrc) as handle:
            self.assertEqual(handle.read(), self._ALIAS + "\n")
        messages = [c.args[1] for c in m["info"].call_args_list]
        self.assertEqual(m["files_info"].call_args.kwargs,
                         {"path": self.bashrc, "action": "created"})
        self.assertIn("Recorded MONEO version 0.3.5", messages)
        self.assertEqual(m["info"].call_args_list[-1].args[1], "Moneo 0.3.5 installed")

    def test_rerun_keeps_existing_moneo_dir_without_extracting(self):
        # bash: `mkdir Moneo && tar ...` - mkdir fails, tar is skipped, set -e
        # doesn't stop, and the rest runs on the old contents.
        self._install(self._env())
        rc, calls, m = self._install(self._env())
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [(self._APT_PIP, None),
                                 (["./configure_service.sh"], self.service_dir)])
        self.assertEqual(m["warn"].call_args.kwargs, {"path": self.moneo})
        with open(self.bashrc) as handle:
            self.assertEqual(handle.read(), self._ALIAS + "\n")      # not appended twice
        self.assertIn(f"{self.bashrc} already has the line",
                      [c.args[1] for c in m["info"].call_args_list])
        m["files_info"].assert_not_called()

    @unittest.skipUnless(os.name == "posix", "runs a real script")
    def test_configure_service_runs_from_its_folder_with_the_exec_bit(self):
        rc, calls, m = self._install(self._env(), real_configure=True)
        self.assertEqual(rc, 0)
        m["error"].assert_not_called()

    @unittest.skipUnless(os.name == "posix", "runs a real script")
    def test_configure_service_without_exec_bit_fails_clearly(self):
        rc, calls, m = self._install(self._env(), script_mode=0o644, real_configure=True)
        self.assertEqual(rc, 3)
        m["record"].assert_not_called()
        error = m["error"].call_args
        self.assertIn("not executable", error.args[1])
        self.assertEqual((error.kwargs["exists"], error.kwargs["executable"]), (True, False))

    def test_each_failing_command_returns_3_and_stops(self):
        expected = self._expected(self._APT_PIP)
        for index, word in enumerate(["apt-get", "tar", "./configure_service.sh"]):
            with self.subTest(failing=word):
                self.tearDown_moneo()
                rc, calls, m = self._install(self._env(), fail={word: 1})
                self.assertEqual(rc, 3)
                self.assertEqual(calls, expected[:index + 1])
                m["record"].assert_not_called()
                if word == "apt-get":
                    m["download"].assert_not_called()

    def test_download_failure_returns_3(self):
        rc, calls, m = self._install(self._env(), download_error=OSError("404"))
        self.assertEqual(rc, 3)
        self.assertEqual(len(calls), 1)                 # only the pip command
        self.assertFalse(os.path.exists(self.moneo))
        m["record"].assert_not_called()

    def test_missing_version_or_sha256_returns_3_before_anything_runs(self):
        for versions in (json.dumps({}),
                         json.dumps({"moneo": {"common": {"version": "0.3.5"}}})):
            with self.subTest(versions=versions):
                rc, calls, m = self._install(self._env(versions=versions))
                self.assertEqual(rc, 3)
                self.assertEqual(calls, [])
                m["download"].assert_not_called()
                m["config_error"].assert_called_once()


class AznfsTests(unittest.TestCase):
    """install_aznfs.py. Commands (utils.process.exec_program) and the download
    (utils.download.download_and_verify) are mocked; the installer is written
    into a temporary directory (the port's _DOWNLOAD_DIR)."""

    _VERSIONS = json.dumps({"aznfs": {"common": {"version": "3.0.12", "sha256": "s"}}})
    _URL = ("https://github.com/Azure/AZNFS-mount/releases/download/3.0.12/"
            "aznfs_install.sh")
    # One line per sed case; checked against real `sed -i 's/yum/dnf/'` in WSL.
    _INSTALLER = (b"yum install -y yum-utils && yum update\n"
                  b"no match here\n"
                  b"  if command -v yum; then yum -y install aznfs; fi\n"
                  b"yummy yum\r\n"
                  b"\n"
                  b"YUM yum\n"
                  b"last line yum yum")
    _PATCHED = (b"dnf install -y yum-utils && yum update\n"
                b"no match here\n"
                b"  if command -v dnf; then yum -y install aznfs; fi\n"
                b"dnfmy yum\r\n"
                b"\n"
                b"YUM dnf\n"
                b"last line dnf yum")

    def setUp(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.script = os.path.join(self.dir, "aznfs_install.sh")
        patcher = mock.patch.object(install_aznfs, "_DOWNLOAD_DIR", self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _install(self, distribution, *, versions=None, rcs=None, download_error=None):
        """Run install() with exec_program and download_and_verify faked.

        Returns (rc, commands, mocks). The fake installer run records the
        script's contents at the moment bash would run it.
        """
        rcs = rcs or {}
        commands, envs, self.seen_script = [], [], None

        def fake_download(url, sha256, dest_dir="."):
            if download_error:
                raise download_error
            path = os.path.join(dest_dir, os.path.basename(url))
            with open(path, "wb") as handle:
                handle.write(self._INSTALLER)
            return path

        def fake_exec(command, op, **kwargs):
            self.assertEqual(op, "install-aznfs")
            commands.append(command)
            envs.append(kwargs["env"])
            if command[0] == "bash":
                with open(command[1], "rb") as handle:
                    self.seen_script = handle.read()
            return rcs.get(command[0], 0)

        env = {"DISTRIBUTION": distribution, "ARCHITECTURE": "x86_64",
               "COMPONENT_VERSIONS": versions or self._VERSIONS}
        with mock.patch.object(download_module, "download_and_verify",
                               side_effect=fake_download) as download, \
             mock.patch.object(process, "exec_program", side_effect=fake_exec), \
             mock.patch.object(install_aznfs, "log_info") as info, \
             mock.patch.object(install_aznfs, "log_warn") as warn, \
             mock.patch.object(install_aznfs, "log_error") as error, \
             mock.patch.object(component_config, "log_error") as config_error, \
             mock.patch.object(download_module, "log_error") as download_error_log:
            rc = install_aznfs.install(env)
        for run_env in envs:
            self.assertEqual(run_env["AZNFS_NONINTERACTIVE_INSTALL"], "1")
            self.assertEqual(run_env["DISTRIBUTION"], distribution)
        self.assertNotIn("AZNFS_NONINTERACTIVE_INSTALL", env)    # caller's env untouched
        return rc, commands, {"download": download, "info": info, "warn": warn,
                              "error": error, "config_error": config_error,
                              "download_error": download_error_log}

    # ---- pure helpers ----------------------------------------------------------
    def test_install_branch(self):
        cases = {"ubuntu22.04": "apt", "ubuntu24.04": "apt", "ubuntu26.04": "apt",
                 "almalinux8.10": "dnf", "almalinux9.8": "dnf", "rocky9.8": "dnf",
                 "rhel9.8": "dnf", "rhel8.10": "dnf", "azurelinux3.0": "installer",
                 "sles15": "installer", "": "installer"}
        for distro, branch in cases.items():
            with self.subTest(distro=distro):
                self.assertEqual(install_aznfs.install_branch(distro), branch)

    def test_sed_replaces_only_the_first_yum_on_each_line(self):
        self.assertEqual(install_aznfs.sed_yum_to_dnf(self._INSTALLER), (self._PATCHED, 5))
        self.assertEqual(install_aznfs.sed_yum_to_dnf(b"yum\n"), (b"dnf\n", 1))
        self.assertEqual(install_aznfs.sed_yum_to_dnf(b""), (b"", 0))

    # ---- PMC branches ----------------------------------------------------------
    def test_ubuntu_and_rhel_family_install_from_pmc(self):
        cases = {"ubuntu24.04": "apt", "ubuntu26.04": "apt", "ubuntu22.04": "apt",
                 "almalinux9.8": "dnf", "rocky8.10": "dnf", "rhel9.8": "dnf"}
        for distro, manager in cases.items():
            with self.subTest(distro=distro):
                rc, commands, m = self._install(distro)
                self.assertEqual(rc, 0)
                self.assertEqual(commands, [[manager, "install", "-y", "aznfs"]])
                m["download"].assert_not_called()
                m["warn"].assert_not_called()
                m["error"].assert_not_called()
                self.assertEqual(m["info"].call_args_list[0].kwargs,
                                 {"distro": distro, "branch": manager,
                                  "noninteractive": True})

    def test_pmc_failure_returns_3(self):
        rc, commands, m = self._install("ubuntu24.04", rcs={"apt": 100})
        self.assertEqual(rc, 3)
        self.assertEqual(commands, [["apt", "install", "-y", "aznfs"]])
        m["error"].assert_not_called()       # exec_program logs the failed command

    # ---- installer branch ------------------------------------------------------
    def test_azurelinux_runs_the_patched_installer_and_removes_it(self):
        rc, commands, m = self._install("azurelinux3.0")

        self.assertEqual(rc, 0)
        m["download"].assert_called_once_with(self._URL, "s", dest_dir=self.dir)
        self.assertEqual(commands, [["bash", self.script]])
        self.assertEqual(self.seen_script, self._PATCHED)       # sed ran before bash
        self.assertFalse(os.path.exists(self.script))           # rm -f aznfs_install.sh
        m["warn"].assert_not_called()
        m["error"].assert_not_called()
        patched = [c for c in m["info"].call_args_list if c.args[1].startswith("Patched ")]
        self.assertEqual(patched[0].kwargs["lines_changed"], 5)

    def test_other_distros_take_the_installer_branch_with_a_warning(self):
        rc, commands, m = self._install("sles15")
        self.assertEqual(rc, 0)
        self.assertEqual(commands, [["bash", self.script]])
        self.assertEqual(m["warn"].call_args.kwargs, {"distro": "sles15"})

    def test_leftover_installer_is_overwritten_with_a_warning(self):
        with open(self.script, "wb") as handle:
            handle.write(b"old, already patched\n")
        rc, _, m = self._install("azurelinux3.0")
        self.assertEqual(rc, 0)
        self.assertEqual(self.seen_script, self._PATCHED)
        self.assertEqual(m["warn"].call_args.kwargs, {"path": self.script})

    def test_installer_failure_leaves_the_script_and_returns_3(self):
        rc, _, m = self._install("azurelinux3.0", rcs={"bash": 1})
        self.assertEqual(rc, 3)
        self.assertTrue(os.path.exists(self.script))            # set -e stops before rm
        self.assertEqual(m["error"].call_args.kwargs["path"], self.script)

    def test_installer_branch_missing_config_returns_3_before_downloading(self):
        for versions in ("{}", json.dumps({"aznfs": {"common": {"version": "3.0.12"}}})):
            with self.subTest(versions=versions):
                rc, commands, m = self._install("azurelinux3.0", versions=versions)
                self.assertEqual(rc, 3)
                self.assertEqual(commands, [])
                m["download"].assert_not_called()
                m["config_error"].assert_called_once()

    def test_download_failure_returns_3(self):
        rc, commands, m = self._install("azurelinux3.0",
                                        download_error=OSError("404"))
        self.assertEqual(rc, 3)
        self.assertEqual(commands, [])
        self.assertEqual(m["download_error"].call_args.kwargs["url"], self._URL)


if __name__ == "__main__":
    unittest.main()
