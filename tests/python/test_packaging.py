"""Unit tests for the Python packaging layer.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import unittest
from unittest import mock

from utils import package_manager
from utils.package_manager import PackageManager, detect_package_manager
from utils.package_installer import PackageInstaller
from utils import build_config


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
    def test_selects_apt_dep_list(self):
        pm = PackageManager(name="apt-get", path="/usr/bin/apt-get")
        fake = mock.Mock()
        fake.manager = pm
        fake.install_package.return_value = True
        with mock.patch.object(build_config, "PackageInstaller", return_value=fake):
            rc = build_config.install_mpifileutils_deps(env={})
        self.assertEqual(rc, 0)
        fake.install_package.assert_called_once_with(
            ["libbz2-dev", "libattr1-dev", "libarchive-dev", "libssl-dev", "libcap-dev"]
        )

    def test_selects_rpm_dep_list(self):
        pm = PackageManager(name="tdnf", path="/usr/bin/tdnf")
        fake = mock.Mock()
        fake.manager = pm
        fake.install_package.return_value = True
        with mock.patch.object(build_config, "PackageInstaller", return_value=fake):
            rc = build_config.install_mpifileutils_deps(env={})
        self.assertEqual(rc, 0)
        fake.install_package.assert_called_once_with(
            ["bzip2-devel", "libattr-devel", "libarchive-devel"]
        )

    def test_returns_3_when_install_fails(self):
        pm = PackageManager(name="apt-get", path="/usr/bin/apt-get")
        fake = mock.Mock()
        fake.manager = pm
        fake.install_package.return_value = False
        with mock.patch.object(build_config, "PackageInstaller", return_value=fake):
            rc = build_config.install_mpifileutils_deps(env={})
        self.assertEqual(rc, 3)

    def test_returns_3_when_no_manager(self):
        fake = mock.Mock()
        fake.manager = None
        with mock.patch.object(build_config, "PackageInstaller", return_value=fake):
            rc = build_config.install_mpifileutils_deps(env={})
        self.assertEqual(rc, 3)


if __name__ == "__main__":
    unittest.main()
