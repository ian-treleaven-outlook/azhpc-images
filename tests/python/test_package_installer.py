"""Unit tests for install_packages() and manager_for_distribution().

These are the package-manager abstraction the component ports use: a port
says what to install, and the distro decides the command.
"""

from __future__ import annotations

import unittest
from unittest import mock

from utils import package_installer
from utils.package_installer import install_packages
from utils.package_manager import manager_for_distribution

_APT = ["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0"]
_DNF = ["dnf", "install", "-y"]


class ManagerForDistributionTests(unittest.TestCase):
    def test_bash_managers(self):
        cases = {"ubuntu22.04": "apt-get", "ubuntu24.04": "apt-get", "ubuntu26.04": "apt-get",
                 "azurelinux3.0": "dnf", "almalinux8.10": "dnf", "almalinux9.7": "dnf",
                 "rocky8.10": "dnf", "rocky9.7": "dnf", "rhel8.10": "dnf", "rhel9.8": "dnf"}
        for distribution, name in cases.items():
            with self.subTest(distribution=distribution):
                self.assertEqual(manager_for_distribution(distribution).name, name)

    def test_unknown_distribution(self):
        for distribution in ("", "debian12", "sles15"):
            with self.subTest(distribution=distribution):
                self.assertIsNone(manager_for_distribution(distribution))


class InstallPackagesTests(unittest.TestCase):
    def _run(self, packages, distribution, rc=0, **kwargs):
        env = {"DISTRIBUTION": distribution}
        with mock.patch.object(package_installer, "exec_program", return_value=rc) as run, \
             mock.patch.object(package_installer, "log_info") as info, \
             mock.patch.object(package_installer, "log_error") as error:
            ok = install_packages(packages, env, "install-test", **kwargs)
        return ok, run, info, error

    def test_one_command_for_all_packages_on_ubuntu(self):
        ok, run, info, _ = self._run(["libbz2-dev", "libattr1-dev"], "ubuntu24.04")
        self.assertTrue(ok)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], [*_APT, "libbz2-dev", "libattr1-dev"])
        self.assertEqual(run.call_args.args[1], "install-test")
        self.assertEqual(info.call_args.kwargs["package_manager"], "apt-get")

    def test_one_command_for_all_packages_on_rpm_distros(self):
        for distribution in ("azurelinux3.0", "almalinux9.7", "rocky8.10", "rhel9.8"):
            with self.subTest(distribution=distribution):
                ok, run, _, _ = self._run(["bzip2-devel", "libattr-devel"], distribution)
                self.assertTrue(ok)
                self.assertEqual(run.call_args.args[0], [*_DNF, "bzip2-devel", "libattr-devel"])

    def test_a_single_name_is_accepted(self):
        _, run, _, _ = self._run("jq", "rocky9.7")
        self.assertEqual(run.call_args.args[0], [*_DNF, "jq"])

    def test_options_go_before_the_packages(self):
        _, run, _, _ = self._run(["lustre-client"], "almalinux9.7",
                                 options=["--disableexcludes=main", "--refresh"])
        self.assertEqual(run.call_args.args[0],
                         [*_DNF, "--disableexcludes=main", "--refresh", "lustre-client"])
        _, run, _, _ = self._run(["python3-pip"], "ubuntu24.04", options=["--only-upgrade"])
        self.assertEqual(run.call_args.args[0], [*_APT, "--only-upgrade", "python3-pip"])

    def test_pins_local_files_and_cwd_pass_through(self):
        _, run, _, _ = self._run(["./packages-microsoft-prod.deb", "moby-engine=2.0"],
                                 "ubuntu24.04", cwd="/tmp/work")
        self.assertEqual(run.call_args.args[0],
                         [*_APT, "./packages-microsoft-prod.deb", "moby-engine=2.0"])
        self.assertEqual(run.call_args.kwargs["cwd"], "/tmp/work")
        self.assertEqual(run.call_args.kwargs["env"], {"DISTRIBUTION": "ubuntu24.04"})

    def test_failure_logs_and_returns_false(self):
        ok, _, _, error = self._run(["nope"], "ubuntu24.04", rc=100)
        self.assertFalse(ok)
        self.assertEqual(error.call_args.kwargs["exit_code"], 100)
        self.assertEqual(error.call_args.kwargs["packages"], "nope")

    def test_unknown_distribution_fails_without_running_anything(self):
        ok, run, _, error = self._run(["jq"], "debian12")
        self.assertFalse(ok)
        run.assert_not_called()
        self.assertEqual(error.call_args.kwargs["distribution"], "debian12")


if __name__ == "__main__":
    unittest.main()
