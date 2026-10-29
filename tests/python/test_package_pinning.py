"""Unit tests for utils/package_pinning.py (port of bash dnf_pin_packages).

dnf and exec_program are mocked, so nothing is really pinned.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import unittest
from unittest import mock

from utils import package_pinning


class DnfPinPackagesTests(unittest.TestCase):
    def _pin(self, packages, *, dnf_present=True, rc=0, env=None):
        """Call dnf_pin_packages with `which` and exec_program faked."""
        which = "/usr/bin/dnf" if dnf_present else None
        with mock.patch.object(package_pinning.shutil, "which", return_value=which) as found, \
             mock.patch.object(package_pinning, "exec_program", return_value=rc) as run, \
             mock.patch.object(package_pinning, "log_error") as error:
            ok = package_pinning.dnf_pin_packages(packages, "test-op", env=env)
        return ok, run, error, found

    def test_pins_all_packages_in_one_command(self):
        ok, run, error, _ = self._pin(["openmpi", "perftest"])
        self.assertTrue(ok)
        run.assert_called_once_with(["dnf", "versionlock", "add", "openmpi", "perftest"],
                                    "test-op", env=None)
        error.assert_not_called()

    def test_wildcards_and_spaces_are_passed_through_unchanged(self):
        # "libnccl*" is expanded by dnf, not us; a quoted bash "$VAR" holding a
        # space-separated list arrives as one argument and stays one.
        _, run, _, _ = self._pin(["libnccl*", "a b"])
        self.assertEqual(run.call_args.args[0], ["dnf", "versionlock", "add", "libnccl*", "a b"])

    def test_no_dnf_is_success_and_runs_nothing(self):
        ok, run, error, _ = self._pin(["ucx*"], dnf_present=False)
        self.assertTrue(ok)
        run.assert_not_called()
        error.assert_not_called()

    def test_empty_arguments_are_dropped(self):
        _, run, _, _ = self._pin(["", "ucx*", ""])
        self.assertEqual(run.call_args.args[0], ["dnf", "versionlock", "add", "ucx*"])

    def test_only_empty_arguments_is_an_error(self):
        ok, run, error, _ = self._pin(["", ""])
        self.assertFalse(ok)
        run.assert_not_called()
        error.assert_called_once()

    def test_dnf_failure_is_an_error(self):
        ok, _, error, _ = self._pin(["libnccl*"], rc=1)
        self.assertFalse(ok)
        error.assert_called_once()

    def test_dnf_is_looked_up_on_the_build_env_path(self):
        env = {"PATH": "/opt/bin:/usr/bin"}
        _, run, _, found = self._pin(["x"], env=env)
        self.assertEqual(found.call_args.kwargs["path"], "/opt/bin:/usr/bin")
        self.assertIs(run.call_args.kwargs["env"], env)


if __name__ == "__main__":
    unittest.main()
