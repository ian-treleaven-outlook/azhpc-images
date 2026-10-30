"""Unit tests for components/python/remove_unused_packages.py.

exec_program is mocked (where remove_packages calls it), so the tests
never run apt-get.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from components.python import remove_unused_packages
from utils import package_installer
from utils.plans import steps

_BASH_SCRIPT = (Path(__file__).resolve().parents[2]
                / "distros" / "ubuntu22.04" / "remove_unused_packages.sh")
_COMMAND = ["apt-get", "remove", "--auto-remove", "-y", "graphviz"]
_ENV = {"DISTRIBUTION": "ubuntu22.04"}


class InstallTests(unittest.TestCase):
    def _install(self, rc=0):
        with mock.patch.object(package_installer, "exec_program", return_value=rc) as exec_program, \
             mock.patch.object(remove_unused_packages, "log_info") as log_info:
            result = remove_unused_packages.install(_ENV)
        return result, exec_program, log_info

    def test_runs_the_bash_command(self):
        rc, exec_program, _ = self._install()

        self.assertEqual(rc, 0)
        exec_program.assert_called_once_with(_COMMAND, "remove-unused-packages", env=_ENV)

    def test_command_is_the_bash_scripts_word_for_word(self):
        self.assertIn(" ".join(_COMMAND) + "\n", _BASH_SCRIPT.read_text(encoding="utf-8"))

    def test_logs_what_it_removes(self):
        _, _, log_info = self._install()

        # remove_packages logs the command; the port logs the result.
        log_info.assert_called_once()
        self.assertEqual(log_info.call_args.kwargs["packages"], "graphviz")

    def test_apt_get_failure_returns_3(self):
        rc, exec_program, log_info = self._install(rc=100)

        self.assertEqual(rc, 3)
        exec_program.assert_called_once()
        log_info.assert_not_called()                 # no "Removed" line


class StepTests(unittest.TestCase):
    def test_step_runs_the_port_from_the_distro_folder(self):
        step = steps.unused_packages()

        self.assertEqual(step.op, "remove-unused-packages")
        self.assertEqual(step.script, "remove_unused_packages.sh")
        self.assertEqual(step.base, "distro")
        self.assertIs(step.action, remove_unused_packages.install)
        self.assertEqual(step.runner, "python")


if __name__ == "__main__":
    unittest.main()
