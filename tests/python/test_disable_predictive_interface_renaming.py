"""Unit tests for components/python/disable_predictive_interface_renaming.py.

  * The pure helpers, search_text() and add_kernel_parameter(), are tested
    directly with real file contents. The expected outputs were checked against
    real `sed` in WSL.
  * install() is tested with the GRUB settings file pointed at a temporary
    directory and update-grub replaced by a mock, so the tests never touch /etc
    or regenerate a real boot menu.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from components.python import disable_predictive_interface_renaming as renaming
from utils.build_config import BuildConfig, build_plan

# What bash searches for when GRUB_CMDLINE_LINUX is unset (note the leading space).
_SEARCH = " console=tty1 console=ttyS0 earlyprintk=ttyS0"

# A 50-cloudimg-settings.cfg with the console parameters after another value.
_CFG = ('# Set the default commandline\n'
        'GRUB_CMDLINE_LINUX_DEFAULT="quiet console=tty1 console=ttyS0 earlyprintk=ttyS0"\n'
        'GRUB_TERMINAL=console\n')
_CFG_UPDATED = ('# Set the default commandline\n'
                'GRUB_CMDLINE_LINUX_DEFAULT="quiet console=tty1 console=ttyS0 '
                'earlyprintk=ttyS0 net.ifnames=0"\n'
                'GRUB_TERMINAL=console\n')

# The console parameters directly after the opening quote: no space before
# "console=tty1", so sed (and this port) leave the line alone.
_CFG_AT_START = ('GRUB_CMDLINE_LINUX_DEFAULT="console=tty1 console=ttyS0 '
                 'earlyprintk=ttyS0"\n')


# --------------------------------------------------------------------------- #
# Pure helpers: real inputs, no mocks. Expected values match GNU sed's output.
# --------------------------------------------------------------------------- #
class SearchTextTests(unittest.TestCase):
    def test_unset_variable_gives_leading_space(self):
        self.assertEqual(renaming.search_text({}), _SEARCH)

    def test_value_from_environment_is_prepended(self):
        self.assertEqual(renaming.search_text({"GRUB_CMDLINE_LINUX": "quiet"}),
                         "quiet" + _SEARCH)


class AddKernelParameterTests(unittest.TestCase):
    def _add(self, text, search=_SEARCH):
        return renaming.add_kernel_parameter(text, search, "net.ifnames=0")

    def test_adds_parameter_after_console_parameters(self):
        self.assertEqual(self._add(_CFG), _CFG_UPDATED)

    def test_parameters_at_start_of_quoted_value_are_not_matched(self):
        self.assertEqual(self._add(_CFG_AT_START), _CFG_AT_START)

    def test_only_first_occurrence_on_a_line(self):
        line = ('X="a console=tty1 console=ttyS0 earlyprintk=ttyS0 '
                'b console=tty1 console=ttyS0 earlyprintk=ttyS0"\n')
        self.assertEqual(
            self._add(line),
            'X="a console=tty1 console=ttyS0 earlyprintk=ttyS0 net.ifnames=0 '
            'b console=tty1 console=ttyS0 earlyprintk=ttyS0"\n')

    def test_every_matching_line_is_changed(self):
        self.assertEqual(self._add(_CFG + _CFG), _CFG_UPDATED + _CFG_UPDATED)

    def test_rerun_adds_parameter_again(self):
        # Same as running the bash twice; not idempotent, kept for parity.
        self.assertIn('earlyprintk=ttyS0 net.ifnames=0 net.ifnames=0"',
                      self._add(self._add(_CFG)))

    def test_missing_final_newline_is_not_added(self):
        self.assertEqual(self._add(_CFG.rstrip("\n")), _CFG_UPDATED.rstrip("\n"))


# --------------------------------------------------------------------------- #
# install(): temp file + mocked update-grub.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "50-cloudimg-settings.cfg")

    def _install(self, env=None, *, contents=_CFG, grub_rc=0):
        """Run install() with the settings file in a temp dir and update-grub mocked.

        contents=None means the settings file does not exist.
        """
        if contents is not None:
            with open(self.path, "w", encoding="utf-8", newline="") as f:
                f.write(contents)
        env = {"DISTRIBUTION": "ubuntu24.04"} if env is None else env
        with mock.patch.object(renaming, "_GRUB_SETTINGS_FILE", self.path), \
             mock.patch.object(renaming, "exec_program",
                               return_value=grub_rc) as run, \
             mock.patch.object(renaming, "log_error") as log_error:
            rc = renaming.install(env)
        return rc, run, log_error

    def _read(self):
        with open(self.path, encoding="utf-8", newline="") as f:
            return f.read()

    def test_success_edits_file_then_runs_update_grub(self):
        env = {"DISTRIBUTION": "ubuntu24.04"}
        rc, run, log_error = self._install(env)

        self.assertEqual(rc, 0)
        self.assertEqual(self._read(), _CFG_UPDATED)
        run.assert_called_once_with(
            ["update-grub"], "disable-predictive-interface-renaming", env=env)
        log_error.assert_not_called()

    def test_no_match_leaves_file_and_still_runs_update_grub(self):
        rc, run, _ = self._install(contents=_CFG_AT_START)

        self.assertEqual(rc, 0)
        self.assertEqual(self._read(), _CFG_AT_START)
        run.assert_called_once()

    def test_grub_cmdline_linux_from_env_changes_the_search(self):
        # With GRUB_CMDLINE_LINUX="quiet" the search text is
        # "quiet console=tty1 ...", which matches right after the quote.
        cfg = 'GRUB_CMDLINE_LINUX="quiet console=tty1 console=ttyS0 earlyprintk=ttyS0"\n'
        rc, _, _ = self._install({"GRUB_CMDLINE_LINUX": "quiet"}, contents=cfg)

        self.assertEqual(rc, 0)
        self.assertEqual(
            self._read(),
            'GRUB_CMDLINE_LINUX="quiet console=tty1 console=ttyS0 '
            'earlyprintk=ttyS0 net.ifnames=0"\n')

    def test_missing_file_returns_3_without_running_update_grub(self):
        # This is also what happens on RPM distros, which have no such file.
        rc, run, log_error = self._install(contents=None)

        self.assertEqual(rc, 3)
        run.assert_not_called()
        log_error.assert_called_once()

    def test_unwritable_file_returns_3(self):
        real_open = open

        def fail_on_write(path, mode="r", *args, **kwargs):
            if "w" in mode:
                raise PermissionError("permission denied")
            return real_open(path, mode, *args, **kwargs)

        # create=True adds an `open` name to the module, shadowing the builtin
        # only for code in disable_predictive_interface_renaming.
        with mock.patch.object(renaming, "open", create=True,
                               side_effect=fail_on_write):
            rc, run, log_error = self._install()

        self.assertEqual(rc, 3)
        run.assert_not_called()
        self.assertEqual(self._read(), _CFG)
        self.assertIn("permission denied", log_error.call_args[0][1])

    def test_update_grub_failure_returns_3(self):
        rc, run, log_error = self._install(grub_rc=1)

        self.assertEqual(rc, 3)
        run.assert_called_once()
        log_error.assert_called_once()


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg)
                 if s.op == "disable-predictive-interface-renaming"]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, renaming.install)
        self.assertEqual(steps[0].script,
                         "disable_predictive_interface_renaming.sh")
        self.assertEqual(steps[0].base, "distro")


if __name__ == "__main__":
    unittest.main()
