"""Unit tests for components/python/disable_auto_upgrade.py.

  * The pure helper, set_unattended_upgrade_off(), is tested directly with real
    file contents. The expected outputs were checked against real `sed` in WSL.
  * install() is tested with the apt file pointed at a temporary directory and
    systemctl replaced by a mock, so the tests never touch /etc or systemd.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import disable_auto_upgrade
from utils.build_config import BuildConfig, build_plan

_REPO_ROOT = Path(__file__).resolve().parents[2]
# The only disable_auto_upgrade.sh left upstream (#623 deleted the 22.04 and
# 24.04 copies, which were byte-identical to it).
_BASH_SCRIPT = _REPO_ROOT / "distros" / "ubuntu26.04" / "disable_auto_upgrade.sh"

# What Ubuntu ships in /etc/apt/apt.conf.d/20auto-upgrades.
_UBUNTU_FILE = ('APT::Periodic::Update-Package-Lists "1";\n'
                'APT::Periodic::Unattended-Upgrade "1";\n')
_UBUNTU_FILE_OFF = ('APT::Periodic::Update-Package-Lists "1";\n'
                    'APT::Periodic::Unattended-Upgrade "0";\n')

_SYSTEMCTL_CALLS = [
    mock.call(["systemctl", "stop", "unattended-upgrades.service"],
              "disable-auto-upgrade", env=mock.ANY),
    mock.call(["systemctl", "disable", "unattended-upgrades.service"],
              "disable-auto-upgrade", env=mock.ANY),
    mock.call(["systemctl", "mask", "unattended-upgrades.service"],
              "disable-auto-upgrade", env=mock.ANY),
]


# --------------------------------------------------------------------------- #
# Pure helper: real inputs, no mocks. Expected values match GNU sed's output.
# --------------------------------------------------------------------------- #
class SetUnattendedUpgradeOffTests(unittest.TestCase):
    def test_ubuntu_file(self):
        self.assertEqual(
            disable_auto_upgrade.set_unattended_upgrade_off(_UBUNTU_FILE),
            _UBUNTU_FILE_OFF)

    def test_keeps_indentation_and_replaces_rest_of_line(self):
        # `.*` consumes everything after the opening quote, comments included.
        self.assertEqual(
            disable_auto_upgrade.set_unattended_upgrade_off(
                '  APT::Periodic::Unattended-Upgrade "1"; // note\n'),
            '  APT::Periodic::Unattended-Upgrade "0";\n')

    def test_text_before_the_match_is_kept(self):
        # Only the first match on a line is replaced, but `.*` already reached
        # the end of the line, so the second setting is swallowed (sed does this).
        self.assertEqual(
            disable_auto_upgrade.set_unattended_upgrade_off(
                'x APT::Periodic::Unattended-Upgrade "1"; '
                'APT::Periodic::Unattended-Upgrade "1";\n'),
            'x APT::Periodic::Unattended-Upgrade "0";\n')

    def test_unquoted_value_is_not_matched(self):
        line = "APT::Periodic::Unattended-Upgrade 1;\n"
        self.assertEqual(disable_auto_upgrade.set_unattended_upgrade_off(line), line)

    def test_missing_final_newline_is_not_added(self):
        self.assertEqual(
            disable_auto_upgrade.set_unattended_upgrade_off(
                'APT::Periodic::Unattended-Upgrade "1";'),
            'APT::Periodic::Unattended-Upgrade "0";')

    def test_already_off_is_unchanged(self):
        self.assertEqual(
            disable_auto_upgrade.set_unattended_upgrade_off(_UBUNTU_FILE_OFF),
            _UBUNTU_FILE_OFF)


# --------------------------------------------------------------------------- #
# install(): temp file + mocked systemctl.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        # A fresh temporary directory per test; cleaned up automatically.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "20auto-upgrades")

    def _install(self, *, contents=_UBUNTU_FILE, systemctl_rcs=(0, 0, 0)):
        """Run install() with the apt file in a temp dir and systemctl mocked.

        contents=None means the apt file does not exist. systemctl_rcs are the
        return codes of the successive systemctl calls.
        """
        if contents is not None:
            with open(self.path, "w", encoding="utf-8", newline="") as f:
                f.write(contents)
        env = {"DISTRIBUTION": "ubuntu26.04"}
        with mock.patch.object(disable_auto_upgrade, "_AUTO_UPGRADES_FILE",
                               self.path), \
             mock.patch.object(disable_auto_upgrade, "exec_program",
                               side_effect=list(systemctl_rcs)) as run, \
             mock.patch.object(disable_auto_upgrade, "log_error") as log_error:
            rc = disable_auto_upgrade.install(env)
        return rc, run, log_error

    def _read(self):
        with open(self.path, encoding="utf-8", newline="") as f:
            return f.read()

    def test_success_edits_file_then_stops_disables_masks(self):
        rc, run, log_error = self._install()

        self.assertEqual(rc, 0)
        self.assertEqual(self._read(), _UBUNTU_FILE_OFF)
        self.assertEqual(run.call_args_list, _SYSTEMCTL_CALLS)
        log_error.assert_not_called()

    def test_crlf_line_endings_are_preserved(self):
        rc, _, _ = self._install(
            contents='APT::Periodic::Unattended-Upgrade "1";\r\n')

        self.assertEqual(rc, 0)
        # `.*` also consumes the "\r" (sed treats it as an ordinary character).
        self.assertEqual(self._read(), 'APT::Periodic::Unattended-Upgrade "0";\n')

    def test_missing_file_returns_3_without_running_systemctl(self):
        # This is also what happens on RPM distros, which have no apt config.
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
        # only for code in disable_auto_upgrade.
        with mock.patch.object(disable_auto_upgrade, "open", create=True,
                               side_effect=fail_on_write):
            rc, run, log_error = self._install()

        self.assertEqual(rc, 3)
        run.assert_not_called()
        self.assertEqual(self._read(), _UBUNTU_FILE)
        self.assertIn("permission denied", log_error.call_args[0][1])

    def test_each_systemctl_failure_returns_3_and_stops(self):
        for failing in range(3):
            with self.subTest(failing=_SYSTEMCTL_CALLS[failing].args[0][1]):
                rcs = [0, 0, 0]
                rcs[failing] = 1
                rc, run, log_error = self._install(systemctl_rcs=rcs)

                self.assertEqual(rc, 3)
                # Calls up to and including the failing one; nothing after.
                self.assertEqual(run.call_args_list,
                                 _SYSTEMCTL_CALLS[:failing + 1])
                log_error.assert_called()


# --------------------------------------------------------------------------- #
# Bash parity: the port mirrors distros/ubuntu26.04/disable_auto_upgrade.sh,
# the only copy left upstream.
# --------------------------------------------------------------------------- #
class BashSourceTests(unittest.TestCase):
    def setUp(self):
        self.script = _BASH_SCRIPT.read_text(encoding="utf-8")

    def test_docstring_quotes_the_ubuntu2604_script(self):
        # Every non-blank line of the script after the shebang appears, indented
        # by four spaces, in the module docstring.
        doc = disable_auto_upgrade.__doc__
        for line in self.script.splitlines()[1:]:
            if line.strip():
                with self.subTest(line=line):
                    self.assertIn("    " + line + "\n", doc)

    def test_constants_match_the_script(self):
        self.assertIn(
            "sed -i 's/APT::Periodic::Unattended-Upgrade \".*/"
            f"{disable_auto_upgrade._SETTING_OFF}/' "
            f"{disable_auto_upgrade._AUTO_UPGRADES_FILE}\n", self.script)
        self.assertEqual(disable_auto_upgrade._SETTING_PATTERN.pattern,
                         'APT::Periodic::Unattended-Upgrade ".*')
        for call in _SYSTEMCTL_CALLS:
            with self.subTest(verb=call.args[0][1]):
                self.assertIn(" ".join(call.args[0]) + "\n", self.script)

    def test_ubuntu2604_install_sh_still_runs_it(self):
        install_sh = (_REPO_ROOT / "distros" / "ubuntu26.04" / "install.sh")
        self.assertIn("\n./disable_auto_upgrade.sh\n",
                      install_sh.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Wiring: upstream removed disable_auto_upgrade.sh from Ubuntu 22.04 and 24.04
# (#623; Packer handles it now), so the build plan no longer runs this module.
# Only Ubuntu 26.04 still runs the script; the module is kept for the future
# Ubuntu 26.04 build plan (ADO 993).
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_is_no_longer_in_the_build_plan(self):
        for os_name in ("Ubuntu24", "Ubuntu22"):
            with self.subTest(os=os_name):
                cfg = BuildConfig(vendor="NVidia", gpu="A100", os=os_name,
                                  fips=False, spec_path=None)
                ops = [s.op for s in build_plan(cfg)]
                self.assertNotIn("disable-auto-upgrade", ops)


if __name__ == "__main__":
    unittest.main()
