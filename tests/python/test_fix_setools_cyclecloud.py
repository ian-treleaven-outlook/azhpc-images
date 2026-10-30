"""Unit tests for components/python/fix_setools_cyclecloud.py.

  * The legacy lookup, the log messages and the sed edit are compared with
    components/fix_setools_cyclecloud.sh (the sed one against real GNU sed,
    when it is there), so they can't drift from the bash.
  * install() runs with rpm and semanage mocked and setools/__init__.py in a
    temporary directory, so no package or system file is touched.

Run from the repo root:
    python3 -m unittest tests.python.test_fix_setools_cyclecloud -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import fix_setools_cyclecloud as m
from utils.plans import steps

_OP = "fix-setools-cyclecloud"
_BASH_SCRIPT = (Path(__file__).resolve().parents[2]
                / "components" / "fix_setools_cyclecloud.sh")
_ENV = {"DISTRIBUTION": "rocky9.8"}
_LEGACY = '__version__ = pkg_resources.get_distribution("setools").version'
# A cut-down setools/__init__.py as EL9's SETools 4.4 ships it.
_INIT_TEXT = f'import pkg_resources\n\n{_LEGACY}\n\nfrom .exception import *\n'


def _bash_text() -> str:
    return _BASH_SCRIPT.read_text(encoding="utf-8")


class BashParityTests(unittest.TestCase):
    def test_legacy_lookup_is_the_line_bash_greps_for(self):
        self.assertIn(f"grep -Fq '{m._LEGACY_LOOKUP}'", _bash_text())

    def test_messages_are_the_bash_ones(self):
        for message in ("python3-setools is not installed, skipping patch",
                        "setools does not use the legacy version lookup, skipping patch",
                        "ERROR: unable to locate the installed setools/__init__.py",
                        "ERROR: semanage still fails after patching setools",
                        "Verified: semanage works with setools"):
            with self.subTest(message=message):
                self.assertIn(message, _bash_text())

    @unittest.skipUnless(shutil.which("bash") and shutil.which("sed") and os.name == "posix",
                         "needs bash and GNU sed")
    def test_pin_version_matches_the_real_sed(self):
        sed_line = next(line for line in _bash_text().splitlines()
                        if line.startswith("sed -i "))
        texts = [_INIT_TEXT, f"{_LEGACY}  # {_LEGACY}\r\n", "no lookup here\n", _LEGACY]
        with tempfile.TemporaryDirectory() as tmp:
            init_file = Path(tmp) / "__init__.py"
            for text in texts:
                with self.subTest(text=text):
                    init_file.write_bytes(text.encode())
                    script = f"SETOOLS_VERSION=4.4.4; SETOOLS_INIT={init_file}; {sed_line}"
                    subprocess.run(["bash", "-ec", script], check=True)
                    self.assertEqual(init_file.read_bytes().decode(),
                                     m.pin_version(text, "4.4.4"))


class PinVersionTests(unittest.TestCase):
    def test_replaces_the_lookup_with_the_version(self):
        self.assertEqual(m.pin_version(_INIT_TEXT, "4.4.4"),
                         'import pkg_resources\n\n__version__ = "4.4.4"\n\n'
                         'from .exception import *\n')

    def test_only_the_first_match_on_a_line_like_sed_without_g(self):
        self.assertEqual(m.pin_version(f"{_LEGACY} {_LEGACY}\n", "1"),
                         f'__version__ = "1" {_LEGACY}\n')

    def test_keeps_crlf_line_endings(self):
        self.assertEqual(m.pin_version(f"{_LEGACY}\r\nx\r\n", "1"), '__version__ = "1"\r\nx\r\n')


class InstallTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.init_file = Path(tmp.name) / "site-packages" / "setools" / "__init__.py"
        self.init_file.parent.mkdir(parents=True)
        self.init_file.write_text(_INIT_TEXT, encoding="utf-8")
        for target in ("log_info", "log_warn", "log_error"):
            patcher = mock.patch.object(m, target)
            setattr(self, target, patcher.start())
            self.addCleanup(patcher.stop)
        patcher = mock.patch("utils.files.log_info")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _install(self, *, version="4.4.4", file_list=None, semanage_ok=True):
        """Run install() with rpm (version and file list) and semanage mocked."""
        if file_list is None:
            file_list = f"/usr/lib64/python3.9/site-packages/setools/policyrep.so\n{self.init_file}\n"
        with mock.patch.object(m, "installed_package_version", return_value=version) as query, \
             mock.patch.object(m, "run_capture", return_value=(0, file_list)) as rpm_ql, \
             mock.patch.object(m, "run", return_value=semanage_ok) as semanage:
            rc = m.install(_ENV)
        self.query, self.rpm_ql, self.semanage = query, rpm_ql, semanage
        return rc

    def test_patches_the_lookup_and_checks_semanage(self):
        self.assertEqual(self._install(), 0)

        self.assertIn('__version__ = "4.4.4"', self.init_file.read_text(encoding="utf-8"))
        self.assertNotIn(_LEGACY, self.init_file.read_text(encoding="utf-8"))
        self.query.assert_called_once_with("python3-setools", _ENV, _OP,
                                           rpm_format="%{VERSION}", missing_ok=True)
        self.assertEqual(self.rpm_ql.call_args.args[0], ["rpm", "-ql", "python3-setools"])
        self.semanage.assert_called_once_with(["/usr/sbin/semanage", "--help"], _OP, env=_ENV)
        self.log_error.assert_not_called()

    def test_not_installed_skips_with_info(self):
        self.assertEqual(self._install(version=None), 0)
        self.rpm_ql.assert_not_called()
        self.semanage.assert_not_called()
        self.assertIn("not installed, skipping", self.log_info.call_args.args[1])

    def test_newer_setools_without_the_lookup_is_left_alone(self):
        self.init_file.write_text("__version__ = '4.5.1'\n", encoding="utf-8")
        self.assertEqual(self._install(), 0)
        self.assertEqual(self.init_file.read_text(encoding="utf-8"), "__version__ = '4.5.1'\n")
        self.semanage.assert_not_called()
        self.assertIn("skipping patch", self.log_info.call_args.args[1])

    def test_no_init_file_in_the_package_fails(self):
        self.assertEqual(self._install(file_list="/usr/bin/sesearch\n"), 3)
        self.log_error.assert_called_once()
        self.semanage.assert_not_called()

    def test_listed_init_file_that_does_not_exist_fails(self):
        missing = self.init_file.parent / "gone" / "setools" / "__init__.py"
        self.assertEqual(self._install(file_list=f"{missing}\n"), 3)
        self.log_error.assert_called_once()

    def test_two_matching_files_fail_like_bash(self):
        self.assertEqual(self._install(file_list=f"{self.init_file}\n{self.init_file}\n"), 3)
        self.assertEqual(self.init_file.read_text(encoding="utf-8"), _INIT_TEXT)

    def test_semanage_still_failing_fails_the_step(self):
        self.assertEqual(self._install(semanage_ok=False), 3)
        self.assertIn("semanage still fails", self.log_error.call_args.args[1])
        # The file stays patched, as with bash.
        self.assertIn('__version__ = "4.4.4"', self.init_file.read_text(encoding="utf-8"))


class StepWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action(self):
        step = steps.setools_cyclecloud()
        self.assertEqual((step.op, step.script, step.base),
                         (_OP, "fix_setools_cyclecloud.sh", "component"))
        self.assertIs(step.action, m.install)
        self.assertEqual(step.runner, "python")


if __name__ == "__main__":
    unittest.main()
