"""Unit tests for utils/dkms.py (port of check_dkms_status in utils/utilities.sh).

installed_pattern() was compared against the real `grep -q "${kernel}.*: installed"`
in WSL on realistic `dkms status` lines and kernel releases (including ones
with "+" and ones where "." must match any character).

Run from the repo root:
    python3 -m unittest tests.python.test_dkms -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from utils import dkms

_OP = "install-test"
_KERNEL = "5.14.0-687.el9.x86_64"
_INSTALLED = f"lustre-client/2.15.8_35_gdf4872c, {_KERNEL}, x86_64: installed\n"
_BUILT = f"lustre-client/2.15.8_35_gdf4872c, {_KERNEL}, x86_64: built\n"
_OTHER_KERNEL = "lustre-client/2.15.8_35_gdf4872c, 5.14.0-611.el9.x86_64, x86_64: installed\n"


class HelperTests(unittest.TestCase):
    def test_status_lines_mirror_read_loop(self):
        self.assertEqual(dkms.status_lines("a\nb\n"), ["a", "b"])
        # `read` skips a final line without a newline.
        self.assertEqual(dkms.status_lines("a\nb"), ["a"])
        self.assertEqual(dkms.status_lines(""), [])
        # Empty lines still count as entries (found=1).
        self.assertEqual(dkms.status_lines("\n"), [""])

    def test_installed_pattern(self):
        pattern = dkms.installed_pattern(_KERNEL)
        self.assertTrue(pattern.search(_INSTALLED.strip()))
        self.assertFalse(pattern.search(_BUILT.strip()))
        self.assertFalse(pattern.search(_OTHER_KERNEL.strip()))
        self.assertTrue(pattern.search(
            f"m/1, {_KERNEL}, x86_64: installed (WARNING! Diff between built and "
            "installed module!)"))
        # "." is any character in grep's BRE, as in bash.
        self.assertTrue(pattern.search("m/1, 5X14Y0-687.el9.x86_64, x86_64: installed"))
        # "+" is literal in a BRE.
        plus = dkms.installed_pattern("5.15.0-1+custom")
        self.assertTrue(plus.search("m, 5.15.0-1+custom, x86_64: installed"))
        self.assertFalse(plus.search("m, 5.15.0-11custom, x86_64: installed"))

    def test_running_kernel_is_platform_release(self):
        with mock.patch.object(dkms.platform, "release", return_value="6.8.0-1-azure"):
            self.assertEqual(dkms.running_kernel(), "6.8.0-1-azure")


class MakeLogTailTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(dkms, "_DKMS_TREE", self._tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _make_log(self, version, text):
        build = os.path.join(self._tmp.name, "lustre-client", version, "build")
        os.makedirs(build)
        path = os.path.join(build, "make.log")
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return path

    def test_no_logs(self):
        self.assertEqual(dkms.make_log_tails("lustre-client"), "")

    def test_tails_last_80_lines_per_log_in_order(self):
        long_log = "".join(f"line {i}\n" for i in range(1, 101))
        b = self._make_log("2.0", long_log)
        a = self._make_log("1.0", "only\nlast without newline")
        tails = dkms.make_log_tails("lustre-client").split("\n")

        self.assertEqual(tails[:3], [f"--- {a} ---", "only", "last without newline"])
        self.assertEqual(tails[3], f"--- {b} ---")
        self.assertEqual(tails[4:], [f"line {i}" for i in range(21, 101)])

    def test_skips_non_files_and_empty_logs(self):
        os.makedirs(os.path.join(self._tmp.name, "lustre-client", "1.0", "build",
                                 "make.log"))
        empty = self._make_log("2.0", "")
        self.assertEqual(dkms.make_log_tails("lustre-client"), f"--- {empty} ---")


class CheckDkmsStatusTests(unittest.TestCase):
    def _check(self, outputs, modules=("lustre-client",), *, has_dkms=True,
               tails="", kernel=_KERNEL):
        """outputs: module -> stdout of `dkms status -m <module>`."""
        env = {"PATH": "/usr/bin"}
        with mock.patch.object(dkms, "_command_exists",
                               return_value=has_dkms) as exists, \
             mock.patch.object(dkms, "run_capture",
                               side_effect=lambda cmd, op, **kw: (0, outputs[cmd[3]])) \
                as run_capture, \
             mock.patch.object(dkms, "make_log_tails", return_value=tails), \
             mock.patch.object(dkms, "log_info") as log_info, \
             mock.patch.object(dkms, "log_warn") as log_warn, \
             mock.patch.object(dkms, "log_error") as log_error, \
             mock.patch.object(dkms, "log_error_detail") as log_error_detail:
            ok = dkms.check_dkms_status(list(modules), _OP, env, kernel=kernel)
        exists.assert_called_once_with("dkms", env)
        return ok, {"capture": run_capture, "info": log_info, "warn": log_warn,
                    "error": log_error, "detail": log_error_detail, "env": env}

    def test_no_dkms_is_success_with_warning(self):
        ok, m = self._check({}, has_dkms=False)
        self.assertTrue(ok)
        m["capture"].assert_not_called()
        m["warn"].assert_called_once()

    def test_installed(self):
        ok, m = self._check({"lustre-client": _OTHER_KERNEL + _INSTALLED})
        self.assertTrue(ok)
        m["capture"].assert_called_once_with(
            ["dkms", "status", "-m", "lustre-client"], _OP, env=m["env"],
            failure_is_error=False)
        messages = [c.args[1] for c in m["info"].call_args_list]
        self.assertIn(f"DKMS status: {_INSTALLED.strip()}", messages)
        self.assertIn(f"DKMS status: {_OTHER_KERNEL.strip()}", messages)
        self.assertEqual(m["info"].call_args_list[-1].kwargs,
                         {"module": "lustre-client", "kernel": _KERNEL,
                          "dkms_status": "installed"})
        m["error"].assert_not_called()
        m["detail"].assert_not_called()

    def test_no_status_entry(self):
        ok, m = self._check({"lustre-client": ""})
        self.assertFalse(ok)
        m["error"].assert_called_once_with(
            _OP, "DKMS module lustre-client has no status entry",
            module="lustre-client", kernel=_KERNEL)

    def test_not_installed_logs_make_log_tails(self):
        ok, m = self._check({"lustre-client": _BUILT + _OTHER_KERNEL},
                            tails="--- x/make.log ---\nerror: boom")
        self.assertFalse(ok)
        m["detail"].assert_called_once_with(
            _OP, f"DKMS module lustre-client is not installed for {_KERNEL}",
            "--- x/make.log ---\nerror: boom", module="lustre-client", kernel=_KERNEL)

    def test_not_installed_without_make_log(self):
        ok, m = self._check({"lustre-client": _BUILT})
        self.assertFalse(ok)
        self.assertIn("no make.log found", m["detail"].call_args.args[2])

    def test_stops_at_first_failing_module(self):
        ok, m = self._check({"a": _BUILT, "b": _INSTALLED}, modules=("a", "b"))
        self.assertFalse(ok)
        self.assertEqual([c.args[0][3] for c in m["capture"].call_args_list], ["a"])

    def test_checks_every_module(self):
        ok, m = self._check({"a": _INSTALLED, "b": _INSTALLED}, modules=("a", "b"))
        self.assertTrue(ok)
        self.assertEqual([c.args[0][3] for c in m["capture"].call_args_list], ["a", "b"])

    def test_defaults_to_running_kernel(self):
        with mock.patch.object(dkms, "running_kernel", return_value=_KERNEL):
            ok, _m = self._check({"lustre-client": _INSTALLED}, kernel=None)
        self.assertTrue(ok)

    def test_command_exists_uses_env_path(self):
        with mock.patch.object(dkms.shutil, "which", return_value=None) as which:
            self.assertFalse(dkms._command_exists("dkms", {"PATH": "/x"}))
        which.assert_called_once_with("dkms", path="/x")


if __name__ == "__main__":
    unittest.main()
