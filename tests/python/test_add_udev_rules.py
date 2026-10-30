"""Unit tests for components/python/add_udev_rules.py.

install() does two kinds of thing, and the tests handle each differently:

  * It appends text to two files. The tests point the module's file-path
    constants at a temporary directory and check the real file contents, so
    nothing under /etc is touched.
  * It runs udevadm twice. The tests replace utils.process.exec_program with
    a mock, so udevadm never runs; they check which commands would have run,
    in order.

How the mocking works: `mock.patch.object(module, "name", value)` temporarily
replaces `module.name`, only inside the `with` block.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from components.python import add_udev_rules
from utils import files, process
from utils.build_config import BuildConfig, build_plan

_RELOAD = ["udevadm", "control", "--reload-rules"]
_TRIGGER = ["udevadm", "trigger"]

# The last rule line of each block, as it appears in components/add-udev-rules.sh.
_RDMA_RULE = 'ACTION=="add", SUBSYSTEM=="infiniband", PROGRAM="rdma_rename %k NAME_KERNEL"\n'
_ISSM_RULE = 'KERNEL=="issm*", NAME="infiniband/%k"\n'


# --------------------------------------------------------------------------- #
# The rules text itself.
# --------------------------------------------------------------------------- #
class RulesTextTests(unittest.TestCase):
    def test_rdma_block_matches_the_bash_heredoc(self):
        text = add_udev_rules._RDMA_NAMING_RULES
        self.assertTrue(text.startswith(
            "# SPDX-License-Identifier: (GPL-2.0 OR Linux-OpenIB)\n"))
        self.assertTrue(text.endswith("\n\n# Suppressing/overwriting the udev rule "
                                      "rdma_core package. NAME_KERNEL leaves the IB "
                                      "devices names as provided by the kernel.\n"
                                      + _RDMA_RULE))
        # 29 lines, like the heredoc. (Checked byte-for-byte against real bash
        # when the port was written.)
        self.assertEqual(text.count("\n"), 29)

    def test_issm_block_is_one_line(self):
        self.assertEqual(add_udev_rules._ISSM_RULES, _ISSM_RULE)


# --------------------------------------------------------------------------- #
# install(): files in a temporary directory, udevadm mocked.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        # A fresh temporary directory for each test, removed afterwards.
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.rdma_file = os.path.join(self._tmp.name, "60-rdma-persistent-naming.rules")
        self.issm_file = os.path.join(self._tmp.name, "99-issm.rules")

    def _install(self, *, exec_results=(0, 0)):
        """Run install() with the files redirected and exec_program mocked.

        exec_results are the exit codes the fake udevadm returns, one per call.
        Returns (rc, exec_program mock, write_file's log_error mock).
        """
        with mock.patch.object(add_udev_rules, "_RDMA_NAMING_RULES_FILE",
                               self.rdma_file), \
             mock.patch.object(add_udev_rules, "_ISSM_RULES_FILE", self.issm_file), \
             mock.patch.object(process, "exec_program",
                               side_effect=list(exec_results)) as run, \
             mock.patch.object(files, "log_error") as log_error:
            rc = add_udev_rules.install({"DISTRIBUTION": "ubuntu24.04"})
        return rc, run, log_error

    def _read(self, path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    # ---- success paths -----------------------------------------------------
    def test_creates_both_files_then_reloads_and_triggers(self):
        rc, run, log_error = self._install()

        self.assertEqual(rc, 0)
        self.assertEqual(self._read(self.rdma_file), add_udev_rules._RDMA_NAMING_RULES)
        self.assertEqual(self._read(self.issm_file), _ISSM_RULE)
        env = {"DISTRIBUTION": "ubuntu24.04"}
        self.assertEqual(run.call_args_list, [
            mock.call(_RELOAD, "add-udev-rules", cwd=None, env=env),
            mock.call(_TRIGGER, "add-udev-rules", cwd=None, env=env),
        ])
        log_error.assert_not_called()

    def test_appends_to_existing_files(self):
        # `>>` keeps what is already there and adds to the end.
        with open(self.issm_file, "w", encoding="utf-8") as f:
            f.write("existing rule\n")

        rc, _, _ = self._install()

        self.assertEqual(rc, 0)
        self.assertEqual(self._read(self.issm_file), "existing rule\n" + _ISSM_RULE)

    def test_rerun_appends_the_blocks_again(self):
        self._install()
        rc, _, _ = self._install(exec_results=(0, 0))

        self.assertEqual(rc, 0)
        self.assertEqual(self._read(self.issm_file), _ISSM_RULE * 2)
        self.assertEqual(self._read(self.rdma_file),
                         add_udev_rules._RDMA_NAMING_RULES * 2)

    def test_missing_rules_directory_is_created(self):
        # write_file creates missing parents (bash's `>>` would fail).
        self.rdma_file = os.path.join(self._tmp.name, "rules.d", "60.rules")
        rc, _, _ = self._install()
        self.assertEqual(rc, 0)
        self.assertEqual(self._read(self.rdma_file), add_udev_rules._RDMA_NAMING_RULES)

    # ---- failure paths: each must return 3 and stop ------------------------
    def test_unwritable_first_file_returns_3_and_stops(self):
        os.mkdir(self.rdma_file)   # a directory where the file should be

        rc, run, log_error = self._install()

        self.assertEqual(rc, 3)
        self.assertFalse(os.path.exists(self.issm_file))   # second append never ran
        run.assert_not_called()
        log_error.assert_called_once()

    def test_unwritable_second_file_returns_3_before_udevadm(self):
        os.mkdir(self.issm_file)   # a directory where the file should be

        rc, run, log_error = self._install()

        self.assertEqual(rc, 3)
        # Like bash, the first file was already appended to.
        self.assertEqual(self._read(self.rdma_file), add_udev_rules._RDMA_NAMING_RULES)
        run.assert_not_called()
        log_error.assert_called_once()

    def test_reload_failure_returns_3_without_trigger(self):
        rc, run, log_error = self._install(exec_results=(1,))

        self.assertEqual(rc, 3)
        self.assertEqual(run.call_args_list,
                         [mock.call(_RELOAD, "add-udev-rules", cwd=None, env=mock.ANY)])
        log_error.assert_not_called()      # exec_program logs the failed command

    def test_trigger_failure_returns_3(self):
        rc, run, log_error = self._install(exec_results=(0, 127))

        self.assertEqual(rc, 3)
        self.assertEqual(run.call_count, 2)
        log_error.assert_not_called()


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "add-udev-rules"]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, add_udev_rules.install)
        self.assertEqual(steps[0].script, "add-udev-rules.sh")


if __name__ == "__main__":
    unittest.main()
