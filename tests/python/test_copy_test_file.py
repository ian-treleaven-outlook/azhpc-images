"""Unit tests for components/python/copy_test_file.py.

Three kinds of test:

  * The pure helpers (top_level_test_files, gcc_major_version, nvcc_command)
    are tested directly with real inputs.
  * install() is tested against a small fake "tests" tree in a temporary
    directory. The module's path constants (destination, nvcc) are pointed
    into that directory, so real files are copied but nothing under /opt or
    /usr is touched. gcc and nvcc never run: run_capture and exec_program are
    replaced by mocks.
  * component_env() must provide AZHPC_IMAGES_TEST_DIR, and the build plan
    must call this module.

How the mocking works: `mock.patch.object(module, "name", value)` temporarily
replaces `module.name`, only inside the `with` block.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import copy_test_file
from utils import build_config
from utils.build_config import BuildConfig, build_plan


# --------------------------------------------------------------------------- #
# Pure helpers: real inputs, no mocks. Each was compared with real bash once.
# --------------------------------------------------------------------------- #
class TopLevelTestFilesTests(unittest.TestCase):
    def test_matches_names_with_a_dot_except_dotfiles(self):
        names = ["run-tests.sh", "python", ".hidden.sh", "health_checks",
                 "test-matrix_NVIDIA.json", "trailing.", "x..y"]
        self.assertEqual(copy_test_file.top_level_test_files(names),
                         ["run-tests.sh", "test-matrix_NVIDIA.json", "trailing.", "x..y"])

    def test_no_match_returns_empty_list(self):
        self.assertEqual(copy_test_file.top_level_test_files(["python", ".git"]), [])

    def test_the_real_tests_directory(self):
        # The repo's tests/ today: four files, and no directory with a dot in
        # its name (cp without -r would fail on one).
        tests_dir = Path(__file__).resolve().parents[1]
        names = copy_test_file.top_level_test_files(os.listdir(tests_dir))
        self.assertIn("run-tests.sh", names)
        for name in names:
            self.assertTrue((tests_dir / name).is_file(), name)

    def test_real_health_checks_has_no_symlinks(self):
        # Keeps the copytree rerun difference (module docstring) unreachable.
        health_checks = Path(__file__).resolve().parents[1] / "health_checks"
        self.assertTrue(health_checks.is_dir())
        links = [p for p in health_checks.rglob("*") if p.is_symlink()]
        self.assertEqual(links, [])


class GccMajorVersionTests(unittest.TestCase):
    def test_major_only(self):
        self.assertEqual(copy_test_file.gcc_major_version("13\n"), 13)

    def test_full_version_keeps_text_before_first_dot(self):
        # Like `cut -d. -f1`.
        self.assertEqual(copy_test_file.gcc_major_version("9.4.0\n"), 9)

    def test_unusable_output_returns_none(self):
        for output in ("", "\n", "abc\n", "7\n8\n"):
            with self.subTest(output=output):
                self.assertIsNone(copy_test_file.gcc_major_version(output))


class NvccCommandTests(unittest.TestCase):
    def test_gcc_newer_than_6_uses_the_default_compiler(self):
        self.assertEqual(copy_test_file.nvcc_command("nvcc", 7, "a.cu", "a"),
                         ["nvcc", "-lnuma", "a.cu", "-o", "a"])

    def test_gcc_6_uses_gcc_9_2_from_opt(self):
        self.assertEqual(copy_test_file.nvcc_command("nvcc", 6, "a.cu", "a"),
                         ["nvcc", "--compiler-bindir", "/opt/gcc-9.2.0/bin",
                          "-lnuma", "a.cu", "-o", "a"])


# --------------------------------------------------------------------------- #
# install(): real copies inside a temporary directory; gcc and nvcc mocked.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        # A fresh temporary directory for each test, removed afterwards.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)

        # A small stand-in for the repo's tests/ directory.
        self.src = root / "tests"
        ndv4 = self.src / "health_checks" / "NDv4"
        ndv4.mkdir(parents=True)
        (self.src / "python").mkdir()                     # no dot: not copied
        (self.src / "run-tests.sh").write_text("run\n")
        (self.src / "test-matrix_NVIDIA.json").write_text("{}\n")
        (self.src / ".hidden.sh").write_text("hidden\n")  # dotfile: not copied
        (ndv4 / "gpu-copy.cu").write_text("// cuda\n")
        (ndv4 / "check-hardware.sh").write_text("check\n")
        os.symlink("check-hardware.sh", ndv4 / "link.sh")
        self.cufile = str(ndv4 / "gpu-copy.cu")
        self.outfile = str(ndv4 / "gpu-copy")

        self.dest = root / "opt" / "azurehpc" / "test"     # created by install()
        self.nvcc = root / "nvcc"                          # absent unless created
        self.env = {"AZHPC_IMAGES_TEST_DIR": str(self.src)}

    def _install(self, env=None, *, gcc_result=(0, "13\n"), nvcc_rc=0):
        """Run install() with paths redirected and gcc/nvcc mocked.

        The fake nvcc creates the output binary when it "succeeds", so the
        tests can see that it is compiled before health_checks is copied.
        Returns (rc, run_capture mock, exec_program mock, log_error mock).
        """
        def fake_nvcc(command, op, env=None):
            if nvcc_rc == 0:
                Path(command[-1]).write_text("binary\n")
            return nvcc_rc

        with mock.patch.object(copy_test_file, "_DEST_TEST_DIR", str(self.dest)), \
             mock.patch.object(copy_test_file, "_NVCC", str(self.nvcc)), \
             mock.patch.object(copy_test_file, "run_capture",
                               return_value=gcc_result) as gcc, \
             mock.patch.object(copy_test_file, "exec_program",
                               side_effect=fake_nvcc) as nvcc, \
             mock.patch.object(copy_test_file, "log_error") as log_error:
            rc = copy_test_file.install(self.env if env is None else env)
        return rc, gcc, nvcc, log_error

    def _dest_names(self):
        """Every path under the destination, relative to it, sorted."""
        return sorted(str(p.relative_to(self.dest)) for p in self.dest.rglob("*"))

    # ---- success paths -----------------------------------------------------
    def test_without_nvcc_copies_files_and_health_checks_only(self):
        rc, gcc, nvcc, log_error = self._install()

        self.assertEqual(rc, 0)
        self.assertEqual(self._dest_names(), [
            "health_checks",
            "health_checks/NDv4",
            "health_checks/NDv4/check-hardware.sh",
            "health_checks/NDv4/gpu-copy.cu",
            "health_checks/NDv4/link.sh",
            "run-tests.sh",
            "test-matrix_NVIDIA.json",
        ])
        self.assertEqual((self.dest / "run-tests.sh").read_text(), "run\n")
        # Like cp -r, a symbolic link inside the tree is copied as a link.
        link = self.dest / "health_checks" / "NDv4" / "link.sh"
        self.assertEqual(os.readlink(link), "check-hardware.sh")
        gcc.assert_not_called()
        nvcc.assert_not_called()
        log_error.assert_not_called()

    def test_with_nvcc_and_new_gcc_compiles_then_copies_the_binary(self):
        self.nvcc.write_text("")

        rc, gcc, nvcc, log_error = self._install(gcc_result=(0, "13\n"))

        self.assertEqual(rc, 0)
        gcc.assert_called_once_with(["gcc", "-dumpversion"], "copy-test-file",
                                    env=self.env)
        nvcc.assert_called_once_with(
            [str(self.nvcc), "-lnuma", self.cufile, "-o", self.outfile],
            "copy-test-file", env=self.env)
        # The binary is written next to its source (as in bash), then copied.
        self.assertTrue(Path(self.outfile).is_file())
        self.assertEqual(
            (self.dest / "health_checks" / "NDv4" / "gpu-copy").read_text(), "binary\n")
        log_error.assert_not_called()

    def test_with_nvcc_and_gcc_6_uses_gcc_9_2(self):
        self.nvcc.write_text("")

        rc, _, nvcc, _ = self._install(gcc_result=(0, "6.5.0\n"))

        self.assertEqual(rc, 0)
        nvcc.assert_called_once_with(
            [str(self.nvcc), "--compiler-bindir", "/opt/gcc-9.2.0/bin",
             "-lnuma", self.cufile, "-o", self.outfile],
            "copy-test-file", env=self.env)

    def test_merges_into_an_existing_destination(self):
        # Like cp -r: existing files stay, matching files are overwritten.
        old_dir = self.dest / "health_checks" / "NDv4"
        old_dir.mkdir(parents=True)
        (old_dir / "old.sh").write_text("old\n")
        (old_dir / "check-hardware.sh").write_text("stale\n")

        rc, _, _, _ = self._install()

        self.assertEqual(rc, 0)
        self.assertEqual((old_dir / "old.sh").read_text(), "old\n")
        self.assertEqual((old_dir / "check-hardware.sh").read_text(), "check\n")

    def test_rerun_succeeds(self):
        # Without symbolic links in the tree, as in the real tests/health_checks.
        (self.src / "health_checks" / "NDv4" / "link.sh").unlink()
        self._install()
        rc, _, _, _ = self._install()
        self.assertEqual(rc, 0)

    def test_rerun_with_a_symlink_in_the_tree_returns_3(self):
        # Known difference (see the module docstring): cp -r would replace the
        # existing link; copytree refuses. Not reachable with the real tree.
        self._install()
        rc, _, _, log_error = self._install()
        self.assertEqual(rc, 3)
        log_error.assert_called_once()

    # ---- failure paths: each must return 3 and stop ------------------------
    def test_missing_test_dir_variable_returns_3(self):
        rc, _, _, log_error = self._install(env={})

        self.assertEqual(rc, 3)
        self.assertFalse(self.dest.exists())       # nothing was done
        log_error.assert_called_once()

    def test_destination_cannot_be_created_returns_3(self):
        # A regular file where a parent directory should be.
        self.dest.parent.parent.mkdir(parents=True)
        self.dest.parent.write_text("")

        rc, _, _, log_error = self._install()

        self.assertEqual(rc, 3)
        log_error.assert_called_once()

    def test_no_matching_top_level_files_returns_3(self):
        # bash: cp gets the literal "*.*" and fails.
        for name in ("run-tests.sh", "test-matrix_NVIDIA.json"):
            (self.src / name).unlink()

        rc, _, _, log_error = self._install()

        self.assertEqual(rc, 3)
        self.assertFalse((self.dest / "health_checks").exists())
        log_error.assert_called_once()

    def test_directory_matching_the_glob_returns_3(self):
        # cp without -r fails on a directory; the step stops there.
        (self.src / "data.d").mkdir()
        self.nvcc.write_text("")

        rc, gcc, _, log_error = self._install()

        self.assertEqual(rc, 3)
        gcc.assert_not_called()
        self.assertFalse((self.dest / "health_checks").exists())
        log_error.assert_called_once()

    def test_gcc_probe_failure_returns_3_without_compiling(self):
        self.nvcc.write_text("")

        rc, _, nvcc, log_error = self._install(gcc_result=(127, ""))

        self.assertEqual(rc, 3)
        nvcc.assert_not_called()
        self.assertFalse((self.dest / "health_checks").exists())
        log_error.assert_called_once()

    def test_unusable_gcc_version_returns_3_without_compiling(self):
        self.nvcc.write_text("")

        rc, _, nvcc, log_error = self._install(gcc_result=(0, "abc\n"))

        self.assertEqual(rc, 3)
        nvcc.assert_not_called()
        log_error.assert_called_once()

    def test_compile_failure_returns_3_before_copying_health_checks(self):
        self.nvcc.write_text("")

        rc, _, _, log_error = self._install(nvcc_rc=1)

        self.assertEqual(rc, 3)
        self.assertFalse((self.dest / "health_checks").exists())
        log_error.assert_called_once()

    def test_health_checks_copy_failure_returns_3(self):
        # A regular file where the health_checks directory should go.
        self.dest.mkdir(parents=True)
        (self.dest / "health_checks").write_text("")

        rc, _, _, log_error = self._install()

        self.assertEqual(rc, 3)
        log_error.assert_called_once()


# --------------------------------------------------------------------------- #
# component_env() and wiring.
# --------------------------------------------------------------------------- #
class ComponentEnvTests(unittest.TestCase):
    def test_sets_azhpc_images_test_dir_like_set_properties(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        repo = Path("/repo")
        # The two probes read /etc/os-release and run dpkg/rpm; fake them.
        with mock.patch.object(build_config, "_detect_distribution",
                               return_value="ubuntu24.04"), \
             mock.patch.object(build_config, "_detect_arch_distro",
                               return_value="amd64"):
            env = build_config.component_env(repo, cfg)

        # utils/set_properties.sh: export AZHPC_IMAGES_TEST_DIR=$TOP_DIR/tests
        self.assertEqual(env["AZHPC_IMAGES_TEST_DIR"], str(repo / "tests"))
        self.assertEqual(env["TEST_DIR"], str(repo / "tests"))


class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "copy-test-file"]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, copy_test_file.install)
        self.assertEqual(steps[0].script, "copy_test_file.sh")


if __name__ == "__main__":
    unittest.main()
