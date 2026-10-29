"""Unit tests for components/python/install_amd_libs.py.

Two kinds of test:

  * The pure helpers (folder_version, aocl_modulefile) and the small symlink
    helper (make_symlink) run for real; make_symlink works in a temporary folder.
  * install() runs with its external effects replaced. The downloads, tar, the
    vendor installers and cp are faked by FakeCommands below, which records each
    command and creates the files the real command would have created. The
    install folder (/opt/amd), the work folder and the modulefile folder are all
    pointed at a temporary folder, so the tests never touch /opt or the network.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_amd_libs
from utils.build_config import BuildConfig, build_plan
from utils.download import ChecksumError

_AOCL = "5.3.0"
_AOCC = "5.2.0"
_AOCL_URL = ("https://download.amd.com/developer/eula/aocl/aocl-5-3/"
             "aocl-linux-aocc-5.3.0.tar.gz")
_AOCC_URL = ("https://download.amd.com/developer/eula/aocc/aocc-5-2/"
             "aocc-compiler-5.2.0.tar")


def _versions(aocl=True, aocc=True):
    amd = {}
    if aocl:
        amd["aocl"] = {"version": _AOCL, "sha256": "aocl-sha"}
    if aocc:
        amd["aocc"] = {"version": _AOCC, "sha256": "aocc-sha"}
    return json.dumps({"amd": {"common": amd}})


# --------------------------------------------------------------------------- #
# Pure helpers: real inputs, no mocks.
# --------------------------------------------------------------------------- #
class FolderVersionTests(unittest.TestCase):
    # Expected values were produced by the real
    # `echo $V | cut -d'.' -f1-2 --output-delimiter='-'`.
    def test_matches_cut(self):
        cases = {"5.3.0": "5-3", "5.2.0": "5-2", "5.3": "5-3", "5": "5",
                 "10.20.30.40": "10-20", "5..3": "5-"}
        for version, expected in cases.items():
            with self.subTest(version=version):
                self.assertEqual(install_amd_libs.folder_version(version), expected)


class AoclModulefileTests(unittest.TestCase):
    def test_exact_heredoc_text(self):
        self.assertEqual(
            install_amd_libs.aocl_modulefile("/opt/amd"),
            "#%Module 1.0\n"
            "#\n"
            "#  AOCL\n"
            "#\n"
            "prepend-path    LD_LIBRARY_PATH   /opt/amd/lib\n"
            "setenv          AMD_FFTW_INCLUDE  /opt/amd/include\n")


class MakeSymlinkTests(unittest.TestCase):
    """make_symlink was also compared against real `ln` in every case below."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.target = os.path.join(self.root, "target")
        os.makedirs(self.target)
        self.link = os.path.join(self.root, "link")

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_a_new_link(self):
        install_amd_libs.make_symlink(self.target, self.link, replace=False)
        self.assertEqual(os.readlink(self.link), self.target)

    def test_replace_overwrites_an_existing_symlink_to_a_folder(self):
        # ln -sfn: -n makes a symlink to a folder count as a plain file.
        other = os.path.join(self.root, "other")
        os.makedirs(other)
        os.symlink(other, self.link)
        install_amd_libs.make_symlink(self.target, self.link, replace=True)
        self.assertEqual(os.readlink(self.link), self.target)

    def test_replace_creates_the_link_inside_a_real_folder(self):
        # ln -sfn into a real folder puts the link inside it, as ln does.
        os.makedirs(self.link)
        install_amd_libs.make_symlink(self.target, self.link, replace=True)
        self.assertEqual(os.readlink(os.path.join(self.link, "target")), self.target)

    def test_without_replace_an_existing_link_is_an_error(self):
        # ln -s (no -f) fails with "File exists".
        os.symlink("/somewhere", self.link)
        with self.assertRaises(FileExistsError):
            install_amd_libs.make_symlink(self.target, self.link, replace=False)
        self.assertEqual(os.readlink(self.link), "/somewhere")


# --------------------------------------------------------------------------- #
# install(): external effects faked, file system in a temporary folder.
# --------------------------------------------------------------------------- #
class FakeCommands:
    """Stands in for exec_program and download_and_verify.

    Records every command and, so the following steps have something to work
    with, imitates what the real command would leave behind:
      * download: returns <dest_dir>/<file name from the URL>
      * tar -xvf X.tar[.gz]: creates the folder X in the work folder
      * AOCL ./install.sh: creates amd/<version>/aocc/{MT,ST}/{lib,include}
      * cp -r: copies for real (shutil.copytree into the destination)

    fail_at=N makes the Nth command (counting from 0) return exit code 1.
    mt_layout=False makes the AOCL installer produce no MT folder.
    """

    def __init__(self, fail_at=None, mt_layout=True):
        self.commands = []
        self.downloads = []
        self.fail_at = fail_at
        self.mt_layout = mt_layout

    def download(self, url, sha256, dest_dir="."):
        self.downloads.append((url, sha256, dest_dir))
        return Path(dest_dir) / os.path.basename(url)

    def run(self, command, op, *, cwd=None, env=None):
        self.commands.append((command, cwd))
        if len(self.commands) - 1 == self.fail_at:
            return 1
        if command[0] == "tar":
            name = os.path.basename(command[2])
            for suffix in (".tar.gz", ".tar"):
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            os.makedirs(os.path.join(cwd, name, "bin"))
        elif command[0] == "./install.sh" and "-t" in command:
            aocc = os.path.join(cwd, "amd", _AOCL, "aocc")
            for build in (("MT", "ST") if self.mt_layout else ("ST",)):
                for sub in ("lib", "include"):
                    os.makedirs(os.path.join(aocc, build, sub))
        elif command[0] == "cp":
            *sources, dest = command[2:]
            for source in sources:
                shutil.copytree(source, os.path.join(dest, os.path.basename(source)),
                                dirs_exist_ok=True)
        return 0


class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = self._tmp.name
        self.prefix = os.path.join(root, "opt", "amd")
        self.work = os.path.join(root, "work")
        self.modules = os.path.join(root, "modulefiles")
        self.env = {"COMPONENT_VERSIONS": _versions(), "DISTRIBUTION": "ubuntu24.04",
                    "ARCHITECTURE": "x86_64", "MODULE_FILES_DIRECTORY": self.modules}

    def tearDown(self):
        self._tmp.cleanup()

    def _install(self, env=None, fake=None, download_error=None, write_error=None):
        """Run install() with everything external faked; return (rc, fake, record)."""
        fake = fake or FakeCommands()
        download = fake.download
        if download_error is not None:
            # Fail the download whose URL is download_error[0].
            def download(url, sha256, dest_dir="."):
                fake.downloads.append((url, sha256, dest_dir))
                if url == download_error[0]:
                    raise download_error[1]
                return Path(dest_dir) / os.path.basename(url)
        with mock.patch.object(install_amd_libs, "_INSTALL_PREFIX", self.prefix), \
             mock.patch.object(install_amd_libs, "_WORK_DIR", self.work), \
             mock.patch.object(install_amd_libs, "exec_program", side_effect=fake.run), \
             mock.patch.object(install_amd_libs, "download_and_verify",
                               side_effect=download), \
             mock.patch.object(install_amd_libs, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(install_amd_libs, "log_error"):
            rc = install_amd_libs.install(env or self.env)
        return rc, fake, record

    def _expected_commands(self):
        aocl = os.path.join(self.work, "aocl-linux-aocc-5.3.0")
        aocc = os.path.join(self.work, "aocc-compiler-5.2.0")
        aocl_out = os.path.join(aocl, "amd", _AOCL, "aocc")
        return [
            (["tar", "-xvf", f"{self.work}/aocl-linux-aocc-5.3.0.tar.gz"], self.work),
            (["./install.sh", "-t", "amd", "-l", "blis", "fftw", "libflame",
              "-i", "lp64"], aocl),
            # The glob's matches, sorted, as the shell would pass them.
            (["cp", "-r", f"{aocl_out}/MT", f"{aocl_out}/ST", self.prefix], None),
            (["tar", "-xvf", f"{self.work}/aocc-compiler-5.2.0.tar"], self.work),
            (["./install.sh"], aocc),
            (["cp", "-r", aocc, self.prefix], None),
        ]

    # ---- success -----------------------------------------------------------
    def test_installs_aocl_then_aocc(self):
        rc, fake, record = self._install()

        self.assertEqual(rc, 0)
        self.assertEqual(fake.downloads, [(_AOCL_URL, "aocl-sha", self.work),
                                          (_AOCC_URL, "aocc-sha", self.work)])
        self.assertEqual(fake.commands, self._expected_commands())
        self.assertEqual(record.call_args_list,
                         [mock.call("AOCL", _AOCL), mock.call("AOCC", _AOCC)])

        # /opt/amd/lib and /opt/amd/include point at the MT build.
        self.assertEqual(os.readlink(os.path.join(self.prefix, "lib")),
                         os.path.join(self.prefix, "MT", "lib"))
        self.assertEqual(os.readlink(os.path.join(self.prefix, "include")),
                         os.path.join(self.prefix, "MT", "include"))
        # The modulefile, and the amd/aocl symlink to it.
        modulefile = os.path.join(self.modules, "amd", "aocl-5.3.0")
        with open(modulefile, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), install_amd_libs.aocl_modulefile(self.prefix))
        self.assertEqual(os.readlink(os.path.join(self.modules, "amd", "aocl")),
                         modulefile)
        # AOCC copied as a folder; the work folder is gone.
        self.assertTrue(os.path.isdir(os.path.join(self.prefix, "aocc-compiler-5.2.0")))
        self.assertFalse(os.path.exists(self.work))

    def test_rerun_appends_modulefile_and_fails_at_the_aocl_symlink(self):
        # Same as bash: `>>` appends a second copy, then `ln -s` (no -f) fails.
        self.assertEqual(self._install()[0], 0)
        rc, fake, record = self._install()

        self.assertEqual(rc, 3)
        with open(os.path.join(self.modules, "amd", "aocl-5.3.0"), encoding="utf-8") as h:
            self.assertEqual(h.read(), install_amd_libs.aocl_modulefile(self.prefix) * 2)
        record.assert_not_called()
        self.assertEqual(len(fake.commands), 3)         # AOCC was not attempted

    # ---- configuration -----------------------------------------------------
    def test_missing_config_returns_3_before_doing_anything(self):
        for versions in (json.dumps({}), _versions(aocl=False), _versions(aocc=False),
                         json.dumps({"amd": {"common": {
                             "aocl": {"version": _AOCL},                # no sha256
                             "aocc": {"version": _AOCC, "sha256": "s"}}}})):
            with self.subTest(versions=versions):
                env = dict(self.env, COMPONENT_VERSIONS=versions)
                rc, fake, record = self._install(env=env)
                self.assertEqual(rc, 3)
                self.assertEqual(fake.downloads, [])
                self.assertEqual(fake.commands, [])
                record.assert_not_called()

    def test_missing_module_files_directory_returns_3(self):
        env = dict(self.env)
        del env["MODULE_FILES_DIRECTORY"]
        rc, fake, _ = self._install(env=env)
        self.assertEqual(rc, 3)
        self.assertEqual(fake.downloads, [])

    # ---- failures: each returns 3 and stops --------------------------------
    def test_each_failing_command_returns_3_and_stops(self):
        expected = self._expected_commands()
        for index in range(len(expected)):
            with self.subTest(failing=expected[index][0]):
                shutil.rmtree(self.prefix, ignore_errors=True)
                shutil.rmtree(self.modules, ignore_errors=True)
                rc, fake, record = self._install(fake=FakeCommands(fail_at=index))

                self.assertEqual(rc, 3)
                # Everything up to and including the failing command ran; nothing after.
                self.assertEqual(fake.commands, expected[:index + 1])
                # AOCL is recorded only once its part (the first 3 commands) succeeded.
                self.assertEqual(record.call_args_list,
                                 [mock.call("AOCL", _AOCL)] if index >= 3 else [])

    def test_failed_aocl_download_returns_3_and_stops(self):
        rc, fake, record = self._install(download_error=(_AOCL_URL, ChecksumError("bad")))
        self.assertEqual(rc, 3)
        self.assertEqual(fake.commands, [])
        record.assert_not_called()

    def test_failed_aocc_download_returns_3_and_stops(self):
        rc, fake, record = self._install(download_error=(_AOCC_URL, OSError("404")))
        self.assertEqual(rc, 3)
        self.assertEqual(fake.commands, self._expected_commands()[:3])
        record.assert_called_once_with("AOCL", _AOCL)

    def test_incomplete_mt_layout_returns_3(self):
        rc, fake, record = self._install(fake=FakeCommands(mt_layout=False))

        self.assertEqual(rc, 3)
        self.assertEqual(len(fake.commands), 3)          # stopped after the cp
        self.assertFalse(os.path.lexists(os.path.join(self.prefix, "lib")))
        self.assertFalse(os.path.exists(os.path.join(self.modules, "amd")))
        record.assert_not_called()

    def test_unwritable_versions_file_returns_3(self):
        rc, fake, _ = self._install(write_error=PermissionError("denied"))
        self.assertEqual(rc, 3)
        self.assertEqual(len(fake.commands), 3)          # AOCC not attempted


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "install-amd-libs"]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, install_amd_libs.install)
        self.assertEqual(steps[0].script, "install_amd_libs.sh")


if __name__ == "__main__":
    unittest.main()
