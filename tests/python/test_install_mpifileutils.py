"""Unit tests for the Lustre switch in components/python/install_mpifileutils.py.

The rest of the module (deps, config, orchestration) is covered in
test_packaging.py. Here: bash's LUSTRE_CMAKE_FLAG (ON only when
/usr/include/lustre/lustre_user.h exists, upstream #598) and that it reaches
the cmake line. The header path and the build directories point at a
temporary directory; deps, download, extraction, the build and the versions
file are mocked.

Run from the repo root:
    python3 -m unittest tests.python.test_install_mpifileutils -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_mpifileutils as mfu

_OP = "install-mpifileutils"


class LustreFlagTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.header = os.path.join(self.root, "lustre", "lustre_user.h")

    def test_on_only_when_the_header_is_a_file(self):
        self.assertEqual(mfu.lustre_cmake_flag(self.header), "OFF")        # missing
        os.makedirs(self.header)                                            # a directory
        self.assertEqual(mfu.lustre_cmake_flag(self.header), "OFF")
        os.rmdir(self.header)
        Path(self.header).write_text("/* lustre */\n")
        self.assertEqual(mfu.lustre_cmake_flag(self.header), "ON")
        link = os.path.join(self.root, "link.h")                            # [[ -f ]] follows links
        os.symlink(self.header, link)
        self.assertEqual(mfu.lustre_cmake_flag(link), "ON")

    def test_default_path_is_the_bash_one(self):
        self.assertEqual(mfu._LUSTRE_USER_H, "/usr/include/lustre/lustre_user.h")

    def _install(self):
        env = {"COMPONENT_VERSIONS": json.dumps({"mpifileutils": {"common": {
                   "version": "0.12", "url": "https://example/mpifileutils-v0.12.tgz",
                   "sha256": "s"}}}),
               "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "x86_64"}
        paths = {"_LUSTRE_USER_H": self.header,
                 "_INSTALL_PREFIX": os.path.join(self.root, "opt"),
                 "_BUILD_DIR": os.path.join(self.root, "build"),
                 "_SRC_DIR": os.path.join(self.root, "src")}
        with mock.patch.multiple(mfu, **paths), \
             mock.patch.object(mfu, "install_deps", return_value=0), \
             mock.patch.object(mfu, "download_and_verify", return_value="x.tgz"), \
             mock.patch.object(mfu.tarfile, "open"), \
             mock.patch.object(mfu, "exec_program", return_value=0) as build, \
             mock.patch.object(mfu, "write_component_version") as record, \
             mock.patch.object(mfu, "log_info") as info:
            rc = mfu.install(env)
        self.assertEqual(rc, 0)
        record.assert_called_once_with("MPIFILEUTILS", "0.12")
        command = build.call_args.args[0]
        self.assertEqual(command[:2], ["bash", "-c"])
        return command[2], info

    def test_cmake_gets_lustre_off_without_the_header(self):
        script, info = self._install()
        self.assertIn(' -DENABLE_LUSTRE="OFF" ', script)
        messages = [c.args[1] for c in info.call_args_list]
        self.assertIn(f"Lustre support OFF: {self.header} not found", messages)
        self.assertEqual(info.call_args_list[0].kwargs["lustre"], "OFF")

    def test_cmake_gets_lustre_on_with_the_header(self):
        os.makedirs(os.path.dirname(self.header))
        Path(self.header).write_text("/* lustre */\n")
        script, info = self._install()
        self.assertIn(' -DENABLE_LUSTRE="ON" ', script)
        messages = [c.args[1] for c in info.call_args_list]
        self.assertIn(f"Lustre support ON: {self.header} found", messages)
        self.assertEqual(info.call_args_list[-1].kwargs["lustre"], "ON")


if __name__ == "__main__":
    unittest.main()
