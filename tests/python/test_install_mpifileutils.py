"""Unit tests for components/python/install_mpifileutils.py.

install() runs with every external effect mocked (config lookup, download,
exec_program, versions file) and its directories and the Lustre header path
pointed into a temporary directory. The expected commands are the ones
components/install_mpifileutils.sh (upstream 0c2b8f9) runs, in its order and
working directories. install_deps' per-distro commands are also covered in
test_packaging.py.

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
_URL = "https://github.com/hpc/mpifileutils/releases/download/v0.12/mpifileutils-v0.12.tgz"


class PureHelperTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    def test_lustre_flag_on_only_when_the_header_is_a_file(self):
        header = os.path.join(self.root, "lustre", "lustre_user.h")
        self.assertEqual(mfu.lustre_cmake_flag(header), "OFF")             # missing
        os.makedirs(header)                                                 # a directory
        self.assertEqual(mfu.lustre_cmake_flag(header), "OFF")
        os.rmdir(header)
        Path(header).write_text("/* lustre */\n")
        self.assertEqual(mfu.lustre_cmake_flag(header), "ON")
        link = os.path.join(self.root, "link.h")                            # [[ -f ]] follows links
        os.symlink(header, link)
        self.assertEqual(mfu.lustre_cmake_flag(link), "ON")

    def test_bash_paths_and_names(self):
        self.assertEqual(mfu._LUSTRE_USER_H, "/usr/include/lustre/lustre_user.h")
        self.assertEqual(mfu._INSTALL_PREFIX, "/opt/mpifileutils")
        self.assertEqual(mfu._BUILD_DIR, "/tmp/mpifileutils-build")
        self.assertEqual(mfu._SRC_DIR, "/tmp/mpifileutils-src")
        self.assertEqual(mfu.tarball_name("0.12"), "mpifileutils-v0.12.tgz")

    def test_deps_command_branches(self):
        self.assertEqual(mfu.deps_command("ubuntu24.04"), (
            ["apt-get", "install", "-y", "libbz2-dev", "libattr1-dev", "libarchive-dev",
             "libssl-dev", "libcap-dev"], "ubuntu"))
        rpm = ["dnf", "install", "-y", "bzip2-devel", "libattr-devel", "libarchive-devel"]
        self.assertEqual(mfu.deps_command("azurelinux3.0"), (rpm, "azurelinux"))
        self.assertEqual(mfu.deps_command("rocky9.6"), (rpm, "rhel-family"))

    def test_build_script_is_the_bash_build_lines(self):
        with mock.patch.multiple(mfu, _SRC_DIR="/s", _INSTALL_PREFIX="/p"):
            script = mfu.build_script("0.12", "ON")
        self.assertEqual(script, (
            "set -e\n"
            "source /etc/profile.d/modules.sh\n"
            "module load mpi/hpcx\n"
            'cmake "/s/mpifileutils-v0.12" -DCMAKE_INSTALL_PREFIX="/p" -DENABLE_XATTRS=ON '
            '-DENABLE_LIBARCHIVE=ON -DENABLE_LUSTRE="ON" -DCMAKE_BUILD_TYPE=Release '
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5\n"
            "make -j$(nproc)\n"
            "make install\n"
            "module unload mpi/hpcx\n"))


class InstallTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.prefix = os.path.join(self.root, "opt", "mpifileutils")
        self.build = os.path.join(self.root, "build")
        self.src = os.path.join(self.root, "src")
        self.header = os.path.join(self.root, "lustre", "lustre_user.h")
        patcher = mock.patch.multiple(mfu, _LUSTRE_USER_H=self.header,
                                      _INSTALL_PREFIX=self.prefix, _BUILD_DIR=self.build,
                                      _SRC_DIR=self.src)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _fake_exec(self, command, op, *, cwd=None, env=None):
        self.assertEqual(op, _OP)
        self.calls.append((command, cwd, env))
        if command[0] == self.fail_on:
            return 2
        if command[:2] == ["tar", "-xzf"] and self.tar_creates_source:
            os.makedirs(os.path.join(cwd, command[2][:-len(".tgz")]), exist_ok=True)
        return 0

    def _install(self, distro="ubuntu24.04", fail_on=None, url=_URL, config=None,
                 download_error=None, write_error=None, tar_creates_source=True):
        self.calls, self.fail_on, self.tar_creates_source = [], fail_on, tar_creates_source
        if config is None:
            config = {"version": "0.12", "url": url, "sha256": "s"}
        env = {"COMPONENT_VERSIONS": json.dumps({"mpifileutils": {"common": config}}),
               "DISTRIBUTION": distro, "ARCHITECTURE": "x86_64", "PATH": "/usr/bin"}

        def fake_download(url, sha256, dest_dir="."):
            if download_error:
                raise download_error
            path = Path(dest_dir) / url.rsplit("/", 1)[-1]
            path.write_bytes(b"tgz")
            return path

        with mock.patch.object(mfu, "exec_program", side_effect=self._fake_exec), \
             mock.patch.object(mfu, "download_and_verify", side_effect=fake_download) as dl, \
             mock.patch.object(mfu, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(mfu, "log_info") as info, \
             mock.patch.object(mfu, "log_warn") as warn, \
             mock.patch.object(mfu, "log_error") as error:
            rc = mfu.install(env)
        self.env, self.download, self.record = env, dl, record
        self.info, self.warn, self.error = info, warn, error
        return rc

    def _commands(self):
        return [c[0] for c in self.calls]

    def _expected(self, deps, lustre="OFF"):
        return [
            (deps, None),
            (["tar", "-xzf", "mpifileutils-v0.12.tgz"], self.src),
            (["bash", "-c", mfu.build_script("0.12", lustre)], self.build),
            (["rm", "-rf", self.build, self.src], None),
            (["ls", "-la", f"{self.prefix}/bin/"], None),
        ]

    def test_ubuntu_full_command_list(self):
        self.assertEqual(self._install(), 0)
        deps = ["apt-get", "install", "-y", "libbz2-dev", "libattr1-dev", "libarchive-dev",
                "libssl-dev", "libcap-dev"]
        self.assertEqual([(c[0], c[1]) for c in self.calls], self._expected(deps))
        for _, _, env in self.calls:
            self.assertIs(env, self.env)
        self.download.assert_called_once_with(_URL, "s", dest_dir=self.src)
        for directory in (self.prefix, self.build, self.src):
            self.assertTrue(os.path.isdir(directory))
        self.record.assert_called_once_with("MPIFILEUTILS", "0.12")
        self.assertIn('-DENABLE_LUSTRE="OFF"', self._commands()[2][2])
        messages = [c.args[1] for c in self.info.call_args_list]
        self.assertIn("=== Installing mpifileutils 0.12 ===", messages)
        self.assertIn(f"=== mpifileutils installed to {self.prefix} ===", messages)
        self.assertIn(f"Lustre support OFF: {self.header} not found", messages)
        self.assertIn("Recorded MPIFILEUTILS=0.12", messages)
        self.warn.assert_not_called()
        self.error.assert_not_called()

    def test_rpm_distros_use_the_dnf_command(self):
        rpm = ["dnf", "install", "-y", "bzip2-devel", "libattr-devel", "libarchive-devel"]
        for distro in ("azurelinux3.0", "almalinux9.8", "rhel8.10"):
            with self.subTest(distro=distro):
                self.assertEqual(self._install(distro=distro), 0)
                self.assertEqual([(c[0], c[1]) for c in self.calls], self._expected(rpm))

    def test_lustre_header_turns_lustre_on(self):
        os.makedirs(os.path.dirname(self.header))
        Path(self.header).write_text("/* lustre */\n")
        self.assertEqual(self._install(), 0)
        self.assertIn('-DENABLE_LUSTRE="ON"', self._commands()[2][2])
        messages = [c.args[1] for c in self.info.call_args_list]
        self.assertIn(f"Lustre support ON: {self.header} found", messages)
        self.assertEqual(self.info.call_args_list[-1].kwargs["lustre"], "ON")

    def test_tarball_name_comes_from_the_version_not_the_url(self):
        # bash: TARBALL="mpifileutils-v${VERSION}.tgz", whatever the URL's file is.
        self.assertEqual(self._install(url="https://example/other.tar.gz"), 0)
        self.assertEqual(self._commands()[1], ["tar", "-xzf", "mpifileutils-v0.12.tgz"])
        self.assertIn("downloaded other.tar.gz but bash extracts mpifileutils-v0.12.tgz",
                      self.warn.call_args.args[1])

    def test_each_failing_command_stops_the_step(self):
        for index, name in enumerate(("apt-get", "tar", "bash", "rm", "ls")):
            with self.subTest(command=name):
                self.assertEqual(self._install(fail_on=name), 3)
                self.assertEqual(len(self.calls), index + 1)
                self.assertEqual(self._commands()[-1][0], name)
                self.record.assert_not_called()
                self.error.assert_called()

    def test_missing_bin_dir_stops_before_recording(self):
        # bash's `ls -la /opt/mpifileutils/bin/` fails under set -e.
        self.assertEqual(self._install(fail_on="ls"), 3)
        self.assertIn(f"listing {self.prefix}/bin/", self.error.call_args.args[1])
        self.record.assert_not_called()

    def test_source_folder_not_created_by_tar_fails(self):
        self.assertEqual(self._install(tar_creates_source=False), 3)
        self.assertEqual(self._commands()[-1], ["tar", "-xzf", "mpifileutils-v0.12.tgz"])
        self.assertIn("was not created by extracting", self.error.call_args.args[1])

    def test_missing_config_fails_before_installing_anything(self):
        for name, config in (("no version", {"url": _URL, "sha256": "s"}),
                             ("no url", {"version": "0.12", "sha256": "s"}),
                             ("no sha256", {"version": "0.12", "url": _URL})):
            with self.subTest(case=name):
                self.assertEqual(self._install(config=config), 3)
                self.assertEqual(self.calls, [])
                self.download.assert_not_called()
                self.error.assert_called_once()

    def test_download_failure_returns_3(self):
        self.assertEqual(self._install(download_error=OSError("boom")), 3)
        self.assertEqual(len(self.calls), 1)           # only the deps
        self.assertIn("download/verify failed", self.error.call_args.args[1])

    def test_unwritable_versions_file_returns_3(self):
        self.assertEqual(self._install(write_error=PermissionError("denied")), 3)
        self.assertEqual(self._commands()[-1][0], "ls")
        self.assertIn("file operation failed", self.error.call_args.args[1])

    def test_mkdir_failure_returns_3(self):
        Path(self.root, "opt").write_text("a file, not a dir")
        self.assertEqual(self._install(), 3)
        self.assertEqual(len(self.calls), 1)           # only the deps
        self.assertIn("file operation failed", self.error.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
