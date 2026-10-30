"""Unit tests for the shared toolkit helpers (ADO 1001).

utils/process.py run/run_all/run_script/command_exists, utils/distro.py,
utils/files.py, utils/download.py fetch_verified, utils/component_config.py
require_config and utils/services.py. Where it is cheap, the real tool (tar,
bash) runs in a temporary folder.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils import component_config, download, files, process, services
from utils.component_config import require_config
from utils.distro import Distro


class RunTests(unittest.TestCase):
    def test_run_is_true_only_on_exit_0(self):
        with mock.patch.object(process, "exec_program", return_value=0) as run:
            self.assertTrue(process.run(["true"], "op", cwd="/w", env={"A": "1"}))
        run.assert_called_once_with(["true"], "op", cwd="/w", env={"A": "1"})
        with mock.patch.object(process, "exec_program", return_value=2):
            self.assertFalse(process.run(["false"], "op"))

    def test_run_all_stops_at_the_first_failure(self):
        with mock.patch.object(process, "exec_program", side_effect=[0, 1, 0]) as run:
            self.assertFalse(process.run_all([["a"], ["b"], ["c"]], "op"))
        self.assertEqual([c.args[0] for c in run.call_args_list], [["a"], ["b"]])

    def test_run_script_builds_one_set_e_script(self):
        with mock.patch.object(process, "exec_program", return_value=0) as run:
            process.run_script(["cmake .", "make -j 4"], "op", modules=["mpi/hpcx"], cwd="/b")
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["bash", "-c"])
        self.assertEqual(command[2], "set -e\nsource /etc/profile.d/modules.sh\n"
                                     "module load mpi/hpcx\ncmake .\nmake -j 4\n")
        self.assertEqual(run.call_args.kwargs["cwd"], "/b")

    def test_run_script_without_modules(self):
        with mock.patch.object(process, "exec_program", return_value=0) as run:
            process.run_script(["echo hi"], "op")
        self.assertEqual(run.call_args.args[0][2], "set -e\necho hi\n")

    @unittest.skipUnless(shutil.which("bash"), "needs bash")
    def test_run_script_stops_on_the_first_failing_line(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(process, "log_info"), mock.patch.object(process, "log_error"), \
             mock.patch.object(process, "log_error_detail"):
            ok = process.run_script(["false", "touch after"], "op", cwd=tmp)
            self.assertFalse(ok)
            self.assertFalse((Path(tmp) / "after").exists())

    def test_command_exists_uses_the_env_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = Path(tmp) / "mytool"
            tool.write_text("#!/bin/sh\n")
            tool.chmod(0o755)
            self.assertTrue(process.command_exists("mytool", {"PATH": tmp}))
            self.assertFalse(process.command_exists("mytool", {"PATH": "/nonexistent"}))


class DistroTests(unittest.TestCase):
    def test_parts(self):
        d = Distro("almalinux9.8")
        self.assertEqual((d.id, d.version, d.major), ("almalinux", "9.8", "9"))
        self.assertEqual(str(Distro("ubuntu24.04")), "ubuntu24.04")
        self.assertEqual(Distro.from_env({"DISTRIBUTION": "rhel8.10"}).major, "8")
        self.assertEqual(Distro.from_env({}).name, "")

    def test_families(self):
        cases = {  # name: (ubuntu, azurelinux, rhel_family, rpm, family)
            "ubuntu22.04": (True, False, False, False, "ubuntu"),
            "ubuntu26.04": (True, False, False, False, "ubuntu"),
            "azurelinux3.0": (False, True, False, True, "azurelinux"),
            "almalinux8.10": (False, False, True, True, "rhel"),
            "rocky9.8": (False, False, True, True, "rhel"),
            "rhel9.8": (False, False, True, True, "rhel"),
            "": (False, False, False, False, ""),
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                d = Distro(name)
                self.assertEqual((d.is_ubuntu, d.is_azurelinux, d.is_rhel_family, d.is_rpm,
                                  d.family), expected)


class WriteFileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "sub" / "x.conf"

    def _write(self, text, **kw):
        with mock.patch.object(files, "log_info") as info, \
             mock.patch.object(files, "log_error") as error:
            ok = files.write_file(self.path, text, "op", **kw)
        return ok, info, error

    def test_created_then_replaced_then_appended(self):
        ok, info, _ = self._write("a\n", mode=0o600)
        self.assertTrue(ok)
        self.assertEqual(info.call_args.kwargs["action"], "created")
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        _, info, _ = self._write("b\n")
        self.assertEqual(info.call_args.kwargs["action"], "replaced")
        _, info, _ = self._write("c\n", append=True)
        self.assertEqual(info.call_args.kwargs["action"], "appended")
        self.assertEqual(self.path.read_text(), "b\nc\n")

    def test_append_to_a_new_file_is_created(self):
        _, info, _ = self._write("x\n", append=True)
        self.assertEqual(info.call_args.kwargs["action"], "created")

    def test_write_error_returns_false(self):
        with mock.patch("builtins.open", side_effect=PermissionError("denied")):
            ok, _, error = self._write("x")
        self.assertFalse(ok)
        self.assertEqual(error.call_args.kwargs["path"], str(self.path))


class RemoveAndExtractTests(unittest.TestCase):
    def test_remove_paths_like_rm_rf(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            (tree / "a").mkdir(parents=True)
            (tree / "a" / "f").write_text("x")
            single = Path(tmp) / "file.tgz"
            single.write_text("x")
            link = Path(tmp) / "link"
            link.symlink_to(tree)
            files.remove_paths(tree, single, link, Path(tmp) / "missing", op="op")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_extract_command(self):
        with mock.patch.object(files, "run", return_value=True) as run:
            files.extract_tarball("v1.tgz", "op", dest="/src", strip_components=1, cwd="/w")
        self.assertEqual(run.call_args.args[0],
                         ["tar", "-xf", "v1.tgz", "-C", "/src", "--strip-components=1"])
        self.assertEqual(run.call_args.kwargs["cwd"], "/w")

    @unittest.skipUnless(shutil.which("tar"), "needs tar")
    def test_extract_keeps_modes_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "pkg-1.0"
            src.mkdir()
            (src / "run.sh").write_text("#!/bin/sh\n")
            (src / "run.sh").chmod(0o755)
            (src / "link").symlink_to("run.sh")
            subprocess.run(["tar", "-czf", "pkg.tgz", "pkg-1.0"], cwd=tmp, check=True)
            out = Path(tmp) / "out"
            out.mkdir()
            with mock.patch.object(process, "log_info"):
                self.assertTrue(files.extract_tarball(Path(tmp) / "pkg.tgz", "op",
                                                      dest=out, strip_components=1))
            self.assertEqual((out / "run.sh").stat().st_mode & 0o777, 0o755)
            self.assertTrue((out / "link").is_symlink())


class FetchVerifiedTests(unittest.TestCase):
    def test_success_returns_the_path(self):
        with mock.patch.object(download, "download_and_verify", return_value=Path("/w/f.tgz")):
            self.assertEqual(download.fetch_verified("https://x/f.tgz", "ab", "op", "/w"),
                             Path("/w/f.tgz"))

    def test_failures_are_logged_and_return_none(self):
        for exc in (OSError("no route"), download.ChecksumError("bad sum")):
            with self.subTest(exc=type(exc).__name__), \
                 mock.patch.object(download, "download_and_verify", side_effect=exc), \
                 mock.patch.object(download, "log_error") as error:
                self.assertIsNone(download.fetch_verified("https://x/f.tgz", "ab", "op"))
                self.assertEqual(error.call_args.args[0], "op")
                self.assertEqual(error.call_args.kwargs["url"], "https://x/f.tgz")


class RequireConfigTests(unittest.TestCase):
    def _env(self, nccl):
        return {"COMPONENT_VERSIONS": json.dumps({"nccl": {"ubuntu24.04": {"x86_64": nccl}}}),
                "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "x86_64",
                "GPU": "NVIDIA", "SKU": "A100"}

    def _require(self, nccl, *keys):
        with mock.patch.object(component_config, "log_error") as error:
            cfg = require_config("nccl", self._env(nccl), "install-nccl", *keys)
        return cfg, error

    def test_present_keys(self):
        cfg, error = self._require({"version": "2.30", "sha256": "ab",
                                    "driver": {"version": "13.0"}},
                                   "version", "sha256", "driver.version")
        self.assertEqual(cfg["version"], "2.30")
        error.assert_not_called()

    def test_missing_null_or_empty_keys_are_named(self):
        cfg, error = self._require({"version": "", "sha256": None, "url": "null"},
                                   "version", "sha256", "url", "commit")
        self.assertIsNone(cfg)
        self.assertEqual(error.call_args.args[0], "install-nccl")
        self.assertEqual(error.call_args.kwargs["missing"], "version, sha256, url, commit")
        self.assertEqual(error.call_args.kwargs["sku"], "A100")

    def test_no_config_at_all(self):
        env = self._env({"version": "1"})
        env["DISTRIBUTION"] = "rhel9.8"
        with mock.patch.object(component_config, "log_error") as error:
            self.assertIsNone(require_config("nccl", env, "op", "version"))
        self.assertEqual(error.call_args.kwargs["distribution"], "rhel9.8")


class ServicesTests(unittest.TestCase):
    def test_systemctl(self):
        with mock.patch.object(services, "run", return_value=True) as run:
            self.assertTrue(services.systemctl("enable", "--now", "x.service", op="op"))
        run.assert_called_once_with(["systemctl", "enable", "--now", "x.service"], "op", env=None)

    def test_is_active_is_a_probe(self):
        with mock.patch.object(services, "exec_program", return_value=3) as run:
            self.assertFalse(services.service_is_active("x", op="op"))
        self.assertIs(run.call_args.kwargs["failure_is_error"], False)
        self.assertEqual(run.call_args.args[0], ["systemctl", "is-active", "--quiet", "x"])


if __name__ == "__main__":
    unittest.main()
