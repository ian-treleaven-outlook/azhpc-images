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

from utils import (component_config, download, files, package_installer, process,
                   repo_source, services)
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



class MoreFileHelperTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        for name in ("log_info", "log_error"):
            patcher = mock.patch.object(files, name)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_make_dirs_fresh_dir_touch(self):
        self.assertTrue(files.make_dirs(self.tmp / "a" / "b", self.tmp / "c", op="op"))
        (self.tmp / "a" / "b" / "old").write_text("x")
        self.assertTrue(files.fresh_dir(self.tmp / "a", op="op"))
        self.assertEqual(list((self.tmp / "a").iterdir()), [])
        self.assertTrue(files.touch(self.tmp / "d" / "marker", op="op"))
        self.assertTrue((self.tmp / "d" / "marker").is_file())

    def test_make_temp_dir(self):
        made = files.make_temp_dir("op", prefix="azhpc-")
        self.addCleanup(shutil.rmtree, made, True)
        self.assertTrue(made.is_dir())
        self.assertTrue(made.name.startswith("azhpc-"))

    def test_copy_file_into_a_dir_or_to_a_name_with_mode(self):
        src = self.tmp / "src.sh"
        src.write_text("#!/bin/sh\n")
        src.chmod(0o755)
        (self.tmp / "dest").mkdir()
        self.assertTrue(files.copy_file(src, self.tmp / "dest", "op"))
        self.assertEqual((self.tmp / "dest" / "src.sh").stat().st_mode & 0o777, 0o755)
        self.assertTrue(files.copy_file(src, self.tmp / "renamed", "op", mode=0o600))
        self.assertEqual((self.tmp / "renamed").stat().st_mode & 0o777, 0o600)
        self.assertFalse(files.copy_file(self.tmp / "missing", self.tmp / "x", "op"))

    def test_copy_files_glob_and_no_match(self):
        for name in ("a.gpg", "b.gpg", "c.txt"):
            (self.tmp / name).write_text(name)
        (self.tmp / "keys").mkdir()
        self.assertTrue(files.copy_files(self.tmp / "*.gpg", self.tmp / "keys", "op"))
        self.assertEqual(sorted(p.name for p in (self.tmp / "keys").iterdir()),
                         ["a.gpg", "b.gpg"])
        self.assertFalse(files.copy_files(self.tmp / "*.none", self.tmp / "keys", "op"))

    def test_move_file_and_symlink(self):
        (self.tmp / "bin").mkdir()
        (self.tmp / "tool").write_text("x")
        self.assertTrue(files.move_file(self.tmp / "tool", self.tmp / "bin", "op"))
        self.assertTrue((self.tmp / "bin" / "tool").is_file())
        link = self.tmp / "link"
        self.assertTrue(files.symlink("bin/tool", link, "op"))
        self.assertFalse(files.symlink("other", link, "op"))            # ln -s: exists
        self.assertTrue(files.symlink("other", link, "op", force=True))  # ln -sf
        self.assertEqual(os.readlink(link), "other")

    def test_file_has_line_and_ensure_line_like_grep_qxF(self):
        path = self.tmp / "bashrc"
        path.write_bytes(b"alias a=b\r\nexport X=1\n")
        self.assertTrue(files.file_has_line(path, "export X=1"))
        self.assertFalse(files.file_has_line(path, "alias a=b"))   # grep -x sees the \r
        self.assertFalse(files.file_has_line(self.tmp / "missing", "x"))
        self.assertTrue(files.ensure_line(path, "export X=1", "op"))
        self.assertTrue(files.ensure_line(path, "alias m=n", "op"))
        self.assertEqual(path.read_bytes(), b"alias a=b\r\nexport X=1\nalias m=n\n")

    def test_edit_file_keeps_crlf_and_odd_bytes(self):
        path = self.tmp / "gridd.conf"
        path.write_bytes(b"FeatureType=0\r\nName=\xff\xfe\r\n")
        self.assertTrue(files.edit_file(path, lambda t: t.replace("FeatureType=0\r\n", ""), "op"))
        self.assertEqual(path.read_bytes(), b"Name=\xff\xfe\r\n")

    def test_edit_file_missing_unless_create(self):
        self.assertFalse(files.edit_file(self.tmp / "none", lambda t: t + "x\n", "op"))
        self.assertTrue(files.edit_file(self.tmp / "new", lambda t: t + "x\n", "op", create=True))
        self.assertEqual((self.tmp / "new").read_text(), "x\n")


class MorePackageHelperTests(unittest.TestCase):
    def _run(self, func, *args, distribution="ubuntu24.04", capture=(0, ""), **kwargs):
        env = {"DISTRIBUTION": distribution}
        with mock.patch.object(package_installer, "exec_program", return_value=0) as run, \
             mock.patch.object(package_installer, "run_capture", return_value=capture) as cap, \
             mock.patch.object(package_installer, "log_info"), \
             mock.patch.object(package_installer, "log_error") as error:
            result = func(*args, env, "op", **kwargs)
        return result, run, cap, error

    def test_refresh_package_index(self):
        _, run, _, _ = self._run(package_installer.refresh_package_index)
        self.assertEqual(run.call_args.args[0], ["apt-get", "update"])
        _, run, _, _ = self._run(package_installer.refresh_package_index, distribution="rocky9.8")
        self.assertEqual(run.call_args.args[0], ["dnf", "makecache", "--refresh"])

    def test_remove_packages(self):
        _, run, _, _ = self._run(package_installer.remove_packages, ["patchelf"], purge=True)
        self.assertEqual(run.call_args.args[0], ["apt-get", "purge", "-y", "patchelf"])
        _, run, _, _ = self._run(package_installer.remove_packages, "azsec-monitor",
                                 distribution="azurelinux3.0")
        self.assertEqual(run.call_args.args[0], ["dnf", "remove", "-y", "azsec-monitor"])
        _, run, _, _ = self._run(package_installer.remove_packages, ["graphviz"],
                                 options=["--auto-remove"])
        self.assertEqual(run.call_args.args[0],
                         ["apt-get", "remove", "--auto-remove", "-y", "graphviz"])

    def test_update_packages(self):
        _, run, _, _ = self._run(package_installer.update_packages, ["WALinuxAgent"],
                                 distribution="almalinux9.8")
        self.assertEqual(run.call_args.args[0], ["dnf", "update", "-y", "WALinuxAgent"])
        _, run, _, _ = self._run(package_installer.update_packages, ["python3-pip"])
        self.assertEqual(run.call_args.args[0],
                         ["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0",
                          "--only-upgrade", "python3-pip"])

    def test_installed_package_version(self):
        version, _, cap, _ = self._run(package_installer.installed_package_version, "rccl",
                                       capture=(0, "2.27.7\n"))
        self.assertEqual(version, "2.27.7")
        self.assertEqual(cap.call_args.args[0], ["dpkg-query", "-W", "-f", "${Version}", "rccl"])
        version, _, cap, _ = self._run(package_installer.installed_package_version, "rccl",
                                       distribution="azurelinux3.0", capture=(0, "2.27.7"),
                                       rpm_format="%{VERSION}")
        self.assertEqual(cap.call_args.args[0],
                         ["rpm", "-q", "--queryformat", "%{VERSION}", "rccl"])
        version, _, _, error = self._run(package_installer.installed_package_version, "x",
                                         capture=(1, ""))
        self.assertIsNone(version)
        error.assert_called_once()

    def test_installed_package_version_missing_ok_is_not_an_error(self):
        # bash `if ! rpm -q pkg`: "not installed" is an answer, not a failure.
        version, _, cap, error = self._run(package_installer.installed_package_version,
                                           "python3-setools", distribution="rocky9.8",
                                           capture=(1, "package python3-setools is not "
                                                       "installed\n"),
                                           missing_ok=True)
        self.assertIsNone(version)
        error.assert_not_called()
        self.assertFalse(cap.call_args.kwargs["failure_is_error"])

    def test_unknown_distribution(self):
        ok, run, _, error = self._run(package_installer.remove_packages, ["x"], distribution="sles15")
        self.assertFalse(ok)
        run.assert_not_called()
        error.assert_called_once()


class MoreProcessDownloadVersionTests(unittest.TestCase):
    def test_command_path_and_cpu_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = Path(tmp) / "screen"
            tool.write_text("#!/bin/sh\n")
            tool.chmod(0o755)
            self.assertEqual(process.command_path("screen", {"PATH": tmp}), str(tool))
            self.assertIsNone(process.command_path("screen", {"PATH": "/nonexistent"}))
        self.assertEqual(process.cpu_count(), len(os.sched_getaffinity(0)))

    def test_fetch_without_checksum(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(download, "download") as get:
            self.assertEqual(download.fetch("https://x/v1.tar.gz", "op", tmp),
                             Path(tmp) / "v1.tar.gz")
        get.assert_called_once()
        with mock.patch.object(download, "download", side_effect=ValueError("bad url")), \
             mock.patch.object(download, "log_error") as error:
            self.assertIsNone(download.fetch("nope", "op"))
        error.assert_called_once()

    def test_fetch_verified_catches_a_malformed_url(self):
        with mock.patch.object(download, "download_and_verify", side_effect=ValueError("x")), \
             mock.patch.object(download, "log_error"):
            self.assertIsNone(download.fetch_verified("nope", "ab", "op"))

    def test_record_version(self):
        with mock.patch.object(component_config, "write_component_version",
                               return_value=Path("/opt/v.txt")) as write, \
             mock.patch.object(component_config, "log_info") as info:
            self.assertTrue(component_config.record_version("NCCL", "2.30", "op"))
        write.assert_called_once_with("NCCL", "2.30")
        self.assertEqual(info.call_args.kwargs["version"], "2.30")
        with mock.patch.object(component_config, "write_component_version",
                               side_effect=PermissionError("denied")), \
             mock.patch.object(component_config, "log_error") as error:
            self.assertFalse(component_config.record_version("NCCL", "2.30", "op"))
        error.assert_called_once()


class GitHelperTests(unittest.TestCase):
    def test_configured_ref(self):
        self.assertEqual(repo_source.configured_ref({"ref": "main", "branch": "b"}, "1.0"), "main")
        self.assertEqual(repo_source.configured_ref({"branch": "b"}, "1.0"), "b")
        self.assertEqual(repo_source.configured_ref({"ref": "", "branch": "b"}, "1.0"), "v1.0")
        self.assertEqual(repo_source.configured_ref({"ref": None, "branch": False}, "1.0"), "v1.0")

    def test_git_clone_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "nvloom").mkdir()                       # what git clone would create
            with mock.patch.object(repo_source, "run", return_value=True) as run, \
                 mock.patch.object(repo_source, "log_info"):
                out = repo_source.git_clone("https://github.com/NVIDIA/nvloom.git", "op",
                                            ref="v1.2", commit="abc", cwd=tmp)
            self.assertEqual(out, os.path.join(tmp, "nvloom"))
            self.assertEqual([c.args[0] for c in run.call_args_list], [
                ["git", "clone", "--branch", "v1.2", "https://github.com/NVIDIA/nvloom.git"],
                ["git", "checkout", "abc"]])
            self.assertEqual(run.call_args_list[1].kwargs["cwd"], os.path.join(tmp, "nvloom"))

    def test_git_clone_missing_checkout_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(repo_source, "run", return_value=True), \
             mock.patch.object(repo_source, "log_error") as error:
            self.assertIsNone(repo_source.git_clone("https://x/repo", "op", cwd=tmp))
        error.assert_called_once()

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_real_clone_of_a_local_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            origin = Path(tmp) / "origin"
            git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
            subprocess.run(["git", "init", "-q", "-b", "main", str(origin)], check=True)
            subprocess.run([*git, "-C", str(origin), "commit", "-q", "--allow-empty", "-m", "x"],
                           check=True)
            with mock.patch.object(process, "log_info"), mock.patch.object(repo_source, "log_info"):
                out = repo_source.git_clone(str(origin), "op", ref="main", dest="copy", cwd=tmp)
            self.assertEqual(out, os.path.join(tmp, "copy"))
            self.assertTrue((Path(out) / ".git").is_dir())


if __name__ == "__main__":
    unittest.main()
