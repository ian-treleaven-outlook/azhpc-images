"""Unit tests for components/python/install_azsecpack_prereqs.py.

  * The pure helpers (install_command, first_line) are tested directly.
  * page_size() is checked against real `getconf PAGE_SIZE` on the machine
    running the tests (read-only), and its error handling with a mock.
  * install() runs with page_size() mocked, PATH pointed at a temporary
    directory, exec_program replaced by a fake that "installs" nmap by creating
    an executable file there, and run_capture (`nmap --version`) mocked. So the
    real shutil.which() search is exercised, and no package manager runs.

Run from the repo root:
    python3 -m unittest tests.python.test_install_azsecpack_prereqs -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_azsecpack_prereqs as m
from utils.build_config import BuildConfig, build_plan

_OP = "install-azsecpack-prereqs"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASH_SCRIPT = _REPO_ROOT / "components" / "install_azsecpack_prereqs.sh"

_4K = 4096
_64K = 65536
_VERSION_OUTPUT = ("Nmap version 7.94SVN ( https://nmap.org )\n"
                   "Platform: aarch64-unknown-linux-gnu\n")
_VERSION_LINE = "Nmap version 7.94SVN ( https://nmap.org )"


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
class InstallCommandTests(unittest.TestCase):
    def test_ubuntu_uses_apt_get(self):
        for distro in ("ubuntu22.04", "ubuntu24.04", "ubuntu26.04"):
            with self.subTest(distro=distro):
                self.assertEqual(m.install_command(distro),
                                 ["apt-get", "install", "-y", "nmap"])

    def test_everything_else_uses_dnf(self):
        # bash's else branch: Azure Linux and the RHEL family, and also an
        # empty DISTRIBUTION (it doesn't contain "ubuntu").
        for distro in ("azurelinux3.0", "almalinux8.10", "almalinux9.x",
                       "rhel8.10", "rhel9.x", "rocky8.10", "rocky9.x", ""):
            with self.subTest(distro=distro):
                self.assertEqual(m.install_command(distro),
                                 ["dnf", "install", "-y", "nmap"])

    def test_returns_a_copy(self):
        m.install_command("ubuntu24.04").append("extra")
        self.assertEqual(m.install_command("ubuntu24.04"),
                         ["apt-get", "install", "-y", "nmap"])


class FirstLineTests(unittest.TestCase):
    def test_first_line_of_nmap_version(self):
        self.assertEqual(m.first_line(_VERSION_OUTPUT), _VERSION_LINE)

    def test_empty_output(self):
        self.assertEqual(m.first_line(""), "")

    def test_no_trailing_newline(self):
        self.assertEqual(m.first_line("only line"), "only line")


class PageSizeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("getconf"), "getconf not available")
    def test_matches_getconf_page_size(self):
        # The bash command, run read-only on the test machine.
        out = subprocess.run(["getconf", "PAGE_SIZE"], capture_output=True,
                             text=True, check=True).stdout
        self.assertEqual(m.page_size(), int(out))

    def test_asks_sysconf_for_the_page_size(self):
        with mock.patch.object(m.os, "sysconf", return_value=_64K) as sysconf:
            self.assertEqual(m.page_size(), _64K)
        sysconf.assert_called_once_with("SC_PAGE_SIZE")


# --------------------------------------------------------------------------- #
# install(): temp PATH, fake package manager, mocked page size and version.
# --------------------------------------------------------------------------- #
class _FakeExec:
    """Stands in for exec_program: records commands, imitates the install.

    With rc=0 and creates_nmap=True it writes an executable `nmap` into
    bin_dir, like a real package install puts it on PATH.
    """

    def __init__(self, bin_dir, rc=0, creates_nmap=True):
        self.bin_dir = bin_dir
        self.rc = rc
        self.creates_nmap = creates_nmap
        self.calls = []

    def __call__(self, command, op, *, env=None, **kwargs):
        self.calls.append((command, op, env))
        if self.rc == 0 and self.creates_nmap:
            _make_nmap(self.bin_dir)
        return self.rc


def _make_nmap(bin_dir):
    path = os.path.join(bin_dir, "nmap")
    with open(path, "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\n")
    os.chmod(path, 0o755)
    return path


class InstallTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.bin_dir = tmp.name

    def _env(self, distribution="ubuntu24.04"):
        return {"DISTRIBUTION": distribution, "PATH": self.bin_dir}

    def _install(self, env, *, size=_64K, fake=None):
        """Run install(); return (rc, fake exec, run_capture mock, log mocks)."""
        fake = fake or _FakeExec(self.bin_dir)
        with mock.patch.object(m, "page_size", return_value=size), \
             mock.patch.object(m, "exec_program", side_effect=fake), \
             mock.patch.object(m, "run_capture",
                               return_value=(0, _VERSION_OUTPUT)) as capture, \
             mock.patch.object(m, "log_info") as log_info, \
             mock.patch.object(m, "log_error") as log_error:
            rc = m.install(env)
        return rc, fake, capture, log_info, log_error

    def test_4k_pages_do_nothing(self):
        rc, fake, capture, log_info, log_error = self._install(self._env(),
                                                               size=_4K)
        self.assertEqual(rc, 0)
        self.assertEqual(fake.calls, [])
        capture.assert_not_called()
        log_error.assert_not_called()
        self.assertEqual(log_info.call_args.kwargs["page_size"], _4K)

    def test_4k_pages_do_nothing_even_without_nmap_on_rpm(self):
        rc, fake, _, _, _ = self._install(self._env("azurelinux3.0"), size=_4K)
        self.assertEqual(rc, 0)
        self.assertEqual(fake.calls, [])

    def test_64k_with_nmap_present_logs_version_and_installs_nothing(self):
        nmap = _make_nmap(self.bin_dir)
        env = self._env()
        rc, fake, capture, log_info, log_error = self._install(env)

        self.assertEqual(rc, 0)
        self.assertEqual(fake.calls, [])
        capture.assert_called_once_with(["nmap", "--version"], _OP, env=env,
                                        failure_is_error=False)
        log_error.assert_not_called()
        msg, fields = log_info.call_args.args[1], log_info.call_args.kwargs
        self.assertEqual(msg, f"nmap already installed: {_VERSION_LINE}")
        self.assertEqual(fields["path"], nmap)
        self.assertEqual(fields["version"], _VERSION_LINE)
        # The page-size decision is logged first.
        self.assertEqual(log_info.call_args_list[0].kwargs["page_size"], _64K)

    def test_64k_installs_with_apt_get_on_ubuntu(self):
        env = self._env("ubuntu24.04")
        rc, fake, capture, log_info, log_error = self._install(env)

        self.assertEqual(rc, 0)
        self.assertEqual(fake.calls,
                         [(["apt-get", "install", "-y", "nmap"], _OP, env)])
        capture.assert_called_once_with(["nmap", "--version"], _OP, env=env,
                                        failure_is_error=False)
        log_error.assert_not_called()
        msg, fields = log_info.call_args.args[1], log_info.call_args.kwargs
        self.assertEqual(msg, f"OK - nmap installed: {_VERSION_LINE}")
        self.assertEqual(fields["path"], os.path.join(self.bin_dir, "nmap"))
        self.assertEqual(fields["package_manager"], "apt-get")

    def test_64k_installs_with_dnf_on_rpm_distros(self):
        for distro in ("azurelinux3.0", "almalinux9.x", "rhel9.x", "rocky8.10"):
            with self.subTest(distro=distro):
                path = os.path.join(self.bin_dir, "nmap")
                if os.path.exists(path):
                    os.remove(path)
                env = self._env(distro)
                rc, fake, _, log_info, log_error = self._install(env)

                self.assertEqual(rc, 0)
                self.assertEqual(fake.calls,
                                 [(["dnf", "install", "-y", "nmap"], _OP, env)])
                log_error.assert_not_called()
                self.assertEqual(log_info.call_args.kwargs["package_manager"],
                                 "dnf")

    def test_install_failure_returns_3(self):
        env = self._env()
        rc, fake, capture, _, log_error = self._install(
            env, fake=_FakeExec(self.bin_dir, rc=100))

        self.assertEqual(rc, 3)
        self.assertEqual(len(fake.calls), 1)
        capture.assert_not_called()
        log_error.assert_called_once()
        self.assertEqual(log_error.call_args.kwargs["exit_code"], 100)
        self.assertEqual(log_error.call_args.kwargs["command"],
                         "apt-get install -y nmap")

    def test_nmap_missing_after_install_returns_3(self):
        env = self._env("rocky9.x")
        rc, fake, capture, _, log_error = self._install(
            env, fake=_FakeExec(self.bin_dir, creates_nmap=False))

        self.assertEqual(rc, 3)
        self.assertEqual(len(fake.calls), 1)
        capture.assert_not_called()
        log_error.assert_called_once()
        self.assertEqual(
            log_error.call_args.args[1],
            "nmap package installed but nmap binary is not available")
        self.assertEqual(log_error.call_args.kwargs["search_path"], self.bin_dir)

    def test_non_executable_nmap_does_not_count(self):
        # `command -v` only finds executable files; shutil.which agrees.
        with open(os.path.join(self.bin_dir, "nmap"), "w") as f:
            f.write("")
        rc, fake, _, _, _ = self._install(
            self._env(), fake=_FakeExec(self.bin_dir, creates_nmap=False))
        self.assertEqual(rc, 3)
        self.assertEqual(len(fake.calls), 1)

    def test_nmap_version_failure_is_not_fatal(self):
        # bash: `$(nmap --version 2>/dev/null | head -1 || true)`.
        _make_nmap(self.bin_dir)
        with mock.patch.object(m, "page_size", return_value=_64K), \
             mock.patch.object(m, "run_capture", return_value=(1, "")), \
             mock.patch.object(m, "log_info") as log_info:
            rc = m.install(self._env())
        self.assertEqual(rc, 0)
        self.assertEqual(log_info.call_args.args[1], "nmap already installed: ")

    def test_unreadable_page_size_returns_3(self):
        for exc in (OSError("sysconf failed"), ValueError("unknown name")):
            with self.subTest(exc=type(exc).__name__):
                with mock.patch.object(m, "page_size", side_effect=exc), \
                     mock.patch.object(m, "exec_program") as run, \
                     mock.patch.object(m, "log_error") as log_error:
                    rc = m.install(self._env())
                self.assertEqual(rc, 3)
                run.assert_not_called()
                self.assertIn(str(exc), log_error.call_args.args[1])


# --------------------------------------------------------------------------- #
# Bash parity: the commands and threshold match the script.
# --------------------------------------------------------------------------- #
class BashSourceTests(unittest.TestCase):
    def setUp(self):
        self.script = _BASH_SCRIPT.read_text(encoding="utf-8")

    def test_commands_match_the_script(self):
        self.assertIn("        " + " ".join(m._APT_INSTALL) + "\n", self.script)
        self.assertIn("        " + " ".join(m._DNF_INSTALL) + "\n", self.script)
        self.assertIn('[[ $DISTRIBUTION == *"ubuntu"* ]]', self.script)
        self.assertIn("PAGE_SIZE=$(getconf PAGE_SIZE)\n", self.script)
        self.assertIn(f'if [ "$PAGE_SIZE" = "{m._PAGE_SIZE_64K}" ]; then',
                      self.script)

    def test_docstring_quotes_the_script(self):
        doc = m.__doc__
        for line in self.script.splitlines()[1:]:
            if line.strip():
                with self.subTest(line=line):
                    self.assertIn("    " + line + "\n", doc)


# --------------------------------------------------------------------------- #
# Wiring: the Step runs this module on every distro/SKU, as install.sh does.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_this_module(self):
        for vendor, gpu in (("NVidia", "A100"), ("NVidia", "GB200"),
                            ("AMD", "MI300X")):
            with self.subTest(gpu=gpu):
                cfg = BuildConfig(vendor=vendor, gpu=gpu, os="Ubuntu24",
                                  fips=False, spec_path=None)
                steps = [s for s in build_plan(cfg)
                         if s.op == "install-azsecpack-prereqs"]
                self.assertEqual(len(steps), 1)
                self.assertEqual(steps[0].script, "install_azsecpack_prereqs.sh")
                self.assertIs(steps[0].action, m.install)


if __name__ == "__main__":
    unittest.main()
