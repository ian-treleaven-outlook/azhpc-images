"""Unit tests for components/python/install_mofed.py.

  * The pure helpers are tested directly; release_version() is also compared
    with real grep + sed when they are available (read-only).
  * The package tables and the openibd drop-in are compared with the text of
    components/install_mofed.sh, so they can't drift from the bash.
  * install() runs with install_packages, run_capture, record_version and
    systemctl mocked, and the drop-in written to a temporary directory, so no
    package manager, systemd or /etc is touched.

Run from the repo root:
    python3 -m unittest tests.python.test_install_mofed -v
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_mofed as m
from utils.plans import steps

_OP = "install-mofed"
_BASH_SCRIPT = Path(__file__).resolve().parents[2] / "components" / "install_mofed.sh"

_DNF_LIST = (
    "Installed Packages\n"
    "mlnx-ofa_kernel.x86_64             24.10-20_6.6.96.2.azl3     @azurelinux-official-base\n"
    "mlnx-ofa_kernel-devel.x86_64       24.10-20_6.6.96.2.azl3     @azurelinux-official-base\n"
    "mlnx-ofa_kernel-hwe-devel.aarch64  24.10-21_6.12.40.1.azl3    @azurelinux-official-base\n"
    "rdma-core.x86_64                   2410mlnx54-1.azl3          @azurelinux-official-base\n"
)
_OFED_INFO = "OFED-internal-24.10-0.7.0:\n\nmlnx-ofa_kernel:\nmlnx_ofed/mlnx-ofa_kernel-4.0.git...\n"
_SED = r"s/.*\s\+\([0-9.]\+-[0-9]\+\)_.*/\1/"


def _bash_dnf_installs() -> list[list[str]]:
    """The package lists of every `dnf install -y` in install_mofed.sh, in order."""
    text = _BASH_SCRIPT.read_text(encoding="utf-8").replace("\\\n", " ")
    return [line.split("dnf install -y", 1)[1].split()
            for line in text.splitlines() if "dnf install -y" in line]


def _bash_heredoc(path: str) -> str:
    text = _BASH_SCRIPT.read_text(encoding="utf-8")
    match = re.search(rf"cat > {re.escape(path)} <<EOF\n(.*?)^EOF$", text, re.S | re.M)
    return match.group(1)


# --------------------------------------------------------------------------- #
# Pure helpers and tables.
# --------------------------------------------------------------------------- #
class ArchitectureKeyTests(unittest.TestCase):
    def test_aarch64(self):
        self.assertEqual(m.architecture_key({"ARCHITECTURE": "aarch64"}), "aarch64")

    def test_anything_else_is_x86_64_like_bash_else(self):
        for arch in ("x86_64", "", "ppc64le"):
            with self.subTest(arch=arch):
                self.assertEqual(m.architecture_key({"ARCHITECTURE": arch}), "x86_64")
        self.assertEqual(m.architecture_key({}), "x86_64")


class ReleaseVersionTests(unittest.TestCase):
    def test_x86_64_devel_package(self):
        self.assertEqual(m.release_version(_DNF_LIST, "mlnx-ofa_kernel-devel"), "24.10-20")

    def test_aarch64_hwe_devel_package(self):
        self.assertEqual(m.release_version(_DNF_LIST, "mlnx-ofa_kernel-hwe-devel"), "24.10-21")

    def test_not_installed_gives_empty(self):
        self.assertEqual(m.release_version("Installed Packages\n", "mlnx-ofa_kernel-devel"), "")

    def test_line_without_the_version_shape_is_kept_whole_like_sed(self):
        line = "mlnx-ofa_kernel-devel.x86_64  weird  @repo"
        self.assertEqual(m.release_version(line, "mlnx-ofa_kernel-devel"), line)

    @unittest.skipUnless(shutil.which("bash") and shutil.which("grep") and shutil.which("sed")
                         and os.name == "posix", "needs bash, grep and GNU sed")
    def test_matches_real_grep_and_sed(self):
        for package in ("mlnx-ofa_kernel-devel", "mlnx-ofa_kernel-hwe-devel", "missing"):
            with self.subTest(package=package):
                out = subprocess.run(
                    ["bash", "-c", f'grep {package} | sed "$1"', "_", _SED],
                    input=_DNF_LIST, capture_output=True, text=True).stdout
                self.assertEqual(m.release_version(_DNF_LIST, package), out.rstrip("\n"))


class BashParityTests(unittest.TestCase):
    def test_package_lists_and_order_match_the_bash_dnf_installs(self):
        deps, aarch64, x86_64, user_space = _bash_dnf_installs()
        self.assertEqual(m._BUILD_DEPENDENCIES, deps)
        self.assertEqual(m._KERNEL_PACKAGES["aarch64"], aarch64)
        self.assertEqual(m._KERNEL_PACKAGES["x86_64"], x86_64)
        self.assertEqual(m._USER_SPACE_PACKAGES, user_space)

    def test_openibd_drop_in_matches_the_bash_heredoc(self):
        self.assertEqual(m._OPENIBD_OVERRIDE, _bash_heredoc(m._OPENIBD_DROP_IN))


# --------------------------------------------------------------------------- #
# install(): every external effect mocked; the drop-in goes to a temp dir.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.drop_in = os.path.join(tmp.name, "openibd.service.d", "override.conf")
        for name in ("log_info", "log_warn", "log_error"):
            patcher = mock.patch.object(m, name)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

    def _install(self, env=None, *, install_rc=(True, True, True),
                 dnf_list=(0, _DNF_LIST), ofed_info=(0, _OFED_INFO),
                 record=True, systemctl=True):
        env = env or {"DISTRIBUTION": "azurelinux3.0", "ARCHITECTURE": "x86_64"}

        def capture(command, op, **kwargs):
            return dnf_list if command[0] == "dnf" else ofed_info

        with mock.patch.object(m, "_OPENIBD_DROP_IN", self.drop_in), \
             mock.patch("utils.files.log_info"), \
             mock.patch.object(m, "install_packages", side_effect=list(install_rc)) as pkgs, \
             mock.patch.object(m, "run_capture", side_effect=capture) as run_capture, \
             mock.patch.object(m, "record_version", return_value=record) as record_version, \
             mock.patch.object(m, "systemctl", return_value=systemctl) as systemctl_mock:
            rc = m.install(env)
        self.pkgs, self.run_capture = pkgs, run_capture
        self.record_version, self.systemctl = record_version, systemctl_mock
        return rc

    def _installed_lists(self):
        return [c.args[0] for c in self.pkgs.call_args_list]

    def test_x86_64_happy_path(self):
        env = {"DISTRIBUTION": "azurelinux3.0", "ARCHITECTURE": "x86_64"}
        self.assertEqual(self._install(env), 0)

        self.assertEqual(self._installed_lists(), [
            m._BUILD_DEPENDENCIES, m._KERNEL_PACKAGES["x86_64"], m._USER_SPACE_PACKAGES])
        for call in self.pkgs.call_args_list:
            self.assertEqual(call.args[1:], (env, _OP))
        self.record_version.assert_called_once_with("OFED", "24.10-0.7.0", _OP)
        with open(self.drop_in, encoding="utf-8") as f:
            self.assertEqual(f.read(), m._OPENIBD_OVERRIDE)
        self.assertEqual([c.args for c in self.systemctl.call_args_list],
                         [("daemon-reload",), ("enable", "openibd")])
        released = [c for c in self.log_info.call_args_list
                    if c.kwargs.get("release_version")]
        self.assertEqual(released[0].kwargs["release_version"], "24.10-20")
        self.log_error.assert_not_called()

    def test_aarch64_uses_the_hwe_kernel_packages_and_release_package(self):
        env = {"DISTRIBUTION": "azurelinux3.0", "ARCHITECTURE": "aarch64"}
        self.assertEqual(self._install(env), 0)
        self.assertEqual(self._installed_lists()[1], m._KERNEL_PACKAGES["aarch64"])
        released = [c for c in self.log_info.call_args_list
                    if c.kwargs.get("release_version")]
        self.assertEqual(released[0].kwargs["release_version"], "24.10-21")

    def test_other_distribution_installs_nothing_but_carries_on_like_bash(self):
        rc = self._install({"DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "x86_64"})
        self.assertEqual(rc, 0)
        self.pkgs.assert_not_called()
        self.log_warn.assert_called()
        self.record_version.assert_called_once()

    def test_failing_install_stops_at_that_command(self):
        rc = self._install(install_rc=(True, False))
        self.assertEqual(rc, 3)
        self.assertEqual(self.pkgs.call_count, 2)
        self.run_capture.assert_not_called()
        self.systemctl.assert_not_called()

    def test_missing_release_version_only_warns(self):
        rc = self._install(dnf_list=(1, ""))
        self.assertEqual(rc, 0)
        self.log_warn.assert_called_once()
        dnf_call = self.run_capture.call_args_list[0]
        self.assertEqual(dnf_call.args[0], ["dnf", "list", "installed"])
        self.assertIs(dnf_call.kwargs["failure_is_error"], False)

    def test_failing_ofed_info_returns_3(self):
        rc = self._install(ofed_info=(127, ""))
        self.assertEqual(rc, 3)
        self.record_version.assert_not_called()
        self.systemctl.assert_not_called()
        self.log_error.assert_called_once()

    def test_unparsable_ofed_info_returns_3(self):
        rc = self._install(ofed_info=(0, "MLNX_OFED_LINUX-24.10\n"))
        self.assertEqual(rc, 3)
        self.assertEqual(self.log_error.call_args.kwargs["first_line"], "MLNX_OFED_LINUX-24.10")

    def test_failing_record_version_returns_3(self):
        self.assertEqual(self._install(record=False), 3)
        self.systemctl.assert_not_called()

    def test_failing_systemctl_returns_3(self):
        self.assertEqual(self._install(systemctl=False), 3)
        self.assertEqual(self.systemctl.call_count, 1)

    def test_unwritable_drop_in_returns_3(self):
        with mock.patch.object(m, "write_file", return_value=False):
            self.assertEqual(self._install(), 3)
        self.systemctl.assert_not_called()


class StepWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        step = steps.mofed()
        self.assertEqual(step.op, "install-mofed")
        self.assertEqual(step.script, "install_mofed.sh")
        self.assertIs(step.action, m.install)
        self.assertEqual(step.base, "component")


if __name__ == "__main__":
    unittest.main()
