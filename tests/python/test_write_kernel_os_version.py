"""Unit tests for components/python/write_kernel_os_version.py.

Two kinds of test:

  * The pure helper, azure_linux_os_version(), is tested directly with real
    inputs. It only transforms a string, so nothing needs to be faked.
  * install() is tested with its external effects replaced by mocks: reading
    the kernel version, running `rpm`, and writing the versions file.
    The tests check its decisions: what it records, which command it runs,
    and how it reacts when something fails. They never touch /opt or run rpm.

How the mocking works: `mock.patch.object(module, "name")` temporarily replaces
`module.name` with a fake object, only inside the `with` block. We patch the
names as seen *inside* write_kernel_os_version (for example its imported
`run_capture`), because that is the name install() actually calls.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import unittest
from unittest import mock

from components.python import write_kernel_os_version
from utils.build_config import BuildConfig, build_plan

_KERNEL = "6.6.64.2-1.azl3"
_RPM_OUTPUT = "azurelinux-release-3.0-25.azl3.noarch\n"   # rpm ends with a newline
_OS_VERSION = "azurelinux-release-3.0-25"


# --------------------------------------------------------------------------- #
# Pure helper: real inputs, no mocks.
# --------------------------------------------------------------------------- #
class AzureLinuxOsVersionTests(unittest.TestCase):
    def test_strips_the_12_character_suffix(self):
        # Mirrors bash ${os_version::-12}: ".azl3.noarch" is exactly 12 characters.
        self.assertEqual(
            write_kernel_os_version.azure_linux_os_version(_RPM_OUTPUT), _OS_VERSION)

    def test_empty_output_returns_none(self):
        self.assertIsNone(write_kernel_os_version.azure_linux_os_version(""))

    def test_output_with_nothing_left_after_the_suffix_returns_none(self):
        # Exactly 12 characters: removing the suffix would leave an empty
        # version, so we treat it as unusable rather than record "".
        self.assertIsNone(
            write_kernel_os_version.azure_linux_os_version(".azl3.noarch\n"))


# --------------------------------------------------------------------------- #
# install(): external effects mocked.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def _install(self, env, *, kernel=_KERNEL, rpm_result=(0, _RPM_OUTPUT),
                 write_error=None):
        """Run install(env) with all of its external effects mocked.

        Returns the exit code plus the mocks, so each test can check what
        install() tried to do:
          * record    - the fake write_component_version (what was recorded)
          * rpm       - the fake run_capture (whether/how rpm was run)
          * log_error - the fake error logger (whether a failure was reported)

        write_error, when given, is an exception the fake writer raises, to
        simulate the versions file being unwritable.
        """
        with mock.patch.object(write_kernel_os_version.platform, "release",
                               return_value=kernel), \
             mock.patch.object(write_kernel_os_version, "run_capture",
                               return_value=rpm_result) as rpm, \
             mock.patch.object(write_kernel_os_version, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(write_kernel_os_version, "log_error") as log_error:
            rc = write_kernel_os_version.install(env)
        return rc, record, rpm, log_error

    # ---- success paths -----------------------------------------------------
    def test_ubuntu_records_kernel_only(self):
        rc, record, rpm, log_error = self._install({"DISTRIBUTION": "ubuntu24.04"})

        self.assertEqual(rc, 0)
        record.assert_called_once_with("KERNEL", _KERNEL)
        rpm.assert_not_called()          # the OS lookup is Azure Linux only
        log_error.assert_not_called()

    def test_azure_linux_records_kernel_then_os(self):
        env = {"DISTRIBUTION": "azurelinux3.0"}
        rc, record, rpm, log_error = self._install(env)

        self.assertEqual(rc, 0)
        # Same order as bash: KERNEL first, then OS.
        self.assertEqual(record.call_args_list, [
            mock.call("KERNEL", _KERNEL),
            mock.call("OS", _OS_VERSION),
        ])
        # The exact command, logged under this step's op label, with the build env.
        rpm.assert_called_once_with(["rpm", "-qf", "/etc/os-release"],
                                    "write-kernel-os-version", env=env)
        log_error.assert_not_called()

    def test_missing_distribution_records_kernel_only(self):
        rc, record, rpm, _ = self._install({})

        self.assertEqual(rc, 0)
        record.assert_called_once_with("KERNEL", _KERNEL)
        rpm.assert_not_called()

    # ---- failure paths: each must return 3 and log an error ----------------
    def test_unknown_kernel_version_returns_3(self):
        rc, record, _, log_error = self._install({"DISTRIBUTION": "ubuntu24.04"},
                                                 kernel="")

        self.assertEqual(rc, 3)
        record.assert_not_called()       # nothing half-recorded
        log_error.assert_called_once()

    def test_rpm_failure_returns_3(self):
        rc, record, _, log_error = self._install({"DISTRIBUTION": "azurelinux3.0"},
                                                 rpm_result=(1, ""))

        self.assertEqual(rc, 3)
        # Like bash, KERNEL was already written before rpm ran; OS never is.
        record.assert_called_once_with("KERNEL", _KERNEL)
        log_error.assert_called_once()

    def test_unusable_rpm_output_returns_3(self):
        rc, record, _, log_error = self._install({"DISTRIBUTION": "azurelinux3.0"},
                                                 rpm_result=(0, ""))

        self.assertEqual(rc, 3)
        record.assert_called_once_with("KERNEL", _KERNEL)
        log_error.assert_called_once()

    def test_unwritable_versions_file_returns_3_without_raising(self):
        # PermissionError is a kind of OSError, which is what a failed write raises.
        rc, _, rpm, log_error = self._install(
            {"DISTRIBUTION": "azurelinux3.0"},
            write_error=PermissionError("/opt/azurehpc: permission denied"))

        self.assertEqual(rc, 3)
        rpm.assert_not_called()          # stopped at the first failure
        self.assertIn("permission denied", log_error.call_args[0][1])


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "write-kernel-os-version"]

        self.assertEqual(len(steps), 1)
        # assertIs checks it is the very same function object, not just similar.
        self.assertIs(steps[0].action, write_kernel_os_version.install)
        # script= stays so parity tests work and removing action= reverts to bash.
        self.assertEqual(steps[0].script, "write_kernel_os_version.sh")


if __name__ == "__main__":
    unittest.main()
