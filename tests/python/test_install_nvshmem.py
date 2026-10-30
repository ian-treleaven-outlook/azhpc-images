"""Unit tests for components/python/install_nvshmem.py.

Two kinds of test:

  * The pure helpers (cuda_major_version, first_word, parse_dnf_version,
    parse_apt_version) are tested directly with realistic inputs. Their
    results were also compared once against real bash grep/awk/cut in WSL.
  * install() is tested with its external effects mocked: running dnf/apt
    (exec_program, both the module's and utils.package_installer's, which
    runs the Azure Linux install, as one mock), listing installed packages (run_capture) and writing the
    versions file. The cuda config comes from a small COMPONENT_VERSIONS JSON in
    the env, so the real require_config() resolves it. Nothing touches /opt or runs
    a package manager.

Run from the repo root:
    python3 -m unittest tests.python.test_install_nvshmem -v
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_nvshmem
from utils import component_config, package_installer
from utils.build_config import BuildConfig, build_plan

_OP = "install-nvshmem"

# tests/python/<this file> -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Realistic `apt list --installed` stdout. The "WARNING: apt does not have a
# stable CLI interface" line goes to stderr, so it isn't part of stdout.
_APT_LIST = (
    "Listing...\n"
    "libnvshmem3-cuda-13/unknown,now 3.4.5-1 arm64 [installed]\n"
    "libnvshmem3-dev-cuda-13/unknown,now 3.4.5-1 arm64 [installed]\n"
    "libnuma1/noble,now 2.0.18-1build1 arm64 [installed,automatic]\n"
)

# Realistic `dnf list installed` stdout.
_DNF_LIST = (
    "Installed Packages\n"
    "bash.aarch64                           5.2.15-3.azl3     @System\n"
    "libnvshmem3-cuda-13.aarch64            3.4.5-1           @cuda-rhel9-sbsa\n"
    "libnvshmem3-devel-cuda-13.aarch64      3.4.5-1           @cuda-rhel9-sbsa\n"
    "libnvshmem3-static-cuda-13.aarch64     3.4.5-1           @cuda-rhel9-sbsa\n"
)

_DNF_INSTALL = [
    "dnf", "install", "-y", "--nogpgcheck",
    "--repofrompath=cuda-rhel9-sbsa,"
    "https://developer.download.nvidia.com/compute/cuda/repos/rhel9/sbsa",
    "libnvshmem3-cuda-13", "libnvshmem3-devel-cuda-13", "libnvshmem3-static-cuda-13",
]
# Bash: `apt install ...` without -y; install_packages() writes apt-get and,
# with assume_yes=False, also leaves out -y.
_APT_INSTALL = ["apt-get", "install", "-o", "Dpkg::Use-Pty=0",
                "libnvshmem3-cuda-13", "libnvshmem3-dev-cuda-13"]


def _env(distribution, driver_version="13.0"):
    """A minimal build env whose cuda config has the given driver.version.

    driver_version=None leaves the cuda config out entirely.
    """
    versions = {}
    if driver_version is not None:
        versions["cuda"] = {distribution: {"aarch64": {
            "driver": {"version": driver_version, "distribution": "x"}}}}
    return {
        "DISTRIBUTION": distribution,
        "ARCHITECTURE": "aarch64",
        "COMPONENT_VERSIONS": json.dumps(versions),
    }


# --------------------------------------------------------------------------- #
# Pure helpers: real inputs, no mocks.
# --------------------------------------------------------------------------- #
class CudaMajorVersionTests(unittest.TestCase):
    def test_takes_the_part_before_the_first_dot(self):
        # Mirrors `cut -d. -f1`.
        self.assertEqual(install_nvshmem.cuda_major_version("13.0"), "13")
        self.assertEqual(install_nvshmem.cuda_major_version("12.9.1"), "12")

    def test_value_without_a_dot_is_used_whole(self):
        self.assertEqual(install_nvshmem.cuda_major_version("13"), "13")

    def test_non_numeric_or_empty_returns_none(self):
        # Bash would carry on with "null" or "" in the package names.
        for value in ("", "null", "x.0", ".0"):
            with self.subTest(value=value):
                self.assertIsNone(install_nvshmem.cuda_major_version(value))


class FirstWordTests(unittest.TestCase):
    def test_returns_the_first_word_across_all_values(self):
        # Like bash's unquoted $var: split on any whitespace, empties dropped.
        self.assertEqual(install_nvshmem.first_word(["", "3.4.5-1", "3.0-1"]), "3.4.5-1")

    def test_no_words_returns_none(self):
        self.assertIsNone(install_nvshmem.first_word([]))
        self.assertIsNone(install_nvshmem.first_word(["", " "]))


class ParseAptVersionTests(unittest.TestCase):
    def test_realistic_output(self):
        # cut -d' ' -f2 of the one matching line; "[installed]" is field 4.
        self.assertEqual(install_nvshmem.parse_apt_version(_APT_LIST, "13"), "3.4.5-1")

    def test_trailing_slash_stops_prefix_matches(self):
        # "libnvshmem3-cuda-1/" does not match "libnvshmem3-cuda-13/...".
        self.assertIsNone(install_nvshmem.parse_apt_version(_APT_LIST, "1"))

    def test_several_matches_record_only_the_first(self):
        # e.g. the package installed for two architectures (multiarch).
        output = ("libnvshmem3-cuda-13/unknown,now 3.4.5-1 arm64 [installed]\n"
                  "libnvshmem3-cuda-13/unknown,now 3.3.9-1 armhf [installed]\n")
        self.assertEqual(install_nvshmem.parse_apt_version(output, "13"), "3.4.5-1")

    def test_double_space_gives_an_empty_field_like_cut(self):
        # cut splits on each single space, so field 2 is empty here and bash's
        # word splitting moves on to the next matching line's value.
        output = ("libnvshmem3-cuda-13/unknown,now  3.4.5-1 arm64 [installed]\n"
                  "libnvshmem3-cuda-13/x,now 3.0-1 arm64 [installed]\n")
        self.assertEqual(install_nvshmem.parse_apt_version(output, "13"), "3.0-1")

    def test_line_without_a_space_is_printed_whole_like_cut(self):
        self.assertEqual(
            install_nvshmem.parse_apt_version("libnvshmem3-cuda-13/nospace\n", "13"),
            "libnvshmem3-cuda-13/nospace")

    def test_package_missing_returns_none(self):
        self.assertIsNone(install_nvshmem.parse_apt_version("Listing...\n", "13"))
        self.assertIsNone(install_nvshmem.parse_apt_version("", "13"))


class ParseDnfVersionTests(unittest.TestCase):
    def test_realistic_output(self):
        # Only the runtime package's line contains "libnvshmem3-cuda-13".
        self.assertEqual(install_nvshmem.parse_dnf_version(_DNF_LIST, "13"), "3.4.5-1")

    def test_several_matches_record_only_the_first(self):
        # Without a trailing "/", "libnvshmem3-cuda-1" also matches cuda-12/13.
        output = ("libnvshmem3-cuda-12.aarch64   3.2.0-1   @System\n"
                  "libnvshmem3-cuda-13.aarch64   3.4.5-1   @System\n")
        self.assertEqual(install_nvshmem.parse_dnf_version(output, "1"), "3.2.0-1")

    def test_tabs_count_as_field_separators_like_awk(self):
        output = "\tlibnvshmem3-cuda-13.aarch64\t3.4.5-1\t@System\n"
        self.assertEqual(install_nvshmem.parse_dnf_version(output, "13"), "3.4.5-1")

    def test_name_only_line_returns_none(self):
        # dnf wraps a long name onto its own line; awk's $2 is then empty.
        output = ("libnvshmem3-cuda-13.aarch64\n"
                  "                  3.4.5-1   @System\n")
        self.assertIsNone(install_nvshmem.parse_dnf_version(output, "13"))

    def test_package_missing_returns_none(self):
        self.assertIsNone(install_nvshmem.parse_dnf_version("Installed Packages\n", "13"))


# --------------------------------------------------------------------------- #
# install(): external effects mocked.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def _install(self, env, *, exec_rc=0, list_result=None, write_error=None):
        """Run install(env) with its external effects mocked.

        Returns the exit code plus the mocks: run (exec_program), capture
        (run_capture), record (utils.component_config.write_component_version)
        and log_error. install_packages()'s own error log is kept as
        self.pkg_error, and require_config()'s / record_version()'s as
        self.config_error.
        """
        if list_result is None:
            output = _DNF_LIST if env["DISTRIBUTION"] == "azurelinux3.0" else _APT_LIST
            list_result = (0, output)
        run = mock.Mock(return_value=exec_rc)
        with mock.patch.object(package_installer, "exec_program", run), \
             mock.patch.object(package_installer, "log_info"), \
             mock.patch.object(package_installer, "log_error") as pkg_error, \
             mock.patch.object(component_config, "log_error") as config_error, \
             mock.patch.object(component_config, "log_info"), \
             mock.patch.object(install_nvshmem, "run_capture",
                               return_value=list_result) as capture, \
             mock.patch.object(component_config, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(install_nvshmem, "log_warn") as log_warn, \
             mock.patch.object(install_nvshmem, "log_error") as log_error:
            rc = install_nvshmem.install(env)
        self.log_warn = log_warn
        self.pkg_error = pkg_error
        self.config_error = config_error
        return rc, run, capture, record, log_error

    # ---- success paths -----------------------------------------------------
    def test_azure_linux_installs_from_the_rhel9_sbsa_repo(self):
        env = _env("azurelinux3.0")
        rc, run, capture, record, log_error = self._install(env)

        self.assertEqual(rc, 0)
        run.assert_called_once_with(_DNF_INSTALL, _OP, cwd=None, env=env)
        capture.assert_called_once_with(["dnf", "list", "installed"], _OP, env=env)
        record.assert_called_once_with("NVSHMEM", "3.4.5-1")
        log_error.assert_not_called()

    def test_ubuntu_installs_with_apt_and_no_dash_y(self):
        env = _env("ubuntu24.04")
        rc, run, capture, record, log_error = self._install(env)

        self.assertEqual(rc, 0)
        # Exactly as bash: no -y (see _install_ubuntu for why it is kept).
        run.assert_called_once_with(_APT_INSTALL, _OP, cwd=None, env=env)
        capture.assert_called_once_with(["apt", "list", "--installed"], _OP, env=env)
        record.assert_called_once_with("NVSHMEM", "3.4.5-1")
        log_error.assert_not_called()
        # The missing -y is announced before apt runs.
        self.log_warn.assert_called_once()
        self.assertEqual(self.log_warn.call_args.kwargs["packages"],
                         "libnvshmem3-cuda-13 libnvshmem3-dev-cuda-13")

    def test_azure_linux_does_not_warn_about_apt(self):
        self._install(_env("azurelinux3.0"))
        self.log_warn.assert_not_called()

    def test_cuda_major_comes_from_the_cuda_driver_version(self):
        env = _env("ubuntu22.04", driver_version="12.9")
        output = "libnvshmem3-cuda-12/unknown,now 3.2.5-1 amd64 [installed]\n"
        rc, run, _, record, _ = self._install(env, list_result=(0, output))

        self.assertEqual(rc, 0)
        run.assert_called_once_with(
            ["apt-get", "install", "-o", "Dpkg::Use-Pty=0",
             "libnvshmem3-cuda-12", "libnvshmem3-dev-cuda-12"],
            _OP, cwd=None, env=env)
        record.assert_called_once_with("NVSHMEM", "3.2.5-1")

    def test_real_versions_json_gives_cuda_13_for_gb200(self):
        # Uses the repo's versions.json, as component_env() would.
        versions = (_REPO_ROOT / "versions.json").read_text(encoding="utf-8")
        for distribution in ("ubuntu24.04", "azurelinux3.0"):
            with self.subTest(distribution=distribution):
                env = {"DISTRIBUTION": distribution, "ARCHITECTURE": "aarch64",
                       "GPU": "NVIDIA", "SKU": "GB200", "COMPONENT_VERSIONS": versions}
                rc, run, _, _, _ = self._install(env)
                self.assertEqual(rc, 0)
                self.assertIn("libnvshmem3-cuda-13", run.call_args[0][0])

    # ---- failure paths: each must return 3 and log an error ----------------
    def test_missing_cuda_config_returns_3(self):
        for driver_version in (None, "null", ""):
            with self.subTest(driver_version=driver_version):
                rc, run, _, record, _ = self._install(_env("ubuntu24.04", driver_version))

                self.assertEqual(rc, 3)
                run.assert_not_called()
                record.assert_not_called()
                self.config_error.assert_called_once()

    def test_non_numeric_cuda_version_returns_3(self):
        rc, run, _, record, log_error = self._install(_env("ubuntu24.04", "x.0"))

        self.assertEqual(rc, 3)
        run.assert_not_called()
        record.assert_not_called()
        log_error.assert_called_once()
        self.assertIn("'x.0' has no numeric major version", log_error.call_args[0][1])

    def test_unsupported_distro_returns_3_without_installing(self):
        # Bash would install nothing and record an empty NVSHMEM version. Azure
        # Linux is an exact match in bash, so azurelinux2.0 is unsupported too.
        for distribution in ("almalinux9.8", "azurelinux2.0"):
            with self.subTest(distribution=distribution):
                rc, run, capture, record, log_error = self._install(_env(distribution))

                self.assertEqual(rc, 3)
                run.assert_not_called()
                capture.assert_not_called()
                record.assert_not_called()
                self.assertIn(distribution, log_error.call_args[0][1])

    def test_install_failure_returns_3_and_stops(self):
        for distribution in ("ubuntu24.04", "azurelinux3.0"):
            with self.subTest(distribution=distribution):
                rc, _, capture, record, log_error = self._install(
                    _env(distribution), exec_rc=100)
                self.assertEqual(rc, 3)
                capture.assert_not_called()
                record.assert_not_called()
                # install_packages() logs the Azure Linux failure itself.
                errors = self.pkg_error if distribution == "azurelinux3.0" else log_error
                errors.assert_called()

    def test_list_failure_returns_3(self):
        for distribution in ("ubuntu24.04", "azurelinux3.0"):
            with self.subTest(distribution=distribution):
                # run_capture logs the failing command itself.
                rc, _, capture, record, _ = self._install(
                    _env(distribution), list_result=(1, ""))
                self.assertEqual(rc, 3)
                capture.assert_called_once()
                record.assert_not_called()

    def test_package_not_listed_returns_3(self):
        # Bash would record an empty version here.
        for distribution in ("ubuntu24.04", "azurelinux3.0"):
            with self.subTest(distribution=distribution):
                rc, _, _, record, log_error = self._install(
                    _env(distribution), list_result=(0, "Listing...\n"))
                self.assertEqual(rc, 3)
                record.assert_not_called()
                log_error.assert_called()

    def test_unwritable_versions_file_returns_3_without_raising(self):
        rc, _, _, record, _ = self._install(
            _env("ubuntu24.04"),
            write_error=PermissionError("/opt/azurehpc: permission denied"))

        self.assertEqual(rc, 3)
        record.assert_called_once_with("NVSHMEM", "3.4.5-1")
        # record_version() logs the failure itself.
        self.assertIn("permission denied", self.config_error.call_args[0][1])


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="GB200", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "install-nvshmem"]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, install_nvshmem.install)
        self.assertEqual(steps[0].script, "install_nvshmem.sh")


if __name__ == "__main__":
    unittest.main()
