"""Unit tests for components/python/install_pmix.py.

Two kinds of test:

  * The pure helpers (version parsing, the apt source line, and the
    `dnf list installed | grep | sed` port) are tested directly with real inputs.
  * install() is tested per distro with its external effects mocked:
    utils.process.exec_program (apt, dnf, curl, rpm via run/run_all) and
    utils.package_installer.exec_program (the apt-get/dnf installs) share one
    mock, plus run_capture and write_component_version. The /etc paths and
    the scratch work directory are
    module-level constants, which the tests point at a temporary directory, so
    the real repo/pin files are genuinely copied but nothing under /etc is touched.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_pmix
from utils import component_config, files, package_installer, process
from utils.build_config import BuildConfig, build_plan
from utils.distro import Distro

_OP = "install-pmix"
# The real components/ directory, so the shipped slurm-repo files are used.
_COMPONENT_DIR = str(Path(__file__).resolve().parents[2] / "components")
_VERSION = "4.2.9-2"
# What install_packages() runs (bash: `apt install -y` / `dnf -y install`).
_APT_INSTALL = ["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0"]
_DNF_INSTALL = ["dnf", "install", "-y"]

_DNF_LIST = (
    "Installed Packages\n"
    "hwloc-devel.x86_64        2.9.2-1.azl3   @azurelinux-official-base\n"
    "pmix.aarch64              5.0.1-1.azl3   @azurelinux-official-base\n"
    "pmix.x86_64               5.0.1-1.azl3   @azurelinux-official-base\n"
    "pmix-devel.x86_64         5.0.1-1.azl3   @azurelinux-official-base\n"
    "pmix-tools.x86_64         5.0.1-1.azl3   @azurelinux-official-base\n"
)


def _env(distribution="ubuntu24.04", architecture="x86_64", arch_distro="amd64",
         pmix=True):
    versions = {}
    if pmix:
        versions["pmix"] = {"common": {"version": _VERSION}}
    return {
        "COMPONENT_VERSIONS": json.dumps(versions),
        "COMPONENT_DIR": _COMPONENT_DIR,
        "DISTRIBUTION": distribution,
        "ARCHITECTURE": architecture,
        "ARCHITECTURE_DISTRO": arch_distro,
    }


# --------------------------------------------------------------------------- #
# Pure helpers: real inputs, no mocks.
# --------------------------------------------------------------------------- #
class VersionHelperTests(unittest.TestCase):
    def test_apt_source_line_is_exactly_what_bash_echoes(self):
        self.assertEqual(
            install_pmix.apt_source_line("amd64", "slurm-ubuntu-noble",
                                         "/usr/share/keyrings/microsoft-prod.gpg"),
            "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] "
            "https://packages.microsoft.com/repos/slurm-ubuntu-noble/ insiders main\n")

    def test_rhel_extra_repo_matches_the_bash_if_elif(self):
        def extra_repo(distribution, architecture="x86_64"):
            return install_pmix.rhel_extra_repo(Distro(distribution), architecture)

        self.assertEqual(extra_repo("rhel9.6"), "codeready-builder-for-rhel-9-x86_64-rhui-rpms")
        self.assertEqual(extra_repo("rhel8.10", "aarch64"),
                         "codeready-builder-for-rhel-8-aarch64-rhui-rpms")
        # rhel* wins over the major-version branches, for any major.
        self.assertEqual(extra_repo("rhel10.0"), "codeready-builder-for-rhel-10-x86_64-rhui-rpms")
        self.assertEqual(extra_repo("almalinux9.8"), "crb")
        self.assertEqual(extra_repo("rocky9.6"), "crb")
        self.assertEqual(extra_repo("rocky8.10"), "powertools")
        self.assertIsNone(extra_repo("almalinux10.0"))


class BashPackageListTests(unittest.TestCase):
    """The package lists, against the install lines of components/install_pmix.sh."""

    def test_package_lists_match_bash(self):
        bash = (Path(_COMPONENT_DIR) / "install_pmix.sh").read_text(encoding="utf-8")
        cases = [
            ("apt install -y ", install_pmix.ubuntu_packages("${PMIX_VERSION}")),
            ("dnf -y install ", install_pmix._AZURE_LINUX_PACKAGES[0]),
            ("dnf -y install ", install_pmix._AZURE_LINUX_PACKAGES[1]),
            ("dnf -y install ", install_pmix.rhel_packages("${PMIX_VERSION}",
                                                           "${OS_MAJOR_VERSION}")),
        ]
        for prefix, packages in cases:
            line = prefix + " ".join(packages)
            with self.subTest(line=line):
                # Bash's Ubuntu line ends with a "# libmunge-dev" comment.
                self.assertRegex(bash, re.escape(line) + r"( #[^\n]*)?\n")


class AzureLinuxPmixVersionTests(unittest.TestCase):
    """Expected values were checked against real `grep -i | sed` in bash."""

    def test_picks_the_pmix_line_for_x86_64(self):
        self.assertEqual(install_pmix.azure_linux_pmix_version(_DNF_LIST, "x86_64"),
                         "5.0.1-1")

    def test_picks_the_pmix_line_for_aarch64(self):
        text = "pmix.aarch64   5.0.2-3.azl3   @azurelinux-official-base\n"
        self.assertEqual(install_pmix.azure_linux_pmix_version(text, "aarch64"),
                         "5.0.2-3")

    def test_other_architecture_is_not_matched_on_aarch64(self):
        text = "pmix.x86_64   5.0.1-1.azl3   @x\n"
        self.assertIsNone(install_pmix.azure_linux_pmix_version(text, "aarch64"))

    def test_tab_separated_columns(self):
        self.assertEqual(
            install_pmix.azure_linux_pmix_version("pmix.x86_64\t5.0.1-1.azl3\t@System\n",
                                                  "x86_64"),
            "5.0.1-1")

    def test_grep_is_case_insensitive_and_first_match_wins(self):
        # Bash records only the first word of the multi-line result.
        text = "PMIX.X86_64   5.0.2-3.azl3   @System\npmix.x86_64   5.0.1-1.azl3   @x\n"
        self.assertEqual(install_pmix.azure_linux_pmix_version(text, "x86_64"),
                         "5.0.2-3")

    def test_unescaped_dot_also_matches_openpmix(self):
        # grep's "pmix.x86_64" is a regex, so "openpmix.x86_64" matches too.
        text = "openpmix.x86_64   4.2.9-12.el9   @System\npmix.x86_64   5.0.1-1.azl3   @x\n"
        self.assertEqual(install_pmix.azure_linux_pmix_version(text, "x86_64"),
                         "4.2.9-12")

    def test_dash_in_repo_name_does_not_confuse_the_parse(self):
        text = "pmix.x86_64     5.0.1-1.azl3 @repo-1.2-3.x\n"
        self.assertEqual(install_pmix.azure_linux_pmix_version(text, "x86_64"),
                         "5.0.1-1")

    def test_line_sed_cannot_parse_returns_none(self):
        # Bash would record "pmix_x86_64" (sed passes the line through unchanged).
        text = "pmix_x86_64   1.2.3.azl3   @x\n"
        self.assertIsNone(install_pmix.azure_linux_pmix_version(text, "x86_64"))

    def test_no_pmix_line_returns_none(self):
        text = "Installed Packages\nhwloc.x86_64  2.9.2-1.azl3  @x\n"
        self.assertIsNone(install_pmix.azure_linux_pmix_version(text, "x86_64"))


# --------------------------------------------------------------------------- #
# install(): external effects mocked, /etc paths redirected to a temp dir.
# --------------------------------------------------------------------------- #
class _InstallTestBase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        # Redirect every system path the module writes or checks.
        self.paths = {
            "_SLURM_APT_LIST": os.path.join(self.root, "slurm.list"),
            "_SLURM_APT_PIN": os.path.join(self.root, "slurm-repository-pin-990"),
            "_MS_PROD_APT_LIST": os.path.join(self.root, "microsoft-prod.list"),
            "_SLURM_YUM_REPO": os.path.join(self.root, "slurm.repo"),
            "_MS_PROD_YUM_REPO": os.path.join(self.root, "microsoft-prod.repo"),
            "_WORK_DIR": os.path.join(self.root, "work"),
        }
        for name, value in self.paths.items():
            patcher = mock.patch.object(install_pmix, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _touch(self, key):
        Path(self.paths[key]).write_text("existing\n", encoding="utf-8")

    def _install(self, env, *, fail_on=None, capture=(0, _DNF_LIST), write_error=None):
        """Run install(env) with commands and the version writer mocked.

        fail_on: the first word(s) of a command that should return 1.
        Returns (rc, commands, exec_mock, capture_mock, record_mock).
        """
        commands = []

        def fake_exec(command, op, *, cwd=None, env=None):
            commands.append(command)
            if fail_on and command[:len(fail_on)] == fail_on:
                return 1
            # curl -O saves the download into cwd; simulate that file.
            if command[0] == "curl":
                Path(cwd, command[-1].rsplit("/", 1)[-1]).write_text("pkg", encoding="utf-8")
            return 0

        # Commands run in utils.process (run, run_all) and utils.package_installer
        # (installs); one mock at both keeps one ordered record of every command.
        with mock.patch.object(process, "exec_program", side_effect=fake_exec) as ex, \
             mock.patch.object(package_installer, "exec_program", new=ex), \
             mock.patch.object(install_pmix, "run_capture", return_value=capture) as cap, \
             mock.patch.object(install_pmix, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(install_pmix, "log_warn") as log_warn, \
             mock.patch.object(install_pmix, "log_error"), \
             mock.patch.object(component_config, "log_error"), \
             mock.patch.object(files, "log_info"), \
             mock.patch.object(files, "log_error"), \
             mock.patch.object(package_installer, "log_info"), \
             mock.patch.object(package_installer, "log_error"):
            rc = install_pmix.install(env)
        self.log_warn = log_warn
        return rc, commands, ex, cap, record

    def _read(self, key):
        return Path(self.paths[key]).read_text(encoding="utf-8")


class UbuntuTests(_InstallTestBase):
    _APT = [
        ["apt", "update"],
        [*_APT_INSTALL, f"pmix={_VERSION}", "libevent-dev", "libhwloc-dev"],
        ["apt-mark", "hold", f"pmix={_VERSION}", "libevent-dev", "libhwloc-dev"],
    ]

    def test_noble_without_ms_repo_installs_the_ms_package_first(self):
        env = _env("ubuntu24.04")
        rc, commands, ex, _, record = self._install(env)

        self.assertEqual(rc, 0)
        self.assertEqual(
            self._read("_SLURM_APT_LIST"),
            "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] "
            "https://packages.microsoft.com/repos/slurm-ubuntu-noble/ insiders main\n")
        pin = Path(_COMPONENT_DIR, "slurm-repo", "slurm-u.pin").read_text(encoding="utf-8")
        self.assertEqual(self._read("_SLURM_APT_PIN"), pin)
        self.assertEqual(commands, [
            ["curl", "-sSL", "-O",
             "https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb"],
            [*_APT_INSTALL, "./packages-microsoft-prod.deb"],
            *self._APT,
        ])
        # The downloads run in the work dir, with the build env.
        work = self.paths["_WORK_DIR"]
        self.assertEqual(ex.call_args_list[0], mock.call(commands[0], _OP, cwd=work, env=env))
        self.assertEqual(ex.call_args_list[1], mock.call(commands[1], _OP, cwd=work, env=env))
        self.assertFalse(os.path.exists(work))           # cleaned up
        record.assert_called_once_with("PMIX", _VERSION)

    def test_jammy_with_ms_repo_skips_the_download_and_overwrites_slurm_list(self):
        self._touch("_MS_PROD_APT_LIST")
        Path(self.paths["_SLURM_APT_LIST"]).write_text("old line\n", encoding="utf-8")
        rc, commands, _, _, record = self._install(_env("ubuntu22.04", arch_distro="arm64"))

        self.assertEqual(rc, 0)
        self.assertEqual(
            self._read("_SLURM_APT_LIST"),
            "deb [arch=arm64 signed-by=/etc/apt/trusted.gpg.d/microsoft-prod.gpg] "
            "https://packages.microsoft.com/repos/slurm-ubuntu-jammy/ insiders main\n")
        self.assertEqual(commands, self._APT)
        record.assert_called_once_with("PMIX", _VERSION)

    def test_unsupported_ubuntu_version_returns_3_before_touching_anything(self):
        rc, commands, _, _, record = self._install(_env("ubuntu20.04"))

        self.assertEqual(rc, 3)
        self.assertEqual(commands, [])
        self.assertFalse(os.path.exists(self.paths["_SLURM_APT_LIST"]))
        record.assert_not_called()

    def test_curl_failure_returns_3_and_stops(self):
        rc, commands, _, _, record = self._install(_env("ubuntu24.04"), fail_on=["curl"])

        self.assertEqual(rc, 3)
        self.assertEqual([c[0] for c in commands], ["curl"])
        self.assertFalse(os.path.exists(self.paths["_WORK_DIR"]))
        record.assert_not_called()

    def test_ms_package_install_failure_returns_3_and_stops(self):
        # bash (#622): apt install -y ./packages-microsoft-prod.deb (was dpkg -i).
        rc, commands, _, _, record = self._install(
            _env("ubuntu24.04"), fail_on=[*_APT_INSTALL, "./packages-microsoft-prod.deb"])

        self.assertEqual(rc, 3)
        self.assertEqual([c[0] for c in commands], ["curl", "apt-get"])
        self.assertFalse(os.path.exists(self.paths["_WORK_DIR"]))
        record.assert_not_called()

    def test_ubuntu26_uses_hpcx_pmix_and_does_nothing(self):
        # Returns before the config lookup, so no pmix version is needed.
        with mock.patch.object(install_pmix, "log_info") as info:
            rc, commands, _, cap, record = self._install(_env("ubuntu26.04", pmix=False))

        self.assertEqual(rc, 0)
        self.assertEqual(commands, [])
        cap.assert_not_called()
        record.assert_not_called()
        self.assertFalse(os.path.exists(self.paths["_SLURM_APT_LIST"]))
        info.assert_called_once()
        self.assertEqual(info.call_args.args[1],
                         "Ubuntu 26.04 uses the PMIx 5 stack bundled with HPC-X.")
        self.assertTrue(info.call_args.kwargs["skipped"])

    def test_apt_install_failure_returns_3_and_skips_the_hold(self):
        self._touch("_MS_PROD_APT_LIST")
        rc, commands, _, _, record = self._install(_env("ubuntu24.04"),
                                                   fail_on=["apt-get", "install"])

        self.assertEqual(rc, 3)
        self.assertEqual(commands, self._APT[:2])
        record.assert_not_called()

    def test_apt_update_failure_returns_3(self):
        self._touch("_MS_PROD_APT_LIST")
        rc, commands, _, _, _ = self._install(_env("ubuntu24.04"), fail_on=["apt", "update"])

        self.assertEqual(rc, 3)
        self.assertEqual(commands, self._APT[:1])

    def test_apt_mark_failure_returns_3(self):
        self._touch("_MS_PROD_APT_LIST")
        rc, _, _, _, record = self._install(_env("ubuntu24.04"), fail_on=["apt-mark"])

        self.assertEqual(rc, 3)
        record.assert_not_called()

    def test_missing_pin_file_returns_3(self):
        env = _env("ubuntu24.04")
        env["COMPONENT_DIR"] = self.root            # no slurm-repo/ here
        rc, commands, _, _, record = self._install(env)

        self.assertEqual(rc, 3)
        self.assertEqual(commands, [])
        record.assert_not_called()


class AzureLinuxTests(_InstallTestBase):
    _DNF = [
        [*_DNF_INSTALL, "pmix", "pmix-devel", "pmix-tools"],
        [*_DNF_INSTALL, "hwloc-devel", "libevent-devel", "munge-devel"],
    ]

    def test_records_the_installed_version_not_versions_json(self):
        env = _env("azurelinux3.0")
        rc, commands, _, cap, record = self._install(env)

        self.assertEqual(rc, 0)
        self.assertEqual(commands, self._DNF)
        cap.assert_called_once_with(["dnf", "list", "installed"], _OP, env=env)
        record.assert_called_once_with("PMIX", "5.0.1-1")
        self.log_warn.assert_not_called()       # only one pmix.x86_64 line

    def test_several_matching_lines_warn_with_count_and_recorded_value(self):
        dnf_list = "openpmix.x86_64   4.2.9-12.el9   @System\n" + _DNF_LIST
        rc, _, _, _, record = self._install(_env("azurelinux3.0"),
                                            capture=(0, dnf_list))

        self.assertEqual(rc, 0)
        record.assert_called_once_with("PMIX", "4.2.9-12")
        self.log_warn.assert_called_once()
        self.assertEqual(self.log_warn.call_args.kwargs,
                         {"matches": 2, "recorded": "4.2.9-12"})

    def test_aarch64_uses_the_aarch64_line(self):
        dnf_list = _DNF_LIST.replace("pmix.aarch64              5.0.1-1",
                                     "pmix.aarch64              5.0.9-7")
        rc, _, _, _, record = self._install(
            _env("azurelinux3.0", architecture="aarch64", arch_distro="aarch64"),
            capture=(0, dnf_list))

        self.assertEqual(rc, 0)
        record.assert_called_once_with("PMIX", "5.0.9-7")

    def test_first_dnf_install_failure_returns_3_and_stops(self):
        rc, commands, _, cap, record = self._install(_env("azurelinux3.0"),
                                                     fail_on=[*_DNF_INSTALL, "pmix"])

        self.assertEqual(rc, 3)
        self.assertEqual(commands, self._DNF[:1])
        cap.assert_not_called()
        record.assert_not_called()

    def test_second_dnf_install_failure_returns_3(self):
        rc, _, _, cap, record = self._install(_env("azurelinux3.0"),
                                              fail_on=[*_DNF_INSTALL, "hwloc-devel"])

        self.assertEqual(rc, 3)
        cap.assert_not_called()
        record.assert_not_called()

    def test_dnf_list_failure_returns_3(self):
        rc, _, _, _, record = self._install(_env("azurelinux3.0"), capture=(1, ""))

        self.assertEqual(rc, 3)
        record.assert_not_called()

    def test_no_installed_pmix_line_returns_3(self):
        rc, _, _, _, record = self._install(_env("azurelinux3.0"),
                                            capture=(0, "Installed Packages\n"))

        self.assertEqual(rc, 3)
        record.assert_not_called()


class RhelTests(_InstallTestBase):
    def _expected(self, major, extra):
        commands = [["dnf", "config-manager", "--set-enabled", extra]] if extra else []
        return commands + [
            ["dnf", "update", "-y"],
            [*_DNF_INSTALL, f"pmix-{_VERSION}.el{major}",
             "hwloc-devel", "libevent-devel", "munge-devel"],
        ]

    def _repo_text(self, major):
        return Path(_COMPONENT_DIR, "slurm-repo", f"slurm-el{major}.repo").read_text(
            encoding="utf-8")

    def test_alma9_without_ms_repo_installs_rpm_and_enables_crb(self):
        env = _env("almalinux9.8")
        rc, commands, ex, _, record = self._install(env)

        self.assertEqual(rc, 0)
        self.assertEqual(self._read("_SLURM_YUM_REPO"), self._repo_text("9"))
        self.assertEqual(commands, [
            ["curl", "-sSL", "-O",
             "https://packages.microsoft.com/config/rhel/9/packages-microsoft-prod.rpm"],
            ["rpm", "-i", "packages-microsoft-prod.rpm"],
            *self._expected("9", "crb"),
        ])
        work = self.paths["_WORK_DIR"]
        self.assertEqual(ex.call_args_list[1], mock.call(commands[1], _OP, cwd=work, env=env))
        self.assertFalse(os.path.exists(work))
        record.assert_called_once_with("PMIX", _VERSION)

    def test_rocky8_with_ms_repo_enables_powertools(self):
        self._touch("_MS_PROD_YUM_REPO")
        rc, commands, _, _, record = self._install(_env("rocky8.10"))

        self.assertEqual(rc, 0)
        self.assertEqual(self._read("_SLURM_YUM_REPO"), self._repo_text("8"))
        self.assertEqual(commands, self._expected("8", "powertools"))
        record.assert_called_once_with("PMIX", _VERSION)

    def test_rhel_enables_the_rhui_codeready_builder_repo(self):
        # bash (#620): RHEL uses codeready-builder-for-rhel-<major>-<arch>-rhui-rpms
        # instead of crb/powertools.
        for distro, arch, major in (("rhel9.6", "x86_64", "9"), ("rhel8.10", "aarch64", "8")):
            with self.subTest(distro=distro):
                self._touch("_MS_PROD_YUM_REPO")
                rc, commands, _, _, record = self._install(_env(distro, architecture=arch))

                self.assertEqual(rc, 0)
                self.assertEqual(self._read("_SLURM_YUM_REPO"), self._repo_text(major))
                self.assertEqual(commands, self._expected(
                    major, f"codeready-builder-for-rhel-{major}-{arch}-rhui-rpms"))
                record.assert_called_once_with("PMIX", _VERSION)

    def test_rhel_without_ms_repo_installs_rpm_first(self):
        rc, commands, _, _, _ = self._install(_env("rhel9.6"))

        self.assertEqual(rc, 0)
        self.assertEqual(commands[:3], [
            ["curl", "-sSL", "-O",
             "https://packages.microsoft.com/config/rhel/9/packages-microsoft-prod.rpm"],
            ["rpm", "-i", "packages-microsoft-prod.rpm"],
            ["dnf", "config-manager", "--set-enabled",
             "codeready-builder-for-rhel-9-x86_64-rhui-rpms"]])

    def test_other_major_enables_no_extra_repo(self):
        # Fake component dir with a slurm-el10.repo, so the copy succeeds.
        # (A distro install_packages() has no manager for, e.g. centos7.9,
        # would now fail at the pmix install; bash's else branch used dnf.)
        component_dir = os.path.join(self.root, "components")
        os.makedirs(os.path.join(component_dir, "slurm-repo"))
        Path(component_dir, "slurm-repo", "slurm-el10.repo").write_text("[el10]\n",
                                                                        encoding="utf-8")
        self._touch("_MS_PROD_YUM_REPO")
        env = _env("almalinux10.0")
        env["COMPONENT_DIR"] = component_dir
        rc, commands, _, _, _ = self._install(env)

        self.assertEqual(rc, 0)
        self.assertEqual(self._read("_SLURM_YUM_REPO"), "[el10]\n")
        self.assertEqual(commands, self._expected("10", None))

    def test_major_without_a_repo_file_returns_3(self):
        rc, commands, _, _, record = self._install(_env("rhel10.0"))

        self.assertEqual(rc, 3)
        self.assertEqual(commands, [])
        record.assert_not_called()

    def test_distribution_without_digits_returns_3(self):
        rc, commands, _, _, record = self._install(_env("mystery"))

        self.assertEqual(rc, 3)
        self.assertEqual(commands, [])
        record.assert_not_called()

    def test_rpm_failure_returns_3_and_stops(self):
        rc, commands, _, _, record = self._install(_env("almalinux9.8"), fail_on=["rpm"])

        self.assertEqual(rc, 3)
        self.assertEqual([c[0] for c in commands], ["curl", "rpm"])
        record.assert_not_called()

    def test_config_manager_failure_returns_3_and_stops(self):
        self._touch("_MS_PROD_YUM_REPO")
        rc, commands, _, _, _ = self._install(_env("almalinux9.8"),
                                              fail_on=["dnf", "config-manager"])

        self.assertEqual(rc, 3)
        self.assertEqual(commands, self._expected("9", "crb")[:1])

    def test_dnf_update_failure_returns_3_and_stops(self):
        self._touch("_MS_PROD_YUM_REPO")
        rc, commands, _, _, _ = self._install(_env("almalinux9.8"),
                                              fail_on=["dnf", "update"])

        self.assertEqual(rc, 3)
        self.assertEqual(commands, self._expected("9", "crb")[:2])

    def test_dnf_install_failure_returns_3(self):
        self._touch("_MS_PROD_YUM_REPO")
        rc, _, _, _, record = self._install(_env("almalinux9.8"),
                                            fail_on=_DNF_INSTALL)

        self.assertEqual(rc, 3)
        record.assert_not_called()


class ConfigAndWriteTests(_InstallTestBase):
    def test_missing_config_returns_3(self):
        rc, commands, _, _, record = self._install(_env("ubuntu24.04", pmix=False))

        self.assertEqual(rc, 3)
        self.assertEqual(commands, [])
        record.assert_not_called()

    def test_unwritable_versions_file_returns_3_without_raising(self):
        self._touch("_MS_PROD_APT_LIST")
        rc, _, _, _, record = self._install(
            _env("ubuntu24.04"), write_error=PermissionError("permission denied"))

        self.assertEqual(rc, 3)
        record.assert_called_once()

    def test_unwritable_slurm_list_returns_3(self):
        # Point the list into a "directory" that is a file: the write fails.
        Path(self.root, "not-a-dir").write_text("", encoding="utf-8")
        with mock.patch.object(install_pmix, "_SLURM_APT_LIST",
                               os.path.join(self.root, "not-a-dir", "slurm.list")):
            rc, commands, _, _, _ = self._install(_env("ubuntu24.04"))

        self.assertEqual(rc, 3)
        self.assertEqual(commands, [])

    def test_missing_sources_list_dir_is_created(self):
        # write_file creates the parent directory (bash's `>` would fail).
        self._touch("_MS_PROD_APT_LIST")
        slurm_list = os.path.join(self.root, "sources.list.d", "slurm.list")
        with mock.patch.object(install_pmix, "_SLURM_APT_LIST", slurm_list):
            rc, _, _, _, _ = self._install(_env("ubuntu24.04"))

        self.assertEqual(rc, 0)
        self.assertTrue(Path(slurm_list).read_text(encoding="utf-8").startswith("deb [arch="))


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "install-pmix"]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, install_pmix.install)
        self.assertEqual(steps[0].script, "install_pmix.sh")


if __name__ == "__main__":
    unittest.main()
