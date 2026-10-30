"""Unit tests for components/python/install_lustre_client.py.

  * Generated files are compared byte for byte with fixtures rendered by the
    real bash functions and echo/tee lines of install_lustre_client.sh in WSL
    (tests/python/fixtures/install_lustre_client/), and each quoted heredoc is
    checked against the .sh file so the two can't drift.
  * The text helpers mirroring grep/cut/sed pipelines, and the kernel 7.0
    skip pattern, were compared against real bash in WSL on representative
    input.
  * install() runs with exec_program (one mock for this module's and
    utils.package_installer's, so all commands are recorded in order),
    config_for, check_dkms_status and write_component_version mocked, `uname -r` replaced, and every /etc path
    redirected into a temporary directory.

Run from the repo root:
    python3 -m unittest tests.python.test_install_lustre_client -v
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_lustre_client as lustre
from utils import package_installer
from utils.build_config import BuildConfig, build_plan

_OP = "install-lustre"
_REPO = Path(__file__).resolve().parents[2]
_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "install_lustre_client"
_SCRIPT = (_REPO / "components" / "install_lustre_client.sh").read_text(encoding="utf-8")

_KERNEL = "5.14.0-687.el9.x86_64"
_UBUNTU_V = "2.15.8-35-gdf4872c"
_NOBLE_V = "2.17.0-24-gf517bc4"
_EL_V = "2.15.8_35_gdf4872c"

# What install_packages() runs for bash's `apt-get install -y` / `dnf install -y`.
_APT = ["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0"]
_DNF = ["dnf", "install", "-y"]

_LSB = {
    "jammy": ('DISTRIB_ID=Ubuntu\nDISTRIB_RELEASE=22.04\nDISTRIB_CODENAME=jammy\n'
              'DISTRIB_DESCRIPTION="Ubuntu 22.04.5 LTS"\n'),
    "noble": ('DISTRIB_ID=Ubuntu\nDISTRIB_RELEASE=24.04\nDISTRIB_CODENAME=noble\n'
              'DISTRIB_DESCRIPTION="Ubuntu 24.04.3 LTS"\n'),
    "resolute": ('DISTRIB_ID=Ubuntu\nDISTRIB_RELEASE=26.04\nDISTRIB_CODENAME=resolute\n'
                 'DISTRIB_DESCRIPTION="Ubuntu 26.04 LTS"\n'),
}
_OS_RELEASE = {
    "ubuntu22.04": 'NAME="Ubuntu"\nVERSION_ID="22.04"\nVERSION="22.04.5 LTS (Jammy Jellyfish)"\n',
    "ubuntu24.04": 'NAME="Ubuntu"\nVERSION_ID="24.04"\nVERSION="24.04.3 LTS (Noble Numbat)"\n',
    "ubuntu26.04": 'NAME="Ubuntu"\nVERSION_ID="26.04"\nVERSION="26.04 LTS (Resolute Raccoon)"\n',
    "almalinux9.8": 'NAME="AlmaLinux"\nVERSION="9.8 (Moss Jungle Cat)"\nVERSION_ID="9.8"\n',
    "rocky8.10": 'NAME="Rocky Linux"\nVERSION="8.10 (Green Obsidian)"\nVERSION_ID="8.10"\n',
    "azurelinux3.0": 'NAME="Microsoft Azure Linux"\nVERSION_ID="3.0"\n',
}


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def _heredoc(function: str) -> str:
    """The body of the quoted heredoc inside `function` in the .sh file."""
    body = re.search(rf"^{function}\(\) {{\n(.*?)^}}\n", _SCRIPT, re.M | re.S).group(1)
    return re.search(r"<<'EOF'\n(.*?\n)EOF\n", body, re.S).group(1)


# --------------------------------------------------------------------------- #
# Pure helpers and parity of generated text.
# --------------------------------------------------------------------------- #
class HelperTests(unittest.TestCase):
    def test_ubuntu_version_from_os_release(self):
        f = lustre.ubuntu_version_from_os_release
        self.assertEqual(f(_OS_RELEASE["ubuntu22.04"]), "22.04")
        self.assertEqual(f(_OS_RELEASE["ubuntu24.04"]), "24.04")
        self.assertEqual(f("VERSION_ID=24.04\n"), "24.04")     # unquoted
        self.assertEqual(f("VERSION_ID\n"), "VERSION_ID")     # cut prints lines without "="
        self.assertEqual(f('X_VERSION_ID="1"\nVERSION_ID="22.04"\n'), "1\n22.04")
        self.assertEqual(f(""), "")

    def test_os_major_from_os_release(self):
        f = lustre.os_major_from_os_release
        self.assertEqual(f(_OS_RELEASE["almalinux9.8"]), "9")
        self.assertEqual(f(_OS_RELEASE["rocky8.10"]), "8")
        self.assertEqual(f(_OS_RELEASE["azurelinux3.0"]), "3")
        self.assertEqual(f('NAME="Rocky"\nVERSION_ID="8.10"'), "8")  # no final newline
        for bad in ["VERSION_ID=9.8\n", 'VERSION_ID="abc"\n', "", ' VERSION_ID="9"\n']:
            with self.subTest(bad=bad):
                self.assertEqual(f(bad), "")

    def test_lsb_codename(self):
        f = lustre.lsb_codename
        self.assertEqual(f(_LSB["jammy"]), "jammy")
        self.assertEqual(f('DISTRIB_CODENAME="noble"\n'), "noble")
        self.assertEqual(f("DISTRIB_CODENAME='noble'"), "noble")
        self.assertEqual(f("DISTRIB_CODENAME=a\nDISTRIB_CODENAME=b\n"), "b")
        self.assertEqual(f("  DISTRIB_CODENAME=jammy  \n"), "jammy")
        self.assertEqual(f("DISTRIB_ID=Ubuntu\n"), "")
        self.assertEqual(f("DISTRIB_CODENAME=\n"), "")

    def test_underscore_version(self):
        self.assertEqual(lustre.underscore_version("2.15.8-35-gdf4872c"), _EL_V)
        self.assertEqual(lustre.underscore_version(_EL_V), _EL_V)

    def test_skips_kernel_matches_bash(self):
        # Results of bash's [[ "$k" == "7.0" || "$k" == 7.0.* || "$k" == 7.0-* ]]
        # for each value, from WSL.
        skipped = ["7.0", "7.0.0", "7.0.0-1004-azure", "7.0-rc1", "7.0.", "7.0-"]
        installed = ["7.00", "7.01.2", "17.0", "6.8.0-1030-azure", "7.1.0", "7", "",
                     " 7.0", "7.0 ", "7.0_x", _KERNEL]
        for kernel in skipped:
            with self.subTest(kernel=kernel):
                self.assertTrue(lustre.skips_kernel(kernel))
        for kernel in installed:
            with self.subTest(kernel=kernel):
                self.assertFalse(lustre.skips_kernel(kernel))

    def test_current_kernel_version(self):
        with mock.patch.object(lustre, "_kernel_release", return_value=_KERNEL):
            self.assertEqual(lustre.current_kernel_version({"KERNEL_VERSION": "7.0"}),
                             ("7.0", "KERNEL_VERSION"))
            # ${KERNEL_VERSION:-...}: unset and empty both fall back to uname -r.
            self.assertEqual(lustre.current_kernel_version({}), (_KERNEL, "uname -r"))
            self.assertEqual(lustre.current_kernel_version({"KERNEL_VERSION": ""}),
                             (_KERNEL, "uname -r"))

    def test_rendered_files_match_bash(self):
        cases = {
            "amlfs.list.jammy-amd64": lustre.amlfs_apt_source(
                "amd64", "/etc/apt/trusted.gpg.d/microsoft-prod.gpg", "jammy"),
            "amlfs.list.noble-arm64": lustre.amlfs_apt_source(
                "arm64", "/usr/share/keyrings/microsoft-prod.gpg", "noble"),
            "amlfs.repo.el9": lustre.amlfs_yum_repo("9"),
            "amlfs.repo.el8": lustre.amlfs_yum_repo("8"),
            "sysconfig-lustre.conf": lustre._NO_O2IB_CONF,
            "sysconfig-dkms-lustre.conf": lustre._NO_O2IB_TR_CONF,
            "jammy-dkms.conf": lustre._LEGACY_DKMS_CONF + lustre.skip_artifact_block(
                11, lustre._UBUNTU_SKIP_DESCRIPTION),
            "el-dkms.conf": lustre.skip_artifact_block(3, lustre._EL_SKIP_DESCRIPTION),
            "el-dkms.conf.rerun": 2 * lustre.skip_artifact_block(
                3, lustre._EL_SKIP_DESCRIPTION),
        }
        for name, text in cases.items():
            with self.subTest(fixture=name):
                self.assertEqual(text, _fixture(name))

    def test_heredocs_match_the_bash_script(self):
        self.assertEqual(lustre._NO_O2IB_CONF, _heredoc("configure_lustre_dkms_no_o2ib"))
        self.assertEqual(lustre._NO_O2IB_TR_CONF,
                         _heredoc("configure_lustre_dkms_no_o2ib_with_tr_workaround"))
        self.assertEqual(lustre._LEGACY_DKMS_CONF,
                         _heredoc("configure_legacy_lustre_dkms_no_o2ib"))
        # The literal (unexpanded) DKMS variable survives.
        self.assertIn("--with-linux=$kernel_source_dir ", lustre._LEGACY_DKMS_CONF)

    def test_constants_match_the_bash_script(self):
        for text in [lustre._UBUNTU_SKIP_DESCRIPTION, lustre._EL_SKIP_DESCRIPTION,
                     lustre._MICROSOFT_KEY_URL, *lustre._SIGNED_BY.values(),
                     "Skipping Lustre client installation for kernel version ",
                     'CURRENT_KERNEL_VERSION=${KERNEL_VERSION:-$(uname -r)}',
                     f'"${{CURRENT_KERNEL_VERSION}}" == "{lustre._SKIPPED_KERNEL}"',
                     "dnf install -y " + " ".join(lustre._EL_DNF_OPTIONS)
                     + ' "${LUSTRE_PACKAGES[@]}"']:
            with self.subTest(text=text):
                self.assertIn(text, _SCRIPT)

    def test_bash_dropped_the_lu20071_patch_and_jammy_dkms_workaround(self):
        # Upstream removed both (#610, #633/#643); the port must not bring them back.
        for gone in ["lu20071", "lu-20071", "jammy_amd", "dkms_3_4_1"]:
            with self.subTest(text=gone):
                self.assertNotIn(gone, _SCRIPT)
                self.assertEqual([name for name in vars(lustre) if gone in name.lower()], [])
        self.assertFalse((_REPO / "components" / "patches" /
                          "lustre-client-lu-20071-timer-container-of.patch").exists())


# --------------------------------------------------------------------------- #
# install(): external effects mocked, system paths in a temporary directory.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        paths = {
            "_LSB_RELEASE": "etc/lsb-release",
            "_OS_RELEASE": "etc/os-release",
            "_APT_SOURCE": "etc/apt/sources.list.d/amlfs.list",
            "_YUM_REPO": "etc/yum.repos.d/amlfs.repo",
            "_DKMS_CONF_DIR": "etc/dkms",
            "_UBUNTU_DKMS_SYSCONFIG": "etc/sysconfig/dkms-lustre",
            "_EL_SYSCONFIG": "etc/sysconfig/lustre",
        }
        for name, rel in paths.items():
            patcher = mock.patch.object(lustre, name, self.p(rel))
            patcher.start()
            self.addCleanup(patcher.stop)
        os.makedirs(self.p("etc/yum.repos.d"))
        os.makedirs(self.p("etc/apt/sources.list.d"))
        patcher = mock.patch.object(lustre, "_kernel_release", return_value=_KERNEL)
        patcher.start()
        self.addCleanup(patcher.stop)

    def p(self, rel: str) -> str:
        return os.path.join(self.root, rel)

    def read(self, rel: str) -> str:
        return Path(self.p(rel)).read_text(encoding="utf-8")

    def _system(self, distribution, codename=None):
        """Write the os-release/lsb-release the step reads."""
        os.makedirs(self.p("etc"), exist_ok=True)
        Path(self.p("etc/os-release")).write_text(_OS_RELEASE[distribution])
        if codename:
            Path(self.p("etc/lsb-release")).write_text(_LSB[codename])

    def _env(self, distribution, **extra):
        env = {"DISTRIBUTION": distribution, "ARCHITECTURE": "x86_64",
               "ARCHITECTURE_DISTRO": "amd64", "GPU": "NVIDIA", "SKU": "H100",
               "TARGET_NODE_TYPE": "azure_vm_regular",
               "COMPONENT_DIR": str(_REPO / "components"), "PATH": "/usr/bin"}
        env.update(extra)
        return env

    def _install(self, env, *, version=_UBUNTU_V, fail_on=None, dkms_ok=True,
                 write_error=None):
        """Run install(env) with everything external mocked; return (rc, mocks)."""
        cfg = None if version is None else {"version": version}

        exec_program = mock.Mock(side_effect=lambda c, op, **kw: 1 if c == fail_on else 0)
        with mock.patch.object(lustre, "config_for", return_value=cfg) as config_for, \
             mock.patch.object(lustre, "exec_program", exec_program), \
             mock.patch.object(package_installer, "exec_program", exec_program), \
             mock.patch.object(package_installer, "log_info"), \
             mock.patch.object(package_installer, "log_error"), \
             mock.patch.object(lustre, "check_dkms_status",
                               return_value=dkms_ok) as check, \
             mock.patch.object(lustre, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(lustre, "log_info") as log_info, \
             mock.patch.object(lustre, "log_warn") as log_warn, \
             mock.patch.object(lustre, "log_error") as log_error:
            rc = lustre.install(env)
        for call in exec_program.call_args_list:
            self.assertEqual(call.args[1], _OP)
            self.assertIs(call.kwargs["env"], env)
        return rc, {"config_for": config_for, "exec": exec_program, "check": check,
                    "record": record, "info": log_info, "warn": log_warn,
                    "error": log_error}

    @staticmethod
    def _commands(m):
        return [c.args[0] for c in m["exec"].call_args_list]

    @staticmethod
    def _messages(log_mock):
        return [c.args[1] for c in log_mock.call_args_list]

    def _info_fields(self, m, message):
        for call in m["info"].call_args_list:
            if call.args[1] == message:
                return call.kwargs
        self.fail(f"no log_info {message!r}; got {self._messages(m['info'])}")

    def _assert_no_side_effects(self, m):
        m["exec"].assert_not_called()
        m["record"].assert_not_called()
        self.assertFalse(os.path.exists(self.p("etc/apt/sources.list.d/amlfs.list")))
        self.assertFalse(os.path.exists(self.p("etc/yum.repos.d/amlfs.repo")))
        self.assertFalse(os.path.exists(self.p("etc/dkms")))
        self.assertFalse(os.path.exists(self.p("etc/sysconfig")))

    # ---- Ubuntu ----------------------------------------------------------------
    def test_ubuntu_2404(self):
        self._system("ubuntu24.04", "noble")
        env = self._env("ubuntu24.04", ARCHITECTURE="aarch64", ARCHITECTURE_DISTRO="arm64")
        rc, m = self._install(env, version=_NOBLE_V)

        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m), [
            ["apt-get", "update"],
            [*_APT, f"lustre-client-{_NOBLE_V}"],
            [*_APT, f"amlfs-lustre-client-dkms-{_NOBLE_V}"],
        ])
        m["config_for"].assert_called_once_with("lustre", env)
        self.assertEqual(self.read("etc/apt/sources.list.d/amlfs.list"),
                         _fixture("amlfs.list.noble-arm64"))
        self.assertEqual(self.read("etc/sysconfig/dkms-lustre"),
                         _fixture("sysconfig-dkms-lustre.conf"))
        self.assertFalse(os.path.exists(self.p("etc/dkms")))
        m["check"].assert_called_once_with(["lustre-client-modules"], _OP, env,
                                           kernel=_KERNEL)
        m["record"].assert_called_once_with("LUSTRE", _NOBLE_V)
        m["warn"].assert_not_called()
        m["error"].assert_not_called()

        self.assertEqual(
            self._info_fields(m, f"Kernel version {_KERNEL} is not 7.0; installing "
                                 "the Lustre client"),
            {"kernel_version": _KERNEL, "kernel_source": "uname -r", "skipped": False})
        self.assertEqual(self._info_fields(m, f"Installing the Lustre client {_NOBLE_V}"),
                         {"version": _NOBLE_V, "distro": "ubuntu24.04", "arch": "aarch64",
                          "gpu": "NVIDIA", "kernel": _KERNEL, "family": "ubuntu"})
        self.assertEqual(self._info_fields(m, "Ubuntu 24.04 (noble)")["codename"], "noble")
        apt_source = self.p("etc/apt/sources.list.d/amlfs.list")
        self.assertEqual(self._info_fields(m, f"Created {apt_source}"),
                         {"path": apt_source, "action": "created"})
        self.assertEqual(self._info_fields(m, f"Recorded LUSTRE version {_NOBLE_V}"),
                         {"component": "LUSTRE", "version": _NOBLE_V})
        self.assertEqual(self._info_fields(m, "Lustre client installed"),
                         {"version": _NOBLE_V, "distro": "ubuntu24.04", "kernel": _KERNEL,
                          "dkms_module": "lustre-client-modules"})

    def test_ubuntu_rerun_replaces_apt_source_and_sysconfig(self):
        self._system("ubuntu24.04", "noble")
        os.makedirs(self.p("etc/sysconfig"))
        Path(self.p("etc/sysconfig/dkms-lustre")).write_text("# from the package\n")
        Path(self.p("etc/apt/sources.list.d/amlfs.list")).write_text("old\n")
        rc, m = self._install(self._env("ubuntu24.04"), version=_NOBLE_V)

        self.assertEqual(rc, 0)
        self.assertEqual(self.read("etc/apt/sources.list.d/amlfs.list"),
                         lustre.amlfs_apt_source("amd64", lustre._SIGNED_BY["24.04"], "noble"))
        self.assertEqual(self.read("etc/sysconfig/dkms-lustre"),
                         _fixture("sysconfig-dkms-lustre.conf"))
        sysconfig = self.p("etc/sysconfig/dkms-lustre")
        self.assertEqual(self._info_fields(m, f"Replaced {sysconfig}")["action"], "replaced")

    def test_ubuntu_2204_same_for_every_gpu(self):
        # The Jammy AMD dkms 3.4.1 upgrade is gone (#633/#643): AMD and NVIDIA
        # builds now run the same commands and write the same files.
        for gpu in ("NVIDIA", "AMD"):
            with self.subTest(gpu=gpu):
                self._system("ubuntu22.04", "jammy")
                env = self._env("ubuntu22.04", GPU=gpu)
                rc, m = self._install(env)

                self.assertEqual(rc, 0)
                self.assertEqual(self._commands(m), [
                    ["apt-get", "update"],
                    [*_APT, f"lustre-client-{_UBUNTU_V}"],
                    [*_APT, f"amlfs-lustre-client-dkms-{_UBUNTU_V}"],
                ])
                self.assertEqual(self.read("etc/apt/sources.list.d/amlfs.list"),
                                 _fixture("amlfs.list.jammy-amd64"))
                self.assertEqual(
                    self.read(f"etc/dkms/lustre-client-modules-{_UBUNTU_V}.conf"),
                    _fixture("jammy-dkms.conf"))
                self.assertFalse(os.path.exists(self.p("etc/sysconfig")))
                m["check"].assert_called_once_with(["lustre-client-modules"], _OP, env,
                                                   kernel=_KERNEL)
                m["record"].assert_called_once_with("LUSTRE", _UBUNTU_V)
                m["warn"].assert_not_called()
                self.assertIn("Ubuntu 22.04: legacy DKMS config (--with-o2ib=no, "
                              "skip record 11)", self._messages(m["info"]))
                os.remove(self.p(f"etc/dkms/lustre-client-modules-{_UBUNTU_V}.conf"))
                os.rmdir(self.p("etc/dkms"))

    def test_ubuntu_command_failures_stop(self):
        commands = [
            ["apt-get", "update"],
            [*_APT, f"lustre-client-{_NOBLE_V}"],
            [*_APT, f"amlfs-lustre-client-dkms-{_NOBLE_V}"],
        ]
        for index, failing in enumerate(commands):
            with self.subTest(failing=failing):
                self._system("ubuntu24.04", "noble")
                rc, m = self._install(self._env("ubuntu24.04"), version=_NOBLE_V,
                                      fail_on=failing)
                self.assertEqual(rc, 3)
                self.assertEqual(self._commands(m), commands[:index + 1])
                m["check"].assert_not_called()
                m["record"].assert_not_called()
                self.assertIn("Lustre client installation failed",
                              self._messages(m["error"]))
                # A command failure is logged by exec_program (and
                # install_packages), not repeated here.
                self.assertEqual(len(m["error"].call_args_list), 1)

    def test_ubuntu_dkms_check_failure(self):
        self._system("ubuntu24.04", "noble")
        rc, m = self._install(self._env("ubuntu24.04"), version=_NOBLE_V, dkms_ok=False)
        self.assertEqual(rc, 3)
        m["record"].assert_not_called()

    def test_ubuntu_unknown_version_fails_before_writing(self):
        self._system("ubuntu22.04", "jammy")
        Path(self.p("etc/os-release")).write_text('VERSION_ID="20.04"\n')
        rc, m = self._install(self._env("ubuntu22.04"))
        self.assertEqual(rc, 3)
        self._assert_no_side_effects(m)
        self.assertIn("20.04", m["error"].call_args_list[0].args[1])

    def test_ubuntu_missing_codename_fails_before_writing(self):
        self._system("ubuntu22.04")
        Path(self.p("etc/lsb-release")).write_text("DISTRIB_ID=Ubuntu\n")
        rc, m = self._install(self._env("ubuntu22.04"))
        self.assertEqual(rc, 3)
        self._assert_no_side_effects(m)

    def test_ubuntu_missing_arch_fails_before_writing(self):
        self._system("ubuntu22.04", "jammy")
        env = self._env("ubuntu22.04")
        del env["ARCHITECTURE_DISTRO"]
        rc, m = self._install(env)
        self.assertEqual(rc, 3)
        self._assert_no_side_effects(m)

    def test_ubuntu_missing_lsb_release(self):
        self._system("ubuntu22.04")
        rc, m = self._install(self._env("ubuntu22.04"))
        self.assertEqual(rc, 3)
        self._assert_no_side_effects(m)
        self.assertIn("file operation failed", m["error"].call_args.args[1])

    def test_apt_source_write_error(self):
        self._system("ubuntu22.04", "jammy")
        os.rmdir(self.p("etc/apt/sources.list.d"))
        Path(self.p("etc/apt/sources.list.d")).write_text("a file, not a dir")
        rc, m = self._install(self._env("ubuntu22.04"))
        self.assertEqual(rc, 3)
        m["exec"].assert_not_called()
        m["record"].assert_not_called()

    # ---- RHEL family ---------------------------------------------------------------
    _EL_COMMANDS = [
        ["rpm", "--import", "https://packages.microsoft.com/keys/microsoft.asc"],
        [*_DNF, "--disableexcludes=main", "--refresh",
         f"lustre-client-dkms-{_EL_V}", f"lustre-client-{_EL_V}-devel"],
    ]

    def test_rhel(self):
        self._system("almalinux9.8")
        env = self._env("almalinux9.8")
        rc, m = self._install(env, version="2.15.8-35-gdf4872c")   # "-" -> "_"

        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m), self._EL_COMMANDS)
        self.assertEqual(self.read("etc/yum.repos.d/amlfs.repo"), _fixture("amlfs.repo.el9"))
        self.assertEqual(self.read("etc/sysconfig/lustre"), _fixture("sysconfig-lustre.conf"))
        # Only the record-3 skip: the LU-20071 PATCH line is gone (#610).
        self.assertEqual(self.read(f"etc/dkms/lustre-client-{_EL_V}.conf"),
                         _fixture("el-dkms.conf"))
        self.assertEqual(os.listdir(self.p("etc/dkms")), [f"lustre-client-{_EL_V}.conf"])
        m["check"].assert_called_once_with(["lustre-client"], _OP, env, kernel=_KERNEL)
        m["record"].assert_called_once_with("LUSTRE", _EL_V)
        m["warn"].assert_not_called()
        m["error"].assert_not_called()
        self.assertEqual(self._info_fields(m, f"RHEL family: el9, Lustre version {_EL_V}"),
                         {"distro": "almalinux9.8", "os_major": "9", "version": _EL_V})
        self.assertEqual(self._info_fields(m, "Lustre client installed")["dkms_module"],
                         "lustre-client")

    def test_rhel8(self):
        self._system("rocky8.10")
        rc, m = self._install(self._env("rocky8.10"), version=_EL_V)

        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m), self._EL_COMMANDS)
        self.assertEqual(self.read("etc/yum.repos.d/amlfs.repo"), _fixture("amlfs.repo.el8"))
        self.assertEqual(self.read(f"etc/dkms/lustre-client-{_EL_V}.conf"),
                         _fixture("el-dkms.conf"))

    def test_rhel_rerun_appends_again_and_warns(self):
        self._system("almalinux9.8")
        env = self._env("almalinux9.8")
        rc, first = self._install(env, version=_EL_V)
        self.assertEqual(rc, 0)
        first["warn"].assert_not_called()

        rc, second = self._install(env, version=_EL_V)
        self.assertEqual(rc, 0)
        self.assertEqual(self.read(f"etc/dkms/lustre-client-{_EL_V}.conf"),
                         _fixture("el-dkms.conf.rerun"))
        conf = self.p(f"etc/dkms/lustre-client-{_EL_V}.conf")
        self.assertEqual([c.kwargs["path"] for c in second["warn"].call_args_list], [conf])
        self.assertEqual(second["warn"].call_args.kwargs["appended"],
                         '# EL ko2iblnd is record 3; --with-o2ib=no intentionally '
                         'skips it. | BUILD_EXCLUSIVE_KERNEL[3]="^$"')
        self.assertEqual(self._info_fields(second, f"Appended {conf}"),
                         {"path": conf, "action": "appended"})
        self.assertEqual(self._info_fields(second, f"Replaced {self.p('etc/yum.repos.d/amlfs.repo')}")
                         ["action"], "replaced")

    def test_rhel_command_failures_stop(self):
        for index, failing in enumerate(self._EL_COMMANDS):
            with self.subTest(failing=failing[0]):
                self._system("almalinux9.8")
                rc, m = self._install(self._env("almalinux9.8"), version=_EL_V,
                                      fail_on=failing)
                self.assertEqual(rc, 3)
                self.assertEqual(self._commands(m), self._EL_COMMANDS[:index + 1])
                m["check"].assert_not_called()
                m["record"].assert_not_called()
                self.assertEqual(len(m["error"].call_args_list), 1)

    def test_rhel_rpm_import_failure_writes_nothing(self):
        self._system("almalinux9.8")
        rc, m = self._install(self._env("almalinux9.8"), version=_EL_V,
                              fail_on=self._EL_COMMANDS[0])
        self.assertEqual(rc, 3)
        self.assertFalse(os.path.exists(self.p("etc/yum.repos.d/amlfs.repo")))
        self.assertFalse(os.path.exists(self.p("etc/dkms")))

    def test_rhel_dkms_check_failure(self):
        self._system("almalinux9.8")
        rc, m = self._install(self._env("almalinux9.8"), version=_EL_V, dkms_ok=False)
        self.assertEqual(rc, 3)
        m["record"].assert_not_called()

    def test_rhel_empty_os_major_fails_before_writing(self):
        self._system("almalinux9.8")
        Path(self.p("etc/os-release")).write_text("VERSION_ID=9.8\n")
        rc, m = self._install(self._env("almalinux9.8"), version=_EL_V)
        self.assertEqual(rc, 3)
        self._assert_no_side_effects(m)

    def test_azurelinux_warns_then_fails_on_missing_version(self):
        self._system("azurelinux3.0")
        rc, m = self._install(self._env("azurelinux3.0"), version=None)
        self.assertEqual(rc, 3)
        self.assertEqual(m["warn"].call_args.kwargs, {"distro": "azurelinux3.0"})
        self.assertIn("does not run install_lustre_client.sh", m["warn"].call_args.args[1])
        self._assert_no_side_effects(m)

    def test_azurelinux_with_a_version_follows_the_rhel_branch(self):
        self._system("azurelinux3.0")
        rc, m = self._install(self._env("azurelinux3.0"), version=_EL_V)
        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m), self._EL_COMMANDS)
        self.assertIn("amlfs-el3", self.read("etc/yum.repos.d/amlfs.repo"))
        m["warn"].assert_called_once()

    # ---- common ----------------------------------------------------------------
    def test_missing_version(self):
        for cfg_version in (None, ""):
            with self.subTest(version=cfg_version):
                self._system("ubuntu24.04", "noble")
                rc, m = self._install(self._env("ubuntu24.04"), version=cfg_version)
                self.assertEqual(rc, 3)
                self._assert_no_side_effects(m)
                self.assertIn("could not resolve lustre version from versions.json",
                              self._messages(m["error"]))
                m["warn"].assert_not_called()

    # ---- kernel 7.0 skip ---------------------------------------------------------
    def test_kernel_7_0_skips_without_changing_anything(self):
        cases = [
            # (distribution, lustre version, KERNEL_VERSION, uname -r, source)
            ("ubuntu24.04", _NOBLE_V, "7.0", _KERNEL, "KERNEL_VERSION"),
            ("ubuntu24.04", _NOBLE_V, None, "7.0.0-1004-azure", "uname -r"),
            ("ubuntu24.04", _NOBLE_V, "", "7.0.0-1004-azure", "uname -r"),
            # Ubuntu 26.04 has no "lustre" entry; bash skips before using it.
            ("ubuntu26.04", None, "7.0", "7.0.0-14-generic", "KERNEL_VERSION"),
            ("almalinux9.8", _EL_V, "7.0-rc1", _KERNEL, "KERNEL_VERSION"),
        ]
        for distro, version, requested, uname, source in cases:
            with self.subTest(distro=distro, kernel_version=requested, uname=uname):
                self._system(distro)
                extra = {} if requested is None else {"KERNEL_VERSION": requested}
                env = self._env(distro, **extra)
                with mock.patch.object(lustre, "_kernel_release", return_value=uname):
                    rc, m = self._install(env, version=version)

                self.assertEqual(rc, 0)
                m["config_for"].assert_called_once_with("lustre", env)   # looked up first
                self._assert_no_side_effects(m)
                m["check"].assert_not_called()
                m["error"].assert_not_called()
                current = requested or uname
                self.assertEqual(
                    self._info_fields(m, "Skipping Lustre client installation for "
                                         f"kernel version {current}"),
                    {"kernel_version": current, "kernel_source": source,
                     "distro": distro, "skipped": True})

    def test_kernel_version_overrides_uname(self):
        # Packer's KERNEL_VERSION wins over the running kernel, as in bash; the
        # DKMS check still uses `uname -r` (bash check_dkms_status does).
        self._system("ubuntu24.04", "noble")
        env = self._env("ubuntu24.04", KERNEL_VERSION="6.8")
        with mock.patch.object(lustre, "_kernel_release", return_value="7.0.0-1004-azure"):
            rc, m = self._install(env, version=_NOBLE_V)

        self.assertEqual(rc, 0)
        self.assertEqual(len(self._commands(m)), 3)
        m["check"].assert_called_once_with(["lustre-client-modules"], _OP, env,
                                           kernel="7.0.0-1004-azure")
        self.assertEqual(
            self._info_fields(m, "Kernel version 6.8 is not 7.0; installing the "
                                 "Lustre client"),
            {"kernel_version": "6.8", "kernel_source": "KERNEL_VERSION", "skipped": False})

    def test_ubuntu_2604_without_a_7_0_kernel_fails_before_writing(self):
        # Bash would write "signed-by=" (no SIGNED_BY for 26.04); with no
        # "lustre" entry for 26.04 it would also install "lustre-client-null".
        for version, message in [(None, "could not resolve lustre version from "
                                        "versions.json"),
                                 (_NOBLE_V, "no apt signing key known for Ubuntu "
                                            "'26.04' (bash sets SIGNED_BY only for "
                                            "22.04 and 24.04)")]:
            with self.subTest(version=version):
                self._system("ubuntu26.04", "resolute")
                rc, m = self._install(self._env("ubuntu26.04", KERNEL_VERSION="6.17"),
                                      version=version)
                self.assertEqual(rc, 3)
                self._assert_no_side_effects(m)
                self.assertIn(message, self._messages(m["error"]))

    def test_record_failure(self):
        self._system("ubuntu24.04", "noble")
        rc, m = self._install(self._env("ubuntu24.04"), version=_NOBLE_V,
                              write_error=OSError("disk full"))
        self.assertEqual(rc, 3)
        self.assertEqual(m["error"].call_args.kwargs,
                         {"component": "LUSTRE", "version": _NOBLE_V})


class KernelReleaseTests(unittest.TestCase):
    def test_kernel_release_is_platform_release(self):
        with mock.patch.object(lustre.platform, "release", return_value="6.8.0-1-azure"):
            self.assertEqual(lustre._kernel_release(), "6.8.0-1-azure")



class WiringTests(unittest.TestCase):
    def test_step_is_wired_before_mpifileutils(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        ops = [s.op for s in build_plan(cfg)]
        steps = [s for s in build_plan(cfg) if s.op == _OP]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, lustre.install)
        self.assertEqual(steps[0].script, "install_lustre_client.sh")
        self.assertEqual(ops[ops.index(_OP) + 1], "install-mpifileutils")


if __name__ == "__main__":
    unittest.main()
