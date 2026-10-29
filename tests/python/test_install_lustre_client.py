"""Unit tests for components/python/install_lustre_client.py.

  * Generated files are compared byte for byte with fixtures rendered by the
    real bash functions and echo/tee lines of install_lustre_client.sh in WSL
    (tests/python/fixtures/install_lustre_client/), and each quoted heredoc is
    checked against the .sh file so the two can't drift.
  * The text helpers mirroring grep/cut/sed pipelines were compared against
    real bash in WSL on representative input.
  * install() runs with exec_program, run_capture, config_for,
    download_and_verify, check_dkms_status and write_component_version mocked,
    and every /etc and /lib/modules path redirected into a temporary directory.

Run from the repo root:
    python3 -m unittest tests.python.test_install_lustre_client -v
"""

from __future__ import annotations

import builtins
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_lustre_client as lustre
from utils.build_config import BuildConfig, build_plan

_OP = "install-lustre"
_REPO = Path(__file__).resolve().parents[2]
_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "install_lustre_client"
_SCRIPT = (_REPO / "components" / "install_lustre_client.sh").read_text(encoding="utf-8")
_PATCH = _REPO / "components" / "patches" / "lustre-client-lu-20071-timer-container-of.patch"

# Upstream changed install_lustre_client.sh after this port was written: it
# removed the LU-20071 patch (and its patch file) and the Jammy DKMS 3.4.1
# workaround, and skips 7.0 kernels. Builds run the bash script until ADO 988
# updates the port (Step.resync). The tests below compare the port with the old
# script or need the deleted patch file, so they are skipped until then.
_STALE = "install_lustre_client.sh changed upstream; port resync in ADO 988"

_KERNEL = "5.14.0-687.el9.x86_64"
_UBUNTU_V = "2.15.8-35-gdf4872c"
_NOBLE_V = "2.17.0-24-gf517bc4"
_EL_V = "2.15.8_35_gdf4872c"

_LSB = {
    "jammy": ('DISTRIB_ID=Ubuntu\nDISTRIB_RELEASE=22.04\nDISTRIB_CODENAME=jammy\n'
              'DISTRIB_DESCRIPTION="Ubuntu 22.04.5 LTS"\n'),
    "noble": ('DISTRIB_ID=Ubuntu\nDISTRIB_RELEASE=24.04\nDISTRIB_CODENAME=noble\n'
              'DISTRIB_DESCRIPTION="Ubuntu 24.04.3 LTS"\n'),
}
_OS_RELEASE = {
    "ubuntu22.04": 'NAME="Ubuntu"\nVERSION_ID="22.04"\nVERSION="22.04.5 LTS (Jammy Jellyfish)"\n',
    "ubuntu24.04": 'NAME="Ubuntu"\nVERSION_ID="24.04"\nVERSION="24.04.3 LTS (Noble Numbat)"\n',
    "almalinux9.8": 'NAME="AlmaLinux"\nVERSION="9.8 (Moss Jungle Cat)"\nVERSION_ID="9.8"\n',
    "rocky8.10": 'NAME="Rocky Linux"\nVERSION="8.10 (Green Obsidian)"\nVERSION_ID="8.10"\n',
    "azurelinux3.0": 'NAME="Microsoft Azure Linux"\nVERSION_ID="3.0"\n',
}
_DKMS_DEB = "dkms_3.4.1-1ubuntu1_all.deb"


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

    def test_dkms_version_from_output(self):
        f = lustre.dkms_version_from_output
        self.assertEqual(f("dkms-3.0.3\n"), "3.0.3")
        self.assertEqual(f("dkms-3.4.1"), "3.4.1")
        self.assertEqual(f("x\ndkms-2.8.7\n"), "2.8.7")
        self.assertEqual(f("dkms-3.0\ndkms-3.1\n"), "3.0\n3.1")
        self.assertEqual(f("dkms: 3.4.1\n"), "")
        self.assertEqual(f("dkms-\n"), "")
        self.assertEqual(f(""), "")

    def test_underscore_version(self):
        self.assertEqual(lustre.underscore_version("2.15.8-35-gdf4872c"), _EL_V)
        self.assertEqual(lustre.underscore_version(_EL_V), _EL_V)

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
            "el-dkms.conf.no-patch": lustre.skip_artifact_block(
                3, lustre._EL_SKIP_DESCRIPTION),
            "el-dkms.conf": lustre.skip_artifact_block(3, lustre._EL_SKIP_DESCRIPTION)
                            + lustre._LU20071_PATCH_LINE,
        }
        for name, text in cases.items():
            with self.subTest(fixture=name):
                self.assertEqual(text, _fixture(name))

    @unittest.skip(_STALE)
    def test_heredocs_match_the_bash_script(self):
        self.assertEqual(lustre._NO_O2IB_CONF, _heredoc("configure_lustre_dkms_no_o2ib"))
        self.assertEqual(lustre._NO_O2IB_TR_CONF,
                         _heredoc("configure_lustre_dkms_no_o2ib_with_tr_workaround"))
        self.assertEqual(lustre._LEGACY_DKMS_CONF,
                         _heredoc("configure_legacy_lustre_dkms_no_o2ib"))
        self.assertEqual(lustre._LU20071_PATCH_LINE,
                         _heredoc("configure_lustre_dkms_lu20071_patch"))
        # The literal (unexpanded) DKMS variable survives.
        self.assertIn("--with-linux=$kernel_source_dir ", lustre._LEGACY_DKMS_CONF)

    @unittest.skip(_STALE)
    def test_constants_match_the_bash_script(self):
        for text in [lustre._JAMMY_DKMS_URL, lustre._JAMMY_DKMS_SHA256,
                     lustre._UBUNTU_SKIP_DESCRIPTION, lustre._EL_SKIP_DESCRIPTION,
                     lustre._LU20071_PATCH_SOURCE, lustre._MICROSOFT_KEY_URL,
                     *lustre._SIGNED_BY.values()]:
            with self.subTest(text=text):
                self.assertIn(text, _SCRIPT)


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
            "_MODULES_DIR": "lib/modules",
            "_WORK_DIR": "work/lustre-build",
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

    def _system(self, distribution, codename=None, timer_h=None):
        """Write the os-release/lsb-release (and timer.h) the step reads."""
        os.makedirs(self.p("etc"), exist_ok=True)
        Path(self.p("etc/os-release")).write_text(_OS_RELEASE[distribution])
        if codename:
            Path(self.p("etc/lsb-release")).write_text(_LSB[codename])
        if timer_h is not None:
            header = self.p(f"lib/modules/{_KERNEL}/build/include/linux/timer.h")
            os.makedirs(os.path.dirname(header))
            Path(header).write_text(timer_h)

    def _env(self, distribution, **extra):
        env = {"DISTRIBUTION": distribution, "ARCHITECTURE": "x86_64",
               "ARCHITECTURE_DISTRO": "amd64", "GPU": "NVIDIA", "SKU": "H100",
               "TARGET_NODE_TYPE": "azure_vm_regular",
               "COMPONENT_DIR": str(_REPO / "components"), "PATH": "/usr/bin"}
        env.update(extra)
        return env

    def _install(self, env, *, version=_UBUNTU_V, fail_on=None, dkms_ok=True,
                 has_dkms=True, dkms_version="dkms-3.0.3\n", compare_rc=1,
                 download_error=None, write_error=None):
        """Run install(env) with everything external mocked; return (rc, mocks)."""
        cfg = None if version is None else {"version": version}
        seen = {}

        def fake_download(url, sha256, dest_dir="."):
            seen["work_dir_existed"] = os.path.isdir(dest_dir)
            if download_error:
                raise download_error
            os.makedirs(dest_dir, exist_ok=True)
            deb = Path(dest_dir) / Path(url).name
            deb.write_bytes(b"deb")
            return deb

        def fake_capture(command, op, **kwargs):
            if command[:2] == ["dkms", "--version"]:
                return (0, dkms_version)
            return (compare_rc, "")

        with mock.patch.object(lustre, "config_for", return_value=cfg) as config_for, \
             mock.patch.object(lustre, "exec_program",
                               side_effect=lambda c, op, **kw: 1 if c == fail_on else 0) \
                as exec_program, \
             mock.patch.object(lustre, "run_capture", side_effect=fake_capture) as capture, \
             mock.patch.object(lustre, "_command_exists", return_value=has_dkms), \
             mock.patch.object(lustre, "download_and_verify",
                               side_effect=fake_download) as download, \
             mock.patch.object(lustre, "check_dkms_status",
                               return_value=dkms_ok) as check, \
             mock.patch.object(lustre, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(lustre, "log_info") as log_info, \
             mock.patch.object(lustre, "log_warn") as log_warn, \
             mock.patch.object(lustre, "log_error") as log_error:
            rc = lustre.install(env)
        for call in exec_program.call_args_list + capture.call_args_list:
            self.assertEqual(call.args[1], _OP)
            self.assertIs(call.kwargs["env"], env)
        return rc, {"config_for": config_for, "exec": exec_program, "capture": capture,
                    "download": download, "check": check, "record": record,
                    "info": log_info, "warn": log_warn, "error": log_error, **seen}

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
            ["apt-get", "install", "-y", f"lustre-client-{_NOBLE_V}"],
            ["apt-get", "install", "-y", f"amlfs-lustre-client-dkms-{_NOBLE_V}"],
        ])
        m["config_for"].assert_called_once_with("lustre", env)
        self.assertEqual(self.read("etc/apt/sources.list.d/amlfs.list"),
                         _fixture("amlfs.list.noble-arm64"))
        self.assertEqual(self.read("etc/sysconfig/dkms-lustre"),
                         _fixture("sysconfig-dkms-lustre.conf"))
        self.assertFalse(os.path.exists(self.p("etc/dkms")))
        m["capture"].assert_not_called()
        m["download"].assert_not_called()
        m["check"].assert_called_once_with(["lustre-client-modules"], _OP, env,
                                           kernel=_KERNEL)
        m["record"].assert_called_once_with("LUSTRE", _NOBLE_V)
        m["warn"].assert_not_called()
        m["error"].assert_not_called()

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

    def test_ubuntu_2204_nvidia(self):
        self._system("ubuntu22.04", "jammy")
        env = self._env("ubuntu22.04")
        rc, m = self._install(env)

        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m), [
            ["apt-get", "update"],
            ["apt-get", "install", "-y", f"lustre-client-{_UBUNTU_V}"],
            ["apt-get", "install", "-y", f"amlfs-lustre-client-dkms-{_UBUNTU_V}"],
        ])
        self.assertEqual(self.read("etc/apt/sources.list.d/amlfs.list"),
                         _fixture("amlfs.list.jammy-amd64"))
        self.assertEqual(self.read(f"etc/dkms/lustre-client-modules-{_UBUNTU_V}.conf"),
                         _fixture("jammy-dkms.conf"))
        self.assertFalse(os.path.exists(self.p("etc/sysconfig")))
        m["capture"].assert_not_called()
        m["download"].assert_not_called()
        m["check"].assert_called_once_with(["lustre-client-modules"], _OP, env,
                                           kernel=_KERNEL)
        m["record"].assert_called_once_with("LUSTRE", _UBUNTU_V)
        m["warn"].assert_not_called()
        self.assertIn("dkms 3.4.1 workaround not needed (only for ubuntu22.04 with AMD GPUs)",
                      self._messages(m["info"]))

    def test_ubuntu_2204_amd_without_dkms_installs_dkms_341(self):
        self._system("ubuntu22.04", "jammy")
        rc, m = self._install(self._env("ubuntu22.04", GPU="AMD"), has_dkms=False)

        self.assertEqual(rc, 0)
        deb = os.path.join(self.p("work/lustre-build"), _DKMS_DEB)
        self.assertEqual(self._commands(m), [
            ["apt-get", "update"],
            ["apt-get", "install", "-y", f"lustre-client-{_UBUNTU_V}"],
            ["apt-get", "install", "-y", deb],
            ["apt-get", "install", "-y", f"amlfs-lustre-client-dkms-{_UBUNTU_V}"],
        ])
        m["capture"].assert_not_called()
        m["download"].assert_called_once_with(
            lustre._JAMMY_DKMS_URL, lustre._JAMMY_DKMS_SHA256,
            dest_dir=self.p("work/lustre-build"))
        self.assertFalse(m["work_dir_existed"])
        self.assertFalse(os.path.exists(self.p("work/lustre-build")))
        self.assertEqual(self.read(f"etc/dkms/lustre-client-modules-{_UBUNTU_V}.conf"),
                         _fixture("jammy-dkms.conf"))
        self.assertIn("dkms is not installed yet", self._messages(m["info"]))

    def test_ubuntu_2204_amd_old_dkms_installs_dkms_341(self):
        self._system("ubuntu22.04", "jammy")
        os.makedirs(self.p("work/lustre-build"))
        Path(self.p("work/lustre-build/" + _DKMS_DEB)).write_text("stale")
        rc, m = self._install(self._env("ubuntu22.04", GPU="AMD"),
                              dkms_version="dkms-3.0.3\n", compare_rc=1)

        self.assertEqual(rc, 0)
        self.assertEqual([c.args[0] for c in m["capture"].call_args_list], [
            ["dkms", "--version"],
            ["dpkg", "--compare-versions", "3.0.3", "ge", "3.4.1"],
        ])
        for call in m["capture"].call_args_list:
            self.assertIs(call.kwargs["failure_is_error"], False)
        self.assertFalse(m["work_dir_existed"])     # stale deb removed first
        m["download"].assert_called_once()
        self.assertEqual(self._info_fields(m, "dkms 3.0.3 is older than 3.4.1"),
                         {"dkms_version": "3.0.3"})

    def test_ubuntu_2204_amd_new_dkms_is_kept(self):
        self._system("ubuntu22.04", "jammy")
        rc, m = self._install(self._env("ubuntu22.04", GPU="AMD"),
                              dkms_version="dkms-3.4.1\n", compare_rc=0)

        self.assertEqual(rc, 0)
        m["download"].assert_not_called()
        self.assertEqual(len(self._commands(m)), 3)
        self.assertIn("dkms 3.4.1 is at least 3.4.1; keeping it", self._messages(m["info"]))

    def test_ubuntu_2204_amd_unparsable_dkms_version_installs(self):
        self._system("ubuntu22.04", "jammy")
        rc, m = self._install(self._env("ubuntu22.04", GPU="AMD"), dkms_version="")

        self.assertEqual(rc, 0)
        self.assertEqual([c.args[0] for c in m["capture"].call_args_list],
                         [["dkms", "--version"]])
        m["download"].assert_called_once()

    def test_ubuntu_2204_amd_download_failure(self):
        self._system("ubuntu22.04", "jammy")
        rc, m = self._install(self._env("ubuntu22.04", GPU="AMD"), has_dkms=False,
                              download_error=OSError("network down"))

        self.assertEqual(rc, 3)
        self.assertEqual(len(self._commands(m)), 2)            # no dkms package
        m["check"].assert_not_called()
        m["record"].assert_not_called()
        self.assertFalse(os.path.exists(self.p("work/lustre-build")))

    def test_ubuntu_2204_amd_deb_install_failure(self):
        self._system("ubuntu22.04", "jammy")
        deb = os.path.join(self.p("work/lustre-build"), _DKMS_DEB)
        rc, m = self._install(self._env("ubuntu22.04", GPU="AMD"), has_dkms=False,
                              fail_on=["apt-get", "install", "-y", deb])

        self.assertEqual(rc, 3)
        self.assertEqual(self._commands(m)[-1], ["apt-get", "install", "-y", deb])
        self.assertFalse(os.path.exists(self.p("etc/dkms")))
        self.assertFalse(os.path.exists(self.p("work/lustre-build")))

    def test_ubuntu_command_failures_stop(self):
        commands = [
            ["apt-get", "update"],
            ["apt-get", "install", "-y", f"lustre-client-{_NOBLE_V}"],
            ["apt-get", "install", "-y", f"amlfs-lustre-client-dkms-{_NOBLE_V}"],
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
                # A command failure is logged by exec_program, not repeated.
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
        ["dnf", "install", "-y", "--disableexcludes=main", "--refresh",
         f"lustre-client-dkms-{_EL_V}", f"lustre-client-{_EL_V}-devel"],
    ]

    def test_rhel_with_from_timer_kernel(self):
        self._system("almalinux9.8", timer_h="#define from_timer(var, t, f) x\n")
        env = self._env("almalinux9.8")
        rc, m = self._install(env, version="2.15.8-35-gdf4872c")   # "-" -> "_"

        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m), self._EL_COMMANDS)
        self.assertEqual(self.read("etc/yum.repos.d/amlfs.repo"), _fixture("amlfs.repo.el9"))
        self.assertEqual(self.read("etc/sysconfig/lustre"), _fixture("sysconfig-lustre.conf"))
        self.assertEqual(self.read(f"etc/dkms/lustre-client-{_EL_V}.conf"),
                         _fixture("el-dkms.conf.no-patch"))
        self.assertFalse(os.path.exists(self.p("etc/dkms/lustre-client/patches")))
        m["check"].assert_called_once_with(["lustre-client"], _OP, env, kernel=_KERNEL)
        m["record"].assert_called_once_with("LUSTRE", _EL_V)
        m["warn"].assert_not_called()
        m["error"].assert_not_called()
        self.assertEqual(
            self._info_fields(m, "LU-20071 patch not needed: the kernel still has "
                                 "from_timer()")["lu20071_patch"], False)
        self.assertEqual(self._info_fields(m, f"RHEL family: el9, Lustre version {_EL_V}"),
                         {"distro": "almalinux9.8", "os_major": "9", "version": _EL_V})
        self.assertEqual(self._info_fields(m, "Lustre client installed")["dkms_module"],
                         "lustre-client")

    @unittest.skip(_STALE)
    def test_rhel_kernel_without_from_timer_gets_lu20071_patch(self):
        self._system("almalinux9.8", timer_h="#define timer_container_of(v, t, f) x\n")
        rc, m = self._install(self._env("almalinux9.8"), version=_EL_V)

        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m), self._EL_COMMANDS)
        self.assertEqual(self.read(f"etc/dkms/lustre-client-{_EL_V}.conf"),
                         _fixture("el-dkms.conf"))
        copied = self.p("etc/dkms/lustre-client/patches/lu-20071-timer-container-of.patch")
        self.assertEqual(Path(copied).read_bytes(), _PATCH.read_bytes())
        self.assertTrue(self._info_fields(
            m, "LU-20071 patch needed: the kernel's timer.h has no from_timer() "
               "(RHEL 9.8 kernels 5.14.0-687+)")["lu20071_patch"])
        m["warn"].assert_not_called()

    def test_rhel_without_kernel_headers_skips_patch(self):
        self._system("rocky8.10")
        rc, m = self._install(self._env("rocky8.10"), version=_EL_V)

        self.assertEqual(rc, 0)
        self.assertEqual(self.read("etc/yum.repos.d/amlfs.repo"), _fixture("amlfs.repo.el8"))
        self.assertEqual(self.read(f"etc/dkms/lustre-client-{_EL_V}.conf"),
                         _fixture("el-dkms.conf.no-patch"))
        header = self.p(f"lib/modules/{_KERNEL}/build/include/linux/timer.h")
        self.assertIn(f"LU-20071 patch not needed: no kernel header {header}",
                      self._messages(m["info"]))

    @unittest.skip(_STALE)
    def test_rhel_rerun_appends_again_and_warns(self):
        self._system("almalinux9.8", timer_h="no timer helpers\n")
        env = self._env("almalinux9.8")
        rc, first = self._install(env, version=_EL_V)
        self.assertEqual(rc, 0)
        first["warn"].assert_not_called()

        rc, second = self._install(env, version=_EL_V)
        self.assertEqual(rc, 0)
        self.assertEqual(self.read(f"etc/dkms/lustre-client-{_EL_V}.conf"),
                         _fixture("el-dkms.conf.rerun"))
        conf = self.p(f"etc/dkms/lustre-client-{_EL_V}.conf")
        warned = [c.kwargs["path"] for c in second["warn"].call_args_list]
        self.assertEqual(warned, [conf, conf])        # skip block, then PATCH line
        self.assertEqual(second["warn"].call_args_list[0].kwargs["appended"],
                         '# EL ko2iblnd is record 3; --with-o2ib=no intentionally '
                         'skips it. | BUILD_EXCLUSIVE_KERNEL[3]="^$"')
        self.assertEqual(self._info_fields(second, f"Appended {conf}"),
                         {"path": conf, "action": "appended"})
        self.assertEqual(self._info_fields(second, f"Replaced {self.p('etc/yum.repos.d/amlfs.repo')}")
                         ["action"], "replaced")

    @unittest.skip(_STALE)
    def test_rhel_unreadable_timer_header_applies_patch_with_warning(self):
        self._system("almalinux9.8", timer_h="#define from_timer(a, b, c) x\n")
        header = self.p(f"lib/modules/{_KERNEL}/build/include/linux/timer.h")
        real_open = builtins.open

        def fake_open(path, *args, **kwargs):
            if path == header:
                raise PermissionError(13, "Permission denied", header)
            return real_open(path, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=fake_open):
            rc, m = self._install(self._env("almalinux9.8"), version=_EL_V)

        self.assertEqual(rc, 0)
        self.assertEqual(self.read(f"etc/dkms/lustre-client-{_EL_V}.conf"),
                         _fixture("el-dkms.conf"))
        self.assertEqual([c.kwargs["path"] for c in m["warn"].call_args_list], [header])

    def test_rhel_missing_patch_file_fails(self):
        self._system("almalinux9.8", timer_h="nothing\n")
        rc, m = self._install(self._env("almalinux9.8", COMPONENT_DIR=self.p("nowhere")),
                              version=_EL_V)
        self.assertEqual(rc, 3)
        self.assertEqual(self._commands(m), self._EL_COMMANDS[:1])   # no dnf install
        m["record"].assert_not_called()

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

    def test_command_exists_uses_env_path(self):
        with mock.patch.object(lustre.shutil, "which", return_value="/usr/sbin/dkms") as which:
            self.assertTrue(lustre._command_exists("dkms", {"PATH": "/usr/sbin"}))
        which.assert_called_once_with("dkms", path="/usr/sbin")


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
