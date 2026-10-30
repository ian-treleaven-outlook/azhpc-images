"""Unit tests for components/python/install_doca.py.

  * Pure helpers are tested directly. parse_apt_cache_version,
    strip_dnf_excludes, rpm_marker_provides and
    installed_doca_packages were also compared with the real awk/sed/sort
    pipelines in WSL, and apt_marker_control / rpm_marker_spec with the
    heredocs rendered by real bash.
  * Parity tests compare every file text with the heredoc in
    components/install_doca.sh, so the two can't drift apart.
  * install() runs with commands (exec_program, wherever the toolkit runs
    them, recorded in one ordered list), downloads, config, DKMS check and
    the versions file mocked; every path it writes (/etc/dkms, dnf.conf, /tmp
    marker files, openibd drop-in, openib.conf, the apt pin) points into a
    temporary directory. The patches are read from the real components/patches.

Run from the repo root:
    python3 -m unittest tests.python.test_doca -v
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_doca
from utils import component_config, download, files, ofed, package_installer, process
from utils.distro import Distro
from utils.download import ChecksumError
from utils.build_config import BuildConfig, build_plan

_OP = "install-doca"
_REPO = Path(__file__).resolve().parents[2]
_BASH = (_REPO / "components" / "install_doca.sh").read_text(encoding="utf-8")
_PATCHES = _REPO / "components" / "patches"
_MARKER = "hpcx-provides-doca-ofed-deps"
_KERNEL = "5.14.0-611.el9.x86_64"
# What install_packages() runs on Ubuntu / the RPM distros.
_APT = ["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0"]
_DNF = ["dnf", "install", "-y"]

_UBUNTU_CFG = {"version": "3.2.3", "sha256": "abc",
               "url": "https://example/DOCA_v3.2.3/host/doca-host_3.2.3-019000-25.10-ubuntu2404_amd64.deb"}
_RHEL_CFG = {"version": "3.2.2", "sha256": "def",
             "url": "https://example/DOCA_v3.2.2/host/doca-host-3.2.2-035000_25.10_rhel9.x86_64.rpm"}
_PRIVATE_CFG = {"version": "3.1.0-091513-25.07-ubuntu2404", "source": "private",
                "repo_file": "doca-host_3.1.0-091513-25.07-ubuntu2404_arm64.deb", "sha256": "123"}
_APT_VERSIONS = {"openmpi": "4.1.7rc1-1.2510066", "ucx": "1.19.0-1.2510066",
                 "sharp": "3.12.0.MLNX20250916-1"}
_OFED_OUT = "OFED-internal-25.10-1.7.1:\n\nInstalled Packages:\n"
_REQUIRES = ("hcoll = 4.8.3230\nlibibverbs >= 2510\nopenmpi = 4.1.7rc1-1.2510066\n"
             "sharp\nucx >= 1.19.0\n")
_INSTALLED = "rdma-core doca\nlibibverbs doca\nmlnx-tools doca\nbash @System\n"
_BASEOS = "bash\nlibibverbs\nrdma-core\n"
_DPLL_NEW = "int (*ffo_get)(const struct dpll_pin *pin, struct dpll_ffo_param *ffo,\n"


def _heredoc(opening: str, closing: str = "EOF") -> str:
    """The text of the heredoc that starts on the line `opening` in install_doca.sh."""
    start = _BASH.index(opening + "\n") + len(opening) + 1
    end = _BASH.index(f"\n{closing}\n", start)
    return _BASH[start:end + 1]


# --------------------------------------------------------------------------- #
# Pure helpers and parity with the bash heredocs.
# --------------------------------------------------------------------------- #
class ParseTests(unittest.TestCase):
    def test_parse_apt_cache_version(self):
        parse = install_doca.parse_apt_cache_version
        self.assertEqual(parse("Package: openmpi\nVersion: 4.1.5-1\nArch: amd64\n"), "4.1.5-1")
        self.assertEqual(parse("Version: 4.1.5-1\nVersion: 5.0.0-2\n"), "4.1.5-1")  # exit
        self.assertEqual(parse("Version: 1.0 extra\n"), "1.0")                      # $2
        self.assertIsNone(parse("Package: openmpi\n"))
        self.assertIsNone(parse("Version:\nVersion: 2\n"))     # awk prints "" and exits
        self.assertIsNone(parse(""))

    def test_strip_dnf_excludes_like_sed(self):
        strip = install_doca.strip_dnf_excludes
        self.assertEqual(strip("[main]\ngpgcheck=1\nexclude=kernel* mft\ninstallonly=3\n"),
                         "[main]\ngpgcheck=1\ninstallonly=3\n")
        self.assertEqual(strip("[main]\nexclude=x"), "[main]\n")
        self.assertEqual(strip("a\nexclude=y\nb"), "a\nb")
        self.assertEqual(strip("exclude=a\n exclude=b\nexcludex=1\n"),
                         " exclude=b\nexcludex=1\n")
        self.assertEqual(strip("[main]\n"), "[main]\n")

    def test_rpm_marker_provides(self):
        output = ("openmpi = 4.1.7rc1-1\nucx >= 1.19.0\nsharp\nucx-devel\n"
                  "libibverbs >= 2510\nucx-cma <\nhcoll(x86-64) = 4.8\n"
                  "openmpi = 4.1.7rc1-1\n  ucx-knem   =   1.2\n")
        self.assertEqual(install_doca.rpm_marker_provides(output), [
            "openmpi = 4.1.7rc1-1", "sharp", "ucx = 1.19.0", "ucx-cma", "ucx-devel",
            "ucx-knem = 1.2"])
        self.assertEqual(install_doca.rpm_marker_provides("libibverbs\n"), [])

    def test_installed_doca_packages(self):
        output = "rdma-core doca\nlibibverbs doca-ofed\nkernel @System\nlonely\nrdma-core doca\n"
        self.assertEqual(install_doca.installed_doca_packages(output),
                         ["libibverbs", "rdma-core"])

    def test_baseos_repo_name(self):
        def name(distro, arch):
            return install_doca.baseos_repo_name(Distro(distro), arch)
        self.assertEqual(name("almalinux9.8", "x86_64"), "baseos")
        self.assertEqual(name("rocky8.10", "x86_64"), "baseos")
        self.assertEqual(name("rhel9.8", "x86_64"), "rhel-9-for-x86_64-baseos-rhui-rpms")
        self.assertEqual(name("rhel8.10", "aarch64"), "rhel-8-for-aarch64-baseos-rhui-rpms")

    def test_bash_branch(self):
        for distro, branch in (("ubuntu22.04", "ubuntu"), ("ubuntu26.04", "ubuntu"),
                               ("azurelinux3.0", "azurelinux"), ("almalinux9.8", "rhel"),
                               ("rocky8.10", "rhel"), ("rhel9.8", "rhel"), ("", "rhel")):
            with self.subTest(distro=distro):
                self.assertEqual(install_doca.bash_branch(Distro(distro)), branch)

    def test_expand_glob(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "m_2_all.deb").touch()
            Path(root, "m_1_all.deb").touch()
            pattern = os.path.join(root, "m_*_all.deb")
            self.assertEqual(install_doca.expand_glob(pattern),
                             [os.path.join(root, "m_1_all.deb"), os.path.join(root, "m_2_all.deb")])
            missing = os.path.join(root, "x_*.deb")
            self.assertEqual(install_doca.expand_glob(missing), [missing])


class BashParityTests(unittest.TestCase):
    def test_dkms_pin(self):
        self.assertEqual(install_doca._DKMS_PIN_CONTENT,
                         _heredoc("    cat > /etc/apt/preferences.d/doca-dkms-pin <<PIN", "PIN"))

    def test_openibd_override(self):
        self.assertEqual(install_doca._OPENIBD_OVERRIDE, _heredoc(
            "cat > /etc/systemd/system/openibd.service.d/override.conf <<EOF"))

    def test_ipoib_lines(self):
        self.assertIn('echo -e "\\n# Load IPoIB\\nIPOIB_LOAD=no" | sudo tee -a '
                      '/etc/infiniband/openib.conf', _BASH)
        self.assertEqual(install_doca._IPOIB_OFF, "\n# Load IPoIB\nIPOIB_LOAD=no\n")

    def test_apt_marker_control(self):
        text = _heredoc('    cat > "${marker_control}" <<EOF')
        for sharp, provides in ((None, ""), ("3.12-1", ", sharp (= 3.12-1)")):
            expected = (text.replace("${HPCX_DOCA_OFED_DEPS_MARKER}", _MARKER)
                        .replace("${openmpi_version}", "4.1.7-1")
                        .replace("${ucx_version}", "1.19-1")
                        .replace("${sharp_provides}", provides)
                        .replace("${DOCA_VERSION}", "3.2.3"))
            self.assertNotIn("$", expected)
            self.assertEqual(install_doca.apt_marker_control("4.1.7-1", "1.19-1", "3.2.3", sharp),
                             expected)

    def test_rpm_marker_spec(self):
        header = (_heredoc('    cat > "${spec_file}" <<EOF')
                  .replace("${HPCX_DOCA_OFED_DEPS_MARKER}", _MARKER)
                  .replace("${DOCA_VERSION}", "3.2.2"))
        body = _heredoc("    cat >> \"${spec_file}\" <<'EOF'")
        self.assertIn("printf 'Provides: %s\\n' \"${provides[@]}\" >> \"${spec_file}\"", _BASH)
        self.assertEqual(install_doca.rpm_marker_spec("3.2.2", ["openmpi = 4", "ucx"]),
                         header + "Provides: openmpi = 4\nProvides: ucx\n" + body)

    def test_dkms_conf_lines_and_patch_names(self):
        for line in ('PATCH+=("peer-mem-gpl.patch")', 'PATCH+=("dpll-ffo-param.patch")',
                     'PATCH+=("xdp-fragments.patch")'):
            self.assertIn(f"\n{line}\nEOF\n", _BASH)
        for source, name in (install_doca._PEER_MEM_PATCH, install_doca._DPLL_PATCH,
                             install_doca._XDP_PATCH):
            self.assertIn(f'"${{COMPONENT_DIR}}/patches/{source}" "${{patch_dir}}/{name}"', _BASH)
            self.assertTrue((_PATCHES / source).is_file())
        self.assertIn(f"local module_version={install_doca._XDP_MODULE_VERSION}\n", _BASH)
        self.assertIn("grep -q 'struct dpll_ffo_param \\*ffo,'", _BASH)


# --------------------------------------------------------------------------- #
# install(): external effects mocked, files in a temporary directory.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # ofed_info runs in utils/ofed.py (record_ofed_version).
        mock.patch.object(ofed, "run_capture", side_effect=self._fake_capture).start()
        self.ofed_error = mock.patch.object(ofed, "log_error").start()
        self.addCleanup(mock.patch.stopall)
        root = Path(self._tmp.name)
        self.top = root / "top"
        (self.top / "internal_bits").mkdir(parents=True)
        self.dkms = root / "dkms"
        self.kernels = root / "kernels"
        self.tmp = root / "tmp"
        self.tmp.mkdir()
        self.dnf_conf = root / "dnf.conf"
        self.dnf_conf_text = "[main]\ngpgcheck=1\nexclude=kernel* mft\n"
        self.dnf_conf.write_text(self.dnf_conf_text)
        self.dropin = root / "openibd.service.d"
        self.openib = root / "openib.conf"
        self.pin = root / "doca-dkms-pin"
        for name, value in (("_DKMS_ETC_DIR", self.dkms), ("_KERNEL_SRC_DIR", self.kernels),
                            ("_MARKER_WORK_DIR", self.tmp), ("_DNF_CONF", self.dnf_conf),
                            ("_OPENIBD_DROPIN_DIR", self.dropin), ("_OPENIB_CONF", self.openib),
                            ("_DKMS_PIN", self.pin)):
            patcher = mock.patch.object(install_doca, name, str(value))
            patcher.start()
            self.addCleanup(patcher.stop)

    def _fake_exec(self, command, op, *, cwd=None, env=None, failure_is_error=True):
        self.calls.append((command, cwd))
        if command == [*_DNF, "doca-ofed"]:
            self.dnf_conf_during = self.dnf_conf.read_text()
        if command == self.fail_on:
            return 1
        if command[:2] == ["dpkg", "--compare-versions"]:
            return self.dpkg_rc
        if command[0] == "equivs-build":
            Path(cwd, f"{_MARKER}_3.2.3_all.deb").touch()
        if command[0] == "rpmbuild" and self.rpm_built:
            rpms = Path(command[2].split(" ", 1)[1], "RPMS", "noarch")
            rpms.mkdir(parents=True)
            (rpms / f"{_MARKER}-3.2.2-1.noarch.rpm").touch()
            self.spec_text = Path(command[4]).read_text()
        return 0

    def _fake_capture(self, command, op, *, env=None, failure_is_error=True):
        self.captures.append(command)
        if command[:2] == ["apt-cache", "show"]:
            version = self.apt_versions.get(command[2])
            return (0, f"Package: {command[2]}\nVersion: {version}\n" if version else "")
        if command == ["ofed_info"]:
            return self.ofed
        if "--requires" in command:
            return (0, self.requires)
        if "--installed" in command:
            return (0, self.installed)
        return (0, self.baseos)

    def _fake_download(self, url, sha256, dest_dir="."):
        self.downloads.append((url, sha256, str(dest_dir)))
        if self.download_error:
            raise self.download_error
        os.makedirs(dest_dir, exist_ok=True)
        path = Path(dest_dir, os.path.basename(url))
        path.write_bytes(b"pkg")
        return path

    def _install(self, distro="ubuntu24.04", cfg=_UBUNTU_CFG, sku="A100",
                 node_type="azure_vm_regular", arch="x86_64", *, fail_on=None, dpkg_rc=0,
                 apt_versions=_APT_VERSIONS, ofed=(0, _OFED_OUT), requires=_REQUIRES,
                 installed=_INSTALLED, baseos=_BASEOS, rpmbuild_present=True,
                 rpm_built=True, dkms_ok=True, dpll=None, download_error=None,
                 checksum_ok=True, write_error=None):
        self.calls, self.captures, self.downloads = [], [], []
        self.fail_on, self.dpkg_rc, self.apt_versions = fail_on, dpkg_rc, apt_versions
        self.ofed, self.requires, self.installed, self.baseos = ofed, requires, installed, baseos
        self.rpm_built, self.download_error = rpm_built, download_error
        self.dnf_conf_during = self.spec_text = None
        if dpll is not None:
            header = self.kernels / _KERNEL / "include" / "linux" / "dpll.h"
            header.parent.mkdir(parents=True, exist_ok=True)
            header.write_text(dpll)
        self.env = {"DISTRIBUTION": distro, "ARCHITECTURE": arch, "SKU": sku,
                    "TARGET_NODE_TYPE": node_type, "TOP_DIR": str(self.top),
                    "COMPONENT_DIR": str(_REPO / "components"), "PATH": "/usr/bin"}
        self.work = os.path.join(str(self.top), "distros", distro)
        with mock.patch.object(install_doca, "config_for", return_value=cfg), \
             mock.patch.object(component_config, "config_for", return_value=cfg), \
             mock.patch.object(component_config, "log_error") as config_error, \
             mock.patch.object(install_doca, "exec_program", side_effect=self._fake_exec), \
             mock.patch.object(process, "exec_program", side_effect=self._fake_exec), \
             mock.patch.object(package_installer, "exec_program", side_effect=self._fake_exec), \
             mock.patch.object(package_installer, "log_info"), \
             mock.patch.object(package_installer, "log_error"), \
             mock.patch.object(files, "log_info") as file_info, \
             mock.patch.object(install_doca, "run_capture", side_effect=self._fake_capture), \
             mock.patch.object(download, "download_and_verify",
                               side_effect=self._fake_download), \
             mock.patch.object(download, "log_error") as download_error, \
             mock.patch.object(install_doca, "verify_checksum",
                               return_value=checksum_ok) as checksum, \
             mock.patch.object(install_doca, "command_exists", return_value=rpmbuild_present), \
             mock.patch.object(install_doca, "check_dkms_status", return_value=dkms_ok) as dkms, \
             mock.patch.object(install_doca, "running_kernel", return_value=_KERNEL), \
             mock.patch.object(component_config, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(component_config, "log_info") as config_info, \
             mock.patch.object(install_doca, "log_info") as info, \
             mock.patch.object(install_doca, "log_warn") as warn, \
             mock.patch.object(install_doca, "log_error") as error:
            rc = install_doca.install(self.env)
        self.checksum, self.dkms_check, self.record = checksum, dkms, record
        self.info, self.warn, self.error = info, warn, error
        self.file_info, self.config_error = file_info, config_error
        self.config_info = config_info
        self.download_error_log = download_error
        return rc

    # ---- helpers for expectations -----------------------------------------
    def _commands(self):
        return [c[0] for c in self.calls]

    def _messages(self, logger):
        return {c.args[1]: c.kwargs for c in logger.call_args_list}

    def _ubuntu_commands(self, deb, version="3.2.3"):
        return [
            [*_APT, deb],
            ["apt-get", "update"],
            [*_APT, "equivs"],
            ["dpkg", "--compare-versions", version, "lt", "3.3.0"],
            ["equivs-build", str(self.tmp / _MARKER)],
            [*_APT, str(self.tmp / f"{_MARKER}_3.2.3_all.deb")],
            [*_APT, "doca-ofed"],
            ["systemctl", "daemon-reload"],
            ["systemctl", "enable", "openibd"],
        ]

    def _rpm_marker_commands(self):
        topdir = str(self.tmp / f"{_MARKER}-rpmbuild")
        return [
            ["rpmbuild", "--define", f"_topdir {topdir}", "-bb",
             os.path.join(topdir, "SPECS", f"{_MARKER}.spec")],
            ["rpm", "-Uvh", "--replacepkgs",
             os.path.join(topdir, "RPMS", "noarch", f"{_MARKER}-3.2.2-1.noarch.rpm")],
        ]

    def _rhel_commands(self, pin="baseos.excludepkgs=libibverbs rdma-core"):
        return ([["rpm", "-i", os.path.basename(_RHEL_CFG["url"])], ["dnf", "clean", "all"]]
                + self._rpm_marker_commands()
                + [[*_DNF, "doca-ofed"]]
                + ([["dnf", "config-manager", "--save", f"--setopt={pin}"]] if pin else [])
                + [["systemctl", "daemon-reload"], ["systemctl", "enable", "openibd"]])

    def _deb(self, cfg=_UBUNTU_CFG):
        return os.path.realpath(os.path.join(self.work, os.path.basename(cfg["url"])))

    # ---- Ubuntu -------------------------------------------------------------
    def test_ubuntu_full_flow(self):
        self.assertEqual(self._install(), 0)
        self.assertEqual(self._commands(), self._ubuntu_commands(self._deb()))
        self.assertEqual(self.calls[0][1], self.work)              # bash's working dir
        self.assertEqual(self.downloads, [(_UBUNTU_CFG["url"], "abc", self.work)])
        self.assertEqual(self.captures, [["apt-cache", "show", "openmpi"],
                                         ["apt-cache", "show", "ucx"],
                                         ["apt-cache", "show", "sharp"], ["ofed_info"]])
        self.assertEqual(self.pin.read_text(), install_doca._DKMS_PIN_CONTENT)
        # The PeerDirect patch for mlnx-ofed-kernel; no DPLL, no XDP (24.04).
        self.assertEqual((self.dkms / "mlnx-ofed-kernel.conf").read_text(),
                         'PATCH+=("peer-mem-gpl.patch")\n')
        patches = self.dkms / "mlnx-ofed-kernel" / "patches"
        self.assertEqual(os.listdir(patches), ["peer-mem-gpl.patch"])
        self.assertEqual((patches / "peer-mem-gpl.patch").read_bytes(),
                         (_PATCHES / "mlnx-ofa-kernel-peer-mem-gpl.patch").read_bytes())
        self.assertEqual(sorted(os.listdir(self.dkms)), ["mlnx-ofed-kernel",
                                                         "mlnx-ofed-kernel.conf"])
        self.dkms_check.assert_called_once_with(
            ["mlnx-ofed-kernel", "iser", "isert", "srp"], _OP, self.env)
        # Marker files cleaned up, versions recorded, openibd configured.
        self.assertEqual(os.listdir(self.tmp), [])
        self.assertEqual(self.record.call_args_list, [mock.call("DOCA", "3.2.3"),
                                                      mock.call("OFED", "25.10-1.7.1")])
        self.assertEqual((self.dropin / "override.conf").read_text(),
                         install_doca._OPENIBD_OVERRIDE)
        self.assertFalse(self.openib.exists())          # A100 VM uses IPoIB
        self.warn.assert_not_called()
        self.error.assert_not_called()

    def test_ubuntu_marker_control_provides_sharp_before_3_3_0(self):
        with mock.patch.object(install_doca, "write_file",
                               wraps=install_doca.write_file) as write:
            self.assertEqual(self._install(), 0)
        written = {c.args[0]: c.args[1] for c in write.call_args_list}
        self.assertEqual(written[str(self.tmp / _MARKER)], install_doca.apt_marker_control(
            _APT_VERSIONS["openmpi"], _APT_VERSIONS["ucx"], "3.2.3", _APT_VERSIONS["sharp"]))

    def test_doca_3_3_0_does_not_provide_sharp(self):
        cfg = dict(_UBUNTU_CFG, version="3.3.0")
        with mock.patch.object(install_doca, "apt_marker_control",
                               wraps=install_doca.apt_marker_control) as control:
            self.assertEqual(self._install(cfg=cfg, dpkg_rc=1, arch="aarch64",
                                           node_type="baremetal_3p", sku="GB200"), 0)
        control.assert_called_once_with(_APT_VERSIONS["openmpi"], _APT_VERSIONS["ucx"],
                                        "3.3.0", None)
        self.assertNotIn(["apt-cache", "show", "sharp"], self.captures)

    def test_dpkg_compare_error_warns_and_skips_sharp(self):
        self.assertEqual(self._install(dpkg_rc=2), 0)
        self.assertNotIn(["apt-cache", "show", "sharp"], self.captures)
        self.warn.assert_called_once()
        self.assertEqual(self.warn.call_args.kwargs["exit_code"], 2)

    def test_ubuntu22_04_doca_3_2_3_adds_the_xdp_patch(self):
        self.assertEqual(self._install("ubuntu22.04"), 0)
        conf = self.dkms / "mlnx-ofed-kernel-25.10.OFED.25.10.3.1.8.1-1.conf"
        self.assertEqual(conf.read_text(), 'PATCH+=("xdp-fragments.patch")\n')
        patches = self.dkms / "mlnx-ofed-kernel" / "patches"
        self.assertEqual(sorted(os.listdir(patches)), ["peer-mem-gpl.patch",
                                                       "xdp-fragments.patch"])
        self.assertEqual((patches / "xdp-fragments.patch").read_bytes(),
                         (_PATCHES / "mlnx-ofed-kernel-xdp-fragments.patch").read_bytes())
        # The patch overlays are in place before doca-ofed builds the modules.
        self.assertLess(self._commands().index(["equivs-build", str(self.tmp / _MARKER)]),
                        self._commands().index([*_APT, "doca-ofed"]))

    def test_xdp_patch_only_for_3_2_3_on_22_04(self):
        for distro, version in (("ubuntu22.04", "3.2.2"), ("ubuntu24.04", "3.2.3"),
                                ("ubuntu26.04", "3.4.0")):
            with self.subTest(distro=distro, version=version):
                self.assertEqual(self._install(distro, cfg=dict(_UBUNTU_CFG, version=version),
                                               dpkg_rc=1), 0)
                self.assertFalse(list(self.dkms.glob("mlnx-ofed-kernel-*.conf")))

    def test_v100_skips_the_peer_mem_patch(self):
        self.assertEqual(self._install("ubuntu22.04", sku="V100"), 0)
        self.assertFalse((self.dkms / "mlnx-ofed-kernel.conf").exists())
        self.assertEqual(os.listdir(self.dkms / "mlnx-ofed-kernel" / "patches"),
                         ["xdp-fragments.patch"])
        self.assertIn("V100", next(m for m in self._messages(self.info) if "V100" in m))

    def test_private_package_from_internal_bits(self):
        package = self.top / "internal_bits" / _PRIVATE_CFG["repo_file"]
        package.write_bytes(b"deb")
        rc = self._install(cfg=_PRIVATE_CFG, arch="aarch64", sku="GB200",
                           node_type="azure_vm_akshost", dpkg_rc=1)
        self.assertEqual(rc, 0)
        self.assertEqual(self.downloads, [])
        self.checksum.assert_called_once_with(str(package), "123")
        self.assertEqual(self._commands()[0], [*_APT, os.path.realpath(str(package))])

    def test_private_checksum_mismatch_returns_3(self):
        rc = self._install(cfg=_PRIVATE_CFG, checksum_ok=False)
        self.assertEqual(rc, 3)
        self.assertEqual(self.calls, [])
        self.record.assert_not_called()           # (verify_checksum logs the mismatch)

    # ---- IPoIB (sku_uses_ipoib) ---------------------------------------------
    def test_ipoib_disabled_on_baremetal_and_ncv6(self):
        for sku, node_type in (("GB200", "baremetal_3p"), ("GB200", "baremetal_1p"),
                               ("NCv6", "azure_vm_regular")):
            with self.subTest(sku=sku, node_type=node_type):
                if self.openib.exists():
                    self.openib.unlink()
                self.assertEqual(self._install(sku=sku, node_type=node_type), 0)
                self.assertEqual(self.openib.read_text(), "\n# Load IPoIB\nIPOIB_LOAD=no\n")
                self.assertIn({"path": str(self.openib), "action": "created"},
                              list(self._messages(self.file_info).values()))

    def test_ipoib_rerun_appends_again_with_warning(self):
        self.openib.write_text("ONBOOT=yes\n")
        self.assertEqual(self._install(node_type="baremetal_3p"), 0)
        self.warn.assert_not_called()
        self.assertEqual(self._install(node_type="baremetal_3p"), 0)
        self.assertEqual(self.openib.read_text(), "ONBOOT=yes\n" + install_doca._IPOIB_OFF * 2)
        self.warn.assert_called_once()
        self.assertEqual(self.warn.call_args.kwargs, {"path": str(self.openib)})

    # ---- Azure Linux ----------------------------------------------------------
    def test_azure_linux_flow(self):
        rc = self._install("azurelinux3.0", cfg=_RHEL_CFG)
        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(), (
            [["rpm", "-i", os.path.basename(_RHEL_CFG["url"])], ["dnf", "clean", "all"]]
            + self._rpm_marker_commands()
            + [[*_DNF, "doca-ofed"],
               ["systemctl", "daemon-reload"], ["systemctl", "enable", "openibd"]]))
        self.assertEqual(self.calls[0][1], self.work)
        self.assertEqual(len(self.downloads), 2)         # bash downloads twice
        self.warn.assert_called_once()
        self.dkms_check.assert_not_called()
        self.assertFalse(self.dkms.exists())
        self.assertEqual(self.dnf_conf.read_text(), self.dnf_conf_text)   # untouched
        self.assertEqual(self.spec_text, install_doca.rpm_marker_spec("3.2.2", [
            "hcoll = 4.8.3230", "openmpi = 4.1.7rc1-1.2510066", "sharp", "ucx = 1.19.0"]))
        self.assertFalse((self.tmp / f"{_MARKER}-rpmbuild").exists())

    def test_rpm_build_installed_when_rpmbuild_missing(self):
        self.assertEqual(self._install("azurelinux3.0", cfg=_RHEL_CFG,
                                       rpmbuild_present=False), 0)
        self.assertEqual(self._commands()[2], [*_DNF, "rpm-build"])

    def test_private_package_on_rpm_distros_fails_like_bash(self):
        for distro in ("azurelinux3.0", "almalinux9.8"):
            with self.subTest(distro=distro):
                (self.top / "internal_bits" / "x.rpm").write_bytes(b"rpm")
                cfg = {"version": "3.2.2", "source": "private", "repo_file": "x.rpm",
                       "sha256": "1"}
                self.assertEqual(self._install(distro, cfg=cfg), 3)
                self.assertEqual(self.calls, [])
                self.assertIn("private package has no url", self.error.call_args_list[0].args[1])

    # ---- RHEL family ----------------------------------------------------------
    def test_rhel_flow(self):
        self.assertEqual(self._install("almalinux9.8", cfg=_RHEL_CFG), 0)
        self.assertEqual(self._commands(), self._rhel_commands())
        self.assertEqual(len(self.downloads), 2)
        # exclude= lifted during the install, then the original restored.
        self.assertEqual(self.dnf_conf_during, "[main]\ngpgcheck=1\n")
        self.assertEqual(self.dnf_conf.read_text(), self.dnf_conf_text)
        self.assertFalse(Path(f"{self.dnf_conf}.bak").exists())
        self.assertEqual((self.dkms / "mlnx-ofa_kernel.conf").read_text(),
                         'PATCH+=("peer-mem-gpl.patch")\n')
        self.dkms_check.assert_called_once_with(
            ["mlnx-ofa_kernel", "iser", "isert", "srp"], _OP, self.env)
        self.assertIn(["dnf", "repoquery", "--installed", "--quiet", "--qf",
                       "%{name} %{from_repo}\\n"], self.captures)
        self.assertIn(["dnf", "repoquery", "--quiet", "--repo=baseos", "--qf", "%{name}\\n",
                       "*"], self.captures)

    def test_rhel_dpll_patch_appended_when_the_kernel_needs_it(self):
        self.assertEqual(self._install("rocky9.8", cfg=_RHEL_CFG, dpll=_DPLL_NEW), 0)
        self.assertEqual((self.dkms / "mlnx-ofa_kernel.conf").read_text(),
                         'PATCH+=("peer-mem-gpl.patch")\nPATCH+=("dpll-ffo-param.patch")\n')
        self.assertEqual(
            (self.dkms / "mlnx-ofa_kernel" / "patches" / "dpll-ffo-param.patch").read_bytes(),
            (_PATCHES / "mlnx-ofa-kernel-dpll-ffo-param.patch").read_bytes())

    def test_rhel_v100_dpll_only(self):
        self.assertEqual(self._install("rocky9.8", cfg=_RHEL_CFG, sku="V100",
                                       dpll=_DPLL_NEW), 0)
        self.assertEqual((self.dkms / "mlnx-ofa_kernel.conf").read_text(),
                         'PATCH+=("dpll-ffo-param.patch")\n')

    def test_rhel_older_dpll_header_gets_no_dpll_patch(self):
        self.assertEqual(self._install("almalinux9.8", cfg=_RHEL_CFG,
                                       dpll="struct dpll_pin_ops {\n"), 0)
        self.assertEqual(os.listdir(self.dkms / "mlnx-ofa_kernel" / "patches"),
                         ["peer-mem-gpl.patch"])

    def test_rhel_baseos_repo_name_on_rhel(self):
        self.assertEqual(self._install("rhel9.8", cfg=_RHEL_CFG), 0)
        self.assertEqual(self._commands(), self._rhel_commands(
            "rhel-9-for-x86_64-baseos-rhui-rpms.excludepkgs=libibverbs rdma-core"))

    def test_rhel_no_conflicts_pins_nothing(self):
        self.assertEqual(self._install("almalinux8.10", cfg=_RHEL_CFG, baseos="bash\n"), 0)
        self.assertEqual(self._commands(), self._rhel_commands(pin=None))

    def test_rhel_no_doca_packages_returns_3(self):
        rc = self._install("almalinux9.8", cfg=_RHEL_CFG, installed="bash @System\n")
        self.assertEqual(rc, 3)
        self.assertIn("no packages found from doca* repos",
                      self.error.call_args_list[0].args[1])
        self.record.assert_not_called()

    def test_rhel_failed_install_still_restores_dnf_conf(self):
        rc = self._install("almalinux9.8", cfg=_RHEL_CFG,
                           fail_on=[*_DNF, "doca-ofed"])
        self.assertEqual(rc, 3)
        self.assertEqual(self.dnf_conf.read_text(), self.dnf_conf_text)
        self.assertFalse(Path(f"{self.dnf_conf}.bak").exists())
        self.dkms_check.assert_not_called()

    # ---- failures -----------------------------------------------------------
    def test_missing_config_returns_3(self):
        for cfg in (None, {}, {"version": "3.2.3"}, {"version": "3.2.3", "url": "u"},
                    {"version": "3.2.3", "source": "private", "sha256": "x"}):
            with self.subTest(cfg=cfg):
                self.assertEqual(self._install(cfg=cfg), 3)
                self.assertEqual(self.calls, [])
                self.config_error.assert_called_once()     # by require_config

    def test_download_failure_returns_3(self):
        self.assertEqual(self._install(download_error=ChecksumError("checksum")), 3)
        self.assertEqual(self.calls, [])
        self.assertIn("checksum", self.download_error_log.call_args_list[0].args[1])

    def test_each_failing_ubuntu_command_stops_the_step(self):
        commands = self._ubuntu_commands(os.path.realpath(os.path.join(
            str(self.top), "distros", "ubuntu24.04", os.path.basename(_UBUNTU_CFG["url"]))))
        for index, command in enumerate(commands):
            if command[0] == "dpkg":
                continue                         # an answer, not a failure
            with self.subTest(command=command[:3]):
                self.assertEqual(self._install(fail_on=command), 3)
                self.assertEqual(self._commands(), commands[:index + 1])
                # exec_program (mocked here) logs the failed command; a failed
                # distro branch also logs its consequence.
                if command[0] != "systemctl":
                    self.error.assert_called()

    def test_each_failing_rhel_command_stops_the_step(self):
        commands = self._rhel_commands()
        for index, command in enumerate(commands):
            with self.subTest(command=command[:3]):
                self.assertEqual(self._install("almalinux9.8", cfg=_RHEL_CFG,
                                               fail_on=command), 3)
                self.assertEqual(self._commands(), commands[:index + 1])
                self.assertEqual(self.dnf_conf.read_text(), self.dnf_conf_text)

    def test_unreadable_apt_versions_return_3(self):
        for missing in ("openmpi", "ucx", "sharp"):
            with self.subTest(missing=missing):
                versions = {k: v for k, v in _APT_VERSIONS.items() if k != missing}
                self.assertEqual(self._install(apt_versions=versions), 3)
                self.assertEqual(self.error.call_args_list[0].args[1],
                                 f"could not read {missing} version from DOCA repo")
                self.assertNotIn("equivs-build", [c[0] for c in self._commands()])

    def test_rpm_marker_failures_return_3(self):
        self.assertEqual(self._install("azurelinux3.0", cfg=_RHEL_CFG, requires="bash\n"), 3)
        self.assertIn("could not derive", self.error.call_args_list[0].args[1])
        self.assertNotIn("rpmbuild", [c[0] for c in self._commands()])
        self.assertEqual(self._install("azurelinux3.0", cfg=_RHEL_CFG, rpm_built=False), 3)
        self.assertIn("failed to build", self.error.call_args_list[0].args[1])

    def test_dkms_check_failure_returns_3(self):
        self.assertEqual(self._install(dkms_ok=False), 3)
        self.record.assert_not_called()
        self.assertNotIn(["systemctl", "daemon-reload"], self._commands())

    def test_ofed_info_failure_returns_3(self):
        for ofed in ((1, ""), (0, "garbage\n")):
            with self.subTest(ofed=ofed):
                self.assertEqual(self._install(ofed=ofed), 3)
                self.record.assert_called_once_with("DOCA", "3.2.3")
                self.assertIn("OFED version", self.ofed_error.call_args.args[1])
                self.assertFalse(self.dropin.exists())

    def test_unwritable_versions_file_returns_3(self):
        self.assertEqual(self._install(write_error=PermissionError("denied")), 3)
        self.assertIn("denied", self.config_error.call_args.args[1])     # record_version

    # ---- logging --------------------------------------------------------------
    def test_key_log_lines(self):
        self.assertEqual(self._install("ubuntu22.04"), 0)
        start = self.info.call_args_list[0]
        self.assertEqual(start.args[1],
                         "Installing DOCA-OFED 3.2.3 on ubuntu22.04 (ubuntu branch)")
        for key, value in (("version", "3.2.3"), ("source", "public"), ("branch", "ubuntu"),
                           ("url", _UBUNTU_CFG["url"]), ("sku", "A100")):
            self.assertEqual(start.kwargs[key], value)
        messages = self._messages(self.info)
        conf = str(self.dkms / "mlnx-ofed-kernel.conf")
        self.assertEqual(self._messages(self.file_info)[f"Wrote {conf} (created)"],
                         {"path": conf, "action": "created"})
        patch = self.dkms / "mlnx-ofed-kernel" / "patches" / "xdp-fragments.patch"
        self.assertIn(f"Copied {_PATCHES / 'mlnx-ofed-kernel-xdp-fragments.patch'} -> "
                      f"{patch} (created)", self._messages(self.file_info))
        marker = [k for k in messages if k.startswith("Building the equivs marker")][0]
        self.assertEqual(messages[marker]["sharp_provided"], True)
        recorded = self._messages(self.config_info)
        self.assertIn("Recorded DOCA 3.2.3", recorded)
        self.assertIn("Recorded OFED 25.10-1.7.1", recorded)
        self.assertEqual(self.info.call_args_list[-1].args[1], "DOCA-OFED 3.2.3 installed")


class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "install-doca"]
        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, install_doca.install)
        self.assertEqual(steps[0].script, "install_doca.sh")


if __name__ == "__main__":
    unittest.main()
