"""Unit tests for components/python/install_rocm.py.

  * Pure helpers (repo texts, package lists) are tested directly, and the
    file texts, package lists and key command lines are compared with the bash
    script itself (components/install_rocm.sh), so an upstream change shows up
    here.
  * install() runs with every external effect mocked: config lookup, download,
    exec_program (wget/gpg/udevadm/systemctl... through utils.process, and the
    apt-get/dnf installs in utils.package_installer, recorded in one ordered
    list), run_capture
    (dpkg-query), the DKMS check, the kernel version and the versions file.
    The files it writes go to a temporary directory, by pointing the module's
    path constants there with mock.patch.object. Nothing touches /etc, /opt,
    /tmp outside the test's own directory, or the network.

Run from the repo root:
    python3 -m unittest tests.python.test_install_rocm -v
"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_rocm
from utils import component_config, download, files, package_installer, process
from utils.build_config import BuildConfig, build_plan
from utils.download import ChecksumError

_REPO = Path(__file__).resolve().parents[2]
_BASH = _REPO / "components" / "install_rocm.sh"

_KERNEL = "6.8.0-1015-azure"
_DEB_URL = ("https://repo.radeon.com/amdgpu-install/6.4.4/ubuntu/noble/"
            "amdgpu-install_6.4.60404-1_all.deb")
_UBUNTU_CFG = {"version": "6.4.4", "url": _DEB_URL, "sha256": "abc123"}
_AZL_CFG = {"version": "7.0.3"}      # versions.json has no url for Azure Linux
_U2604_CFG = {"version": "10.0"}     # nor for Ubuntu 26.04
_AMDGPU_CFG = {"version": "31.50"}
_AMDGPU_DPKG = "1:6.16.6.31050000-2238411.26.04"
_RVS_DPKG = "1.3.0.100000-1"
_PATCH_BYTES = b"--- a/amd/amdgpu/amdgpu_device.c\n+++ b/amd/amdgpu/amdgpu_device.c\n"

_ROCM_PACKAGES = ["rocm-utils", "rocm-developer-tools", "rocm-openmp-sdk",
                  "rocm-opencl-sdk", "rocm-ml-sdk", "migraphx", "migraphx-dev",
                  "rpp", "rpp-dev"]

# What install_packages() runs (bash's `apt install -y` / `apt-get install -y`
# and `dnf -y install` / `dnf install -y`, normalised).
_APT = ["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0"]
_DNF = ["dnf", "install", "-y"]

_TAIL_UBUNTU = [
    ["udevadm", "control", "--reload-rules"],
    ["sudo", "udevadm", "trigger"],
    ["update-initramfs", "-c", "-k", _KERNEL],
    ["systemctl", "start", "rocmstartup"],
    ["systemctl", "enable", "rocmstartup"],
]

_UBUNTU_COMMANDS = [
    [*_APT, "./amdgpu-install_6.4.60404-1_all.deb"],
    ["amdgpu-install", "-y", "--usecase=graphics"],
    [*_APT, *_ROCM_PACKAGES],
    [*_APT, "rocm-bandwidth-test"],
] + _TAIL_UBUNTU

# Ubuntu 22.04 also installs the GCC 12 C++ headers, before rocm-bandwidth-test.
_UBUNTU2204_COMMANDS = (_UBUNTU_COMMANDS[:3]
                        + [[*_APT, "libstdc++-12-dev"]]
                        + _UBUNTU_COMMANDS[3:])

_AZL_COMMANDS = [
    [*_DNF, "azurelinux-repos-amd"],
    [*_DNF, f"kernel-drivers-gpu-{_KERNEL}"],
    [*_DNF, "amdgpu", "amdgpu-firmware", "amdgpu-headers"],
    ["dnf", "makecache", "--refresh"],      # bash: dnf repolist --refresh
    [*_DNF, "rocm-dev", "rocm-validation-suite", "rocm-bandwidth-test"],
    [*_DNF, "rocm-smi-lib", "rocm-core", "rocm-device-libs",
     "rocm-llvm", "rocm-validation-suite"],
    [*_DNF, "rocm-bandwidth-test"],
    ["udevadm", "control", "--reload-rules"],
    ["sudo", "udevadm", "trigger"],
    ["systemctl", "start", "rocmstartup"],
    ["systemctl", "enable", "rocmstartup"],
]

# Exactly what the three `echo -e ... | tee [-a]` lines produce (checked
# against real bash when the port was written).
_EXPECTED_UNIT = ("[Unit]\n\nDescription=Runs /etc/init.d/initamdgpu.sh\n\n\n"
                  "[Service]\n\nExecStart=/etc/init.d/initamdgpu.sh\n\n\n"
                  "[Install]\n\nWantedBy=multi-user.target\n")


def _bash_heredoc(start: str) -> str:
    """The body of the heredoc that starts on the bash line `start` (stripped).

    Bash writes the body lines plus a newline after the last one.
    """
    lines = _BASH.read_text(encoding="utf-8").split("\n")
    first = [line.strip() for line in lines].index(start) + 1
    last = lines.index("EOF", first)
    return "\n".join(lines[first:last]) + "\n"


class PureHelperTests(unittest.TestCase):
    def test_repo_text_expands_the_version(self):
        self.assertEqual(install_rocm.amd_rocm_repo_text("7.0.3"), (
            "[amd_rocm]\n"
            'name="AMD ROCM packages repo for Azure Linux 3.0"\n'
            "baseurl=https://repo.radeon.com/rocm/azurelinux3/7.0.3/main/\n"
            "enabled=1\nrepo_gpgcheck=0\ngpgcheck=0\nsslverify=0\n"))

    def test_package_lists_use_the_kernel(self):
        self.assertEqual(install_rocm.azure_linux_driver_packages(_KERNEL), [
            ["azurelinux-repos-amd"], [f"kernel-drivers-gpu-{_KERNEL}"],
            ["amdgpu", "amdgpu-firmware", "amdgpu-headers"]])
        self.assertEqual(install_rocm.ubuntu2604_prerequisites(_KERNEL),
                         ["ca-certificates", "wget", "gnupg", "dkms",
                          f"linux-headers-{_KERNEL}"])

    def test_service_unit_text(self):
        self.assertEqual(install_rocm._SERVICE_UNIT, _EXPECTED_UNIT)

    def test_init_script_is_literal(self):
        # Quoted heredoc: shell syntax kept, not expanded.
        text = install_rocm._INIT_SCRIPT_TEXT
        self.assertTrue(text.startswith("#!/bin/sh\nat_count=0\n"))
        self.assertIn("at_count=$(($at_count + 1))", text)
        self.assertIn("sudo modprobe amdgpu ip_block_mask=0x7f", text)
        self.assertTrue(text.endswith("done\n\nexit 0\n"))

    def test_ubuntu2604_helpers(self):
        url = install_rocm.amdgpu_repo_url("31.50")
        self.assertEqual(url, "https://repo.radeon.com/amdgpu/31.50/ubuntu")
        self.assertEqual(install_rocm.amdgpu_list_text(url),
                         "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdgpu.gpg] "
                         "https://repo.radeon.com/amdgpu/31.50/ubuntu resolute main\n")
        self.assertEqual(install_rocm.ubuntu2604_package("10.0"), "amdrocm-core-sdk10.0")


class BashTextTests(unittest.TestCase):
    """The port's file texts and key command lines, against the new bash."""

    def test_heredocs_match_bash(self):
        self.assertEqual(install_rocm._AMDGPU_DKMS_CONF,
                         _bash_heredoc("cat > /etc/dkms/amdgpu.conf <<'EOF'"))
        self.assertEqual(install_rocm._AMDROCM_SOURCES, _bash_heredoc(
            "cat > /etc/apt/sources.list.d/amdrocm-stable.sources <<EOF"))
        self.assertEqual(install_rocm._RVS_LIST, _bash_heredoc(
            "cat > /etc/apt/sources.list.d/rvs.list <<EOF"))
        amdgpu_list = _bash_heredoc("cat > /etc/apt/sources.list.d/amdgpu.list <<EOF")
        self.assertEqual(install_rocm.amdgpu_list_text("URL"),
                         amdgpu_list.replace("${driver_url}", "URL"))
        self.assertEqual(install_rocm._UDEV_RULES, _bash_heredoc(
            "cat <<'EOF' > /etc/udev/rules.d/99-amdgpu-permissive.rules"))
        self.assertEqual(install_rocm._INIT_SCRIPT_TEXT,
                         _bash_heredoc("cat <<'EOF' > /tmp/tempinit.sh"))

    def test_paths_urls_and_packages_match_bash(self):
        bash = _BASH.read_text(encoding="utf-8")
        for value in (install_rocm._DKMS_PATCH_SOURCE, install_rocm._DKMS_PATCH_DIR,
                      install_rocm._DKMS_PATCH_NAME, install_rocm._DKMS_CONF_FILE,
                      install_rocm._KEYRINGS_DIR, install_rocm._AMDROCM_KEY_URL,
                      install_rocm._AMDGPU_KEY_URL, install_rocm._AMDROCM_KEY_DOWNLOAD,
                      install_rocm._AMDGPU_KEY_DOWNLOAD, install_rocm._AMDROCM_KEYRING,
                      install_rocm._AMDGPU_KEYRING, install_rocm._AMDROCM_SOURCES_FILE,
                      install_rocm._AMDGPU_LIST_FILE, install_rocm._RVS_LIST_FILE,
                      install_rocm._ROCM_INFO_VERSION_FILE, install_rocm._LD_SO_CONF_FILE,
                      install_rocm._RVS_PACKAGE):
            with self.subTest(value=value):
                self.assertIn(value, bash)
        for line in (
                'apt install -y ca-certificates wget gnupg dkms "linux-headers-$(uname -r)"',
                "install -d -m 0755 /etc/apt/keyrings",
                "gpg --batch --yes --dearmor -o /etc/apt/keyrings/amdrocm.gpg "
                "/tmp/amdrocm-packages.key",
                "chmod 0644 /etc/apt/keyrings/amdrocm.gpg /etc/apt/keyrings/amdgpu.gpg",
                "apt install -y amdgpu-dkms amdgpu-dkms-firmware",
                "check_dkms_status amdgpu",
                'apt install -y "$rocm_package" amdrocm10-rvs',
                "echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf",
                "amdgpu-install -y --usecase=graphics\n",
                "apt-get install -y libstdc++-12-dev"):
            with self.subTest(line=line):
                self.assertIn(line, bash)
        block = bash.split("apt-get install -y \\\n", 1)[1].split("\n    # else", 1)[0]
        self.assertEqual(block.replace("\\", " ").split(), install_rocm._ROCM_PACKAGES)

    def test_azure_linux_package_lists_match_bash(self):
        # One list per bash command, in bash's order.
        bash = _BASH.read_text(encoding="utf-8")
        lines = ["dnf install -y azurelinux-repos-amd",
                 "dnf -y install kernel-drivers-gpu-$(uname -r)",
                 "dnf -y install amdgpu amdgpu-firmware amdgpu-headers"]
        lists = install_rocm.azure_linux_driver_packages("$(uname -r)")
        lists += install_rocm._AZURE_LINUX_ROCM_PACKAGES
        lines += ["dnf install -y " + " ".join(p)
                  for p in install_rocm._AZURE_LINUX_ROCM_PACKAGES]
        for line, packages in zip(lines, lists):
            with self.subTest(line=line):
                self.assertIn(f"    {line}\n", bash)
                self.assertEqual(line.split(maxsplit=3)[3], " ".join(packages))
        positions = [bash.index(f"    {line}\n") for line in lines]
        self.assertEqual(positions, sorted(positions))

    def test_repo_patch_file_exists(self):
        self.assertTrue((_REPO / "components" / install_rocm._DKMS_PATCH_SOURCE).is_file())


class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.root = root
        self.paths = {
            "_WORK_DIR": str(root / "work"),
            "_AMD_ROCM_REPO_FILE": str(root / "amd_rocm.repo"),
            "_UDEV_RULES_FILE": str(root / "99-amdgpu-permissive.rules"),
            "_BLACKLIST_FILE": str(root / "blacklist.conf"),
            "_INIT_SCRIPT": str(root / "initamdgpu.sh"),
            "_SERVICE_FILE": str(root / "rocmstartup.service"),
            "_DKMS_PATCH_DIR": str(root / "dkms" / "amdgpu" / "patches"),
            "_DKMS_CONF_FILE": str(root / "amdgpu.conf"),
            "_KEYRINGS_DIR": str(root / "keyrings"),
            "_AMDROCM_KEY_DOWNLOAD": str(root / "amdrocm-packages.key"),
            "_AMDGPU_KEY_DOWNLOAD": str(root / "amdgpu-packages.key"),
            "_AMDROCM_KEYRING": str(root / "keyrings" / "amdrocm.gpg"),
            "_AMDGPU_KEYRING": str(root / "keyrings" / "amdgpu.gpg"),
            "_AMDROCM_SOURCES_FILE": str(root / "amdrocm-stable.sources"),
            "_AMDGPU_LIST_FILE": str(root / "amdgpu.list"),
            "_RVS_LIST_FILE": str(root / "rvs.list"),
            "_ROCM_INFO_VERSION_FILE": str(root / "info-version"),
            "_LD_SO_CONF_FILE": str(root / "rocm.conf"),
        }
        for name, value in self.paths.items():
            patcher = mock.patch.object(install_rocm, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # COMPONENT_DIR with the DKMS patch in it.
        self.component_dir = root / "components"
        patch = self.component_dir / install_rocm._DKMS_PATCH_SOURCE
        patch.parent.mkdir(parents=True)
        patch.write_bytes(_PATCH_BYTES)
        # What the ROCm 10 SDK writes.
        Path(self.paths["_ROCM_INFO_VERSION_FILE"]).write_text("10.0.0\n", encoding="utf-8")

    def _u2604_commands(self):
        p = self.paths
        return [
            [*_APT, "ca-certificates", "wget", "gnupg", "dkms",
             f"linux-headers-{_KERNEL}"],
            ["install", "-d", "-m", "0755", p["_KEYRINGS_DIR"]],
            ["wget", "-O", p["_AMDROCM_KEY_DOWNLOAD"],
             "https://stable.repo.amd.com/rocm/gpg/packages.gpg"],
            ["gpg", "--batch", "--yes", "--dearmor", "-o", p["_AMDROCM_KEYRING"],
             p["_AMDROCM_KEY_DOWNLOAD"]],
            ["wget", "-O", p["_AMDGPU_KEY_DOWNLOAD"], "https://repo.radeon.com/rocm/rocm.gpg.key"],
            ["gpg", "--batch", "--yes", "--dearmor", "-o", p["_AMDGPU_KEYRING"],
             p["_AMDGPU_KEY_DOWNLOAD"]],
            ["chmod", "0644", p["_AMDROCM_KEYRING"], p["_AMDGPU_KEYRING"]],
            ["apt-get", "update"],                 # bash: apt update
            [*_APT, "amdgpu-dkms", "amdgpu-dkms-firmware"],
            [*_APT, "amdrocm-core-sdk10.0", "amdrocm10-rvs"],
            ["ldconfig"],
        ] + _TAIL_UBUNTU

    def _download(self, url, sha256, dest_dir="."):
        # Like the real helper: dest_dir is created and the file lands in it
        # under the URL's name.
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        path = Path(dest_dir) / Path(url).name
        path.write_bytes(b"deb")
        self.downloaded = path
        return path

    def _install(self, distro, cfg, *, fail_on=None, rc=1, kernel=_KERNEL,
                 write_error=None, amdgpu_cfg=_AMDGPU_CFG, dkms_ok=True,
                 dpkg=None, env_extra=None, watch=None):
        """Run install() with mocks; fail_on = a command (list) that returns rc.

        watch(command) is called before each command runs, to look at the
        files at that point.
        """
        def fake_exec(command, op, *, cwd=None, env=None):
            if watch:
                watch(command)
            if command == fail_on:
                return rc
            if command[:2] == ["wget", "-O"]:
                Path(command[2]).write_bytes(b"key")      # what wget -O would do
            return 0

        def fake_config(component, env):
            return {"rocm": cfg, "amdgpu": amdgpu_cfg}[component]

        versions = {"amdgpu-dkms": (0, _AMDGPU_DPKG), "amdrocm10-rvs": (0, _RVS_DPKG)}
        versions.update(dpkg or {})

        def fake_capture(command, op, *, cwd=None, env=None, failure_is_error=True):
            return versions[command[-1]]

        env = {"DISTRIBUTION": distro, "COMPONENT_DIR": str(self.component_dir)}
        env.update(env_extra or {})
        # One mock for exec_program where commands run (utils.process for
        # run/run_all/systemctl, utils.package_installer for installs), so
        # call_args_list is the ordered record of every command. The toolkit
        # helpers log their own results, so their log calls share the
        # module's mocks too.
        with mock.patch.object(component_config, "config_for",
                               side_effect=fake_config) as config, \
             mock.patch.object(download, "download_and_verify",
                               side_effect=self._download) as fetch, \
             mock.patch.object(process, "exec_program", side_effect=fake_exec) as run, \
             mock.patch.object(package_installer, "exec_program", new=run), \
             mock.patch.object(package_installer, "run_capture",
                               side_effect=fake_capture) as capture, \
             mock.patch.object(install_rocm, "check_dkms_status",
                               return_value=dkms_ok) as dkms, \
             mock.patch.object(install_rocm.platform, "release", return_value=kernel), \
             mock.patch.object(component_config, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(install_rocm, "log_info") as info, \
             mock.patch.object(install_rocm, "log_warn") as warn, \
             mock.patch.object(install_rocm, "log_error") as error, \
             mock.patch.object(package_installer, "log_info"), \
             mock.patch.object(package_installer, "log_error", new=error), \
             mock.patch.object(files, "log_info", new=info), \
             mock.patch.object(component_config, "log_info", new=info), \
             mock.patch.object(files, "log_error", new=error), \
             mock.patch.object(download, "log_error", new=error), \
             mock.patch.object(component_config, "log_error", new=error):
            result = install_rocm.install(env)
        self.run_mock, self.download, self.record = run, fetch, record
        self.config, self.capture, self.dkms = config, capture, dkms
        self.info, self.warn, self.error = info, warn, error
        return result

    def _commands(self):
        return [c.args[0] for c in self.run_mock.call_args_list]

    def _read(self, key):
        return Path(self.paths[key]).read_text(encoding="utf-8")

    def _info_messages(self):
        return [c.args[1] for c in self.info.call_args_list]

    def _info_by_message(self):
        return {c.args[1]: c.kwargs for c in self.info.call_args_list}

    def _warns_about(self, text):
        """The log_warn calls (of the last install) whose message contains text."""
        return [c for c in self.warn.call_args_list if text in c.args[1]]

    def _patch_target(self):
        return Path(self.paths["_DKMS_PATCH_DIR"], "pci-resize-resource-exclude-bars.patch")

    def _patch_source(self):
        return os.path.join(str(self.component_dir), install_rocm._DKMS_PATCH_SOURCE)

    # ---- success paths ------------------------------------------------------
    def test_ubuntu_success(self):
        self.assertEqual(self._install("ubuntu24.04", _UBUNTU_CFG), 0)
        self.assertEqual(self._commands(), _UBUNTU_COMMANDS)
        self.download.assert_called_once_with(_DEB_URL, "abc123",
                                              dest_dir=self.paths["_WORK_DIR"])
        # apt install of ./<deb> runs inside the work dir.
        self.assertEqual(self.run_mock.call_args_list[0].kwargs["cwd"],
                         self.paths["_WORK_DIR"])
        self.assertFalse(os.path.exists(self.paths["_WORK_DIR"]))   # deb + dir removed
        self.record.assert_called_once_with("ROCM", "6.4.4")

        self.assertEqual(self._read("_UDEV_RULES_FILE"),
                         'KERNEL=="kfd", MODE="0666"\n'
                         'SUBSYSTEM=="drm", KERNEL=="renderD*", MODE="0666"\n')
        self.assertEqual(self._read("_BLACKLIST_FILE"), "blacklist amdgpu\n")
        self.assertEqual(self._read("_INIT_SCRIPT"), install_rocm._INIT_SCRIPT_TEXT)
        self.assertEqual(self._read("_SERVICE_FILE"), _EXPECTED_UNIT)
        self.assertTrue(os.stat(self.paths["_INIT_SCRIPT"]).st_mode & 0o111)
        self.assertEqual(stat.S_IMODE(os.stat(self.paths["_SERVICE_FILE"]).st_mode), 0o644)
        self.assertFalse(os.path.exists(self.paths["_AMD_ROCM_REPO_FILE"]))
        # DKMS patch + config; none of the 26.04 apt setup.
        self.assertEqual(self._patch_target().read_bytes(), _PATCH_BYTES)
        self.assertEqual(self._read("_DKMS_CONF_FILE"), install_rocm._AMDGPU_DKMS_CONF)
        self.assertFalse(os.path.exists(self.paths["_AMDGPU_LIST_FILE"]))
        self.assertFalse(os.path.exists(self.paths["_LD_SO_CONF_FILE"]))
        self.capture.assert_not_called()
        self.dkms.assert_not_called()
        self.config.assert_called_once()           # "rocm" only
        self.error.assert_not_called()

    def test_ubuntu2204_adds_libstdcxx_12(self):
        self.assertEqual(self._install("ubuntu22.04", _UBUNTU_CFG), 0)
        self.assertEqual(self._commands(), _UBUNTU2204_COMMANDS)
        self.assertIn("ubuntu22.04: also installing libstdc++-12-dev for hipcc",
                      self._info_messages())

    def test_ubuntu_dkms_patch_is_set_up_before_amdgpu_install(self):
        seen = {}

        def watch(command):
            if command[0] == "amdgpu-install":
                seen["conf"] = os.path.exists(self.paths["_DKMS_CONF_FILE"])
            elif command[0] == "apt-get" and command[-1].startswith("./"):
                seen["conf_at_deb"] = os.path.exists(self.paths["_DKMS_CONF_FILE"])
        self.assertEqual(self._install("ubuntu24.04", _UBUNTU_CFG, watch=watch), 0)
        self.assertEqual(seen, {"conf_at_deb": False, "conf": True})

    def test_real_patch_file_is_copied_byte_for_byte(self):
        real_components = str(_REPO / "components")
        self.assertEqual(self._install("ubuntu24.04", _UBUNTU_CFG,
                                       env_extra={"COMPONENT_DIR": real_components}), 0)
        self.assertEqual(self._patch_target().read_bytes(),
                         (_REPO / "components" / install_rocm._DKMS_PATCH_SOURCE).read_bytes())

    def test_ubuntu2604_success(self):
        self.assertEqual(self._install("ubuntu26.04", _U2604_CFG), 0)
        self.assertEqual(self._commands(), self._u2604_commands())
        self.download.assert_not_called()
        self.assertEqual([c.args[0] for c in self.config.call_args_list], ["rocm", "amdgpu"])
        self.dkms.assert_called_once_with(["amdgpu"], "install-rocm", mock.ANY,
                                          kernel=_KERNEL)
        self.assertEqual([c.args[0] for c in self.capture.call_args_list], [
            ["dpkg-query", "-W", "-f", "${Version}", "amdgpu-dkms"],
            ["dpkg-query", "-W", "-f", "${Version}", "amdrocm10-rvs"]])
        # AMDGPU and RVS in the branch, then ROCM (from .info/version) later.
        self.assertEqual(self.record.call_args_list, [
            mock.call("AMDGPU", _AMDGPU_DPKG), mock.call("RVS", _RVS_DPKG),
            mock.call("ROCM", "10.0.0")])
        # Files.
        self.assertEqual(self._read("_AMDROCM_SOURCES_FILE"), install_rocm._AMDROCM_SOURCES)
        self.assertEqual(self._read("_AMDGPU_LIST_FILE"),
                         "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdgpu.gpg] "
                         "https://repo.radeon.com/amdgpu/31.50/ubuntu resolute main\n")
        self.assertEqual(self._read("_RVS_LIST_FILE"), install_rocm._RVS_LIST)
        self.assertEqual(self._read("_LD_SO_CONF_FILE"), "/opt/rocm/lib\n")
        self.assertEqual(self._patch_target().read_bytes(), _PATCH_BYTES)
        self.assertEqual(self._read("_DKMS_CONF_FILE"), install_rocm._AMDGPU_DKMS_CONF)
        # rm -f of the downloaded keys.
        self.assertFalse(os.path.exists(self.paths["_AMDROCM_KEY_DOWNLOAD"]))
        self.assertFalse(os.path.exists(self.paths["_AMDGPU_KEY_DOWNLOAD"]))
        # Shared tail: blacklist (26.04 is *ubuntu*), udev, service.
        self.assertEqual(self._read("_BLACKLIST_FILE"), "blacklist amdgpu\n")
        self.assertEqual(self._read("_SERVICE_FILE"), _EXPECTED_UNIT)
        self.assertFalse(os.path.exists(self.paths["_WORK_DIR"]))
        self.error.assert_not_called()

    def test_ubuntu2604_dkms_patch_before_amdgpu_dkms_install(self):
        order = []

        def watch(command):
            if command == [*_APT, "amdgpu-dkms", "amdgpu-dkms-firmware"]:
                order.append(("dkms-install", os.path.exists(self.paths["_DKMS_CONF_FILE"])))
            if command == ["apt-get", "update"]:
                order.append(("apt-update", os.path.exists(self.paths["_AMDGPU_LIST_FILE"]),
                              os.path.exists(self.paths["_DKMS_CONF_FILE"])))
        self.assertEqual(self._install("ubuntu26.04", _U2604_CFG, watch=watch), 0)
        # Sources written before apt update; DKMS config after it, before amdgpu-dkms.
        self.assertEqual(order, [("apt-update", True, False), ("dkms-install", True)])

    def test_ubuntu2604_logs(self):
        self._install("ubuntu26.04", _U2604_CFG)
        start = self.info.call_args_list[0]
        self.assertIn("Installing ROCm 10.0 on ubuntu26.04", start.args[1])
        self.assertEqual(start.kwargs["amdgpu_version"], "31.50")
        by_message = self._info_by_message()
        branch = by_message["Ubuntu 26.04: installing ROCm from AMD's apt repos"]
        self.assertEqual((branch["rocm_package"], branch["amdgpu_repo"]),
                         ("amdrocm-core-sdk10.0", "https://repo.radeon.com/amdgpu/31.50/ubuntu"))
        installed = [kw for msg, kw in by_message.items()
                     if msg.startswith("Installed ROCm SDK reports version 10.0.0")]
        self.assertEqual((installed[0]["installed_version"], installed[0]["config_version"]),
                         ("10.0.0", "10.0"))
        self.assertIn(f"Recorded AMDGPU {_AMDGPU_DPKG}", by_message)
        self.assertIn(f"Recorded RVS {_RVS_DPKG}", by_message)
        self.assertIn("Recorded ROCM 10.0.0", by_message)
        self.assertEqual(by_message[f"Wrote {self.paths['_AMDGPU_LIST_FILE']} (created)"]
                         ["action"], "created")
        self.assertIn(f"Copied {self._patch_source()} -> {self._patch_target()} (created)",
                      by_message)
        self.assertEqual(self._info_messages()[-1], "ROCm 10.0.0 installed and configured")

    def test_ubuntu2604_rerun_replaces_files(self):
        self.assertEqual(self._install("ubuntu26.04", _U2604_CFG), 0)
        self.assertEqual(self._install("ubuntu26.04", _U2604_CFG), 0)
        self.assertEqual(self._read("_LD_SO_CONF_FILE"), "/opt/rocm/lib\n")
        self.assertEqual(self._read("_DKMS_CONF_FILE"), install_rocm._AMDGPU_DKMS_CONF)
        by_message = self._info_by_message()
        self.assertEqual(by_message[f"Wrote {self.paths['_RVS_LIST_FILE']} (replaced)"]
                         ["action"], "replaced")
        self.assertIn(f"Copied {self._patch_source()} -> {self._patch_target()} (replaced)",
                      by_message)

    def test_azure_linux_success(self):
        self.assertEqual(self._install("azurelinux3.0", _AZL_CFG), 0)
        self.assertEqual(self._commands(), _AZL_COMMANDS)
        self.download.assert_not_called()
        self.assertEqual(self._read("_AMD_ROCM_REPO_FILE"),
                         install_rocm.amd_rocm_repo_text("7.0.3"))
        # No blacklist / initramfs / DKMS patch on Azure Linux.
        self.assertFalse(os.path.exists(self.paths["_BLACKLIST_FILE"]))
        self.assertFalse(os.path.exists(self.paths["_DKMS_CONF_FILE"]))
        self.assertEqual(self._read("_SERVICE_FILE"), _EXPECTED_UNIT)
        self.record.assert_called_once_with("ROCM", "7.0.3")

    def test_azure_linux_does_not_need_component_dir(self):
        self.assertEqual(self._install("azurelinux3.0", _AZL_CFG,
                                       env_extra={"COMPONENT_DIR": ""}), 0)

    def test_appends_and_overwrites_like_bash(self):
        for key in ("_AMD_ROCM_REPO_FILE", "_UDEV_RULES_FILE", "_SERVICE_FILE"):
            Path(self.paths[key]).write_text("old\n", encoding="utf-8")
        self.assertEqual(self._install("azurelinux3.0", _AZL_CFG), 0)
        # >> appends to the repo file; > replaces the others.
        self.assertEqual(self._read("_AMD_ROCM_REPO_FILE"),
                         "old\n" + install_rocm.amd_rocm_repo_text("7.0.3"))
        self.assertFalse(self._read("_UDEV_RULES_FILE").startswith("old"))
        self.assertEqual(self._read("_SERVICE_FILE"), _EXPECTED_UNIT)
        # "old" is not an [amd_rocm] section, so no rerun warning.
        self.assertEqual(self._warns_about("already has"), [])
        by_message = self._info_by_message()
        self.assertEqual(by_message[f"Wrote {self.paths['_UDEV_RULES_FILE']} (replaced)"]
                         ["action"], "replaced")
        self.assertEqual(by_message[f"Wrote {self.paths['_AMD_ROCM_REPO_FILE']} (appended)"]
                         ["action"], "appended")
        self.assertEqual(by_message[f"Wrote {self.paths['_INIT_SCRIPT']} (created)"]["action"],
                         "created")

        Path(self.paths["_BLACKLIST_FILE"]).write_text("blacklist foo\n", encoding="utf-8")
        Path(self.paths["_DKMS_CONF_FILE"]).write_text("old\n", encoding="utf-8")
        self.assertEqual(self._install("ubuntu24.04", _UBUNTU_CFG), 0)
        self.assertEqual(self._read("_BLACKLIST_FILE"), "blacklist foo\nblacklist amdgpu\n")
        self.assertEqual(self._read("_DKMS_CONF_FILE"), install_rocm._AMDGPU_DKMS_CONF)
        self.assertEqual(self._warns_about("already has"), [])

    def test_rerun_appends_warn_when_the_text_is_already_there(self):
        self.assertEqual(self._install("azurelinux3.0", _AZL_CFG), 0)
        self.assertEqual(self._install("azurelinux3.0", _AZL_CFG), 0)
        self.assertEqual(self._read("_AMD_ROCM_REPO_FILE"),
                         install_rocm.amd_rocm_repo_text("7.0.3") * 2)
        self.assertEqual(len(self._warns_about("already has an [amd_rocm]")), 1)

        self.assertEqual(self._install("ubuntu24.04", _UBUNTU_CFG), 0)
        self.assertEqual(self._warns_about("already has"), [])
        self.assertEqual(self._install("ubuntu24.04", _UBUNTU_CFG), 0)
        self.assertEqual(self._read("_BLACKLIST_FILE"), "blacklist amdgpu\n" * 2)
        self.assertEqual(len(self._warns_about("already has 'blacklist amdgpu'")), 1)

    def test_warns_before_starting_rocmstartup_on_every_distro(self):
        for distro, cfg in (("ubuntu24.04", _UBUNTU_CFG), ("azurelinux3.0", _AZL_CFG),
                            ("ubuntu26.04", _U2604_CFG)):
            with self.subTest(distro=distro):
                self.assertEqual(self._install(distro, cfg), 0)
                warns = self._warns_about("Starting rocmstartup")
                self.assertEqual(len(warns), 1)
                self.assertEqual(warns[0].kwargs["service"], "rocmstartup")
                # start, then enable, at the end - as in bash.
                self.assertEqual(self._commands()[-2:], [["systemctl", "start", "rocmstartup"],
                                                         ["systemctl", "enable", "rocmstartup"]])

    def test_ubuntu_warns_that_amdgpu_is_blacklisted(self):
        self.assertEqual(self._install("ubuntu24.04", _UBUNTU_CFG), 0)
        self.assertEqual(len(self._warns_about("Blacklisting amdgpu")), 1)
        self.assertEqual(self._install("ubuntu26.04", _U2604_CFG), 0)
        self.assertEqual(len(self._warns_about("Blacklisting amdgpu")), 1)
        self.assertEqual(self._install("azurelinux3.0", _AZL_CFG), 0)
        self.assertEqual(self._warns_about("Blacklisting amdgpu"), [])

    def test_logs_why_rocm_runs_before_pmix_and_mpi(self):
        self._install("azurelinux3.0", _AZL_CFG)
        self.assertIn("before PMIx/MPI", self._info_messages()[1])

    def test_udev_reload_failure_skips_trigger_and_continues(self):
        # bash: `udevadm control --reload-rules && sudo udevadm trigger` under
        # set -e - a failure before && only skips the trigger.
        rc = self._install("ubuntu24.04", _UBUNTU_CFG,
                           fail_on=["udevadm", "control", "--reload-rules"])
        self.assertEqual(rc, 0)
        self.assertNotIn(["sudo", "udevadm", "trigger"], self._commands())
        self.assertIn(["systemctl", "enable", "rocmstartup"], self._commands())
        self.assertEqual(len(self._warns_about("udev rules reload failed")), 1)

    # ---- logging ------------------------------------------------------------
    def test_key_log_lines(self):
        self._install("ubuntu24.04", _UBUNTU_CFG)
        start = self.info.call_args_list[0]
        self.assertIn("Installing ROCm 6.4.4 on ubuntu24.04", start.args[1])
        self.assertEqual(start.kwargs["source"], _DEB_URL)
        self.assertEqual(start.kwargs["kernel"], _KERNEL)
        messages = self._info_messages()
        self.assertIn("Ubuntu: installing ROCm with amdgpu-install", messages)
        by_message = self._info_by_message()
        packages = [kw for msg, kw in by_message.items()
                    if msg.startswith("Installing ROCm without MIVisionX")]
        self.assertEqual(packages[0]["packages"], _ROCM_PACKAGES)
        self.assertIn(f"Wrote {self.paths['_UDEV_RULES_FILE']} (created)", messages)
        self.assertIn(f"Wrote {self.paths['_DKMS_CONF_FILE']} (created)", messages)
        self.assertIn(f"Wrote {self.paths['_BLACKLIST_FILE']} (created)", messages)
        self.assertIn("Recorded ROCM 6.4.4", messages)
        self.assertIn("Started and enabled rocmstartup.service", messages)
        self.assertEqual(messages[-1], "ROCm 6.4.4 installed and configured")

    # ---- failures -----------------------------------------------------------
    def test_unsupported_distro_fails_before_any_change(self):
        rc = self._install("rocky9.8", _UBUNTU_CFG)
        self.assertEqual(rc, 3)
        self.run_mock.assert_not_called()
        self.record.assert_not_called()
        self.assertFalse(os.path.exists(self.paths["_UDEV_RULES_FILE"]))
        self.assertIn("not supported", self.error.call_args.args[1])

    def test_missing_config_returns_3(self):
        for distro in ("ubuntu24.04", "ubuntu26.04", "azurelinux3.0"):
            for cfg in (None, {}, {"url": _DEB_URL}):
                with self.subTest(distro=distro, cfg=cfg):
                    self.assertEqual(self._install(distro, cfg), 3)
                    self.run_mock.assert_not_called()
                    self.error.assert_called_once()

    def test_ubuntu_missing_url_or_sha_returns_3(self):
        self.assertEqual(self._install("ubuntu24.04", {"version": "6.4.4"}), 3)
        self.download.assert_not_called()
        self.assertIn("missing url, sha256", self.error.call_args.args[1])

    def test_ubuntu2604_missing_amdgpu_version_returns_3(self):
        for amdgpu_cfg in (None, {}, {"version": ""}):
            with self.subTest(amdgpu_cfg=amdgpu_cfg):
                self.assertEqual(self._install("ubuntu26.04", _U2604_CFG,
                                               amdgpu_cfg=amdgpu_cfg), 3)
                self.run_mock.assert_not_called()
                self.assertIn("'amdgpu' config", self.error.call_args.args[1])

    def test_ubuntu_missing_patch_returns_3_before_any_change(self):
        for distro, cfg in (("ubuntu24.04", _UBUNTU_CFG), ("ubuntu26.04", _U2604_CFG)):
            for component_dir in ("", str(self.root / "nowhere")):
                with self.subTest(distro=distro, component_dir=component_dir):
                    rc = self._install(distro, cfg,
                                       env_extra={"COMPONENT_DIR": component_dir})
                    self.assertEqual(rc, 3)
                    self.run_mock.assert_not_called()
                    self.download.assert_not_called()
                    self.assertIn("DKMS patch not found", self.error.call_args.args[1])

    def test_empty_kernel_returns_3(self):
        self.assertEqual(self._install("azurelinux3.0", _AZL_CFG, kernel=""), 3)
        self.run_mock.assert_not_called()

    def test_download_failure_returns_3(self):
        with mock.patch.object(self, "_download", side_effect=ChecksumError("checksum")):
            self.assertEqual(self._install("ubuntu24.04", _UBUNTU_CFG), 3)
        self.run_mock.assert_not_called()
        self.assertIn("download of", self.error.call_args.args[1])
        self.assertFalse(os.path.exists(self.paths["_WORK_DIR"]))

    def test_each_failing_command_stops_the_step(self):
        cases = [("ubuntu24.04", _UBUNTU_CFG, _UBUNTU_COMMANDS),
                 ("ubuntu22.04", _UBUNTU_CFG, _UBUNTU2204_COMMANDS),
                 ("ubuntu26.04", _U2604_CFG, self._u2604_commands()),
                 ("azurelinux3.0", _AZL_CFG, _AZL_COMMANDS)]
        for distro, cfg, commands in cases:
            for index, command in enumerate(commands):
                if command[:2] == ["udevadm", "control"]:
                    continue        # tolerated, see the dedicated test
                with self.subTest(distro=distro, command=command):
                    self.assertEqual(self._install(distro, cfg, fail_on=command), 3)
                    self.assertEqual(self._commands(), commands[:index + 1])
                    # (The failure is logged by exec_program, mocked here.)
                    # ROCM is recorded only after the udev trigger, like bash.
                    trigger = commands.index(["sudo", "udevadm", "trigger"])
                    recorded = [c.args[0] for c in self.record.call_args_list]
                    self.assertEqual("ROCM" in recorded, index > trigger)

    def test_ubuntu2604_dkms_check_failure_returns_3(self):
        self.assertEqual(self._install("ubuntu26.04", _U2604_CFG, dkms_ok=False), 3)
        commands = self._commands()
        self.assertEqual(commands[-1], [*_APT, "amdgpu-dkms", "amdgpu-dkms-firmware"])
        self.record.assert_not_called()
        self.assertIn("amdgpu DKMS module is not installed", self.error.call_args.args[1])

    def test_ubuntu2604_missing_or_empty_info_version_returns_3(self):
        info = Path(self.paths["_ROCM_INFO_VERSION_FILE"])
        for content in (None, "", "\n"):
            with self.subTest(content=content):
                if content is None:
                    info.unlink(missing_ok=True)
                else:
                    info.write_text(content, encoding="utf-8")
                self.assertEqual(self._install("ubuntu26.04", _U2604_CFG), 3)
                self.record.assert_not_called()
                self.assertNotIn(["ldconfig"], self._commands())
                self.error.assert_called_once()

    def test_ubuntu2604_dpkg_query_failure_returns_3(self):
        cases = [({"amdgpu-dkms": (1, "")}, []),
                 ({"amdgpu-dkms": (0, "")}, []),
                 ({"amdrocm10-rvs": (1, "")}, ["AMDGPU"])]
        for dpkg, recorded in cases:
            with self.subTest(dpkg=dpkg):
                self.assertEqual(self._install("ubuntu26.04", _U2604_CFG, dpkg=dpkg), 3)
                self.assertEqual([c.args[0] for c in self.record.call_args_list], recorded)
                self.assertNotIn(["ldconfig"], self._commands())
                self.assertIn("dpkg-query", self.error.call_args.args[1])

    def test_file_write_error_returns_3(self):
        os.makedirs(self.paths["_UDEV_RULES_FILE"])      # a directory can't be opened
        self.assertEqual(self._install("ubuntu24.04", _UBUNTU_CFG), 3)
        self.assertNotIn(["udevadm", "control", "--reload-rules"], self._commands())
        self.assertIn("could not write", self.error.call_args.args[1])

    def test_dkms_conf_write_error_returns_3(self):
        os.makedirs(self.paths["_DKMS_CONF_FILE"])
        self.assertEqual(self._install("ubuntu24.04", _UBUNTU_CFG), 3)
        self.assertNotIn(["amdgpu-install", "-y", "--usecase=graphics"], self._commands())
        self.assertIn("could not write", self.error.call_args.args[1])
        self.assertFalse(os.path.exists(self.paths["_WORK_DIR"]))

    def test_dkms_patch_dir_error_returns_3(self):
        # A file where the patch directory goes: bash's `mkdir -p` fails.
        Path(self.paths["_DKMS_PATCH_DIR"]).parent.mkdir(parents=True)
        Path(self.paths["_DKMS_PATCH_DIR"]).write_text("", encoding="utf-8")
        self.assertEqual(self._install("ubuntu24.04", _UBUNTU_CFG), 3)
        self.assertNotIn(["amdgpu-install", "-y", "--usecase=graphics"], self._commands())
        self.assertIn("could not create directory", self.error.call_args.args[1])
        self.assertFalse(os.path.exists(self.paths["_WORK_DIR"]))

    def test_unwritable_versions_file_returns_3(self):
        rc = self._install("azurelinux3.0", _AZL_CFG,
                           write_error=PermissionError("denied"))
        self.assertEqual(rc, 3)
        self.assertNotIn(["systemctl", "start", "rocmstartup"], self._commands())


class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="AMD", gpu="MI300X", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "install-rocm"]
        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, install_rocm.install)
        self.assertEqual(steps[0].script, "install_rocm.sh")


if __name__ == "__main__":
    unittest.main()
