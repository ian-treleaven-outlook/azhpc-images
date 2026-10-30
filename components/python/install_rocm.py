"""install_rocm.py — Python port of components/install_rocm.sh.

What this step does
-------------------
ROCm is AMD's GPU software stack (kernel driver, runtime, compilers, math and
communication libraries). On AMD GPU images (MI200 / MI300X / MI300HF) this
step:

  1. Installs ROCm, one branch per distro (bash's if/elif, see _BRANCHES):
       * Ubuntu 26.04  ROCm 10 from AMD's apt repos: signing keys, three apt
                       source files (ROCm core, the amdgpu driver version from
                       versions.json's "amdgpu" entry, RVS), the amdgpu DKMS
                       driver (checked with check_dkms_status), then the ROCm
                       core SDK and RVS. Records AMDGPU and RVS package
                       versions and adds /opt/rocm/lib to the linker path.
       * other Ubuntu  the `amdgpu-install` .deb from versions.json, then
                       `amdgpu-install -y --usecase=graphics` plus an explicit
                       ROCm package list (Ubuntu 22.04 also libstdc++-12-dev)
                       and rocm-bandwidth-test.
       * Azure Linux   the AMD repo package, GPU kernel drivers and amdgpu
                       packages, an [amd_rocm] dnf repo for the ROCm version,
                       then the ROCm packages.
     On every Ubuntu the amdgpu DKMS patch (a backport of a pci_resize_resource()
     API change, so amdgpu builds on newer kernels) is copied to
     /etc/dkms/amdgpu/patches/ with an /etc/dkms/amdgpu.conf that applies it
     only when the driver source lacks AMD's own fix.
  2. Writes udev rules giving every user access to the GPUs, reloads udev.
  3. Records ROCM in the component-versions file.
  4. Ubuntu only: blacklists amdgpu and rebuilds the initramfs.
  5. Installs /etc/init.d/initamdgpu.sh and rocmstartup.service, which at boot
     waits (up to ~15 minutes) for all the GPUs to appear, then loads amdgpu
     with the right options for the GPU model; starts and enables the service.

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets install() as the `action=` of the "install-rocm"
Step (AMD images only); deleting `action=` runs the bash script again. The
step runs before install-pmix / install-mpis, as in distros/*/install.sh, so
the HPC-X step can rebuild UCX with ROCm support. install() returns 0 or 3.

Behaviour kept from bash
------------------------
  * Commands, arguments, order and stop-on-failure points are bash's, with
    one `set -e` quirk: in `udevadm control --reload-rules && sudo udevadm
    trigger` a failed reload only skips the trigger, and the step carries on
    (with a warning; the rules apply from the next boot anyway). `sudo` stays.
  * Appends stay appends (amd_rocm.repo, blacklist.conf): a rerun adds the
    text again, with a warning when it is already there. Every other file is
    overwritten.
  * Every Ubuntu (26.04 too) blacklists amdgpu, so only rocmstartup's script
    loads it at boot, once all the GPUs are present (logged as a warning).
  * `systemctl start rocmstartup` runs on the build VM too, though its script
    then polls up to ~15 minutes for GPUs the VM may not have (warned).
    Upstream #653 briefly waited for it in Packer, then reverted that.
  * Ubuntu 26.04 records ROCM from /opt/rocm/core/.info/version (e.g.
    "10.0.0"), not versions.json's "10.0"; both are logged. Its RVS package
    has a hard-coded "10" (amdrocm10-rvs) while the SDK name follows
    versions.json.
  * The DKMS patch and config are written just before amdgpu-dkms is
    installed (apt on 26.04, amdgpu-install elsewhere).
  * Ubuntu 22.04 and 24.04 share the amdgpu-install path; bash keeps the old
    per-release if/else (and a ROCm 7.0 TODO) commented out.

Deliberate differences from bash
--------------------------------
  * Installs go through install_packages(): `apt-get install -y -o
    Dpkg::Use-Pty=0` / `dnf install -y` (bash: `apt install -y`, `apt-get
    install -y`, `dnf -y install`).
  * Unsupported distros fail before changing anything. Bash has no else
    branch: it would install no ROCm yet still write the udev rules, the boot
    service and a ROCM version (versions.json has Rocky entries bash ignores).
  * Missing config fails up front (require_config): the ROCm version
    everywhere, url/sha256 on Ubuntu other than 26.04, the "amdgpu" version
    on 26.04 (bash's `jq -e` stops there too). On Ubuntu, a missing
    COMPONENT_DIR or DKMS patch fails up front rather than at bash's `cp`.
    An empty `uname -r` fails too.
  * Ubuntu 26.04: a missing/empty .info/version or a failing/empty
    `dpkg-query` fails the step. Bash's AMDGPU query runs inside a command
    argument, which `set -e` doesn't check, so bash would record "".
  * The amdgpu-install .deb is downloaded into a scratch directory
    (_WORK_DIR, emptied first and removed afterwards) instead of the build dir.
  * The init script and unit file are written in place instead of via
    /tmp/tempinit.sh + cp and ./rocmstartup.service + mv; the bytes are the
    same (the unit text was checked against real bash). The script then gets
    `chmod +x`; the unit gets 0644, what mv of a tee-created file has under
    umask 022.
  * write_file() creates missing parent directories; `rm -f`s go through
    remove_paths(), which never fails the build.
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
from dataclasses import dataclass
from typing import Callable

from utils.component_config import require_config, write_component_version
from utils.distro import Distro
from utils.dkms import check_dkms_status
from utils.download import fetch_verified
from utils.files import remove_paths, write_file
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import run, run_all, run_capture
from utils.services import systemctl

_OP = "install-rocm"

# ---- Paths (module-level so tests can point them at a temporary directory) --
_WORK_DIR = "/tmp/azhpc-rocm"
_AMD_ROCM_REPO_FILE = "/etc/yum.repos.d/amd_rocm.repo"
_UDEV_RULES_FILE = "/etc/udev/rules.d/99-amdgpu-permissive.rules"
_BLACKLIST_FILE = "/etc/modprobe.d/blacklist.conf"
_INIT_SCRIPT = "/etc/init.d/initamdgpu.sh"
_SERVICE_FILE = "/etc/systemd/system/rocmstartup.service"
_SERVICE_NAME = "rocmstartup"

# The amdgpu DKMS patch: its source (relative to COMPONENT_DIR), where DKMS
# looks for it, the name it gets there, and the DKMS config that selects it.
_DKMS_PATCH_SOURCE = "patches/amdgpu-pci-resize-resource-exclude-bars.patch"
_DKMS_PATCH_DIR = "/etc/dkms/amdgpu/patches"
_DKMS_PATCH_NAME = "pci-resize-resource-exclude-bars.patch"
_DKMS_CONF_FILE = "/etc/dkms/amdgpu.conf"

# Ubuntu 26.04 apt setup. The keyring paths also appear, literally, in the apt
# source texts below (as in bash), so tests that move these files don't change
# what the source files say.
_KEYRINGS_DIR = "/etc/apt/keyrings"
_AMDROCM_KEY_URL = "https://stable.repo.amd.com/rocm/gpg/packages.gpg"
_AMDGPU_KEY_URL = "https://repo.radeon.com/rocm/rocm.gpg.key"
_AMDROCM_KEY_DOWNLOAD = "/tmp/amdrocm-packages.key"
_AMDGPU_KEY_DOWNLOAD = "/tmp/amdgpu-packages.key"
_AMDROCM_KEYRING = "/etc/apt/keyrings/amdrocm.gpg"
_AMDGPU_KEYRING = "/etc/apt/keyrings/amdgpu.gpg"
_AMDROCM_SOURCES_FILE = "/etc/apt/sources.list.d/amdrocm-stable.sources"
_AMDGPU_LIST_FILE = "/etc/apt/sources.list.d/amdgpu.list"
_RVS_LIST_FILE = "/etc/apt/sources.list.d/rvs.list"
_ROCM_INFO_VERSION_FILE = "/opt/rocm/core/.info/version"
_LD_SO_CONF_FILE = "/etc/ld.so.conf.d/rocm.conf"
_ROCM_LIB_DIR = "/opt/rocm/lib"

# ---- Packages, one list per bash install command ----------------------------
_AMDGPU_DKMS_PACKAGE = "amdgpu-dkms"
_RVS_PACKAGE = "amdrocm10-rvs"          # bash hard-codes the "10"
# The kernel headers are added by ubuntu2604_prerequisites().
_UBUNTU2604_PREREQUISITES = ["ca-certificates", "wget", "gnupg", "dkms"]
_UBUNTU2604_DRIVER_PACKAGES = [_AMDGPU_DKMS_PACKAGE, "amdgpu-dkms-firmware"]

# Ubuntu 22.04 / 24.04: the `rocm` metapackage minus MIVisionX (and rocDecode,
# which only MIVisionX pulls in). Bash's reason: they pull FFmpeg (plus Qt,
# cJSON and mbedTLS on 24.04) packages with publishing-blocking CVEs whose
# Ubuntu fixes need Pro/ESM. Same order as bash.
_ROCM_PACKAGES = ["rocm-utils", "rocm-developer-tools", "rocm-openmp-sdk",
                  "rocm-opencl-sdk", "rocm-ml-sdk", "migraphx", "migraphx-dev",
                  "rpp", "rpp-dev"]
# Installed between _ROCM_PACKAGES and rocm-bandwidth-test. Ubuntu 22.04:
# rocdecode-dev used to pull this in; ROCm's clang picks the newest GCC (12),
# so hipcc needs its C++ headers.
_UBUNTU_EXTRA_PACKAGES = {"ubuntu22.04": ["libstdc++-12-dev"]}
_BANDWIDTH_TEST_PACKAGES = ["rocm-bandwidth-test"]

# Azure Linux, after the [amd_rocm] repo is added (rocm-validation-suite and
# rocm-bandwidth-test are listed twice, as in bash). The ones before it need
# the kernel version; see azure_linux_driver_packages().
_AZURE_LINUX_ROCM_PACKAGES = [
    ["rocm-dev", "rocm-validation-suite", "rocm-bandwidth-test"],
    ["rocm-smi-lib", "rocm-core", "rocm-device-libs", "rocm-llvm",
     "rocm-validation-suite"],
    ["rocm-bandwidth-test"],
]

# ---- File contents (exact bytes bash writes) --------------------------------
# Quoted heredocs (`<<'EOF'`) are written literally: no $ expansion, and a
# newline after the last line.
_UDEV_RULES = """\
KERNEL=="kfd", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="renderD*", MODE="0666"
"""

# DKMS reads this as shell when it builds amdgpu: it adds the patch unless the
# driver source already has AMD's native fix (..._ALIGN) or our backport
# (..._EXCLUDE_BARS) in all three places.
_AMDGPU_DKMS_CONF = """\
_amdgpu_source="$source_tree/$module-$module_version"
_amdgpu_pci_resize_fixed=false
# AMD's native fix uses ALIGN; EXCLUDE_BARS identifies our backport.
for _amdgpu_flag in HAVE_PCI_RESIZE_RESOURCE_ALIGN HAVE_PCI_RESIZE_RESOURCE_EXCLUDE_BARS; do
    if grep -Fq "$_amdgpu_flag" "$_amdgpu_source/amd/dkms/configure" &&
       grep -Fq "$_amdgpu_flag" "$_amdgpu_source/amd/dkms/config/config.h.in" &&
       grep -Fq "$_amdgpu_flag" "$_amdgpu_source/amd/amdgpu/amdgpu_device.c"; then
        _amdgpu_pci_resize_fixed=true
        break
    fi
done
if [[ "$_amdgpu_pci_resize_fixed" == false ]]; then
    PATCH+=("pci-resize-resource-exclude-bars.patch")
fi
"""

# Ubuntu 26.04 apt sources (unquoted heredocs, but only amdgpu.list has a
# variable in it; see amdgpu_list_text()).
_AMDROCM_SOURCES = """\
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/amdrocm.gpg
"""
_RVS_LIST = ("deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] "
             "https://stable.repo.amd.com/rocm/extras/rvs/packages/ubuntu2604/ stable main\n")

_LD_SO_CONF = f"{_ROCM_LIB_DIR}\n"

# PCI IDs: 1002:740c is MI200 (16 per node), 1002:74b5 MI300X and 1002:74bd
# MI300HF (8 per node). The $(...) are for /bin/sh at boot (quoted heredoc).
_INIT_SCRIPT_TEXT = """\
#!/bin/sh
at_count=0
while [ $at_count -le 90 ]
do
    if [ $(lspci -d 1002:74b5 | wc -l) -eq 8 -o $(lspci -d 1002:74bd | wc -l) -eq 8 -o $(lspci -d 1002:740c | wc -l) -eq 16 ]; then
       echo Required number of GPUs found
       at_count=91
       sleep 120s
       echo doing Modprobe for amdgpu
       if [ $(lspci -d 1002:740c | wc -l) -eq 16 ]; then
          sudo modprobe amdgpu
       else
          sudo modprobe -r hyperv_drm
          sudo modprobe amdgpu ip_block_mask=0x7f
       fi
    else
       sleep 10
       at_count=$(($at_count + 1))
    fi
done

exit 0
"""

# What the three `echo -e '...' | tee [-a] rocmstartup.service` lines write:
# echo -e turns each \n into a newline and adds one more at the end, hence
# three blank lines between sections. Verified byte for byte against bash.
_SERVICE_UNIT = (
    "[Unit]\n\nDescription=Runs /etc/init.d/initamdgpu.sh\n\n\n"
    "[Service]\n\nExecStart=/etc/init.d/initamdgpu.sh\n\n\n"
    "[Install]\n\nWantedBy=multi-user.target\n"
)
_SERVICE_FILE_MODE = 0o644     # a tee-created file moved with mv, umask 022

_BLACKLIST_LINE = "blacklist amdgpu\n"


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
def amd_rocm_repo_text(version: str) -> str:
    """The [amd_rocm] block bash appends to amd_rocm.repo (unquoted heredoc,
    so ${rocm_version} is expanded). The quotes around the name are text."""
    return (
        "[amd_rocm]\n"
        'name="AMD ROCM packages repo for Azure Linux 3.0"\n'
        f"baseurl=https://repo.radeon.com/rocm/azurelinux3/{version}/main/\n"
        "enabled=1\n"
        "repo_gpgcheck=0\n"
        "gpgcheck=0\n"
        "sslverify=0\n"
    )


def amdgpu_repo_url(driver_version: str) -> str:
    """bash: driver_url="https://repo.radeon.com/amdgpu/${driver_version}/ubuntu"."""
    return f"https://repo.radeon.com/amdgpu/{driver_version}/ubuntu"


def amdgpu_list_text(driver_url: str) -> str:
    """/etc/apt/sources.list.d/amdgpu.list; "resolute" is Ubuntu 26.04."""
    return (f"deb [arch=amd64 signed-by=/etc/apt/keyrings/amdgpu.gpg] "
            f"{driver_url} resolute main\n")


def ubuntu2604_package(rocm_version: str) -> str:
    """bash: rocm_package="amdrocm-core-sdk${rocm_version}"."""
    return f"amdrocm-core-sdk{rocm_version}"


def ubuntu2604_prerequisites(kernel: str) -> list[str]:
    """bash: apt install -y ca-certificates wget gnupg dkms "linux-headers-$(uname -r)"."""
    return [*_UBUNTU2604_PREREQUISITES, f"linux-headers-{kernel}"]


def azure_linux_driver_packages(kernel: str) -> list[list[str]]:
    """The Azure Linux dnf installs before the repo file, one list per command."""
    return [
        ["azurelinux-repos-amd"],
        [f"kernel-drivers-gpu-{kernel}"],
        ["amdgpu", "amdgpu-firmware", "amdgpu-headers"],
    ]


# --------------------------------------------------------------------------- #
# Small helpers.
# --------------------------------------------------------------------------- #
def _install_each(package_lists, env: dict[str, str], cwd: str | None = None) -> bool:
    """One install_packages() per list (one per bash command), stopping at the
    first failure."""
    return all(install_packages(packages, env, _OP, cwd=cwd) for packages in package_lists)


# TODO(ADO 1001): toolkit candidate: file_has_line(path, line) -> bool (for the
# "already there; appending again, as bash does" warning before `>>`).
def _has_line(path: str, line: str) -> bool:
    """True if `path` already has `line`; a missing/unreadable file is "no"."""
    try:
        with open(path, encoding="utf-8") as handle:
            return line in handle.read().splitlines()
    except OSError:
        return False


def _append(path: str, text: str, line: str, what: str) -> bool:
    """bash `>>` / `tee -a`: a rerun appends again, as bash does; we warn
    when `line` is already in the file."""
    if _has_line(path, line):
        log_warn(_OP, f"{path} already has {what}; appending it again, as bash "
                      "does on a rerun", path=path)
    return write_file(path, text, _OP, append=True)


# TODO(ADO 1001): toolkit candidate: copy_file(source, dest, op) -> bool
# (bash `mkdir -p <dir>; cp <source> <dest>`, logging created/replaced).
def _copy_file(source: str, target: str) -> None:
    """`mkdir -p` + `cp` (no -p: a new file gets the default mode). Raises
    OSError; install() catches it."""
    os.makedirs(os.path.dirname(target), exist_ok=True)
    action = "replaced" if os.path.exists(target) else "created"
    shutil.copyfile(source, target)
    log_info(_OP, f"Copied {source} to {target} ({action})",
             source=source, path=target, action=action)


# TODO(ADO 1001): toolkit candidate: installed_package_version(package, env, op)
# -> str | None (dpkg-query -W -f='${Version}' / rpm -q --queryformat).
def _dpkg_version(package: str, env: dict[str, str]) -> str | None:
    """bash `dpkg-query -W -f='${Version}' <package>`: the installed version,
    or None (logged) when the query fails or prints nothing."""
    # bash's quotes only stop ${Version} expanding; dpkg-query gets it literally.
    rc, output = run_capture(["dpkg-query", "-W", "-f=${Version}", package], _OP, env=env)
    version = output.rstrip("\n") if rc == 0 else ""
    if not version:
        log_error(_OP, f"could not read the installed {package} version from dpkg-query",
                  package=package, exit_code=rc)
        return None
    return version


def _configure_amdgpu_dkms_patches(component_dir: str) -> bool:
    """bash's configure_amdgpu_dkms_patches: the patch + /etc/dkms/amdgpu.conf."""
    _copy_file(os.path.join(component_dir, _DKMS_PATCH_SOURCE),
               os.path.join(_DKMS_PATCH_DIR, _DKMS_PATCH_NAME))
    if not write_file(_DKMS_CONF_FILE, _AMDGPU_DKMS_CONF, _OP):
        return False
    log_info(_OP, "amdgpu DKMS builds will apply the pci_resize_resource patch "
                  "unless the driver source already has the fix",
             path=_DKMS_CONF_FILE, patch=_DKMS_PATCH_NAME)
    return True


# --------------------------------------------------------------------------- #
# What install() works from, resolved and checked before any change.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Inputs:
    distro: Distro
    config: dict                    # versions.json "rocm"
    driver_version: str | None      # versions.json "amdgpu" (Ubuntu 26.04 only)
    kernel: str                     # `uname -r`
    component_dir: str

    @property
    def rocm_version(self) -> str:
        return str(self.config["version"])


# --------------------------------------------------------------------------- #
# Ubuntu 26.04: ROCm 10 from AMD's apt repos.
# --------------------------------------------------------------------------- #
# TODO(ADO 1001): toolkit candidate: add_apt_signing_key(url, keyring, op, env=)
# -> bool (wget + gpg --dearmor + chmod 0644 + rm of the download).
def _install_amd_apt_keys(env: dict[str, str]) -> bool:
    if not run_all([
        ["install", "-d", "-m", "0755", _KEYRINGS_DIR],
        ["wget", "-O", _AMDROCM_KEY_DOWNLOAD, _AMDROCM_KEY_URL],
        ["gpg", "--batch", "--yes", "--dearmor", "-o", _AMDROCM_KEYRING,
         _AMDROCM_KEY_DOWNLOAD],
        ["wget", "-O", _AMDGPU_KEY_DOWNLOAD, _AMDGPU_KEY_URL],
        ["gpg", "--batch", "--yes", "--dearmor", "-o", _AMDGPU_KEYRING,
         _AMDGPU_KEY_DOWNLOAD],
        ["chmod", "0644", _AMDROCM_KEYRING, _AMDGPU_KEYRING],
    ], _OP, env=env):
        return False
    log_info(_OP, "Installed the AMD apt signing keys",
             keyrings=[_AMDROCM_KEYRING, _AMDGPU_KEYRING],
             key_urls=[_AMDROCM_KEY_URL, _AMDGPU_KEY_URL])
    remove_paths(_AMDROCM_KEY_DOWNLOAD, _AMDGPU_KEY_DOWNLOAD, op=_OP)
    return True


def _add_amd_apt_repos(env: dict[str, str], kernel: str, driver_url: str) -> bool:
    """Prerequisites, signing keys, the three source files, `apt update`."""
    if not (install_packages(ubuntu2604_prerequisites(kernel), env, _OP)
            and _install_amd_apt_keys(env)):
        return False
    sources = [(_AMDROCM_SOURCES_FILE, _AMDROCM_SOURCES),
               (_AMDGPU_LIST_FILE, amdgpu_list_text(driver_url)),
               (_RVS_LIST_FILE, _RVS_LIST)]
    if not all(write_file(path, text, _OP) for path, text in sources):
        return False
    # TODO(ADO 1001): toolkit candidate: refresh_package_index(env, op) -> bool
    # (apt update / dnf repolist --refresh).
    return run(["apt", "update"], _OP, env=env)


def _install_amdgpu_dkms(env: dict[str, str], inputs: _Inputs) -> bool:
    if not (_configure_amdgpu_dkms_patches(inputs.component_dir)
            and install_packages(_UBUNTU2604_DRIVER_PACKAGES, env, _OP)):
        return False
    # check_dkms_status logs the status lines and, on failure, the make.log tails.
    if not check_dkms_status(["amdgpu"], _OP, env, kernel=inputs.kernel):
        log_error(_OP, "ROCm install stopped: the amdgpu DKMS module is not "
                       "installed for the running kernel", kernel=inputs.kernel)
        return False
    return True


def _read_installed_rocm_version(config_version: str) -> str | None:
    """bash: rocm_version=$(cat /opt/rocm/core/.info/version)."""
    try:
        with open(_ROCM_INFO_VERSION_FILE, encoding="utf-8") as handle:
            installed = handle.read().rstrip("\n")      # what "$(...)" strips
    except OSError as exc:
        log_error(_OP, f"could not read the installed ROCm version: {exc}",
                  path=_ROCM_INFO_VERSION_FILE)
        return None
    if not installed:
        log_error(_OP, f"{_ROCM_INFO_VERSION_FILE} is empty; no ROCm version to record",
                  path=_ROCM_INFO_VERSION_FILE)
        return None
    log_info(_OP, f"Installed ROCm SDK reports version {installed}; recording that "
                  f"instead of versions.json's {config_version}, as bash does",
             path=_ROCM_INFO_VERSION_FILE, installed_version=installed,
             config_version=config_version)
    return installed


def _record_package_versions(env: dict[str, str]) -> bool:
    """Record AMDGPU and RVS from dpkg, in bash's order."""
    for component, package in (("AMDGPU", _AMDGPU_DKMS_PACKAGE), ("RVS", _RVS_PACKAGE)):
        version = _dpkg_version(package, env)
        if version is None:
            return False
        write_component_version(component, version)
        log_info(_OP, f"Recorded {component}={version}", component=component,
                 version=version, package=package)
    return True


def _add_rocm_to_ld_path(env: dict[str, str]) -> bool:
    # The SDK ships no ld.so.conf entry, so anything linking ROCm (e.g. Open
    # MPI's accelerator component) can't find libamdhip64 without it.
    if not (write_file(_LD_SO_CONF_FILE, _LD_SO_CONF, _OP)
            and run(["ldconfig"], _OP, env=env)):
        return False
    log_info(_OP, f"Added {_ROCM_LIB_DIR} to the dynamic linker path",
             path=_LD_SO_CONF_FILE, lib_dir=_ROCM_LIB_DIR)
    return True


def _install_ubuntu2604(env: dict[str, str], inputs: _Inputs) -> str | None:
    """Returns the ROCm version to record: what the installed SDK reports."""
    driver_url = amdgpu_repo_url(inputs.driver_version)
    rocm_package = ubuntu2604_package(inputs.rocm_version)
    log_info(_OP, "Ubuntu 26.04: installing ROCm from AMD's apt repos",
             handler="ubuntu26.04", rocm_package=rocm_package,
             amdgpu_version=inputs.driver_version, amdgpu_repo=driver_url,
             rvs_package=_RVS_PACKAGE)
    if not (_add_amd_apt_repos(env, inputs.kernel, driver_url)
            and _install_amdgpu_dkms(env, inputs)
            and install_packages([rocm_package, _RVS_PACKAGE], env, _OP)):
        return None
    installed = _read_installed_rocm_version(inputs.rocm_version)
    if installed is None or not (_record_package_versions(env)
                                 and _add_rocm_to_ld_path(env)):
        return None
    return installed


# --------------------------------------------------------------------------- #
# Ubuntu 22.04 / 24.04: amdgpu-install.
# --------------------------------------------------------------------------- #
def _install_rocm_without_mivisionx(env: dict[str, str], distro: Distro) -> bool:
    # Bash keeps its old `amdgpu-install --usecase=graphics,rocm` branches
    # commented out, with TODOs to go back to the full `rocm` usecase once
    # ROCm 7.0 is used again.
    log_info(_OP, "Installing ROCm without MIVisionX/rocDecode (their "
                  "dependencies have CVEs that block publishing)",
             usecase="graphics", packages=_ROCM_PACKAGES)
    if not run(["amdgpu-install", "-y", "--usecase=graphics"], _OP,
               cwd=_WORK_DIR, env=env):
        return False
    extra = _UBUNTU_EXTRA_PACKAGES.get(distro.name, [])
    if extra:
        log_info(_OP, f"{distro.name}: also installing {' '.join(extra)} for hipcc",
                 packages=extra)
    package_lists = [_ROCM_PACKAGES] + ([extra] if extra else []) + [_BANDWIDTH_TEST_PACKAGES]
    return _install_each(package_lists, env, cwd=_WORK_DIR)


def _install_ubuntu(env: dict[str, str], inputs: _Inputs) -> str | None:
    log_info(_OP, "Ubuntu: installing ROCm with amdgpu-install",
             handler="ubuntu", usecase="graphics")
    # An empty work directory, so a failed run's leftovers can't be picked up;
    # removed whatever happens (our addition).
    remove_paths(_WORK_DIR, op=_OP)
    try:
        deb = fetch_verified(inputs.config["url"], inputs.config["sha256"], _OP, _WORK_DIR)
        if deb is None:
            return None
        log_info(_OP, f"Downloaded and verified {deb.name}", path=str(deb))
        # `./<file>` makes apt treat it as a local .deb, relative to cwd.
        if not (install_packages([f"./{deb.name}"], env, _OP, cwd=_WORK_DIR)
                and _configure_amdgpu_dkms_patches(inputs.component_dir)
                and _install_rocm_without_mivisionx(env, inputs.distro)):
            return None
        remove_paths(deb, op=_OP)                       # bash: rm -f ./<deb>
    finally:
        remove_paths(_WORK_DIR, op=_OP)
    return inputs.rocm_version


# --------------------------------------------------------------------------- #
# Azure Linux 3.0: AMD's dnf repos.
# --------------------------------------------------------------------------- #
def _install_azurelinux(env: dict[str, str], inputs: _Inputs) -> str | None:
    log_info(_OP, "Azure Linux: installing ROCm from the AMD dnf repos",
             handler="azurelinux")
    version = inputs.rocm_version
    if not (_install_each(azure_linux_driver_packages(inputs.kernel), env)
            and _append(_AMD_ROCM_REPO_FILE, amd_rocm_repo_text(version),
                        "[amd_rocm]", "an [amd_rocm] section")):
        return None
    log_info(_OP, "Added the AMD ROCm dnf repo", repo="amd_rocm", version=version)
    # TODO(ADO 1001): toolkit candidate: refresh_package_index (see above).
    if not (run(["dnf", "repolist", "--refresh"], _OP, env=env)
            and _install_each(_AZURE_LINUX_ROCM_PACKAGES, env)):
        return None
    return version


# --------------------------------------------------------------------------- #
# Steps shared by every supported distro.
# --------------------------------------------------------------------------- #
def _setup_udev(env: dict[str, str]) -> bool:
    if not write_file(_UDEV_RULES_FILE, _UDEV_RULES, _OP):
        return False
    # `udevadm control --reload-rules && sudo udevadm trigger` under `set -e`:
    # a failure BEFORE `&&` doesn't stop bash; it only skips the trigger.
    if not run(["udevadm", "control", "--reload-rules"], _OP, env=env):
        log_warn(_OP, "udev rules reload failed; skipping `udevadm trigger` and "
                      "continuing, as bash does (rules apply from next boot)")
        return True
    if not run(["sudo", "udevadm", "trigger"], _OP, env=env):
        return False
    log_info(_OP, "Reloaded udev rules and triggered device events",
             path=_UDEV_RULES_FILE)
    return True


def _blacklist_amdgpu(env: dict[str, str], kernel: str) -> bool:
    """Ubuntu only: stop amdgpu auto-loading and rebuild the initramfs."""
    log_warn(_OP, "Blacklisting amdgpu: it will not auto-load at boot; only "
                  f"{_SERVICE_NAME}.service loads it, once all GPUs are present",
             path=_BLACKLIST_FILE, service=_SERVICE_NAME)
    if not (_append(_BLACKLIST_FILE, _BLACKLIST_LINE, _BLACKLIST_LINE.rstrip("\n"),
                    "'blacklist amdgpu'")
            and run(["update-initramfs", "-c", "-k", kernel], _OP, env=env)):
        return False
    log_info(_OP, "Blacklisted amdgpu and created a new initramfs", kernel=kernel)
    return True


def _install_boot_service(env: dict[str, str]) -> bool:
    """The init script and the unit that runs it at boot; start + enable."""
    log_info(_OP, "Writing gpu mode probe in init.d", path=_INIT_SCRIPT)
    if not write_file(_INIT_SCRIPT, _INIT_SCRIPT_TEXT, _OP):
        return False
    # `chmod +x` adds the execute bits to whatever mode the file has.
    mode = stat.S_IMODE(os.stat(_INIT_SCRIPT).st_mode) | 0o111
    os.chmod(_INIT_SCRIPT, mode)
    if not write_file(_SERVICE_FILE, _SERVICE_UNIT, _OP, mode=_SERVICE_FILE_MODE):
        return False
    # The script polls up to ~15 minutes (91 checks, 10 s apart) for GPUs.
    log_warn(_OP, f"Starting {_SERVICE_NAME} on the build VM, as bash does: its "
                  "script polls for AMD GPUs for up to ~15 minutes and the build "
                  "VM may have none", service=_SERVICE_NAME, script=_INIT_SCRIPT)
    if not (systemctl("start", _SERVICE_NAME, op=_OP, env=env)
            and systemctl("enable", _SERVICE_NAME, op=_OP, env=env)):
        return False
    log_info(_OP, f"Started and enabled {_SERVICE_NAME}.service",
             service=_SERVICE_NAME, path=_SERVICE_FILE, init_mode=oct(mode))
    return True


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Branch:
    """How one of bash's distro branches differs."""
    rocm_keys: tuple[str, ...]      # versions.json "rocm" keys it reads
    install: Callable[[dict, _Inputs], str | None]  # -> the ROCm version to record
    needs_driver_version: bool = False


# Bash's if/elif order: exactly ubuntu26.04, any other *ubuntu*, exactly
# azurelinux3.0; there is no else.
_BRANCHES = {
    "ubuntu26.04": _Branch(("version",), _install_ubuntu2604, needs_driver_version=True),
    "ubuntu": _Branch(("version", "url", "sha256"), _install_ubuntu),
    "azurelinux3.0": _Branch(("version",), _install_azurelinux),
}


def _branch_for(distro: Distro) -> _Branch | None:
    if distro.is_ubuntu:
        return _BRANCHES.get(distro.name, _BRANCHES["ubuntu"])
    return _BRANCHES.get(distro.name)


def _resolve_inputs(env: dict[str, str], distro: Distro, branch: _Branch) -> _Inputs | None:
    """Everything the branch needs, or None (logged) before any change."""
    config = require_config("rocm", env, _OP, *branch.rocm_keys)
    if config is None:
        return None
    driver_version = None
    if branch.needs_driver_version:
        driver = require_config("amdgpu", env, _OP, "version")
        if driver is None:
            return None
        driver_version = str(driver["version"])
    component_dir = env.get("COMPONENT_DIR", "")
    patch = os.path.join(component_dir, _DKMS_PATCH_SOURCE)
    if distro.is_ubuntu and not (component_dir and os.path.isfile(patch)):
        log_error(_OP, f"amdgpu DKMS patch not found at {patch!r} (COMPONENT_DIR unset?)",
                  path=patch)
        return None
    kernel = platform.release()                         # `uname -r`
    if not kernel:
        log_error(_OP, "could not determine the running kernel version (uname -r)")
        return None
    return _Inputs(distro, config, driver_version, kernel, component_dir)


def _log_start(inputs: _Inputs) -> None:
    if inputs.driver_version:
        source = f"apt ({amdgpu_repo_url(inputs.driver_version)} + stable.repo.amd.com)"
    else:
        source = inputs.config.get("url", "dnf (azurelinux-repos-amd + amd_rocm repo)")
    log_info(_OP, f"Installing ROCm {inputs.rocm_version} on {inputs.distro}",
             version=inputs.rocm_version, distro=str(inputs.distro),
             kernel=inputs.kernel, source=source, amdgpu_version=inputs.driver_version)
    log_info(_OP, "ROCm is installed before PMIx/MPI, as in install.sh, so the "
                  "HPC-X step can rebuild UCX with ROCm support")


def _configure_gpus(env: dict[str, str], inputs: _Inputs, version: str) -> bool:
    """Steps 2-5: udev, record ROCM (where bash does), blacklist, boot service."""
    if not _setup_udev(env):
        return False
    write_component_version("ROCM", version)
    log_info(_OP, f"Recorded ROCM={version}", component="ROCM", version=version)
    if inputs.distro.is_ubuntu and not _blacklist_amdgpu(env, inputs.kernel):
        return False
    return _install_boot_service(env)


def install(env: dict[str, str]) -> int:
    """Install ROCm and its boot-time GPU setup. Returns 0 or 3."""
    distro = Distro.from_env(env)
    branch = _branch_for(distro)
    if branch is None:
        log_error(_OP, f"ROCm install is not supported on {distro.name!r} "
                       "(bash would install nothing but still record ROCM)",
                  distro=distro.name)
        return 3
    inputs = _resolve_inputs(env, distro, branch)
    if inputs is None:
        return 3
    _log_start(inputs)
    try:
        version = branch.install(env, inputs)
        if version is None or not _configure_gpus(env, inputs, version):
            return 3
    except OSError as exc:
        # The DKMS patch copy, chmod, and the versions file in /opt.
        log_error(_OP, f"file operation failed: {exc}",
                  path=getattr(exc, "filename", None))
        return 3
    log_info(_OP, f"ROCm {version} installed and configured",
             version=version, distro=distro.name,
             amdgpu_blacklisted=distro.is_ubuntu, service=_SERVICE_NAME)
    return 0
