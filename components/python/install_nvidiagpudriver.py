"""install_nvidiagpudriver.py — Python port of components/install_nvidiagpudriver.sh.

What this step does
-------------------
Installs the NVIDIA data-center GPU driver for every NVIDIA SKU except NCv6
(which uses the GRID driver, install_nvidiagriddriver), including the
rackscale NVLink SKUs GB200/GB300 (VR200 is built as GB200). install() runs:

  1. The driver, per distro family (_DRIVER_INSTALLERS):
       * Azure Linux 3.0  fetch the PMC "nvidia" and NVIDIA "cuda-azl3" repo
                          files (_AZL_REPO_FILES), `dnf install` the driver
                          meta-package with cuda-azl3 disabled, read the
                          version from `dnf list installed`, pin the package.
                          V100 -> "cuda" (proprietary), aarch64 ->
                          "cuda-open-hwe", else "cuda-open".
       * Ubuntu           NVIDIA's CUDA apt repo is already there (the
                          bootstrap's install_cuda_repo_and_dkms.sh, #643). On
                          the MRC network (baremetal_1p) or with nvidia
                          driver.source "private", first install the local
                          driver repo .deb (driver.repo_file) from
                          $TOP_DIR/internal_bits and prefer it over every
                          other source. Then nvidia-driver-pinning-<version>
                          plus cuda-drivers (V100), the -open dkms/driver
                          packages of driver.major_version (rackscale) or
                          nvidia-open (ubuntu_driver_packages). Ubuntu 26.04
                          records the installed nvidia-open version
                          (versions.json only names the branch, e.g. "610").
                          Drop the kms modprobe file, write nvprofiling.conf.
       * anything else    (RHEL family) download and run NVIDIA's .run
                          installer (proprietary for V100, open otherwise);
                          AlmaLinux/Rocky/RHEL also `dkms install` it.
     The driver version is recorded as NVIDIA.
  2. Queue nvidia_peermem for first boot; on baremetal_1p/baremetal_3p add
     NVreg_GrdmaPciTopoCheckOverride=1 to /etc/modprobe.d/nvidia.conf.
  3. Unless azure_vm_akshost: install cuda-toolkit-<major>-<minor> (the RHEL
     family first adds NVIDIA's CUDA dnf repo, rhel_cuda_repo_commands),
     write /etc/profile.d/cuda.sh, record the nvcc version as CUDA, and build
     the CUDA samples (install_cuda_samples.install). The package name,
     cuda.sh text and nvcc parser are shared with install_nvidiagriddriver
     (utils/cuda.py).
  4. Nested bash: install_gdrcopy.sh.
  5. Not rackscale: nested install_nvidia_fabric_manager.sh. Rackscale
     (NVLINK_RACKSCALE=true, any case): nvprofiling and CDMM
     (nvidia-openrm.conf) options, NVIDIA IMEX (recorded as IMEX),
     NVreg_CreateImexChannel0=1, rebuild the initramfs, enable
     nvidia-imex.service.
  6. Nested bash: configure_nvidia_persistence.sh.

The nested scripts stay bash; they run with the same environment from this
step's work dir (bash runs them from its own cwd, the distro build dir).

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets install() as the `action=` of the
"install-nv-driver" Step (NVIDIA, not NCv6). Deleting `action=` runs the bash
script again, which is why the .sh file is kept. install() returns 0 or 3,
logs problems and never lets an exception escape.

Synced with upstream 0c2b8f9: local driver repo for driver.source "private"
and `apt install` of local .debs (#626, #622), CUDA keyring + DKMS moved to
the bootstrap (#643), Ubuntu 26.04 driver version (#614), RHEL
supplementary-repo cccl exclude (#620). The public versions.json has no
nvidia driver.repo_file or driver.source: the MRC and "private" paths need
the out-of-repo 1P overlay, and fail up front without it.

Parity notes
------------
  * The version parsers (dnf list | grep | sed, nvcc | grep | awk | cut,
    dpkg-query | sed 's/-.*//', bash's unquoted ${VAR} keeping the first
    word) and the RHEL supplementary repo id were compared against the real
    tools in WSL. The generated files were rendered with real bash; the tests
    compare against those renders (tests/python/fixtures/install_nvidiagpudriver/).
  * `dnf config-manager --save ... >/dev/null`: the output goes to the debug
    log instead.
  * `curl URL > file` runs as `curl URL`, with Python writing the file.
  * cuda.sh is written once with both lines (bash: tee, then tee -a).
  * Bash runs `sudo update-initramfs`; the build runs as root, so no sudo.

Deliberate differences from bash
--------------------------------
  * Installs go through install_packages() (`apt install pkg -y` ->
    `apt-get install -y -o Dpkg::Use-Pty=0 pkg`). In the RHEL-family branch
    only AlmaLinux/Rocky/RHEL have a known manager; elsewhere the toolkit
    install fails (bash would try dnf).
  * Package-manager toolkit calls: the local repo's `apt update` runs as
    `apt-get update` (refresh_package_index); the Ubuntu 26.04 driver
    version is read with `dpkg-query -W -f '${Version}' nvidia-open`
    (installed_package_version; bash: -f='${Version}', same output).
  * Every versions.json value a branch needs is required before anything is
    installed (_required_keys): nvidia driver.version (Ubuntu, RHEL family),
    driver.sha256 (RHEL family), driver.repo_file (Ubuntu local repo),
    driver.major_version (Ubuntu rackscale, non-V100), imex.version
    (rackscale); cuda driver.version (toolkit) and driver.distribution (RHEL
    family toolkit). Bash carries on with "null".
  * Ubuntu 26.04: an empty driver version from dpkg-query fails the step;
    bash records NVIDIA="".
  * Ubuntu: we log whether a CUDA apt source (/etc/apt/sources.list.d/cuda-*)
    exists and warn if not. Bash doesn't check; apt then fails.
  * Rackscale on a distro other than Azure Linux 3.0 or Ubuntu fails up
    front. Bash installs the driver, toolkit and gdrcopy first, then exits 1
    at the IMEX step.
  * A version that can't be read fails the step instead of recording "" or a
    whole `dnf list` line: the Azure Linux driver version, and the nvcc CUDA
    version (also when `dnf list installed` or nvcc fails, which bash ignores
    without pipefail).
  * A failing curl leaves the repo file untouched; bash's `>` has already
    emptied it.
  * The local-repo keyrings are copied with copy_files, which expands the
    glob in Python; no match fails with a clear message (bash's `cp` fails
    on the literal pattern).
  * `rm -f` of the kms modprobe file goes through remove_paths, which logs
    but never fails the build.
  * Files are written with write_file, which creates missing parent
    directories.
  * Downloads and nested scripts use _WORK_DIR, emptied first and removed at
    the end. Bash ends with `rm -rf *.run *.tar.gz *.rpm` and `rm -rf -- */`
    in the distro build dir, deleting every subdirectory there, including
    other steps' leftovers. The port never deletes anything in the
    repository and logs a warning naming what bash would have removed.

Bash quirks, kept and warned about (log_warn when they apply)
-------------------------------------------------------------
  * Appends are not rerun-safe: nvidia_peermem (modules-load) and the
    NVreg_GrdmaPciTopoCheckOverride / NVreg_CreateImexChannel0 options
    (nvidia.conf) are appended again on every run.
  * `curl` runs without -f, so an HTTP error page is saved as the repo file.
    Warned when the saved file has no "[section]" header.
  * Ubuntu installs `nvidia-imex` unpinned but records the versions.json IMEX
    version, without checking what apt installed.
  * The RHEL-family .run installer and CUDA repo URLs hard-code x86_64.
    Warned when ARCHITECTURE is something else.
  * Ubuntu 26.04 always reads the driver version from the nvidia-open
    package, even when the V100 or rackscale branch installed other
    packages. Warned on those branches.
  * The final build-dir cleanup (see above).
"""

from __future__ import annotations

import os
import platform
import re
from dataclasses import dataclass

from components.python import install_cuda_samples
from utils.component_config import (DEFAULT_NODE_TYPE, config_for, record_version,
                                    require_config)
from utils.cuda import (APT_SOURCES_DIR, CUDA_PROFILE, CUDA_PROFILE_TEXT,
                        NVCC_VERSION_COMMAND, check_cuda_apt_source, cuda_toolkit_package,
                        nvcc_cuda_version)
from utils.distro import Distro
from utils.download import fetch_verified
from utils.files import copy_files, file_has_line, fresh_dir, remove_paths, write_file
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import (install_packages, installed_package_version,
                                     refresh_package_index)
from utils.package_pinning import dnf_pin_packages
from utils.process import run, run_all, run_capture
from utils.services import systemctl
from utils.sku import NETWORK_MODE_MRC, sku_network_mode

# The Step's `op` label in plans/steps.py; used on every log line.
_OP = "install-nv-driver"

# Our own download/build directory, also the cwd of the nested scripts.
_WORK_DIR = "/tmp/nvidia-driver-build"

# Prefixed to every absolute system path this module reads or writes. "" on a
# real build; the tests point it at a temporary directory.
_ROOT = ""

# The only distro handled by the Azure Linux branch (an exact match, as in bash).
_AZURE_LINUX = "azurelinux3.0"

_AKS_HOST = "azure_vm_akshost"
_BAREMETAL = ("baremetal_1p", "baremetal_3p")

# ---- Azure Linux ----------------------------------------------------------
_AZL_REPO_FILES = {
    "aarch64": [
        ("https://packages.microsoft.com/azurelinux/3.0/prod/nvidia/aarch64/config.repo",
         "/etc/yum.repos.d/azurelinux-nvidia-prod.repo"),
        ("https://developer.download.nvidia.com/compute/cuda/repos/azl3/sbsa/cuda-azl3.repo",
         "/etc/yum.repos.d/cuda-azl3.repo"),
    ],
    "x86_64": [
        ("https://packages.microsoft.com/azurelinux/3.0/prod/nvidia/x86_64/config.repo",
         "/etc/yum.repos.d/azurelinux-nvidia-prod.repo"),
        ("https://developer.download.nvidia.com/compute/cuda/repos/azl3/x86_64/cuda-azl3.repo",
         "/etc/yum.repos.d/cuda-azl3.repo"),
    ],
}
# A repo file has at least one "[repo-id]" line; an HTML error page doesn't.
_REPO_SECTION = re.compile(r"^\[[^]\n]+\][ \t]*$", re.MULTILINE)

# ---- Ubuntu -----------------------------------------------------------------
# A driver.source with this value also uses the local driver repo (as MRC does).
_PRIVATE_SOURCE = "private"
_APT_PREFERENCES = "/etc/apt/preferences.d/00-nvidia-prefer"
_APT_PREFERENCES_TEXT = 'Package: *\nPin: origin ""\nPin-Priority: 1001\n'
_KEYRINGS_DIR = "/usr/share/keyrings"
_KMS_CONF = "/etc/modprobe.d/nvidia-graphics-drivers-kms.conf"
# The one Ubuntu release whose recorded driver version comes from dpkg-query.
_UBUNTU_DPKG_VERSION = "ubuntu26.04"
_DPKG_VERSION_PACKAGE = "nvidia-open"
# Why each Ubuntu driver branch (see ubuntu_driver_packages) is chosen.
_UBUNTU_BRANCH_REASON = {
    "cuda-drivers": "V100: proprietary kernel modules",
    "rackscale": "NVLINK_RACKSCALE=true: the open dkms/driver packages of the "
                 "major version and nvidia-modprobe",
    "nvidia-open": "open kernel modules",
}

# ---- RHEL family ------------------------------------------------------------
_RUN_URL = "https://us.download.nvidia.com/tesla/{v}/NVIDIA-Linux-x86_64-{v}.run"
_RHEL_CUDA_REPO = ("https://developer.download.nvidia.com/compute/cuda/repos/"
                   "{d}/x86_64/cuda-{d}.repo")

# ---- Rackscale (IMEX), per family (other families are rejected up front) -----
_IMEX_PACKAGE = {"azurelinux": "nvidia-imex-{version}", "ubuntu": "nvidia-imex"}
_INITRAMFS = {"azurelinux": ["dracut", "--force"],
              "ubuntu": ["update-initramfs", "-u", "-k", "all"]}

# ---- Files written on every distro --------------------------------------------
_NVPROFILING_CONF = "/etc/modprobe.d/nvprofiling.conf"
_NVPROFILING_TEXT = "options nvidia NVreg_RestrictProfilingToAdminUsers=0\n"
_PEERMEM_CONF = "/etc/modules-load.d/nvidia-peermem.conf"
_PEERMEM_LINE = "nvidia_peermem"
_NVIDIA_CONF = "/etc/modprobe.d/nvidia.conf"
_GRDMA_LINE = "options nvidia NVreg_GrdmaPciTopoCheckOverride=1"
_IMEX_CHANNEL_LINE = "options nvidia NVreg_CreateImexChannel0=1"
_OPENRM_CONF = "/etc/modprobe.d/nvidia-openrm.conf"
_OPENRM_TEXT = "options nvidia NVreg_CoherentGPUMemoryMode=driver\n"

# ---- Nested bash scripts (in $COMPONENT_DIR) ----------------------------------
_GDRCOPY_SCRIPT = "install_gdrcopy.sh"
_FABRIC_MANAGER_SCRIPT = "install_nvidia_fabric_manager.sh"
_PERSISTENCE_SCRIPT = "configure_nvidia_persistence.sh"

# What the bash `rm` at the end of the script deletes in the distro build dir.
_BASH_CLEANUP = "rm -rf *.run *.tar.gz *.rpm; rm -rf -- */ (dotglob)"

# Bash word splitting / awk field splitting: spaces, tabs and newlines.
_BLANKS = re.compile(r"[ \t\n]+")


# --------------------------------------------------------------------------- #
# Pure helpers: they only transform values, so tests call them directly.
# --------------------------------------------------------------------------- #
def first_word(text: str) -> str | None:
    """The first word of `text`, or None: what bash's unquoted ${VAR} passes
    as write_component_version's version argument."""
    words = [word for word in _BLANKS.split(text) if word]
    return words[0] if words else None


def is_rackscale(env: dict[str, str]) -> bool:
    """bash: [[ "${NVLINK_RACKSCALE,,}" == "true" ]] (case-insensitive)."""
    return env.get("NVLINK_RACKSCALE", "").lower() == "true"


def azurelinux_driver_package(sku: str, arch: str) -> str:
    """The Azure Linux driver meta-package (bash's AL3_GPU_DRIVER_PACKAGES)."""
    if sku == "V100":
        return "cuda"              # V100 needs the proprietary kernel modules
    if arch == "aarch64":
        return "cuda-open-hwe"
    return "cuda-open"


# sed 's/.*\s\+\([0-9.]\+-[0-9]\+\)_.*/\1/' in Python's regex syntax.
_DNF_VERSION_SED = re.compile(r".*[ \t\n\r\f\v]+([0-9.]+-[0-9]+)_.*")


def azurelinux_driver_version(dnf_list_output: str, package: str) -> str | None:
    """The driver version from `dnf list installed` output, or None.

    Mirrors
        dnf list installed | grep "^${PKG}\\." | sed 's/.*\\s\\+\\([0-9.]\\+-[0-9]\\+\\)_.*/\\1/'
    and the unquoted ${NVIDIA_DRIVER_VERSION} (first word only):

        "cuda-open.x86_64   580.126.20-1_6.6.104.2.4.azl3   @azurelinux-nvidia"
                                                     ->  "580.126.20-1"

    Like sed, a matching line whose version doesn't fit the pattern is kept
    whole; the caller rejects that (bash would record "cuda-open.x86_64").
    """
    grep = re.compile("^" + re.escape(package) + r"\.")
    lines = [line for line in dnf_list_output.split("\n") if grep.search(line)]
    # count=1: sed without the g flag replaces the first match only.
    edited = [_DNF_VERSION_SED.sub(r"\1", line, count=1) for line in lines]
    return first_word("\n".join(edited))


def looks_like_version(value: str | None, pattern: str) -> bool:
    """True if `value` is a whole match for `pattern`."""
    return value is not None and re.fullmatch(pattern, value) is not None


def local_repo_dir(repo_file: str) -> str:
    """bash: ${repo_file%%_*} (everything before the first _)."""
    return repo_file.split("_")[0]


def dpkg_driver_version(dpkg_query_output: str) -> str | None:
    """The driver version from `dpkg-query -W -f='${Version}' nvidia-open`, or None.

    Mirrors  dpkg-query ... | sed 's/-.*//'  and the unquoted
    ${NVIDIA_DRIVER_VERSION} (first word only):  "610.43.02-0ubuntu1" -> "610.43.02"
    """
    edited = [re.sub(r"-.*", "", line, count=1) for line in dpkg_query_output.split("\n")]
    return first_word("\n".join(edited))


def ubuntu_driver_packages(sku: str, rackscale: bool, major: str) -> tuple[str, list[str]]:
    """(branch, packages) bash installs after nvidia-driver-pinning-<version>."""
    if sku == "V100":
        return "cuda-drivers", ["cuda-drivers"]
    if rackscale:
        return "rackscale", [f"nvidia-dkms-{major}-open", f"nvidia-driver-{major}-open",
                             "nvidia-modprobe"]
    return "nvidia-open", ["nvidia-open"]


def rhel_cuda_excludes(cuda_version: str) -> str:
    """The excludepkgs bash sets on the RHEL-family CUDA repo. mft must track
    DOCA, not CUDA; CUDA 13 cccl obsoletes the cuda-cccl-12 CUDA 12 needs."""
    excludes = "mft* kernel-mft*"
    if cuda_version.startswith("12."):      # [[ $V == 12.* ]]
        excludes += " cccl-*"
    return excludes


def rhel_supplementary_repo(distribution: str, arch: str) -> str:
    """The RHUI repo bash also excludes cccl from on RHEL + CUDA 12.

    bash: rhel-${rhel_version%%.*}-for-${ARCHITECTURE}-supplementary-rhui-rpms
          with rhel_version=${DISTRIBUTION#rhel}
    "rhel9.8", "x86_64" -> "rhel-9-for-x86_64-supplementary-rhui-rpms"
    """
    return f"rhel-{Distro(distribution).major}-for-{arch}-supplementary-rhui-rpms"


def rhel_cuda_repo_commands(cuda_distribution: str, cuda_version: str,
                            distro: Distro, arch: str) -> list[list[str]]:
    """The dnf commands bash runs before the RHEL-family toolkit install."""
    commands = [["dnf", "config-manager", "--add-repo",
                 _RHEL_CUDA_REPO.format(d=cuda_distribution)]]
    if cuda_version.startswith("12.") and distro.is_rhel:
        commands.append(["dnf", "config-manager", "--save",
                         f"--setopt={rhel_supplementary_repo(distro.name, arch)}"
                         ".excludepkgs=cccl-*"])
    return commands + [
        ["dnf", "config-manager", "--save",
         f"--setopt=cuda-{cuda_distribution}-x86_64.excludepkgs="
         f"{rhel_cuda_excludes(cuda_version)}"],
        ["dnf", "clean", "expire-cache"],
    ]


def _host(path: str) -> str:
    """`path` on the image being built (under _ROOT in the tests)."""
    return _ROOT + path


def _value(cfg, dotted_key: str) -> str:
    """cfg["a"]["b"] for "a.b", as a string; "" when any part is missing."""
    node = cfg
    for key in dotted_key.split("."):
        node = node.get(key) if isinstance(node, dict) else None
    return "" if node is None else str(node)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass
class _Inputs:
    """Everything resolved from env and versions.json before installing."""
    distro: Distro
    arch: str
    sku: str
    node_type: str
    rackscale: bool
    mrc: bool
    toolkit: bool
    driver_source: str = ""        # nvidia driver.source ("private" -> local repo)
    driver_version: str = ""       # Ubuntu/RHEL: versions.json; Azure Linux: dnf
    driver_sha256: str = ""
    driver_major: str = ""
    repo_file: str = ""
    cuda_version: str = ""
    cuda_distribution: str = ""
    imex_version: str = ""

    @property
    def family(self) -> str:
        """bash's if/elif/else: == "azurelinux3.0", == *"ubuntu"*, everything else."""
        if self.distro.name == _AZURE_LINUX:
            return "azurelinux"
        if self.distro.is_ubuntu:
            return "ubuntu"
        return "rhel"

    @property
    def local_repo(self) -> bool:
        """Ubuntu only. bash: _is_mrc_network || [ "$NVIDIA_DRIVER_SOURCE" = "private" ]"""
        return self.family == "ubuntu" and (self.mrc or self.driver_source == _PRIVATE_SOURCE)


def _build_facts(env: dict[str, str]) -> _Inputs:
    """The build target, before any version is looked up."""
    node_type = env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE
    sku = env.get("SKU", "")
    p = _Inputs(
        distro=Distro.from_env(env),
        arch=env.get("ARCHITECTURE", ""),
        sku=sku,
        node_type=node_type,
        rackscale=is_rackscale(env),
        # bash's _is_mrc_network (TARGET_NODE_TYPE == baremetal_1p).
        mrc=sku_network_mode(sku, node_type) == NETWORK_MODE_MRC,
        toolkit=env.get("TARGET_NODE_TYPE", "") != _AKS_HOST,
    )
    if p.family == "ubuntu":
        p.driver_source = _value(config_for("nvidia", env), "driver.source")
    return p


def _required_keys(p: _Inputs) -> tuple[list[str], list[str]]:
    """The nvidia and cuda versions.json keys this build's branches read."""
    nvidia, cuda = [], []
    if p.family == "ubuntu":
        nvidia.append("driver.version")
        if p.local_repo:
            nvidia.append("driver.repo_file")
        if p.rackscale and p.sku != "V100":
            nvidia.append("driver.major_version")
    elif p.family == "rhel":
        nvidia += ["driver.version", "driver.sha256"]
    if p.rackscale:
        nvidia.append("imex.version")
    if p.toolkit:
        cuda.append("driver.version")
        if p.family == "rhel":
            cuda.append("driver.distribution")
    return nvidia, cuda


def _resolve(env: dict[str, str]) -> _Inputs | None:
    """Read env and versions.json; None (logged) if a needed value is missing."""
    p = _build_facts(env)
    log_info(_OP, "Installing the NVIDIA GPU driver", distro=p.distro.name,
             arch=p.arch, sku=p.sku, node_type=p.node_type, rackscale=p.rackscale,
             nvlink_rackscale=env.get("NVLINK_RACKSCALE", ""), mrc_network=p.mrc,
             driver_source=p.driver_source or "(not set)",
             local_driver_repo=p.local_repo, cuda_toolkit=p.toolkit, work_dir=_WORK_DIR)
    if p.rackscale and p.family == "rhel":
        log_error(_OP, f"Unsupported distribution for nvidia-imex: {p.distro} "
                       "(NVLINK_RACKSCALE=true needs Azure Linux 3.0 or Ubuntu; "
                       "bash exits 1 at the IMEX step)", distro=p.distro.name)
        return None

    # Only components with required keys are looked up: e.g. Azure Linux reads
    # its driver version from dnf, so it needs no nvidia config at all.
    nvidia_keys, cuda_keys = _required_keys(p)
    nvidia = require_config("nvidia", env, _OP, *nvidia_keys) if nvidia_keys else {}
    cuda = require_config("cuda", env, _OP, *cuda_keys) if cuda_keys else {}
    if nvidia is None or cuda is None:
        return None
    # Only what the branches need: e.g. Azure Linux's driver version comes
    # from dnf even when versions.json has one.
    nv = {key: _value(nvidia, key) for key in nvidia_keys}
    cu = {key: _value(cuda, key) for key in cuda_keys}
    p.driver_version = nv.get("driver.version", "")
    p.driver_sha256 = nv.get("driver.sha256", "")
    p.repo_file = nv.get("driver.repo_file", "")
    p.driver_major = nv.get("driver.major_version", "")
    p.imex_version = nv.get("imex.version", "")
    p.cuda_version = cu.get("driver.version", "")
    p.cuda_distribution = cu.get("driver.distribution", "")
    log_info(_OP, "Resolved versions", driver_version=p.driver_version or "(from dnf)",
             cuda_version=p.cuda_version, imex_version=p.imex_version)
    return p


# --------------------------------------------------------------------------- #
# Small jobs shared by the steps. Each returns False (or None) after logging.
# --------------------------------------------------------------------------- #
def _append_line(path: str, line: str) -> bool:
    """bash's `echo <line> >> <path>`.

    BASH QUIRK (kept): the append is not rerun-safe; a line already in the
    file is added again. Warned about when that happens.
    """
    if file_has_line(path, line):
        log_warn(_OP, f"{path} already contains {line!r}; appending it "
                      "again, as bash does (the append is not rerun-safe)",
                 path=path, line=line)
    return write_file(path, line + "\n", _OP, append=True)


def _nested(script: str, env: dict[str, str], why: str) -> bool:
    """Run a nested component script with bash's environment, from _WORK_DIR."""
    # Local: the toolkit has no nested-script runner; this is one run() call.
    path = os.path.join(env.get("COMPONENT_DIR", ""), script)
    log_info(_OP, f"Running nested script {script}: {why}", script=script)
    if not run([path], _OP, cwd=_WORK_DIR, env=env):
        log_error(_OP, f"nested script {script} failed", script=path)
        return False
    return True


def _warn_x86_64_only(p: _Inputs, what: str) -> None:
    if p.arch != "x86_64":
        # BASH QUIRK (kept): the RHEL-family URLs hard-code x86_64.
        log_warn(_OP, f"{what} URL is hard-coded to x86_64 but ARCHITECTURE is "
                      f"{p.arch!r} (bash behaviour, kept)", arch=p.arch)


# --------------------------------------------------------------------------- #
# 1. Driver, per distro family. Each returns the driver version, or None.
# --------------------------------------------------------------------------- #
def _fetch_azl_repo_files(env: dict[str, str], arch: str) -> bool:
    for url, path in _AZL_REPO_FILES[arch]:
        # bash: curl URL > path (no -f: an HTTP error page would be saved).
        rc, text = run_capture(["curl", url], _OP, env=env)
        if rc != 0 or not write_file(_host(path), text, _OP):
            return False
        if not _REPO_SECTION.search(text):
            log_warn(_OP, f"{path} has no [repo] section: curl runs without -f, "
                          "so this may be an HTTP error page (bash behaviour, kept)",
                     path=path, url=url)
    return True


def _dnf_installed_version(env: dict[str, str], package: str) -> str | None:
    rc, listing = run_capture(["dnf", "list", "installed"], _OP, env=env)
    version = azurelinux_driver_version(listing, package) if rc == 0 else None
    if not looks_like_version(version, r"[0-9.]+-[0-9]+"):
        log_error(_OP, f"could not read the installed {package} version from "
                       f"`dnf list installed` (parsed {version!r})",
                  package=package, exit_code=rc)
        return None
    log_info(_OP, f"Installed {package} version is {version}", package=package,
             driver_version=version, version_source="dnf list installed")
    return version


def _driver_azurelinux(env: dict[str, str], p: _Inputs) -> str | None:
    package = azurelinux_driver_package(p.sku, p.arch)
    reason = ("V100 needs the proprietary kernel modules" if p.sku == "V100"
              else "aarch64 uses the HWE open modules" if p.arch == "aarch64"
              else "open kernel modules")
    log_info(_OP, f"Azure Linux: installing driver package {package} from PMC ({reason})",
             package=package, source="packages.microsoft.com", sku=p.sku, arch=p.arch)
    if not _fetch_azl_repo_files(env, "aarch64" if p.arch == "aarch64" else "x86_64"):
        return None
    # The CUDA repo has an identically named 'cuda' meta-package, so it is
    # disabled: every driver package comes from PMC.
    if not install_packages([package], env, _OP, options=["--disablerepo=cuda-azl3*"]):
        return None
    version = _dnf_installed_version(env, package)
    # Keep later dnf operations from moving the PMC-installed driver family.
    if version is None or not dnf_pin_packages([package], _OP, env):
        return None
    return version


def _add_local_repo(env: dict[str, str], p: _Inputs) -> bool:
    """bash's install_from_nvidia_local_repo: install the local NVIDIA driver
    repo and prefer it over every other source."""
    deb = os.path.join(env.get("TOP_DIR", ""), "internal_bits", p.repo_file)
    reason = ("MRC network (baremetal_1p)" if p.mrc
              else f'driver.source is "{_PRIVATE_SOURCE}"')
    log_info(_OP, f"{reason}: adding the local NVIDIA driver repo and preferring it "
                  "over every other source (the CUDA toolkit still comes from the "
                  "online CUDA repo)", repo_file=deb, node_type=p.node_type,
             driver_source=p.driver_source or "(not set)", mrc_network=p.mrc)
    # bash: cp /var/<repo dir>/nvidia-driver-local-*-keyring.gpg /usr/share/keyrings/
    keyrings = _host(f"/var/{local_repo_dir(p.repo_file)}/nvidia-driver-local-*-keyring.gpg")
    # The preference is written before `apt update` so it applies to the refresh.
    return (install_packages([deb], env, _OP)
            and copy_files(keyrings, _host(_KEYRINGS_DIR), _OP)
            and write_file(_host(_APT_PREFERENCES), _APT_PREFERENCES_TEXT, _OP)
            and refresh_package_index(env, _OP))


def _driver_ubuntu(env: dict[str, str], p: _Inputs) -> str | None:
    log_info(_OP, f"Ubuntu: installing NVIDIA driver {p.driver_version} from apt",
             driver_version=p.driver_version, version_source="versions.json")
    check_cuda_apt_source(_OP, _host(APT_SOURCES_DIR))
    if p.local_repo:
        if not _add_local_repo(env, p):
            return None
    else:
        log_info(_OP, "Not the MRC network and driver.source is not \"private\": "
                      "driver packages come from the CUDA repo",
                 node_type=p.node_type, driver_source=p.driver_source or "(not set)")

    branch, packages = ubuntu_driver_packages(p.sku, p.rackscale, p.driver_major)
    log_info(_OP, f"Installing {' '.join(packages)} ({_UBUNTU_BRANCH_REASON[branch]})",
             branch=branch, packages=" ".join(packages), driver_major=p.driver_major)
    # One install per bash command, stopping at the first failure.
    if not (install_packages([f"nvidia-driver-pinning-{p.driver_version}"], env, _OP)
            and install_packages(packages, env, _OP)):
        return None

    version = p.driver_version
    if p.distro.name == _UBUNTU_DPKG_VERSION:
        version = _dpkg_driver_version(env, p, branch)
        if version is None:
            return None
    # bash: rm -f, for an unused file the driver package may create.
    remove_paths(_host(_KMS_CONF), op=_OP)
    if not write_file(_host(_NVPROFILING_CONF), _NVPROFILING_TEXT, _OP):
        return None
    # nvidia-peermem is not loaded at build time; it is queued for first boot
    # (_queue_peermem, and setup_sku_customizations' openibd drop-in).
    return version


def _dpkg_driver_version(env: dict[str, str], p: _Inputs, branch: str) -> str | None:
    """Ubuntu 26.04: the installed nvidia-open version, without the revision."""
    if branch != "nvidia-open":
        # BASH QUIRK (kept): bash asks for nvidia-open's version whatever was
        # installed; on these branches it may not be installed at all.
        log_warn(_OP, f"{p.distro}: reading the driver version from "
                      f"{_DPKG_VERSION_PACKAGE} although the {branch} branch was "
                      "installed (bash behaviour, kept)",
                 distro=p.distro.name, branch=branch)
    installed = installed_package_version(_DPKG_VERSION_PACKAGE, env, _OP)
    if installed is None:
        return None
    version = dpkg_driver_version(installed)
    if not version:
        log_error(_OP, f"could not read the installed {_DPKG_VERSION_PACKAGE} version "
                       f"from {installed!r} (bash's sed 's/-.*//' leaves nothing); bash "
                       "would record an empty NVIDIA version",
                  package=_DPKG_VERSION_PACKAGE, distro=p.distro.name)
        return None
    if not looks_like_version(version, r"[0-9]+(\.[0-9]+)*"):
        log_warn(_OP, f"{_DPKG_VERSION_PACKAGE} version {version!r} doesn't look like "
                      "a driver version; recording it anyway, as bash does",
                 package=_DPKG_VERSION_PACKAGE, driver_version=version)
    log_info(_OP, f"{p.distro}: installed driver version is {version} "
                  f"(versions.json driver.version {p.driver_version} only picks the "
                  "pinning package)", driver_version=version,
             pinning_version=p.driver_version, version_source="dpkg-query nvidia-open")
    return version


def _driver_rhel(env: dict[str, str], p: _Inputs) -> str | None:
    module_type = "proprietary" if p.sku == "V100" else "open"
    url = _RUN_URL.format(v=p.driver_version)
    log_info(_OP, f"RHEL family: installing NVIDIA driver {p.driver_version} from the "
                  f".run installer ({module_type} kernel modules)",
             driver_version=p.driver_version, version_source="versions.json",
             source=url, kernel_module_type=module_type)
    _warn_x86_64_only(p, "The NVIDIA .run driver")
    if (fetch_verified(url, p.driver_sha256, _OP, _WORK_DIR) is None
            or not run(["bash", f"NVIDIA-Linux-x86_64-{p.driver_version}.run", "--silent",
                        "--dkms", f"--kernel-module-type={module_type}"],
                       _OP, cwd=_WORK_DIR, env=env)):
        return None

    if not p.distro.is_rhel_family:
        log_info(_OP, f"{p.distro} is not AlmaLinux/Rocky/RHEL: no dkms install, "
                      "as in bash", distro=p.distro.name)
        return p.driver_version
    kernel = platform.release()            # bash: `uname -r`
    log_info(_OP, f"Registering the driver with dkms for kernel {kernel}", kernel=kernel)
    if not run(["dkms", "install", "--no-depmod", "-m", "nvidia", "-v",
                p.driver_version, "-k", kernel, "--force"], _OP, env=env):
        return None
    return p.driver_version


_DRIVER_INSTALLERS = {"azurelinux": _driver_azurelinux, "ubuntu": _driver_ubuntu,
                      "rhel": _driver_rhel}


# --------------------------------------------------------------------------- #
# 2. nvidia_peermem at first boot, baremetal modprobe option
# --------------------------------------------------------------------------- #
def _queue_peermem(p: _Inputs) -> bool:
    if not _append_line(_host(_PEERMEM_CONF), _PEERMEM_LINE):
        return False
    if p.node_type not in _BAREMETAL:
        log_info(_OP, f"{p.node_type} is not baremetal: no GrdmaPciTopoCheckOverride",
                 node_type=p.node_type)
        return True
    log_info(_OP, f"{p.node_type}: overriding the GPUDirect RDMA PCI topology check",
             node_type=p.node_type)
    return _append_line(_host(_NVIDIA_CONF), _GRDMA_LINE)


# --------------------------------------------------------------------------- #
# 3. CUDA toolkit
# --------------------------------------------------------------------------- #
def _add_rhel_cuda_repo(env: dict[str, str], p: _Inputs) -> bool:
    _warn_x86_64_only(p, "The CUDA dnf repo")
    repo = _RHEL_CUDA_REPO.format(d=p.cuda_distribution)
    excludes = rhel_cuda_excludes(p.cuda_version)
    log_info(_OP, f"Adding CUDA repo {repo} with excludepkgs={excludes}",
             source=repo, excludepkgs=excludes)
    if p.cuda_version.startswith("12.") and p.distro.is_rhel:
        supplementary = rhel_supplementary_repo(p.distro.name, p.arch)
        log_info(_OP, f"RHEL with CUDA 12: also excluding cccl-* from {supplementary}",
                 repo=supplementary, excludepkgs="cccl-*", distro=p.distro.name)
    return run_all(rhel_cuda_repo_commands(p.cuda_distribution, p.cuda_version,
                                           p.distro, p.arch), _OP, env=env)


def _nvcc_cuda_version(env: dict[str, str]) -> str | None:
    rc, output = run_capture(list(NVCC_VERSION_COMMAND), _OP, env=env)
    cuda = nvcc_cuda_version(output) if rc == 0 else None
    if not looks_like_version(cuda, r"[0-9]+(\.[0-9]+)*"):
        log_error(_OP, f"could not read the CUDA version from `nvcc --version` "
                       f"(parsed {cuda!r})", exit_code=rc)
        return None
    log_info(_OP, f"nvcc reports CUDA {cuda}", cuda_version=cuda, version_source="nvcc")
    return cuda


def _install_toolkit(env: dict[str, str], p: _Inputs) -> str | None:
    """Install the toolkit, write cuda.sh, build samples; return the CUDA version."""
    package = cuda_toolkit_package(p.cuda_version)
    log_info(_OP, f"Installing CUDA toolkit {p.cuda_version} ({package})",
             cuda_version=p.cuda_version, package=package)
    # Ubuntu's CUDA repo comes from the bootstrap, Azure Linux's from step 1.
    if ((p.family == "rhel" and not _add_rhel_cuda_repo(env, p))
            or not install_packages([package], env, _OP)):
        log_error(_OP, f"CUDA toolkit installation failed ({package})", package=package)
        return None
    if not write_file(_host(CUDA_PROFILE), CUDA_PROFILE_TEXT, _OP, mode=0o644):
        return None
    cuda = _nvcc_cuda_version(env)
    if cuda is None or not record_version("CUDA", cuda, _OP):
        return None
    log_info(_OP, "Building the CUDA samples (install_cuda_samples)")
    if install_cuda_samples.install(env) != 0:
        log_error(_OP, "CUDA samples installation failed")
        return None
    return cuda


# --------------------------------------------------------------------------- #
# 5. Fabric manager, or (rackscale) IMEX and CDMM
# --------------------------------------------------------------------------- #
def _install_imex(env: dict[str, str], p: _Inputs) -> bool:
    log_info(_OP, "NVLINK_RACKSCALE=true: configuring CDMM and installing NVIDIA IMEX "
                  "instead of fabric manager", imex_version=p.imex_version)
    if not (write_file(_host(_NVPROFILING_CONF), _NVPROFILING_TEXT, _OP)
            and write_file(_host(_OPENRM_CONF), _OPENRM_TEXT, _OP)):
        return False
    if p.family == "ubuntu":
        # BASH QUIRK (kept): apt installs whatever nvidia-imex it picks, but
        # the versions.json version is what gets recorded.
        log_warn(_OP, "Ubuntu installs nvidia-imex without a version; IMEX is "
                      f"recorded as {p.imex_version} from versions.json without "
                      "checking what apt installed (bash behaviour, kept)",
                 imex_version=p.imex_version)
    package = _IMEX_PACKAGE[p.family].format(version=p.imex_version)
    if not (install_packages([package], env, _OP)
            and record_version("IMEX", p.imex_version, _OP)
            and _append_line(_host(_NVIDIA_CONF), _IMEX_CHANNEL_LINE)):
        return False

    initramfs = _INITRAMFS[p.family]
    log_info(_OP, f"Rebuilding the initramfs ({initramfs[0]}) to pick up the modprobe "
                  "options", command=initramfs[0])
    if not (run(initramfs, _OP, env=env)
            and systemctl("enable", "nvidia-imex.service", op=_OP, env=env)):
        return False
    log_info(_OP, "Enabled nvidia-imex.service", service="nvidia-imex")
    return True


def _fabric_manager_or_imex(env: dict[str, str], p: _Inputs) -> bool:
    if p.rackscale:
        return _install_imex(env, p)
    return _nested(_FABRIC_MANAGER_SCRIPT, env, "NVLINK_RACKSCALE is not true, so "
                                                "fabric manager (NVSwitch) instead of IMEX")


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _install_all(env: dict[str, str], p: _Inputs) -> dict[str, str] | None:
    """Steps 1-6; returns the recorded versions, or None on failure."""
    driver = _DRIVER_INSTALLERS[p.family](env, p)
    if driver is None:
        log_error(_OP, "NVIDIA driver installation failed", distro=p.distro.name)
        return None
    if not (record_version("NVIDIA", driver, _OP) and _queue_peermem(p)):
        return None
    versions = {"NVIDIA": driver}

    if p.toolkit:
        cuda = _install_toolkit(env, p)
        if cuda is None:
            return None
        versions["CUDA"] = cuda
    else:
        log_info(_OP, "AKS host (azure_vm_akshost): skipping the CUDA toolkit, cuda.sh "
                      "and the CUDA samples, as in bash", node_type=p.node_type)

    if not (_nested(_GDRCOPY_SCRIPT, env, "GPUDirect RDMA copy library")
            and _fabric_manager_or_imex(env, p)
            and _nested(_PERSISTENCE_SCRIPT, env, "NVIDIA persistence daemon")):
        return None
    if p.rackscale:
        versions["IMEX"] = p.imex_version
    return versions


def install(env: dict[str, str]) -> int:
    """Install the NVIDIA driver, CUDA and the related tools. Returns 0 or 3."""
    p = _resolve(env)
    if p is None or not fresh_dir(_WORK_DIR, _OP):
        return 3
    try:
        versions = _install_all(env, p)
    finally:
        remove_paths(_WORK_DIR, op=_OP)
        # Deliberate difference: bash would delete these in the distro build dir.
        log_warn(_OP, f"Not running bash's final cleanup ({_BASH_CLEANUP}) in the "
                      "distro build dir: it deletes every subdirectory there, "
                      "including other steps' leftovers; the Python port only "
                      f"removes {_WORK_DIR}",
                 build_dir=os.path.join(env.get("TOP_DIR", ""), "distros", p.distro.name))
    if versions is None:
        return 3
    log_info(_OP, "NVIDIA GPU driver installed", distro=p.distro.name, sku=p.sku,
             rackscale=p.rackscale,
             **{k.lower() + "_version": v for k, v in versions.items()})
    return 0
