"""install_nvidiagpudriver.py — Python port of components/install_nvidiagpudriver.sh.

What this step does
-------------------
Installs the NVIDIA data-center GPU driver for every NVIDIA SKU except NCv6
(which uses the GRID driver), including the rackscale NVLink SKUs GB200/GB300
(VR200 is built as GB200). Then, as the bash script does:

  1. Driver, per distro
       * Azure Linux 3.0  fetch the PMC "nvidia" and NVIDIA "cuda-azl3" repo
                          files, `dnf install` the driver meta-package with the
                          cuda-azl3 repo disabled, read the installed version
                          from `dnf list installed`, and pin the package:
                            V100     -> "cuda"           (proprietary modules)
                            aarch64  -> "cuda-open-hwe"
                            else     -> "cuda-open"
       * Ubuntu           add NVIDIA's CUDA apt repo (cuda-keyring); on the MRC
                          network (baremetal_1p) also the local driver repo from
                          $TOP_DIR/internal_bits, preferred over every other
                          source; install nvidia-driver-pinning-<version>, then
                          cuda-drivers (V100), the -open dkms/driver packages of
                          the major version (NVLINK_RACKSCALE=true) or
                          nvidia-open; drop the kms modprobe file and write
                          /etc/modprobe.d/nvprofiling.conf.
       * anything else    (RHEL family) download NVIDIA's .run installer and run
                          it (proprietary modules for V100, open otherwise);
                          on AlmaLinux/Rocky/RHEL also `dkms install` it for the
                          running kernel.
     The driver version is recorded as NVIDIA.
  2. Queue nvidia_peermem for first boot (/etc/modules-load.d), and on
     baremetal_1p/baremetal_3p add NVreg_GrdmaPciTopoCheckOverride=1 to
     /etc/modprobe.d/nvidia.conf.
  3. Unless TARGET_NODE_TYPE is azure_vm_akshost: install the CUDA toolkit
     (cuda-toolkit-<major>-<minor>), write /etc/profile.d/cuda.sh, record the
     version nvcc reports as CUDA, and build the CUDA samples
     (install_cuda_samples.install, the Python port of install_cuda_samples.sh).
  4. Nested bash: install_gdrcopy.sh.
  5. Not rackscale: nested install_nvidia_fabric_manager.sh.
     Rackscale (NVLINK_RACKSCALE=true, any case): nvprofiling and CDMM
     (nvidia-openrm.conf) modprobe options, NVIDIA IMEX (recorded as IMEX),
     NVreg_CreateImexChannel0=1, rebuild the initramfs and enable
     nvidia-imex.service.
  6. Nested bash: configure_nvidia_persistence.sh.

The nested scripts stay bash. They run with the same environment, from this
step's work dir (bash runs them from the distro build dir, its own cwd).

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-nv-driver" Step (NVIDIA, not NCv6). Deleting `action=` makes the
build run the bash script again, which is why the .sh file is kept.
install() returns 0 for success or 3 for failure, logs problems and never lets
an exception escape.

Parity notes
------------
  * The version parsers (dnf list | grep | sed, nvcc | grep | awk | cut, and
    bash's unquoted ${VAR} keeping only the first word) were compared against
    the real tools in WSL. The generated files were rendered with real bash;
    the tests compare against those renders (tests/python/fixtures/
    install_nvidiagpudriver/).
  * `curl URL > file` is run as `curl URL`, with Python writing its output to
    the file.
  * /etc/profile.d/cuda.sh is written once with both lines; bash writes the
    first line with `tee` and appends the second with `tee -a`. Same content.
  * Bash runs `sudo update-initramfs`; the build already runs as root, so the
    port runs update-initramfs without sudo.

Deliberate differences from bash
--------------------------------
  * Every versions.json value a branch needs is checked before anything is
    installed, and a missing one fails the step: nvidia driver.version (Ubuntu,
    RHEL family), driver.sha256 (RHEL family), driver.repo_file (Ubuntu MRC),
    driver.major_version (Ubuntu rackscale, non-V100), imex.version
    (rackscale); cuda driver.version (toolkit) and driver.distribution (Ubuntu,
    and RHEL family toolkit). Bash would carry on with "null" in package
    names, URLs and recorded versions.
  * Rackscale on a distro other than Azure Linux 3.0 or Ubuntu fails up front.
    Bash installs the driver, toolkit and gdrcopy first, then exits 1 at the
    IMEX step.
  * A version that can't be read fails the step instead of recording "" or a
    whole `dnf list` line: the Azure Linux driver version, and the CUDA
    version from nvcc (including when `dnf list installed` or nvcc fails, which
    bash ignores because it doesn't set pipefail).
  * A failing curl leaves the repo file untouched; bash's `>` has already
    emptied it before curl fails.
  * The MRC keyring glob is expanded in Python; no match fails with a clear
    message (bash's `cp` fails on the literal pattern).
  * Downloads and nested scripts use _WORK_DIR, emptied first and removed at
    the end. Bash ends with `rm -rf *.run *.tar.gz *.rpm` and `rm -rf -- */`
    in the distro build dir, deleting every subdirectory there, including ones
    other steps left behind. The port never deletes anything in the repository
    and logs a warning naming what bash would have removed.

Bash quirks, kept and warned about (log_warn when they apply)
-------------------------------------------------------------
  * Appends are not rerun-safe: nvidia_peermem (modules-load) and the
    NVreg_GrdmaPciTopoCheckOverride / NVreg_CreateImexChannel0 options
    (nvidia.conf) are appended again on every run.
  * `curl` runs without -f, so an HTTP error page is saved as the repo file.
    Warned when the saved file has no "[section]" header.
  * Ubuntu installs `nvidia-imex` unpinned but records the versions.json IMEX
    version, without checking that apt installed that version.
  * The RHEL-family .run installer and CUDA repo URLs hard-code x86_64.
    Warned when ARCHITECTURE is something else.
  * The final build-dir cleanup (see above).
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
from dataclasses import dataclass

from components.python import install_cuda_samples
from utils.component_config import DEFAULT_NODE_TYPE, config_for, write_component_version
from utils.download import download_and_verify
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.package_pinning import dnf_pin_packages
from utils.process import exec_program, run_capture
from utils.sku import NETWORK_MODE_MRC, sku_network_mode

# The Step's `op` label in build_config.py; used on every log line.
_OP = "install-nv-driver"

# Our own download/build directory, also the cwd of the nested scripts.
_WORK_DIR = "/tmp/nvidia-driver-build"

# Prefixed to every absolute system path this module reads or writes. "" on a
# real build; the tests point it at a temporary directory.
_ROOT = ""

# The only distro handled by the Azure Linux branch (an exact match, as in bash).
_AZURE_LINUX = "azurelinux3.0"
# Distros that also `dkms install` the .run driver (bash's prefix matches).
_DKMS_PREFIXES = ("almalinux", "rocky", "rhel")

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
_KEYRING_DEB = "cuda-keyring_1.1-1_all.deb"
_KEYRING_URL = ("https://developer.download.nvidia.com/compute/cuda/repos/"
                "{distribution}/{arch}/" + _KEYRING_DEB)
_APT_PREFERENCES = "/etc/apt/preferences.d/00-nvidia-prefer"
_APT_PREFERENCES_TEXT = 'Package: *\nPin: origin ""\nPin-Priority: 1001\n'
_KEYRINGS_DIR = "/usr/share/keyrings"
_KMS_CONF = "/etc/modprobe.d/nvidia-graphics-drivers-kms.conf"

# ---- RHEL family ------------------------------------------------------------
_RUN_URL = "https://us.download.nvidia.com/tesla/{v}/NVIDIA-Linux-x86_64-{v}.run"
_RHEL_CUDA_REPO = ("https://developer.download.nvidia.com/compute/cuda/repos/"
                   "{d}/x86_64/cuda-{d}.repo")

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
_CUDA_PROFILE = "/etc/profile.d/cuda.sh"
# Single-quoted in bash's echo, so the ${...} are written literally.
_CUDA_PROFILE_TEXT = (
    'export PATH="${PATH:+$PATH:}/usr/local/cuda/bin"\n'
    'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/cuda/lib64"\n'
)

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
    followed by the unquoted ${NVIDIA_DRIVER_VERSION} (first word only):

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


def nvcc_cuda_version(nvcc_output: str) -> str | None:
    """The CUDA version from `nvcc --version` output, or None.

    Mirrors  nvcc --version | grep release | awk '{print $6}' | cut -c2-
    and the unquoted ${cuda_version} (first word only):

        "Cuda compilation tools, release 13.0, V13.0.88"  ->  "13.0.88"
    """
    values = []
    for line in nvcc_output.split("\n"):
        if "release" not in line:
            continue
        fields = [field for field in _BLANKS.split(line) if field]
        sixth = fields[5] if len(fields) >= 6 else ""
        values.append(sixth[1:])           # cut -c2-: drop the first character
    return first_word("\n".join(values))


def mrc_repo_dir(repo_file: str) -> str:
    """bash: echo $FILE | awk -F'_' '{print $1}' (everything before the first _)."""
    return repo_file.split("_")[0]


def cuda_toolkit_package(cuda_version: str) -> str:
    """cuda-toolkit-${CUDA_DRIVER_VERSION//./-}: "13.0" -> "cuda-toolkit-13-0"."""
    return "cuda-toolkit-" + cuda_version.replace(".", "-")


def rhel_cuda_excludes(cuda_version: str) -> str:
    """The excludepkgs bash sets on the RHEL-family CUDA repo."""
    excludes = "mft* kernel-mft*"
    if cuda_version.startswith("12."):      # [[ $V == 12.* ]]
        excludes += " cccl-*"
    return excludes


def _host(path: str) -> str:
    """`path` on the image being built (under _ROOT in the tests)."""
    return _ROOT + path


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass
class _Inputs:
    """Everything resolved from env and versions.json before installing."""
    distribution: str
    arch: str
    sku: str
    node_type: str
    rackscale: bool
    mrc: bool
    toolkit: bool
    driver_version: str = ""       # Ubuntu/RHEL: versions.json; Azure Linux: dnf
    driver_sha256: str = ""
    driver_major: str = ""
    repo_file: str = ""
    cuda_version: str = ""
    cuda_distribution: str = ""
    imex_version: str = ""

    @property
    def family(self) -> str:
        if self.distribution == _AZURE_LINUX:
            return "azurelinux"
        if "ubuntu" in self.distribution:      # bash: == *"ubuntu"*
            return "ubuntu"
        return "rhel"


def _value(cfg, *keys) -> str:
    """cfg[keys[0]][keys[1]]..., as a string; "" when any part is missing."""
    node = cfg
    for key in keys:
        node = node.get(key) if isinstance(node, dict) else None
    return "" if node is None else str(node)


def _resolve(env: dict[str, str]) -> _Inputs | None:
    """Read env and versions.json; None (logged) if a needed value is missing."""
    node_type = env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE
    sku = env.get("SKU", "")
    p = _Inputs(
        distribution=env.get("DISTRIBUTION", ""),
        arch=env.get("ARCHITECTURE", ""),
        sku=sku,
        node_type=node_type,
        rackscale=is_rackscale(env),
        # bash's _is_mrc_network (TARGET_NODE_TYPE == baremetal_1p).
        mrc=sku_network_mode(sku, node_type) == NETWORK_MODE_MRC,
        toolkit=env.get("TARGET_NODE_TYPE", "") != _AKS_HOST,
    )
    nvidia = config_for("nvidia", env)
    cuda = config_for("cuda", env)

    needed = {}                    # "<component> <key>" -> resolved value
    if p.family == "ubuntu":
        needed["nvidia driver.version"] = _value(nvidia, "driver", "version")
        needed["cuda driver.distribution"] = _value(cuda, "driver", "distribution")
        if p.mrc:
            needed["nvidia driver.repo_file"] = _value(nvidia, "driver", "repo_file")
        if p.rackscale and p.sku != "V100":
            needed["nvidia driver.major_version"] = _value(nvidia, "driver", "major_version")
    elif p.family == "rhel":
        needed["nvidia driver.version"] = _value(nvidia, "driver", "version")
        needed["nvidia driver.sha256"] = _value(nvidia, "driver", "sha256")
        if p.toolkit:
            needed["cuda driver.distribution"] = _value(cuda, "driver", "distribution")
    if p.toolkit:
        needed["cuda driver.version"] = _value(cuda, "driver", "version")
    if p.rackscale:
        if p.family == "rhel":
            log_error(_OP, f"Unsupported distribution for nvidia-imex: {p.distribution} "
                           "(NVLINK_RACKSCALE=true needs Azure Linux 3.0 or Ubuntu; "
                           "bash exits 1 at the IMEX step)", distro=p.distribution)
            return None
        needed["nvidia imex.version"] = _value(nvidia, "imex", "version")

    missing = [name for name, value in needed.items() if not value]
    if missing:
        log_error(_OP, f"could not resolve {', '.join(missing)} from versions.json",
                  distro=p.distribution, arch=p.arch, sku=p.sku, node_type=p.node_type)
        return None

    p.driver_version = needed.get("nvidia driver.version", "")
    p.driver_sha256 = needed.get("nvidia driver.sha256", "")
    p.repo_file = needed.get("nvidia driver.repo_file", "")
    p.driver_major = needed.get("nvidia driver.major_version", "")
    p.cuda_version = needed.get("cuda driver.version", "")
    p.cuda_distribution = needed.get("cuda driver.distribution", "")
    p.imex_version = needed.get("nvidia imex.version", "")
    return p


# --------------------------------------------------------------------------- #
# Helpers with side effects. Each returns False (or None) after logging a
# failure; exec_program has already logged a failing command's output.
# --------------------------------------------------------------------------- #
def _run(command: list[str], env: dict[str, str], *, cwd: str | None = None) -> bool:
    return exec_program(command, _OP, cwd=cwd, env=env) == 0


def _run_all(commands: list[list[str]], env: dict[str, str], *,
             cwd: str | None = None) -> bool:
    """Run commands in order, stopping at the first failure (like `set -e`)."""
    return all(_run(command, env, cwd=cwd) for command in commands)


def _write_file(path: str, text: str) -> bool:
    """Replace `path` with `text` (bash's `> file` / `tee file`)."""
    action = "replaced" if os.path.lexists(path) else "created"
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    except OSError as exc:
        log_error(_OP, f"could not write {path}: {exc}", path=path)
        return False
    log_info(_OP, f"Wrote {path}", path=path, action=action)
    log_debug(_OP, f"Contents of {path}:\n{text}", path=path)
    return True


def _append_line(path: str, line: str) -> bool:
    """Append `line` to `path`, creating it if needed (bash's `echo ... >> file`).

    BASH QUIRK (kept): the append is not rerun-safe; a line already in the
    file is added again. Warned about when that happens.
    """
    action = "appended" if os.path.lexists(path) else "created"
    try:
        if action == "appended":
            with open(path, encoding="utf-8") as handle:
                if line in handle.read().split("\n"):
                    log_warn(_OP, f"{path} already contains {line!r}; appending it "
                                  "again, as bash does (the append is not rerun-safe)",
                             path=path, line=line)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as exc:
        log_error(_OP, f"could not append to {path}: {exc}", path=path)
        return False
    log_info(_OP, f"Added {line!r} to {path}", path=path, action=action, line=line)
    return True


def _record(component: str, version: str) -> bool:
    """Write one entry to the component-versions file."""
    try:
        write_component_version(component, version)
    except OSError as exc:
        log_error(_OP, f"could not record {component}={version}: {exc}",
                  component=component, version=version)
        return False
    log_info(_OP, f"Recorded {component} version {version}",
             component=component, version=version)
    return True


def _nested(script: str, env: dict[str, str], why: str) -> bool:
    """Run a nested component script with bash's environment, from _WORK_DIR."""
    path = os.path.join(env.get("COMPONENT_DIR", ""), script)
    log_info(_OP, f"Running nested script {script}: {why}", script=script)
    if not _run([path], env, cwd=_WORK_DIR):
        log_error(_OP, f"nested script {script} failed", script=path)
        return False
    log_info(_OP, f"Nested script {script} finished", script=script)
    return True


def _warn_x86_64_only(p: _Inputs, what: str) -> None:
    if p.arch != "x86_64":
        # BASH QUIRK (kept): the RHEL-family URLs hard-code x86_64.
        log_warn(_OP, f"{what} URL is hard-coded to x86_64 but ARCHITECTURE is "
                      f"{p.arch!r} (bash behaviour, kept)", arch=p.arch)


# --------------------------------------------------------------------------- #
# 1. Driver, per distro family. Each returns the driver version, or None.
# --------------------------------------------------------------------------- #
def _driver_azurelinux(env, p: _Inputs) -> str | None:
    package = azurelinux_driver_package(p.sku, p.arch)
    reason = ("V100 needs the proprietary kernel modules" if p.sku == "V100"
              else "aarch64 uses the HWE open modules" if p.arch == "aarch64"
              else "open kernel modules")
    log_info(_OP, f"Azure Linux: installing driver package {package} from PMC ({reason})",
             package=package, source="packages.microsoft.com", sku=p.sku, arch=p.arch)

    arch_key = "aarch64" if p.arch == "aarch64" else "x86_64"
    for url, path in _AZL_REPO_FILES[arch_key]:
        # bash: curl URL > path (no -f: an HTTP error page would be saved).
        rc, text = run_capture(["curl", url], _OP, env=env)
        if rc != 0:
            log_error(_OP, f"could not download {url}", url=url, exit_code=rc)
            return None
        if not _write_file(_host(path), text):
            return None
        if not _REPO_SECTION.search(text):
            log_warn(_OP, f"{path} has no [repo] section: curl runs without -f, "
                          "so this may be an HTTP error page (bash behaviour, kept)",
                     path=path, url=url)

    # The CUDA repo has an identically named 'cuda' meta-package, so it is
    # disabled: every driver package comes from PMC.
    if not _run(["dnf", "install", "-y", "--disablerepo=cuda-azl3*", package], env):
        log_error(_OP, f"could not install driver package {package}", package=package)
        return None

    rc, listing = run_capture(["dnf", "list", "installed"], _OP, env=env)
    version = azurelinux_driver_version(listing, package) if rc == 0 else None
    if not looks_like_version(version, r"[0-9.]+-[0-9]+"):
        log_error(_OP, f"could not read the installed {package} version from "
                       f"`dnf list installed` (parsed {version!r})",
                  package=package, exit_code=rc)
        return None
    log_info(_OP, f"Installed {package} version is {version}", package=package,
             driver_version=version, version_source="dnf list installed")

    # Keep later dnf operations from moving the PMC-installed driver family.
    if not dnf_pin_packages([package], _OP, env):
        return None
    return version


def _driver_ubuntu(env, p: _Inputs) -> str | None:
    cuda_arch = "sbsa" if p.arch == "aarch64" else "x86_64"
    keyring_url = _KEYRING_URL.format(distribution=p.cuda_distribution, arch=cuda_arch)
    log_info(_OP, f"Ubuntu: installing NVIDIA driver {p.driver_version} from the "
                  "NVIDIA CUDA apt repo", driver_version=p.driver_version,
             version_source="versions.json", source=keyring_url)
    if not _run_all([["wget", keyring_url],
                     ["dpkg", "-i", f"./{_KEYRING_DEB}"],
                     ["apt-get", "update"]], env, cwd=_WORK_DIR):
        log_error(_OP, "could not add the NVIDIA CUDA apt repo", url=keyring_url)
        return None

    if p.mrc:
        if not _add_mrc_local_repo(env, p):
            return None
    else:
        log_info(_OP, "Not the MRC network: driver packages come from the CUDA repo",
                 node_type=p.node_type)

    commands = [["apt", "install", f"nvidia-driver-pinning-{p.driver_version}", "-y"]]
    if p.sku == "V100":
        log_info(_OP, "V100: installing cuda-drivers (proprietary kernel modules)")
        commands.append(["apt", "install", "cuda-drivers", "-y"])
    elif p.rackscale:
        major = p.driver_major
        log_info(_OP, f"NVLINK_RACKSCALE=true: installing the {major} open dkms/driver "
                      "packages and nvidia-modprobe", driver_major=major)
        commands.append(["apt", "install", f"nvidia-dkms-{major}-open",
                         f"nvidia-driver-{major}-open", "nvidia-modprobe", "-y"])
    else:
        log_info(_OP, "Installing nvidia-open (open kernel modules)")
        commands.append(["apt", "install", "nvidia-open", "-y"])
    if not _run_all(commands, env):
        log_error(_OP, "NVIDIA driver package installation failed",
                  driver_version=p.driver_version)
        return None

    # Remove an unused configuration file the driver package may create.
    kms = _host(_KMS_CONF)
    try:
        os.remove(kms)
        log_info(_OP, f"Removed {kms}", path=kms, action="removed")
    except FileNotFoundError:
        log_info(_OP, f"{kms} not present; nothing to remove", path=kms)
    except OSError as exc:
        log_error(_OP, f"could not remove {kms}: {exc}", path=kms)
        return None

    if not _write_file(_host(_NVPROFILING_CONF), _NVPROFILING_TEXT):
        return None
    # nvidia-peermem is not loaded at build time; it is queued for first boot
    # (see the modules-load file written later and setup_sku_customizations).
    return p.driver_version


def _add_mrc_local_repo(env, p: _Inputs) -> bool:
    """MRC network: install the local NVIDIA driver repo and prefer it."""
    deb = os.path.join(env.get("TOP_DIR", ""), "internal_bits", p.repo_file)
    log_info(_OP, "MRC network (baremetal_1p): adding the local NVIDIA driver repo and "
                  "preferring it over every other source", repo_file=deb,
             node_type=p.node_type)
    if not _run(["dpkg", "-i", deb], env):
        log_error(_OP, f"could not install the local driver repo {deb}", path=deb)
        return False

    pattern = _host(f"/var/{mrc_repo_dir(p.repo_file)}/nvidia-driver-local-*-keyring.gpg")
    keyrings = sorted(glob.glob(pattern))
    if not keyrings:
        log_error(_OP, f"no keyring matches {pattern}", path=pattern)
        return False
    for keyring in keyrings:
        try:
            copied = shutil.copy(keyring, _host(_KEYRINGS_DIR))
        except OSError as exc:
            log_error(_OP, f"could not copy {keyring}: {exc}", path=keyring)
            return False
        log_info(_OP, f"Copied {keyring} to {copied}", source=keyring, path=copied,
                 action="copied")

    # Set before `apt update` so the priority applies during the refresh.
    if not _write_file(_host(_APT_PREFERENCES), _APT_PREFERENCES_TEXT):
        return False
    return _run(["apt", "update"], env)


def _driver_rhel(env, p: _Inputs) -> str | None:
    module_type = "proprietary" if p.sku == "V100" else "open"
    url = _RUN_URL.format(v=p.driver_version)
    log_info(_OP, f"RHEL family: installing NVIDIA driver {p.driver_version} from the "
                  f".run installer ({module_type} kernel modules)",
             driver_version=p.driver_version, version_source="versions.json",
             source=url, kernel_module_type=module_type)
    _warn_x86_64_only(p, "The NVIDIA .run driver")
    try:
        download_and_verify(url, p.driver_sha256, dest_dir=_WORK_DIR)
    except Exception as exc:
        log_error(_OP, f"download/verify of the driver failed: {exc}", url=url)
        return None
    if not _run(["bash", f"NVIDIA-Linux-x86_64-{p.driver_version}.run", "--silent",
                 "--dkms", f"--kernel-module-type={module_type}"], env, cwd=_WORK_DIR):
        log_error(_OP, "the NVIDIA .run installer failed", driver_version=p.driver_version)
        return None

    if p.distribution.startswith(_DKMS_PREFIXES):
        kernel = platform.release()        # bash: `uname -r`
        log_info(_OP, f"Registering the driver with dkms for kernel {kernel}",
                 kernel=kernel)
        if not _run(["dkms", "install", "--no-depmod", "-m", "nvidia", "-v",
                     p.driver_version, "-k", kernel, "--force"], env):
            return None
    else:
        log_info(_OP, f"{p.distribution} is not AlmaLinux/Rocky/RHEL: no dkms install, "
                      "as in bash", distro=p.distribution)
    return p.driver_version


# --------------------------------------------------------------------------- #
# 3. CUDA toolkit
# --------------------------------------------------------------------------- #
def _install_toolkit(env, p: _Inputs) -> str | None:
    """Install the toolkit, write cuda.sh, build samples; return the CUDA version."""
    package = cuda_toolkit_package(p.cuda_version)
    log_info(_OP, f"Installing CUDA toolkit {p.cuda_version} ({package})",
             cuda_version=p.cuda_version, package=package)
    if p.family == "ubuntu":
        commands = [["apt", "install", "-y", package]]   # repo added with the driver
    elif p.family == "azurelinux":
        commands = [["dnf", "install", "-y", package]]
    else:
        repo = _RHEL_CUDA_REPO.format(d=p.cuda_distribution)
        _warn_x86_64_only(p, "The CUDA dnf repo")
        excludes = rhel_cuda_excludes(p.cuda_version)
        # mft must track DOCA, not CUDA; CUDA 13 cccl obsoletes the cuda-cccl-12
        # that CUDA 12 still needs (see the bash comments).
        log_info(_OP, f"Adding CUDA repo {repo} with excludepkgs={excludes}",
                 source=repo, excludepkgs=excludes)
        commands = [
            ["dnf", "config-manager", "--add-repo", repo],
            ["dnf", "config-manager", "--save",
             f"--setopt=cuda-{p.cuda_distribution}-x86_64.excludepkgs={excludes}"],
            ["dnf", "clean", "expire-cache"],
            ["dnf", "install", "-y", package],
        ]
    if not _run_all(commands, env):
        log_error(_OP, f"CUDA toolkit installation failed ({package})", package=package)
        return None

    profile = _host(_CUDA_PROFILE)
    if not _write_file(profile, _CUDA_PROFILE_TEXT):
        return None
    try:
        os.chmod(profile, 0o644)
    except OSError as exc:
        log_error(_OP, f"could not chmod 644 {profile}: {exc}", path=profile)
        return None
    log_info(_OP, f"Set mode 644 on {profile}", path=profile, mode="644")

    rc, output = run_capture(["bash", "-c", "source /etc/profile; nvcc --version"],
                             _OP, env=env)
    cuda = nvcc_cuda_version(output) if rc == 0 else None
    if not looks_like_version(cuda, r"[0-9]+(\.[0-9]+)*"):
        log_error(_OP, f"could not read the CUDA version from `nvcc --version` "
                       f"(parsed {cuda!r})", exit_code=rc)
        return None
    log_info(_OP, f"nvcc reports CUDA {cuda}", cuda_version=cuda, version_source="nvcc")
    if not _record("CUDA", cuda):
        return None

    log_info(_OP, "Building the CUDA samples (install_cuda_samples)",
             script="install_cuda_samples.sh")
    if install_cuda_samples.install(env) != 0:
        log_error(_OP, "CUDA samples installation failed")
        return None
    log_info(_OP, "CUDA samples installed")
    return cuda


# --------------------------------------------------------------------------- #
# 5. Rackscale: IMEX and CDMM
# --------------------------------------------------------------------------- #
def _install_imex(env, p: _Inputs) -> bool:
    log_info(_OP, "NVLINK_RACKSCALE=true: configuring CDMM and installing NVIDIA IMEX "
                  "instead of fabric manager", imex_version=p.imex_version)
    if not (_write_file(_host(_NVPROFILING_CONF), _NVPROFILING_TEXT)
            and _write_file(_host(_OPENRM_CONF), _OPENRM_TEXT)):
        return False

    if p.family == "azurelinux":
        command = ["dnf", "install", "-y", f"nvidia-imex-{p.imex_version}"]
    else:   # ubuntu (other distros were rejected in _resolve, as bash exits 1)
        # BASH QUIRK (kept): apt installs whatever nvidia-imex it picks, but
        # the versions.json version is what gets recorded.
        log_warn(_OP, "Ubuntu installs nvidia-imex without a version; IMEX is "
                      f"recorded as {p.imex_version} from versions.json without "
                      "checking what apt installed (bash behaviour, kept)",
                 imex_version=p.imex_version)
        command = ["apt-get", "install", "nvidia-imex", "-y"]
    if not _run(command, env):
        log_error(_OP, "NVIDIA IMEX installation failed", imex_version=p.imex_version)
        return False
    if not _record("IMEX", p.imex_version):
        return False

    if not _append_line(_host(_NVIDIA_CONF), _IMEX_CHANNEL_LINE):
        return False

    initramfs = (["dracut", "--force"] if p.family == "azurelinux"
                 else ["update-initramfs", "-u", "-k", "all"])
    log_info(_OP, f"Rebuilding the initramfs ({initramfs[0]}) to pick up the modprobe "
                  "options", command=initramfs[0])
    if not _run(initramfs, env):
        log_error(_OP, "rebuilding the initramfs failed", command=initramfs[0])
        return False

    if not _run(["systemctl", "enable", "nvidia-imex.service"], env):
        log_error(_OP, "could not enable nvidia-imex.service", service="nvidia-imex")
        return False
    log_info(_OP, "Enabled nvidia-imex.service", service="nvidia-imex")
    return True


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _install(env: dict[str, str], p: _Inputs) -> dict[str, str] | None:
    """Steps 1-6; returns the recorded versions, or None on failure."""
    versions: dict[str, str] = {}

    # ---- 1. driver ---------------------------------------------------------
    handler = {"azurelinux": _driver_azurelinux, "ubuntu": _driver_ubuntu,
               "rhel": _driver_rhel}[p.family]
    driver = handler(env, p)
    if driver is None:
        log_error(_OP, "NVIDIA driver installation failed", distro=p.distribution)
        return None
    if not _record("NVIDIA", driver):
        return None
    versions["NVIDIA"] = driver

    # ---- 2. peermem at first boot, baremetal modprobe option --------------
    if not _append_line(_host(_PEERMEM_CONF), _PEERMEM_LINE):
        return None
    if p.node_type in _BAREMETAL:
        log_info(_OP, f"{p.node_type}: overriding the GPUDirect RDMA PCI topology check",
                 node_type=p.node_type)
        if not _append_line(_host(_NVIDIA_CONF), _GRDMA_LINE):
            return None
    else:
        log_info(_OP, f"{p.node_type} is not baremetal: no GrdmaPciTopoCheckOverride",
                 node_type=p.node_type)

    # ---- 3. CUDA toolkit + samples ----------------------------------------
    if p.toolkit:
        cuda = _install_toolkit(env, p)
        if cuda is None:
            return None
        versions["CUDA"] = cuda
    else:
        log_info(_OP, "AKS host (azure_vm_akshost): skipping the CUDA toolkit, cuda.sh "
                      "and the CUDA samples, as in bash", node_type=p.node_type)

    # ---- 4. gdrcopy ---------------------------------------------------------
    if not _nested(_GDRCOPY_SCRIPT, env, "GPUDirect RDMA copy library"):
        return None

    # ---- 5. fabric manager or IMEX ------------------------------------------
    if p.rackscale:
        if not _install_imex(env, p):
            return None
        versions["IMEX"] = p.imex_version
    elif not _nested(_FABRIC_MANAGER_SCRIPT, env,
                     "NVLINK_RACKSCALE is not true, so fabric manager (NVSwitch) "
                     "instead of IMEX"):
        return None

    # ---- 6. persistence daemon ----------------------------------------------
    if not _nested(_PERSISTENCE_SCRIPT, env, "NVIDIA persistence daemon"):
        return None
    return versions


def install(env: dict[str, str]) -> int:
    """Install the NVIDIA driver, CUDA and the related tools. Returns 0 or 3."""
    p = _resolve(env)
    if p is None:
        return 3
    log_info(_OP, "Installing the NVIDIA GPU driver", distro=p.distribution,
             arch=p.arch, sku=p.sku, node_type=p.node_type, rackscale=p.rackscale,
             nvlink_rackscale=env.get("NVLINK_RACKSCALE", ""), mrc_network=p.mrc,
             cuda_toolkit=p.toolkit, driver_version=p.driver_version or "(from dnf)",
             cuda_version=p.cuda_version, imex_version=p.imex_version,
             work_dir=_WORK_DIR)

    try:
        shutil.rmtree(_WORK_DIR, ignore_errors=True)
        os.makedirs(_WORK_DIR, exist_ok=True)
    except OSError as exc:
        log_error(_OP, f"could not create {_WORK_DIR}: {exc}", path=_WORK_DIR)
        return 3

    try:
        versions = _install(env, p)
    finally:
        shutil.rmtree(_WORK_DIR, ignore_errors=True)
        log_info(_OP, f"Removed work dir {_WORK_DIR}", path=_WORK_DIR)
        # Deliberate difference: bash would delete these in the distro build dir.
        log_warn(_OP, f"Not running bash's final cleanup ({_BASH_CLEANUP}) in the "
                      "distro build dir: it deletes every subdirectory there, "
                      "including other steps' leftovers; the Python port only "
                      f"removes {_WORK_DIR}",
                 build_dir=os.path.join(env.get("TOP_DIR", ""), "distros",
                                        p.distribution))
    if versions is None:
        return 3

    log_info(_OP, "NVIDIA GPU driver installed", distro=p.distribution, sku=p.sku,
             rackscale=p.rackscale, **{k.lower() + "_version": v
                                       for k, v in versions.items()})
    return 0
