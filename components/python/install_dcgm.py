"""install_dcgm.py — Python port of components/install_dcgm.sh.

What this step does
-------------------
Installs NVIDIA DCGM (Data Center GPU Manager), the GPU health-monitoring and
diagnostics service, records its version as DCGM in the component-versions
file, and enables and starts the nvidia-dcgm service. The NVIDIA package repos
were already added by the driver/CUDA steps.

DCGM packages are built per CUDA major version (e.g. ...-cuda13), so the step
first asks the installed NVIDIA driver which CUDA version it supports. Like
bash, it compiles and runs a tiny C program that loads libcuda.so.1 and calls
cuDriverGetVersion(), which returns 1000 * major + 10 * minor (13000 for CUDA
13.0). This needs the driver's user-space library, not GPU hardware, so it
works on a build VM without a GPU (nvidia-smi would not).

Per distro family:

  * Ubuntu        install the four datacenter-gpu-manager-4 packages for that
                  CUDA major, pinned to the "dcgm" version in versions.json.
  * Azure Linux   install datacenter-gpu-manager-4-cuda13-<version>
                  (see "Bash bug, mirrored" below).
  * RHEL family   (AlmaLinux, Rocky, RHEL) install the cuda<major> package
                  (unpinned), then read the installed version from `dcgmi`.

When the SKU's CUDA version in versions.json (e.g. 12.9 for a V100) has a
lower major than the driver supports, Ubuntu and RHEL also install the DCGM
packages for that lower major (DCGM needs both; NVIDIA/DCGM issue #254).

The original bash, for comparison (comments trimmed; the C program is in
_CUDA_PROBE_SOURCE below, byte for byte):

    CUDA_DRIVER_SUPPORTED_VERSION=$(
        cuda_probe_dir=$(mktemp -d)
        trap 'rm -rf "${cuda_probe_dir}"' EXIT
        cat > "${cuda_probe_dir}/cuda_driver_api_version.c" <<'EOF'
    ...
    EOF
        gcc "${cuda_probe_dir}/cuda_driver_api_version.c" -ldl -o "${cuda_probe_dir}/cuda_driver_api_version"
        "${cuda_probe_dir}/cuda_driver_api_version"
    )
    CUDA_VERSION=$((CUDA_DRIVER_SUPPORTED_VERSION / 1000))

    cuda_metadata=$(get_component_config "cuda")
    SKU_CUDA_VERSION=$(jq -r '.driver.version' <<< $cuda_metadata | cut -d'.' -f1)

    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        dcgm_metadata=$(get_component_config "dcgm")
        DCGM_VERSION=$(jq -r '.version' <<< $dcgm_metadata)
        apt-get install -y \\
            datacenter-gpu-manager-4-cuda${CUDA_VERSION}=${DCGM_VERSION} \\
            datacenter-gpu-manager-4-core=${DCGM_VERSION} \\
            datacenter-gpu-manager-4-proprietary=${DCGM_VERSION} \\
            datacenter-gpu-manager-4-proprietary-cuda${CUDA_VERSION}=${DCGM_VERSION}
        if [[ "${SKU_CUDA_VERSION}" -lt "${CUDA_VERSION}" ]]; then
            apt-get install -y \\
                datacenter-gpu-manager-4-cuda${SKU_CUDA_VERSION}=${DCGM_VERSION} \\
                datacenter-gpu-manager-4-proprietary-cuda${SKU_CUDA_VERSION}=${DCGM_VERSION}
        fi
    elif [[ $DISTRIBUTION == *"azurelinux"* ]]; then
        dcgm_metadata=$(get_component_config "dcgm")
        DCGM_VERSION=$(jq -r '.version' <<< $dcgm_metadata)
        if [ "$1" = "V100" ]; then
            dnf install -y datacenter-gpu-manager-4-cuda12-${DCGM_VERSION}
        else
            dnf install -y datacenter-gpu-manager-4-cuda13-${DCGM_VERSION}
        fi
    else
        dnf clean expire-cache
        dnf install --assumeyes --setopt=install_weak_deps=True datacenter-gpu-manager-4-cuda${CUDA_VERSION}
        if [[ "${SKU_CUDA_VERSION}" -lt "${CUDA_VERSION}" ]]; then
            dnf install --assumeyes --setopt=install_weak_deps=True datacenter-gpu-manager-4-cuda${SKU_CUDA_VERSION}
        fi
        DCGM_VERSION=$(dcgmi --version | awk '{print $3}')
    fi

    write_component_version "DCGM" ${DCGM_VERSION}

    systemctl --now enable nvidia-dcgm
    systemctl start nvidia-dcgm
    systemctl is-active --quiet nvidia-dcgm     # set -e: a failure stops the script

Bash bug, mirrored: the Azure Linux branch checks `[ "$1" = "V100" ]`, but
install.sh runs install_dcgm.sh with no arguments, so $1 is always empty and
every SKU (V100 included) gets the cuda13 package. This port always installs
cuda13 on Azure Linux too, and does not try to "fix" it from SKU.

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-dcgm" Step. Deleting `action=` from that Step makes the build run the
bash script again, which is why the .sh file is kept.

install() receives the build environment (a dict) and returns 0 for success or
3 for failure. It logs problems and never lets an exception escape.

Deliberate differences from bash
--------------------------------
  * A missing "dcgm" version (Ubuntu, Azure Linux) or a missing/non-numeric
    CUDA driver version for the SKU (Ubuntu, RHEL family, where it is compared)
    fails before installing anything. Bash would install "...=null", or compare
    "null" (which bash arithmetic treats as 0) and install "...-cudanull".
    Azure Linux never uses the SKU CUDA version, so, as in bash, it may be missing.
  * Output from the CUDA probe that isn't a positive integer fails the step.
  * On the RHEL family, a failing `dcgmi --version` or an empty parsed version
    fails the step; bash (no pipefail) would record "".
  * The probe's temporary directory is created by Python's tempfile.mkdtemp()
    (like `mktemp -d`, under $TMPDIR or /tmp) and removed afterwards in all
    cases, like bash's EXIT trap.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile

from utils.component_config import config_for, write_component_version
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.process import exec_program, run_capture

# The Step's `op` label in build_config.py; used on every log line.
_OP = "install-dcgm"

# Where the probe's temporary directory is created. None means Python's default
# temp location, like `mktemp -d`. The tests point it at their own directory.
_PROBE_PARENT_DIR: str | None = None
_PROBE_SOURCE_NAME = "cuda_driver_api_version.c"
_PROBE_BINARY_NAME = "cuda_driver_api_version"

# The C program from install_dcgm.sh's quoted heredoc (<<'EOF'), byte for byte:
# a quoted heredoc expands nothing, and its text ends with a newline after "}".
# A test checks this against the .sh file so the two can't drift apart.
_CUDA_PROBE_SOURCE = r"""#include <dlfcn.h>
#include <stdio.h>

typedef int (*cuDriverGetVersionFn)(int *);

int main(void) {
    int exit_code = 1;
    int cuda_driver_api_version = 0;
    void *cuda_library = NULL;

    cuda_library = dlopen("libcuda.so.1", RTLD_NOW);
    if (cuda_library == NULL) {
        fprintf(stderr, "Unable to load libcuda.so.1: %s\n", dlerror());
        goto cleanup;
    }

    cuDriverGetVersionFn cuDriverGetVersion = (cuDriverGetVersionFn)dlsym(cuda_library, "cuDriverGetVersion");
    if (cuDriverGetVersion == NULL) {
        fprintf(stderr, "Unable to find cuDriverGetVersion in libcuda.so.1: %s\n", dlerror());
        goto cleanup;
    }

    int result = cuDriverGetVersion(&cuda_driver_api_version);
    if (result != 0 || cuda_driver_api_version <= 0) {
        fprintf(stderr, "Unable to query CUDA driver API version from libcuda.so.1\n");
        goto cleanup;
    }

    printf("%d\n", cuda_driver_api_version);
    exit_code = 0;

cleanup:
    if (cuda_library != NULL) {
        dlclose(cuda_library);
    }

    return exit_code;
}
"""
# (The r"""...""" "raw string" keeps the C "\n" escapes as the two characters
# backslash and n, exactly as they appear in the bash heredoc.)

# The DCGM service, and the Azure Linux package's fixed CUDA major (see above).
_SERVICE = "nvidia-dcgm"
_AZURE_LINUX_CUDA_MAJOR = 13

# The dnf options bash passes on the RHEL family.
_RHEL_DNF_INSTALL = ["dnf", "install", "--assumeyes",
                     "--setopt=install_weak_deps=True"]

# What bash word splitting and awk's default field splitting treat as
# separators: spaces, tabs and newlines.
_BLANKS = re.compile(r"[ \t\n]+")


# --------------------------------------------------------------------------- #
# Pure helpers: they only transform values, so tests call them directly.
# --------------------------------------------------------------------------- #
def parse_cuda_driver_version(output: str) -> int | None:
    """The integer the CUDA probe printed (e.g. "13000\\n" -> 13000), or None.

    Bash's $(...) drops the trailing newline and $(( ... )) does the maths. We
    only accept a plain positive integer; anything else is None.
    """
    text = output.strip()
    # fullmatch() requires the WHOLE string to match: digits only.
    if not re.fullmatch(r"[0-9]+", text) or int(text) <= 0:
        return None
    return int(text)


def cuda_major(driver_api_version: int) -> int:
    """$((CUDA_DRIVER_SUPPORTED_VERSION / 1000)): 13000 -> 13, 12090 -> 12.

    `//` is integer division (it drops the remainder), like bash's `/`.
    """
    return driver_api_version // 1000


def sku_cuda_major(cuda_config) -> int | None:
    """The major of cuda.driver.version from versions.json ("12.9" -> 12).

    Mirrors  jq -r '.driver.version' | cut -d'.' -f1 . Returns None when the
    value is missing or its major isn't a number, where bash would carry on
    with "null" (see the module docstring).
    """
    driver = cuda_config.get("driver") if isinstance(cuda_config, dict) else None
    version = driver.get("version") if isinstance(driver, dict) else None
    if version is None:
        return None
    # str() handles a version written as a JSON number; split(".")[0] is cut -f1.
    major = str(version).split(".")[0]
    return int(major) if re.fullmatch(r"[0-9]+", major) else None


def dcgmi_version(output: str) -> str | None:
    """The DCGM version from `dcgmi --version` output, or None.

    Mirrors  DCGM_VERSION=$(dcgmi --version | awk '{print $3}')  followed by the
    unquoted ${DCGM_VERSION}: awk prints field 3 of every line ("" if a line is
    shorter), and write_component_version records only the first word.

        "dcgmi  version: 4.4.1"  ->  "4.4.1"
    """
    values = []
    for line in output.split("\n"):
        fields = [field for field in _BLANKS.split(line) if field]
        values.append(fields[2] if len(fields) >= 3 else "")
    words = [word for word in _BLANKS.split("\n".join(values)) if word]
    return words[0] if words else None


def ubuntu_packages(cuda: int, dcgm_version: str) -> list[str]:
    """The four Ubuntu packages bash installs for CUDA major `cuda`."""
    return [
        f"datacenter-gpu-manager-4-cuda{cuda}={dcgm_version}",
        f"datacenter-gpu-manager-4-core={dcgm_version}",
        f"datacenter-gpu-manager-4-proprietary={dcgm_version}",
        f"datacenter-gpu-manager-4-proprietary-cuda{cuda}={dcgm_version}",
    ]


def ubuntu_extra_packages(sku_cuda: int, dcgm_version: str) -> list[str]:
    """The two extra Ubuntu packages for a lower SKU CUDA major."""
    return [
        f"datacenter-gpu-manager-4-cuda{sku_cuda}={dcgm_version}",
        f"datacenter-gpu-manager-4-proprietary-cuda{sku_cuda}={dcgm_version}",
    ]


# --------------------------------------------------------------------------- #
# Helpers with side effects.
# --------------------------------------------------------------------------- #
def _probe_cuda_driver_version(env: dict[str, str]) -> int | None:
    """Compile and run the C probe; return cuDriverGetVersion(), or None (logged)."""
    # mkdtemp() creates a new, uniquely named directory and returns its path.
    probe_dir = tempfile.mkdtemp(prefix="cuda_probe.", dir=_PROBE_PARENT_DIR)
    source = os.path.join(probe_dir, _PROBE_SOURCE_NAME)
    binary = os.path.join(probe_dir, _PROBE_BINARY_NAME)
    try:
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(_CUDA_PROBE_SOURCE)
        log_debug(_OP, f"Wrote CUDA probe source {source}", path=source)

        if exec_program(["gcc", source, "-ldl", "-o", binary], _OP, env=env) != 0:
            log_error(_OP, "could not compile the CUDA driver probe", path=source)
            return None
        # The probe prints the version on stdout; its errors go to stderr.
        rc, output = run_capture([binary], _OP, env=env)
        if rc != 0:
            log_error(_OP, "the CUDA driver probe could not query libcuda.so.1 "
                           "(is the NVIDIA driver installed?)", exit_code=rc)
            return None
        version = parse_cuda_driver_version(output)
        if version is None:
            log_error(_OP, f"unexpected CUDA driver probe output: {output.strip()!r}")
        return version
    finally:
        # Like bash's `trap 'rm -rf ...' EXIT`: always remove the directory.
        shutil.rmtree(probe_dir, ignore_errors=True)


def _run_all(commands: list[list[str]], env: dict[str, str]) -> bool:
    """Run commands in order, stopping at the first failure (like `set -e`)."""
    for command in commands:
        if exec_program(command, _OP, env=env) != 0:
            return False
    return True


def _config_dcgm_version(env: dict[str, str]) -> str | None:
    """The "dcgm" version from versions.json, or None (logged)."""
    cfg = config_for("dcgm", env)
    if not isinstance(cfg, dict) or not cfg.get("version"):
        log_error(_OP, "could not resolve dcgm version from versions.json",
                  distro=env.get("DISTRIBUTION", ""))
        return None
    return str(cfg["version"])


def _needs_sku_packages(cuda: int, sku_cuda: int) -> bool:
    """Log and return whether extra packages for the SKU's CUDA major are needed."""
    needed = sku_cuda < cuda
    if needed:
        log_info(_OP, f"SKU CUDA major {sku_cuda} is lower than the driver's "
                      f"{cuda}: also installing the cuda{sku_cuda} DCGM packages",
                 cuda_major=cuda, sku_cuda_major=sku_cuda)
    else:
        log_info(_OP, f"SKU CUDA major {sku_cuda} is not lower than the driver's "
                      f"{cuda}: no extra DCGM packages", cuda_major=cuda,
                 sku_cuda_major=sku_cuda)
    return needed


# --------------------------------------------------------------------------- #
# Per-distro handlers. Each returns the DCGM version to record, or None on
# failure (already logged).
# --------------------------------------------------------------------------- #
def _install_ubuntu(env, cuda: int, sku_cuda: int | None) -> str | None:
    dcgm = _config_dcgm_version(env)
    if dcgm is None:
        return None
    if sku_cuda is None:
        log_error(_OP, "could not resolve a numeric cuda driver version for this "
                       "SKU from versions.json")
        return None
    log_info(_OP, f"Ubuntu: installing DCGM {dcgm} for CUDA {cuda} from apt",
             dcgm_version=dcgm, version_source="versions.json", cuda_major=cuda)

    commands = [["apt-get", "install", "-y", *ubuntu_packages(cuda, dcgm)]]
    if _needs_sku_packages(cuda, sku_cuda):
        commands.append(["apt-get", "install", "-y",
                         *ubuntu_extra_packages(sku_cuda, dcgm)])
    return dcgm if _run_all(commands, env) else None


def _install_azurelinux(env, cuda: int, sku_cuda: int | None) -> str | None:
    dcgm = _config_dcgm_version(env)
    if dcgm is None:
        return None
    # BASH BUG (mirrored on purpose): install_dcgm.sh picks cuda12 only when
    # its first argument is "V100", but install.sh never passes one, so it
    # always installs the cuda13 package, whatever the SKU or probed CUDA major.
    log_info(_OP, f"Azure Linux: installing DCGM {dcgm} (always the "
                  f"cuda{_AZURE_LINUX_CUDA_MAJOR} package, like bash)",
             dcgm_version=dcgm, version_source="versions.json", cuda_major=cuda,
             sku_cuda_major=sku_cuda)
    package = f"datacenter-gpu-manager-4-cuda{_AZURE_LINUX_CUDA_MAJOR}-{dcgm}"
    # The bug only changes the result for a V100: bash meant it to get cuda12.
    # Warn so a V100 image build's log shows the package it really received.
    sku = env.get("SKU", "")
    if sku == "V100":
        log_warn(_OP, "Azure Linux V100: installing the cuda13 DCGM package "
                      "because install.sh passes no SKU argument to "
                      "install_dcgm.sh (kept for bash parity)",
                 sku=sku, package=package)
    return dcgm if _run_all([["dnf", "install", "-y", package]], env) else None


def _install_rhel(env, cuda: int, sku_cuda: int | None) -> str | None:
    if sku_cuda is None:
        log_error(_OP, "could not resolve a numeric cuda driver version for this "
                       "SKU from versions.json")
        return None
    log_info(_OP, f"RHEL family: installing the latest DCGM for CUDA {cuda} from dnf",
             distro=env.get("DISTRIBUTION", ""), cuda_major=cuda)

    commands = [
        ["dnf", "clean", "expire-cache"],
        [*_RHEL_DNF_INSTALL, f"datacenter-gpu-manager-4-cuda{cuda}"],
    ]
    if _needs_sku_packages(cuda, sku_cuda):
        commands.append([*_RHEL_DNF_INSTALL, f"datacenter-gpu-manager-4-cuda{sku_cuda}"])
    if not _run_all(commands, env):
        return None

    # The RHEL family installs unpinned, so record what dcgmi reports.
    rc, output = run_capture(["dcgmi", "--version"], _OP, env=env)
    dcgm = dcgmi_version(output) if rc == 0 else None
    if dcgm is None:
        log_error(_OP, f"could not read the DCGM version from `dcgmi --version`: "
                       f"{output.strip()!r}", exit_code=rc)
        return None
    log_info(_OP, f"Installed DCGM version is {dcgm}", dcgm_version=dcgm,
             version_source="dcgmi")
    return dcgm


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def install(env: dict[str, str]) -> int:
    """Install DCGM, record its version and start nvidia-dcgm. Returns 0 or 3."""
    distribution = env.get("DISTRIBUTION", "")
    log_info(_OP, "Installing DCGM", distro=distribution, sku=env.get("SKU", ""),
             node_type=env.get("TARGET_NODE_TYPE", ""))

    # ---- 1. CUDA major supported by the installed driver -------------------
    try:
        driver_api = _probe_cuda_driver_version(env)
    except OSError as exc:
        log_error(_OP, f"could not set up the CUDA driver probe: {exc}")
        return 3
    if driver_api is None:
        return 3
    cuda = cuda_major(driver_api)
    log_info(_OP, f"Probed CUDA driver API version {driver_api} (CUDA {cuda})",
             cuda_driver_api_version=driver_api, cuda_major=cuda)

    # ---- 2. CUDA major of this SKU's driver in versions.json ---------------
    sku_cuda = sku_cuda_major(config_for("cuda", env))
    log_info(_OP, f"SKU CUDA major from versions.json: {sku_cuda}",
             sku_cuda_major=sku_cuda, cuda_major=cuda)

    # ---- 3. packages, per distro --------------------------------------------
    if "ubuntu" in distribution:
        handler = _install_ubuntu
    elif "azurelinux" in distribution:
        handler = _install_azurelinux
    else:  # RHEL family: AlmaLinux, Rocky Linux, RHEL, etc.
        handler = _install_rhel
    dcgm = handler(env, cuda, sku_cuda)
    if dcgm is None:
        log_error(_OP, "DCGM installation failed", distro=distribution)
        return 3

    # ---- 4. record the version ----------------------------------------------
    try:
        write_component_version("DCGM", dcgm)
    except OSError as exc:
        log_error(_OP, f"could not record DCGM={dcgm}: {exc}",
                  component="DCGM", version=dcgm)
        return 3
    log_info(_OP, f"Recorded DCGM version {dcgm}", component="DCGM", version=dcgm)

    # ---- 5. service -----------------------------------------------------------
    log_info(_OP, f"Enabling and starting the {_SERVICE} service", service=_SERVICE)
    if not _run_all([["systemctl", "--now", "enable", _SERVICE],
                     ["systemctl", "start", _SERVICE]], env):
        log_error(_OP, f"could not enable/start the {_SERVICE} service",
                  service=_SERVICE)
        return 3
    if exec_program(["systemctl", "is-active", "--quiet", _SERVICE], _OP, env=env) != 0:
        log_error(_OP, "DCGM is inactive!", service=_SERVICE)
        return 3
    log_info(_OP, f"{_SERVICE} service is active", service=_SERVICE)

    log_info(_OP, "DCGM installed", distro=distribution, dcgm_version=dcgm,
             cuda_major=cuda, sku_cuda_major=sku_cuda)
    return 0
