"""install_nvshmem.py — Python port of components/install_nvshmem.sh.

What this step does
-------------------
NVSHMEM is NVIDIA's library for GPU-to-GPU shared memory across a node or an
NVLink domain. This step installs the NVSHMEM packages that match the image's
CUDA major version and records the installed version in the shared
component-versions file (/opt/azurehpc/component_versions.txt).

The build plan runs it only for GB200 images (Ubuntu 24.04 and Azure Linux 3.0,
aarch64).

  * CUDA major version: the first dot-separated field of the cuda config's
    driver.version, e.g. "13.0" -> "13".
  * Azure Linux 3.0: dnf installs the runtime, devel and static packages from
    NVIDIA's RHEL 9 sbsa (aarch64) CUDA repo, added just for this command.
  * Ubuntu: apt installs the runtime and dev packages.
  * The version comes from the package manager's list of installed packages.

The original bash, for comparison:

    cuda_metadata=$(get_component_config "cuda")
    CUDA_DRIVER_VERSION=$(jq -r '.driver.version' <<< $cuda_metadata)
    CUDA_MAJOR_VERSION=$(echo $CUDA_DRIVER_VERSION | cut -d. -f1)

    if [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
        NVSHMEM_CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/rhel9/sbsa"
        dnf install -y --nogpgcheck \\
            --repofrompath=cuda-rhel9-sbsa,$NVSHMEM_CUDA_REPO \\
            libnvshmem3-cuda-$CUDA_MAJOR_VERSION libnvshmem3-devel-cuda-$CUDA_MAJOR_VERSION libnvshmem3-static-cuda-$CUDA_MAJOR_VERSION
        nvshmem_version=$(dnf list installed | grep libnvshmem3-cuda-$CUDA_MAJOR_VERSION | awk '{print $2}')
    elif [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        apt install libnvshmem3-cuda-$CUDA_MAJOR_VERSION libnvshmem3-dev-cuda-$CUDA_MAJOR_VERSION
        nvshmem_version=$(apt list --installed | grep libnvshmem3-cuda-$CUDA_MAJOR_VERSION/ | cut -d' ' -f2)
    fi

    write_component_version "NVSHMEM" $nvshmem_version

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets this module's install() as the `action=` of the
"install-nvshmem" Step. When a Step has an action, the build calls the Python
function instead of running the bash script. Deleting `action=` from that Step
makes the build run the bash script again, which is why the .sh file is kept.

install() receives the build environment (a dict of environment variables) and
returns 0 for success or 3 for failure. It logs problems and returns 3; it does
not let exceptions escape to the build loop.

Deliberate differences from bash
--------------------------------
Each of these is a case where bash would carry on and record an empty or
meaningless version. Python stops with an error (return 3) instead:

  * A missing cuda config or driver.version, or one whose major part is not a
    number. Bash would build package names such as "libnvshmem3-cuda-null".
  * A distro that is neither azurelinux3.0 nor Ubuntu. Bash installs nothing
    there and records an empty NVSHMEM version.
  * `dnf list installed` / `apt list --installed` failing, or its output not
    containing the package. In bash only the last command of a pipeline decides
    the exit status, so `set -e` doesn't notice; the version is just empty.
  * The Azure Linux install goes through install_packages() (the shared
    package-manager abstraction); the dnf command it runs is bash's, unchanged.

Known quirk kept for parity
---------------------------
The Ubuntu `apt install` has no `-y`. See _install_ubuntu() for what that means;
a warning is logged before it runs, and install_packages() is called with
assume_yes=False to keep it that way.
"""

# Lets type hints such as `str | None` work on Python versions older than 3.10.
from __future__ import annotations

from collections.abc import Callable

from utils.component_config import config_for, write_component_version
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import run_capture

# The Step's `op` label in plans/steps.py. Every log line from this module uses it.
_OP = "install-nvshmem"

# Exact match, as in bash.
_AZURE_LINUX = "azurelinux3.0"

# NVIDIA's CUDA repo for RHEL 9 on sbsa (Arm server, i.e. aarch64). dnf uses it
# only for the one install command, via --repofrompath=<repo id>,<url>.
_CUDA_REPO_ID = "cuda-rhel9-sbsa"
_CUDA_REPO_URL = "https://developer.download.nvidia.com/compute/cuda/repos/rhel9/sbsa"


# --------------------------------------------------------------------------- #
# Pure helpers: they only transform their input, so tests call them directly.
# --------------------------------------------------------------------------- #
def cuda_major_version(driver_version: str) -> str | None:
    """Return the CUDA major version from a driver version like "13.0".

    Mirrors bash's `echo $CUDA_DRIVER_VERSION | cut -d. -f1`: everything before
    the first ".", or the whole value if there is no ".".

        "13.0" -> "13"        "12" -> "12"

    Returns None unless the result is a plain number. Bash would carry on with
    "" or "null". Requiring digits also means the package name contains no
    characters that grep would treat as regular-expression syntax, so the
    plain substring test in the parsers below matches exactly what grep matches.
    """
    # str.split(".") returns a list of the pieces between dots; [0] is the first.
    major = driver_version.strip().split(".")[0]
    # isdigit() is True only for a non-empty string made of digits.
    # isascii() rules out other Unicode digits such as "١".
    if major.isascii() and major.isdigit():
        return major
    return None


def first_word(values: list[str]) -> str | None:
    """Return the value bash's unquoted `$nvshmem_version` would pass first.

    In bash, `nvshmem_version=$(...)` captures every output line, joined by
    newlines. Bash then passes the variable unquoted:
    `write_component_version "NVSHMEM" $nvshmem_version`. Bash splits it on
    spaces, tabs and newlines and drops empty pieces. write_component_version
    only reads its second argument ($2), so only the first word is recorded.

    str.split() with no arguments splits the same way: on any whitespace, with
    no empty pieces. Returns None when there is no word, where bash would
    record an empty version.
    """
    words = "\n".join(values).split()
    return words[0] if words else None


def parse_dnf_version(dnf_output: str, cuda_major: str) -> str | None:
    """Get the NVSHMEM version from `dnf list installed` output.

    Mirrors: dnf list installed | grep libnvshmem3-cuda-$M | awk '{print $2}'

    A typical matching line (name.arch, version, repo):

        libnvshmem3-cuda-13.aarch64    3.4.5-1    @cuda-rhel9-sbsa

      * grep keeps every line that contains the package name anywhere.
      * awk '{print $2}' prints the second whitespace-separated field, or an
        empty line if there isn't one.
    """
    package = f"libnvshmem3-cuda-{cuda_major}"
    values = []
    # grep reads lines separated by "\n". split("\n") does the same.
    # splitlines() would also split on other characters, such as "\r".
    for line in dnf_output.split("\n"):
        if package in line:
            fields = line.split()
            values.append(fields[1] if len(fields) >= 2 else "")
    return first_word(values)


def parse_apt_version(apt_output: str, cuda_major: str) -> str | None:
    """Get the NVSHMEM version from `apt list --installed` output.

    Mirrors: apt list --installed | grep libnvshmem3-cuda-$M/ | cut -d' ' -f2

    A typical matching line (name/suites version arch [status]):

        libnvshmem3-cuda-13/unknown,now 3.4.5-1 arm64 [installed]

      * The trailing "/" makes grep skip names that only start with the
        package name, e.g. libnvshmem3-cuda-1 does not match cuda-13.
      * cut -d' ' -f2 splits on each single space (two spaces in a row give
        an empty field) and prints field 2. cut prints a line that contains
        no space unchanged.
    """
    package = f"libnvshmem3-cuda-{cuda_major}/"
    values = []
    for line in apt_output.split("\n"):
        if package in line:
            if " " in line:
                # split(" ") splits on each single space, keeping empty fields
                # like cut does. (split() with no argument would merge them.)
                values.append(line.split(" ")[1])
            else:
                values.append(line)
    return first_word(values)


# --------------------------------------------------------------------------- #
# Per-distro handlers. Each installs the packages and returns the installed
# version, or None after logging why it couldn't.
# --------------------------------------------------------------------------- #
def _install_azure_linux(cuda_major: str, env: dict[str, str]) -> str | None:
    """Azure Linux 3.0: dnf install from NVIDIA's RHEL 9 sbsa repo."""
    packages = [
        f"libnvshmem3-cuda-{cuda_major}",
        f"libnvshmem3-devel-cuda-{cuda_major}",
        f"libnvshmem3-static-cuda-{cuda_major}",
    ]
    options = ["--nogpgcheck", f"--repofrompath={_CUDA_REPO_ID},{_CUDA_REPO_URL}"]
    if not install_packages(packages, env, _OP, options=options):
        return None

    # run_capture() returns the exit code and stdout. stderr is logged but not
    # returned, so warnings can't get mixed into the text we parse.
    rc, output = run_capture(["dnf", "list", "installed"], _OP, env=env)
    if rc != 0:
        log_error(_OP, "could not list installed packages with dnf")
        return None
    version = parse_dnf_version(output, cuda_major)
    if version is None:
        log_error(_OP, f"libnvshmem3-cuda-{cuda_major} not found in "
                       "`dnf list installed` output")
    return version


def _install_ubuntu(cuda_major: str, env: dict[str, str]) -> str | None:
    """Ubuntu: apt install the runtime and dev packages."""
    # Kept exactly as bash has it: there is no `-y`. apt normally asks "Do you
    # want to continue? [Y/n]" only when it must also install other packages
    # (dependencies) or remove some. The command gets no input
    # (stdin is /dev/null), so if apt does ask, it reads end-of-file, prints
    # "Abort." and fails; we then return 3. Bash stops the same way when its
    # stdin is at end-of-file. Adding -y would change behavior, so the bash
    # quirk is kept (assume_yes=False) and logged as a warning.
    packages = [f"libnvshmem3-cuda-{cuda_major}", f"libnvshmem3-dev-cuda-{cuda_major}"]
    log_warn(_OP, "Installing without -y, as bash's `apt install` does: if apt "
                  "needs to ask for confirmation it reads no answer and aborts",
             packages=" ".join(packages))
    if not install_packages(packages, env, _OP, assume_yes=False):
        log_error(_OP, "apt could not install the NVSHMEM packages (if apt "
                       "printed 'Abort.', it wanted a confirmation; bash has no -y)")
        return None

    # apt prints "WARNING: apt does not have a stable CLI interface..." to
    # stderr here. run_capture keeps stderr out of the returned output.
    rc, output = run_capture(["apt", "list", "--installed"], _OP, env=env)
    if rc != 0:
        log_error(_OP, "could not list installed packages with apt")
        return None
    version = parse_apt_version(output, cuda_major)
    if version is None:
        log_error(_OP, f"libnvshmem3-cuda-{cuda_major} not found in "
                       "`apt list --installed` output")
    return version


def _handler_for(distribution: str) -> Callable[[str, dict[str, str]], str | None] | None:
    """Pick the install handler for a distro, using bash's tests: an exact
    match for Azure Linux, and "contains 'ubuntu'" for Ubuntu."""
    if distribution == _AZURE_LINUX:
        return _install_azure_linux
    if "ubuntu" in distribution:
        return _install_ubuntu
    return None


def _cuda_driver_version(env: dict[str, str]) -> str | None:
    """Return the cuda config's driver.version, or None if it isn't a string."""
    config = config_for("cuda", env)
    # isinstance() guards each level, so a missing or oddly shaped config
    # gives None instead of raising an exception.
    driver = config.get("driver") if isinstance(config, dict) else None
    version = driver.get("version") if isinstance(driver, dict) else None
    return version if isinstance(version, str) else None


def install(env: dict[str, str]) -> int:
    """Install NVSHMEM for the image's CUDA major version and record its version.

    `env` is the build environment made by component_env() in build_config.py.
    Uses DISTRIBUTION plus the values config_for() reads to find the cuda config.

    Returns 0 on success and 3 on failure.
    """
    # ---- 1. CUDA major version (from the cuda config, as in bash) ---------
    driver_version = _cuda_driver_version(env)
    if driver_version is None:
        log_error(_OP, "could not resolve cuda driver.version from versions.json")
        return 3
    cuda_major = cuda_major_version(driver_version)
    if cuda_major is None:
        log_error(_OP, f"cuda driver.version {driver_version!r} has no numeric "
                       "major version")
        return 3

    # ---- 2. Install with the distro's package manager ---------------------
    distribution = env.get("DISTRIBUTION", "")
    handler = _handler_for(distribution)
    if handler is None:
        # Bash would skip the install and record an empty version.
        log_error(_OP, f"NVSHMEM install is not supported on {distribution!r}")
        return 3

    log_info(_OP, f"Installing NVSHMEM for CUDA {cuda_major} on {distribution}")
    version = handler(cuda_major, env)
    if version is None:
        return 3            # the handler has already logged why

    # ---- 3. Record the version --------------------------------------------
    try:
        write_component_version("NVSHMEM", version)
    except OSError as exc:
        log_error(_OP, f"could not record NVSHMEM={version}: {exc}")
        return 3
    log_info(_OP, f"Recorded NVSHMEM version {version}")
    return 0
