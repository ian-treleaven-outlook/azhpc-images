"""install_nvshmem.py — Python port of components/install_nvshmem.sh.

What this step does
-------------------
NVSHMEM is NVIDIA's library for GPU-to-GPU shared memory across a node or an
NVLink domain. This step installs the NVSHMEM packages for the image's CUDA
major version and records the installed version as NVSHMEM. The build plan
runs it only for GB200 images (Ubuntu 24.04 and Azure Linux 3.0, aarch64).

  * CUDA major version: cuda driver.version up to the first "." ("13.0" -> "13";
    bash: `echo $CUDA_DRIVER_VERSION | cut -d. -f1`).
  * The packages, install options and the "what got installed" query differ
    per distro; they are the _RECIPES table below:
      - Azure Linux 3.0: dnf installs the runtime, devel and static packages
        from NVIDIA's RHEL 9 sbsa CUDA repo, added for that command only
        (--repofrompath); the version is read from `dnf list installed`.
      - Ubuntu: apt installs the runtime and dev packages; the version is read
        from `apt list --installed`.

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets install() as the `action=` of the "install-nvshmem"
Step; deleting `action=` runs the bash script again, which is why the .sh file
is kept. install() returns 0 or 3 and never lets an exception escape.

Deliberate differences from bash
--------------------------------
Each is a case where bash carries on and records an empty or meaningless
version; the port stops with 3 instead:

  * A missing cuda driver.version, or one whose major part is not a number
    (bash would install "libnvshmem3-cuda-null").
  * A distro that is neither azurelinux3.0 (an exact match, as in bash) nor
    Ubuntu (bash installs nothing and records an empty version).
  * `dnf list installed` / `apt list --installed` failing or not listing the
    package. Bash's pipeline hides both from `set -e`.

Kept bash quirk
---------------
The Ubuntu `apt install` has no `-y`. apt asks "Do you want to continue?"
only when it must install other packages too or remove some; with no input
(stdin is /dev/null) it then reads end-of-file, prints "Abort." and fails, as
bash does when its stdin is at end-of-file. install_packages() is called with
assume_yes=False to keep that, and a warning is logged before apt runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from utils.component_config import record_version, require_config
from utils.distro import Distro
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import run_capture

# The Step's `op` label in plans/steps.py.
_OP = "install-nvshmem"

# Bash matches Azure Linux exactly, but Ubuntu by substring (== *"ubuntu"*).
_AZURE_LINUX = "azurelinux3.0"

# NVIDIA's CUDA repo for RHEL 9 on sbsa (Arm server, i.e. aarch64).
_CUDA_REPO_ID = "cuda-rhel9-sbsa"
_CUDA_REPO_URL = "https://developer.download.nvidia.com/compute/cuda/repos/rhel9/sbsa"


# --------------------------------------------------------------------------- #
# Pure helpers: they only transform their input, so tests call them directly.
# --------------------------------------------------------------------------- #
def cuda_major_version(driver_version: str) -> str | None:
    """The CUDA major version of a driver version like "13.0", or None.

    Mirrors `cut -d. -f1`: everything before the first ".", or the whole
    value. None unless that is a plain ASCII number (bash would carry on with
    "" or "null"). Digits only also means the package name holds nothing grep
    treats as regex syntax, so the parsers' substring tests match grep exactly.
    """
    major = driver_version.strip().split(".")[0]
    if major.isascii() and major.isdigit():
        return major
    return None


def first_word(values: list[str]) -> str | None:
    """The value bash's unquoted `$nvshmem_version` passes first, or None.

    `nvshmem_version=$(...)` holds every output line; passed unquoted it is
    split on blanks and newlines, and write_component_version only reads $2.
    """
    words = "\n".join(values).split()
    return words[0] if words else None


def parse_dnf_version(dnf_output: str, cuda_major: str) -> str | None:
    """dnf list installed | grep libnvshmem3-cuda-$M | awk '{print $2}'

        libnvshmem3-cuda-13.aarch64    3.4.5-1    @cuda-rhel9-sbsa   ->  3.4.5-1

    grep keeps every line containing the name; awk prints field 2 (or an
    empty line). Lines are split on "\\n" only, as grep does.
    """
    package = f"libnvshmem3-cuda-{cuda_major}"
    values = []
    for line in dnf_output.split("\n"):
        if package in line:
            fields = line.split()
            values.append(fields[1] if len(fields) >= 2 else "")
    return first_word(values)


def parse_apt_version(apt_output: str, cuda_major: str) -> str | None:
    """apt list --installed | grep libnvshmem3-cuda-$M/ | cut -d' ' -f2

        libnvshmem3-cuda-13/unknown,now 3.4.5-1 arm64 [installed]   ->  3.4.5-1

    The trailing "/" stops prefix matches (cuda-1 vs cuda-13). cut splits on
    each single space (two in a row give an empty field) and prints a line
    with no space unchanged.
    """
    package = f"libnvshmem3-cuda-{cuda_major}/"
    values = []
    for line in apt_output.split("\n"):
        if package in line:
            values.append(line.split(" ")[1] if " " in line else line)
    return first_word(values)


# --------------------------------------------------------------------------- #
# Per-distro data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Recipe:
    """How one distro installs NVSHMEM and reads back the installed version."""
    packages: tuple[str, ...]             # "{major}" is the CUDA major version
    options: tuple[str, ...]
    assume_yes: bool
    list_command: tuple[str, ...]
    parse: Callable[[str, str], str | None]


_RECIPES = {
    "azurelinux": _Recipe(
        packages=("libnvshmem3-cuda-{major}", "libnvshmem3-devel-cuda-{major}",
                  "libnvshmem3-static-cuda-{major}"),
        options=("--nogpgcheck", f"--repofrompath={_CUDA_REPO_ID},{_CUDA_REPO_URL}"),
        assume_yes=True,
        list_command=("dnf", "list", "installed"),
        parse=parse_dnf_version),
    "ubuntu": _Recipe(
        packages=("libnvshmem3-cuda-{major}", "libnvshmem3-dev-cuda-{major}"),
        options=(),
        assume_yes=False,                 # BASH QUIRK (kept): `apt install` without -y
        list_command=("apt", "list", "--installed"),
        parse=parse_apt_version),
}


def _recipe_for(distro: Distro) -> _Recipe | None:
    if distro.is_azurelinux and distro.name != _AZURE_LINUX:
        return None
    return _RECIPES.get(distro.family)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def _install_packages(recipe: _Recipe, packages: list[str], env: dict[str, str]) -> bool:
    if not recipe.assume_yes:
        log_warn(_OP, "Installing without -y, as bash's `apt install` does: if apt "
                      "needs to ask for confirmation it reads no answer and aborts",
                 packages=" ".join(packages))
    if install_packages(packages, env, _OP, options=list(recipe.options),
                        assume_yes=recipe.assume_yes):
        return True
    if not recipe.assume_yes:
        log_error(_OP, "apt could not install the NVSHMEM packages (if apt "
                       "printed 'Abort.', it wanted a confirmation; bash has no -y)")
    return False


def _installed_version(recipe: _Recipe, cuda_major: str,
                       env: dict[str, str]) -> str | None:
    """The installed version from the package manager's list (run_capture keeps
    stderr, e.g. apt's "no stable CLI" warning, out of the parsed text)."""
    rc, output = run_capture(list(recipe.list_command), _OP, env=env)
    if rc != 0:
        return None
    version = recipe.parse(output, cuda_major)
    if version is None:
        log_error(_OP, f"libnvshmem3-cuda-{cuda_major} not found in "
                       f"`{' '.join(recipe.list_command)}` output")
    return version


def install(env: dict[str, str]) -> int:
    """Install NVSHMEM for the image's CUDA major version and record its version.

    Returns 0 on success and 3 on failure.
    """
    cfg = require_config("cuda", env, _OP, "driver.version")
    if cfg is None:
        return 3
    driver_version = str(cfg["driver"]["version"])
    cuda_major = cuda_major_version(driver_version)
    if cuda_major is None:
        log_error(_OP, f"cuda driver.version {driver_version!r} has no numeric "
                       "major version")
        return 3

    distro = Distro.from_env(env)
    recipe = _recipe_for(distro)
    if recipe is None:
        log_error(_OP, f"NVSHMEM install is not supported on {distro.name!r}")
        return 3

    packages = [package.format(major=cuda_major) for package in recipe.packages]
    log_info(_OP, f"Installing NVSHMEM for CUDA {cuda_major} on {distro}",
             cuda_major=cuda_major, packages=" ".join(packages))
    if not _install_packages(recipe, packages, env):
        return 3
    version = _installed_version(recipe, cuda_major, env)
    if version is None or not record_version("NVSHMEM", version, _OP):
        return 3
    return 0