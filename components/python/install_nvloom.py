"""install_nvloom.py — Python port of components/install_nvloom.sh.

What this step does
-------------------
nvloom is NVIDIA's benchmark for multi-node NVLink (MNNVL) bandwidth. It only
makes sense on SKUs with rack-scale NVLink, so the build plan runs this step
for GB200 images only (Ubuntu 24.04 and Azure Linux 3.0, aarch64).

  1. Install the build dependencies (compiler toolchain, Boost
     program_options, cmake) with the distro's package manager.
  2. Clone the nvloom source at the pinned tag (v<version> from versions.json).
  3. On Azure Linux, apply the repo's azurelinux_nvloom.patch. It stops
     nvloom linking Boost statically, which fails on Azure Linux.
  4. Load the HPC-X MPI module, then build with cmake + make for the
     Blackwell GPU architecture (CUDA arch 100).
  5. Move the built `nvloom_cli` binary into /opt/nvidia/nvloom and record
     the NVLOOM version.

The original bash, for comparison (the Ubuntu branch is the same as the Azure
Linux one, but with apt packages and no patch):

    dest_dir=/opt/nvidia/nvloom
    mkdir -p $dest_dir

    source /etc/profile.d/modules.sh
    module load mpi/hpcx

    nvloom_metadata=$(get_component_config "nvloom")
    NVLOOM_VERSION=$(jq -r '.version' <<< $nvloom_metadata)
    NVLOOM_DOWNLOAD_URL=$(jq -r '.url' <<< $nvloom_metadata)

    if [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
        dnf install -y build-essential
        dnf install -y boost-devel boost-program-options
        dnf install -y cmake

        git clone $NVLOOM_DOWNLOAD_URL --branch v$NVLOOM_VERSION
        pushd nvloom
        git apply /home/hpcuser/azhpc-images/distros/azurelinux3.0/azurelinux_nvloom.patch
        cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="100" .
        make -j $(nproc)
        mv nvloom_cli $dest_dir
        popd
    elif [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        apt install -y build-essential
        apt install -y libboost-program-options-dev
        apt install -y cmake
        ... same clone / cmake / make / mv, without the patch ...
    fi

    module unload mpi/hpcx
    rm -rf ./nvloom
    write_component_version "NVLOOM" ${NVLOOM_VERSION}

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-nvloom" Step. When a Step has an action, the build calls the Python
function instead of running the bash script. Deleting `action=` from that Step
makes the build run the bash script again, which is why the .sh file is kept.

`module` is a shell function, not a program. A `module load` only changes the
shell it runs in, so Python can't run it as a separate command and keep the
effect. The module load, cmake and make therefore run together in one
`bash -c` script with `set -e` (the same approach as install_mpifileutils.py).
Everything else is plain Python or a single command run with exec_program.

install() returns 0 for success or 3 for failure. It logs problems and returns
3; it does not let exceptions escape to the build loop.

Deliberate differences from bash
--------------------------------
  * Patch path: bash applies the patch from /home/hpcuser/azhpc-images/...,
    which only exists when the repo is checked out in that one user's home.
    We use <TOP_DIR>/distros/azurelinux3.0/azurelinux_nvloom.patch, and fail
    with 3 if TOP_DIR is unset or the patch file is missing.
  * Unsupported distros: bash skips the build but still records NVLOOM, so the
    image would claim a tool it doesn't have. We fail with 3.
  * Checks first: the config, distro and patch are checked before anything is
    installed. Bash finds these problems later (e.g. `git clone --branch vnull`).
  * Work directory: bash clones into its current directory (the distro build
    dir). We clone into _WORK_DIR, delete any copy left by an earlier failed
    run first (in bash, that leftover would make `git clone` fail), and delete
    it after a successful install (bash's `rm -rf ./nvloom`; a failure to
    delete is ignored rather than failing the step).
  * Module load timing: bash loads mpi/hpcx at the very top. Here it is
    loaded just before cmake, which is the only thing that needs it, so a
    broken module setup is found after the dependencies are installed rather
    than before.
"""

# Lets type hints such as `str | None` work on Python versions older than 3.10.
from __future__ import annotations

import os
import shutil

from utils.component_config import config_for, write_component_version
from utils.logger import log_error, log_info
from utils.process import exec_program

# The Step's `op` label in build_config.py. Every log line from this module uses it.
_OP = "install-nvloom"

# Where the nvloom_cli binary is installed (bash's dest_dir).
_DEST_DIR = "/opt/nvidia/nvloom"

# Build area used instead of bash's current directory. `git clone` creates
# the source checkout inside it, in a folder named after the repo ("nvloom").
_WORK_DIR = "/tmp/nvloom-build"
_SOURCE_DIR_NAME = "nvloom"

# The binary the build produces.
_BINARY = "nvloom_cli"

_AZURE_LINUX = "azurelinux3.0"

# The patch's location inside the repo checkout (joined onto TOP_DIR).
_PATCH_IN_REPO = os.path.join("distros", "azurelinux3.0", "azurelinux_nvloom.patch")

# Dependency installs, one command each, in bash's order. Bash runs with
# `set -e`, so the first failing command stops the step. install() does the
# same by stopping at the first non-zero exit code. (We don't use
# PackageInstaller: it carries on after a failed package and picks the
# package manager itself, while bash names dnf/apt explicitly.)
_AZURE_LINUX_DEPS = (
    ["dnf", "install", "-y", "build-essential"],
    ["dnf", "install", "-y", "boost-devel", "boost-program-options"],
    ["dnf", "install", "-y", "cmake"],
)
_UBUNTU_DEPS = (
    ["apt", "install", "-y", "build-essential"],
    ["apt", "install", "-y", "libboost-program-options-dev"],
    ["apt", "install", "-y", "cmake"],
)


# --------------------------------------------------------------------------- #
# Pure helpers: no side effects, so tests call them directly.
# --------------------------------------------------------------------------- #
def dependency_commands(distribution: str) -> tuple[list[str], ...] | None:
    """Return the dependency install commands for a distro, or None if bash
    has no branch for it.

    Uses bash's tests: an exact match for Azure Linux, and "contains 'ubuntu'"
    for Ubuntu.
    """
    if distribution == _AZURE_LINUX:
        return _AZURE_LINUX_DEPS
    if "ubuntu" in distribution:
        return _UBUNTU_DEPS
    return None


def patch_path(top_dir: str) -> str:
    """Where the Azure Linux nvloom patch lives in this repo checkout.

    Replaces bash's hardcoded /home/hpcuser/azhpc-images/... path.
    """
    return os.path.join(top_dir, _PATCH_IN_REPO)


def build_script() -> str:
    """The bash script that loads HPC-X and builds nvloom.

    It runs in the nvloom source directory. These commands share one shell
    because `module load` only affects the shell that runs it. With `set -e`,
    the script stops at the first failing command, as bash's `set -e` does.
    `module unload` is kept for parity with bash; the shell exits right after
    anyway.
    """
    return (
        "set -e\n"
        "source /etc/profile.d/modules.sh\n"
        "module load mpi/hpcx\n"
        'cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc '
        '-DCMAKE_CUDA_ARCHITECTURES="100" .\n'
        "make -j $(nproc)\n"
        "module unload mpi/hpcx\n"
    )


# --------------------------------------------------------------------------- #
# Steps. Each returns True on success, or False after logging why not.
# --------------------------------------------------------------------------- #
def _run_all(commands, env: dict[str, str], cwd: str | None = None) -> bool:
    """Run commands in order, stopping at the first failure (like `set -e`)."""
    for command in commands:
        rc = exec_program(command, _OP, cwd=cwd, env=env)
        if rc != 0:
            log_error(_OP, f"`{' '.join(command)}` failed with exit code {rc}")
            return False
    return True


def _clone(url: str, version: str, env: dict[str, str]) -> str | None:
    """Clone nvloom at tag v<version> into a fresh _WORK_DIR.

    Returns the source directory, or None on failure.
    """
    try:
        # Equivalent of `rm -rf`, then `mkdir -p`: start from an empty work dir.
        shutil.rmtree(_WORK_DIR, ignore_errors=True)
        os.makedirs(_WORK_DIR, exist_ok=True)
    except OSError as exc:
        log_error(_OP, f"could not create {_WORK_DIR}: {exc}")
        return None

    # Same arguments, in the same order, as bash's git clone.
    if not _run_all([["git", "clone", url, "--branch", f"v{version}"]],
                    env, cwd=_WORK_DIR):
        return None

    source_dir = os.path.join(_WORK_DIR, _SOURCE_DIR_NAME)
    # Bash's `pushd nvloom` would fail here if the URL named another repo.
    if not os.path.isdir(source_dir):
        log_error(_OP, f"git clone did not create {source_dir}")
        return None
    return source_dir


def _install_binary(source_dir: str) -> bool:
    """Move the built binary into _DEST_DIR, as `mv nvloom_cli $dest_dir` does.

    _DEST_DIR is a directory, so mv puts the file inside it and replaces any
    existing copy. shutil.move() to the full target path does the same.
    """
    built = os.path.join(source_dir, _BINARY)
    target = os.path.join(_DEST_DIR, _BINARY)
    try:
        shutil.move(built, target)
    except OSError as exc:
        log_error(_OP, f"could not move {built} to {target}: {exc}")
        return False
    return True


def install(env: dict[str, str]) -> int:
    """Build nvloom from source, install nvloom_cli, and record its version.

    `env` is the build environment made by component_env() in build_config.py.
    Uses DISTRIBUTION, TOP_DIR (Azure Linux patch) and the values config_for()
    reads to find the nvloom config.

    Returns 0 on success and 3 on failure.
    """
    # ---- 1. Checks before any changes are made ----------------------------
    config = config_for("nvloom", env)
    # isinstance() guards against a missing (None) or oddly shaped config.
    if not isinstance(config, dict) or not config.get("version") or not config.get("url"):
        log_error(_OP, "could not resolve nvloom version and url from versions.json")
        return 3
    version = config["version"]
    url = config["url"]

    distribution = env.get("DISTRIBUTION", "")
    deps = dependency_commands(distribution)
    if deps is None:
        # Bash would skip the build but still record NVLOOM.
        log_error(_OP, f"nvloom build is not supported on {distribution!r}")
        return 3

    patch = None
    if distribution == _AZURE_LINUX:
        top_dir = env.get("TOP_DIR", "")
        if not top_dir:
            log_error(_OP, "TOP_DIR is not set; cannot find the nvloom patch")
            return 3
        patch = patch_path(top_dir)
        if not os.path.isfile(patch):
            log_error(_OP, f"nvloom patch not found: {patch}")
            return 3

    log_info(_OP, f"Installing nvloom {version} on {distribution}")

    # ---- 2. Destination directory (bash's `mkdir -p $dest_dir`) ----------
    try:
        os.makedirs(_DEST_DIR, exist_ok=True)
    except OSError as exc:
        log_error(_OP, f"could not create {_DEST_DIR}: {exc}")
        return 3

    # ---- 3. Build dependencies -------------------------------------------
    if not _run_all(deps, env):
        return 3

    # ---- 4. Source, patch, build ------------------------------------------
    source_dir = _clone(url, version, env)
    if source_dir is None:
        return 3

    if patch is not None and not _run_all([["git", "apply", patch]], env, cwd=source_dir):
        return 3

    if not _run_all([["bash", "-c", build_script()]], env, cwd=source_dir):
        return 3

    # ---- 5. Install the binary, clean up, record --------------------------
    if not _install_binary(source_dir):
        return 3

    # Bash's `rm -rf ./nvloom`. A leftover copy is harmless (the next run
    # deletes it before cloning), so a failed cleanup doesn't fail the step.
    shutil.rmtree(_WORK_DIR, ignore_errors=True)

    try:
        write_component_version("NVLOOM", version)
    except OSError as exc:
        log_error(_OP, f"could not record NVLOOM={version}: {exc}")
        return 3
    log_info(_OP, f"nvloom {version} installed to {_DEST_DIR}")
    return 0
