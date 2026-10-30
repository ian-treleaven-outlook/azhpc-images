"""install_nvbandwidth_tool.py — Python port of components/install_nvbandwidth_tool.sh.

What this step does
-------------------
nvbandwidth is NVIDIA's GPU memory-bandwidth benchmark. The build plan runs
this step for GB200 images (Ubuntu 24.04 and Azure Linux 3.0, aarch64), but
every distro branch of the bash script is mirrored.

  1. Create /opt/nvidia/nvbandwidth.
  2. Install Boost program_options (and cmake on Azure Linux aarch64) with the
     distro's package manager. Distros the bash script has no branch for
     install nothing and carry on.
  3. Clone the source at the configured ref (versions.json "ref", else
     "branch", else the tag v<version>), check out the pinned "commit" (if
     any), and record where the source came from as NVBANDWIDTH_REPO_COMMIT
     and NVBANDWIDTH_REPO_REF (write_repo_source_info; upstream #650,
     "build evidence").
  4. Load the HPC-X MPI module, then build with cmake (multi-node support,
     -DMULTINODE=1) and make.
  5. Move the built `nvbandwidth` binary into /opt/nvidia/nvbandwidth, delete
     the source and record the NVBANDWIDTH version.

The original bash, for comparison:

    dest_dir=/opt/nvidia/nvbandwidth
    mkdir -p $dest_dir

    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        apt install -y libboost-program-options-dev
    elif [[ $DISTRIBUTION == almalinux* ]]; then
        dnf -y install boost-devel
    elif [[ $DISTRIBUTION == "azurelinux3.0" && $ARCHITECTURE != "aarch64" ]]; then
        dnf -y install boost-devel
    elif [[ $DISTRIBUTION == "azurelinux3.0" && $ARCHITECTURE == "aarch64" ]]; then
        dnf install -y boost-devel boost-program-options
        dnf install -y cmake
    fi

    nvbandwidth_metadata=$(get_component_config "nvbandwidth")
    NVBANDWIDTH_VERSION=$(jq -r '.version' <<< $nvbandwidth_metadata)
    NVBANDWIDTH_DOWNLOAD_URL=$(jq -r '.url' <<< $nvbandwidth_metadata)
    NVBANDWIDTH_COMMIT=$(jq -r '.commit // empty' <<< $nvbandwidth_metadata)
    NVBANDWIDTH_REF=$(jq -r '.ref // .branch // empty' <<< $nvbandwidth_metadata)
    NVBANDWIDTH_REF=${NVBANDWIDTH_REF:-v${NVBANDWIDTH_VERSION}}

    git clone --branch ${NVBANDWIDTH_REF} ${NVBANDWIDTH_DOWNLOAD_URL}

    pushd nvbandwidth
    git checkout ${NVBANDWIDTH_COMMIT}
    write_repo_source_info "NVBANDWIDTH" . \\
        --commit "${NVBANDWIDTH_COMMIT}" \\
        --ref "${NVBANDWIDTH_REF}"
    source /etc/profile.d/modules.sh
    module load mpi/hpcx
    cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DMULTINODE=1 .
    make
    mv ./nvbandwidth $dest_dir
    popd

    rm -rf ./nvbandwidth
    write_component_version "NVBANDWIDTH" ${NVBANDWIDTH_VERSION}

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-nvbandwidth" Step. Deleting `action=` from that Step makes the build
run the bash script again, which is why the .sh file is kept.

`module` is a shell function, not a program, so the module load, cmake and
make run together in one `bash -c` script with `set -e` (the same approach as
install_nvloom.py). Everything else is plain Python or a single command run
with exec_program.

install() returns 0 for success or 3 for failure. It logs problems and returns
3; it does not let exceptions escape to the build loop.

Deliberate differences from bash
--------------------------------
  * The dependency installs go through install_packages(), which normalises
    the spellings: `apt install -y` -> `apt-get install -y -o Dpkg::Use-Pty=0`,
    `dnf -y install` -> `dnf install -y`.
  * Checks first: a missing version or url fails before anything is
    installed. Bash would install the dependencies, then try
    `git clone --branch vnull null`.
  * No pinned commit (versions.json has no "commit"): bash runs a bare
    `git checkout`, which changes nothing, and builds the tip of the ref. The
    port skips that no-op command and logs a warning; write_repo_source_info
    then reads the commit from git, as in bash.
  * Work directory: bash clones into its current directory (the distro build
    dir). We clone into _WORK_DIR, delete any copy left by an earlier failed
    run first (in bash, that leftover would make `git clone` fail), and delete
    it after a successful install (bash's `rm -rf ./nvbandwidth`; a failure to
    delete is ignored rather than failing the step).
  * The checkout and write_repo_source_info run in the source directory, as
    in bash; the port passes its absolute path instead of ".".

Bash quirk, kept and warned about
---------------------------------
Distros without a dependency branch (Rocky, RHEL, anything else) install no
Boost but still build, so cmake/make may fail. A warning is logged.
"""

# Lets type hints such as `str | None` work on Python versions older than 3.10.
from __future__ import annotations

import os
import shutil

from utils.component_config import config_for, write_component_version
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import exec_program
from utils.repo_source import write_repo_source_info

# The Step's `op` label in build_config.py. Every log line from this module uses it.
_OP = "install-nvbandwidth"

# Where the nvbandwidth binary is installed (bash's dest_dir).
_DEST_DIR = "/opt/nvidia/nvbandwidth"

# Build area used instead of bash's current directory. `git clone` creates
# the source checkout inside it, in a folder named after the repo.
_WORK_DIR = "/tmp/nvbandwidth-build"
_SOURCE_DIR_NAME = "nvbandwidth"

# The binary the build produces.
_BINARY = "nvbandwidth"

_AZURE_LINUX = "azurelinux3.0"


# --------------------------------------------------------------------------- #
# Pure helpers: no side effects, so tests call them directly.
# --------------------------------------------------------------------------- #
def dependency_packages(distribution: str, arch: str) -> tuple[list[str], ...]:
    """The dependency package lists bash installs, one per install command, in
    order; () if bash has no branch for this distro (it then installs nothing
    and carries on)."""
    if "ubuntu" in distribution:                       # == *"ubuntu"*
        return (["libboost-program-options-dev"],)
    if distribution.startswith("almalinux"):           # == almalinux*
        return (["boost-devel"],)
    if distribution == _AZURE_LINUX and arch != "aarch64":
        return (["boost-devel"],)
    if distribution == _AZURE_LINUX and arch == "aarch64":
        return (["boost-devel", "boost-program-options"], ["cmake"])
    return ()


def configured_ref(config: dict, version: str) -> str:
    """The git ref to clone, as bash resolves it:

        NVBANDWIDTH_REF=$(jq -r '.ref // .branch // empty' <<< $nvbandwidth_metadata)
        NVBANDWIDTH_REF=${NVBANDWIDTH_REF:-v${NVBANDWIDTH_VERSION}}

    jq's `//` skips a missing (null) or false value, but keeps "" (so an empty
    "ref" hides "branch"); bash's `:-` then turns "" into v<version>. Checked
    against real jq/bash in WSL.
    """
    ref = ""
    for key in ("ref", "branch"):
        value = config.get(key)
        if value is not None and value is not False:
            ref = str(value)
            break
    return ref or f"v{version}"


def configured_commit(config: dict) -> str:
    """bash: $(jq -r '.commit // empty' <<< $nvbandwidth_metadata); "" if unset."""
    value = config.get("commit")
    return "" if value is None or value is False else str(value)


def build_script() -> str:
    """The bash script that loads HPC-X and builds nvbandwidth.

    It runs in the nvbandwidth source directory. These commands share one
    shell because `module load` only affects the shell that runs it. With
    `set -e`, the script stops at the first failing command, as bash's
    `set -e` does.
    """
    return (
        "set -e\n"
        "source /etc/profile.d/modules.sh\n"
        "module load mpi/hpcx\n"
        "cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DMULTINODE=1 .\n"
        "make\n"
    )


# --------------------------------------------------------------------------- #
# Steps. Each returns True on success, or False after logging why not.
# --------------------------------------------------------------------------- #
def _run_all(commands, env: dict[str, str], cwd: str | None = None) -> bool:
    """Run commands in order, stopping at the first failure (like `set -e`)."""
    for command in commands:
        rc = exec_program(command, _OP, cwd=cwd, env=env)
        if rc != 0:
            log_error(_OP, f"`{' '.join(command)}` failed with exit code {rc}",
                      exit_code=rc)
            return False
    return True


def _clone(url: str, ref: str, env: dict[str, str]) -> str | None:
    """Clone nvbandwidth at `ref` into a fresh _WORK_DIR.

    Returns the source directory, or None on failure.
    """
    try:
        # Equivalent of `rm -rf`, then `mkdir -p`: start from an empty work dir.
        shutil.rmtree(_WORK_DIR, ignore_errors=True)
        os.makedirs(_WORK_DIR, exist_ok=True)
    except OSError as exc:
        log_error(_OP, f"could not create {_WORK_DIR}: {exc}", path=_WORK_DIR)
        return None

    # Same arguments, in the same order, as bash's git clone.
    if not _run_all([["git", "clone", "--branch", ref, url]], env, cwd=_WORK_DIR):
        return None

    source_dir = os.path.join(_WORK_DIR, _SOURCE_DIR_NAME)
    # Bash's `pushd nvbandwidth` would fail here if the URL named another repo.
    if not os.path.isdir(source_dir):
        log_error(_OP, f"git clone did not create {source_dir}", path=source_dir,
                  source=url)
        return None
    return source_dir


def _checkout_and_record(source_dir: str, commit: str, ref: str,
                         env: dict[str, str]) -> bool:
    """git checkout <commit>, then write_repo_source_info (bash's order)."""
    if commit:
        if not _run_all([["git", "checkout", commit]], env, cwd=source_dir):
            return False
    else:
        log_warn(_OP, "versions.json pins no nvbandwidth commit: building the tip of "
                      f"{ref} (bash runs a bare `git checkout`, which changes nothing)",
                 ref=ref)
    try:
        recorded = write_repo_source_info("NVBANDWIDTH", source_dir, _OP,
                                          commit=commit or None, ref=ref, env=env)
    except OSError as exc:
        log_error(_OP, f"could not record the nvbandwidth source: {exc}",
                  commit=commit, ref=ref)
        return False
    if not recorded:
        log_error(_OP, "could not record the nvbandwidth source (git failed)",
                  commit=commit, ref=ref)
        return False
    return True


def _install_binary(source_dir: str) -> bool:
    """`mv ./nvbandwidth $dest_dir`: _DEST_DIR is a directory, so mv puts the
    file inside it and replaces any existing copy, as shutil.move() to the
    full target path does."""
    built = os.path.join(source_dir, _BINARY)
    target = os.path.join(_DEST_DIR, _BINARY)
    action = "replaced" if os.path.lexists(target) else "created"
    try:
        shutil.move(built, target)
    except OSError as exc:
        log_error(_OP, f"could not move {built} to {target}: {exc}",
                  source=built, path=target)
        return False
    log_info(_OP, f"Installed {target}", path=target, action=action)
    return True


def install(env: dict[str, str]) -> int:
    """Build nvbandwidth from source, install it, and record its version.

    `env` is the build environment made by component_env() in build_config.py.
    Uses DISTRIBUTION, ARCHITECTURE and the values config_for() reads to find
    the nvbandwidth config.

    Returns 0 on success and 3 on failure.
    """
    # ---- Checks before any changes are made -------------------------------
    config = config_for("nvbandwidth", env)
    config = config if isinstance(config, dict) else {}
    missing = [key for key in ("version", "url") if not config.get(key)]
    if missing:
        log_error(_OP, f"could not resolve nvbandwidth {', '.join(missing)} from "
                       "versions.json", missing=missing,
                  distro=env.get("DISTRIBUTION", ""),
                  node_type=env.get("TARGET_NODE_TYPE", ""))
        return 3
    version = str(config["version"])
    url = str(config["url"])
    commit = configured_commit(config)
    ref = configured_ref(config, version)

    distribution = env.get("DISTRIBUTION", "")
    arch = env.get("ARCHITECTURE", "")
    log_info(_OP, f"Installing nvbandwidth {version} on {distribution}",
             version=version, source=url, ref=ref, commit=commit or "(not pinned)",
             distro=distribution, arch=arch, node_type=env.get("TARGET_NODE_TYPE", ""),
             dest=_DEST_DIR, work_dir=_WORK_DIR)

    # ---- 1. Destination directory (bash's `mkdir -p $dest_dir`) -----------
    try:
        os.makedirs(_DEST_DIR, exist_ok=True)
    except OSError as exc:
        log_error(_OP, f"could not create {_DEST_DIR}: {exc}", path=_DEST_DIR)
        return 3

    # ---- 2. Build dependencies --------------------------------------------
    deps = dependency_packages(distribution, arch)
    if deps:
        log_info(_OP, f"Installing the build dependencies for {distribution}/{arch}",
                 distro=distribution, arch=arch,
                 packages=" ".join(package for packages in deps for package in packages))
        # One install per bash command, stopping at the first failure.
        if not all(install_packages(packages, env, _OP) for packages in deps):
            return 3
    else:
        # BASH QUIRK (kept): no branch for this distro, so no Boost; the build
        # below may then fail.
        log_warn(_OP, f"install_nvbandwidth_tool.sh installs no dependencies on "
                      f"{distribution!r}/{arch!r}; building anyway (bash behaviour, "
                      "kept)", distro=distribution, arch=arch)

    # ---- 3. Source ---------------------------------------------------------
    source_dir = _clone(url, ref, env)
    if source_dir is None:
        return 3
    if not _checkout_and_record(source_dir, commit, ref, env):
        return 3

    # ---- 4. Build ----------------------------------------------------------
    if not _run_all([["bash", "-c", build_script()]], env, cwd=source_dir):
        return 3

    # ---- 5. Install the binary, clean up, record --------------------------
    if not _install_binary(source_dir):
        return 3

    # Bash's `rm -rf ./nvbandwidth`. A leftover copy is harmless (the next run
    # deletes it before cloning), so a failed cleanup doesn't fail the step.
    shutil.rmtree(_WORK_DIR, ignore_errors=True)

    try:
        write_component_version("NVBANDWIDTH", version)
    except OSError as exc:
        log_error(_OP, f"could not record NVBANDWIDTH={version}: {exc}")
        return 3
    log_info(_OP, f"Recorded NVBANDWIDTH version {version}", component="NVBANDWIDTH",
             version=version)
    log_info(_OP, f"nvbandwidth {version} installed to {_DEST_DIR}", version=version,
             commit=commit or "(not pinned)", ref=ref, dest=_DEST_DIR)
    return 0
