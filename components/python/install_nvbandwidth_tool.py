"""install_nvbandwidth_tool.py — Python port of components/install_nvbandwidth_tool.sh.

What this step does
-------------------
nvbandwidth is NVIDIA's GPU memory-bandwidth benchmark. The build plan runs
this step for GB200 images (Ubuntu 24.04 and Azure Linux 3.0, aarch64), but
every distro branch of the bash script is mirrored.

  1. Create /opt/nvidia/nvbandwidth.
  2. Install Boost program_options (and cmake on Azure Linux aarch64) with the
     distro's package manager (dependency_packages()). Distros the bash
     script has no branch for install nothing and carry on.
  3. Clone the source at the configured ref (versions.json "ref", else
     "branch", else the tag v<version>), check out the pinned "commit" (if
     any), and record NVBANDWIDTH_REPO_COMMIT / NVBANDWIDTH_REPO_REF
     (write_repo_source_info; upstream #650, "build evidence").
  4. With the HPC-X MPI module loaded, build with cmake (multi-node support,
     -DMULTINODE=1) and make, in one shell: `module load` only affects the
     shell it runs in (run_script).
  5. Move the built `nvbandwidth` binary into /opt/nvidia/nvbandwidth, delete
     the source and record NVBANDWIDTH.

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets install() as the "install-nvbandwidth" Step's
`action=`; deleting `action=` runs the bash script again, which is why it is
kept.

Bash quirk, kept and warned about
---------------------------------
Distros without a dependency branch (Rocky, RHEL, anything else) install no
Boost but still build, so cmake/make may fail. A warning is logged.

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
    it after a successful install (bash's `rm -rf ./nvbandwidth`, via
    remove_paths, so a failure to delete doesn't fail the step).
  * write_repo_source_info gets the checkout's absolute path instead of ".".
"""

from __future__ import annotations

import os

from utils.component_config import record_version, require_config
from utils.distro import Distro
from utils.files import fresh_dir, make_dirs, move_file, remove_paths
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import run, run_script
from utils.repo_source import configured_ref, git_clone, write_repo_source_info

_OP = "install-nvbandwidth"

# Where the nvbandwidth binary is installed (bash's dest_dir).
_DEST_DIR = "/opt/nvidia/nvbandwidth"

# Build area used instead of bash's current directory. `git clone` creates
# the source checkout inside it, in a folder named after the repo.
_WORK_DIR = "/tmp/nvbandwidth-build"
_SOURCE_DIR_NAME = "nvbandwidth"
_BINARY = "nvbandwidth"

_AZURE_LINUX = "azurelinux3.0"

# Run in the source directory with mpi/hpcx loaded.
_HPCX_MODULE = "mpi/hpcx"
_BUILD_LINES = [
    "cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DMULTINODE=1 .",
    "make",
]


# --------------------------------------------------------------------------- #
# Pure helpers: no side effects, so tests call them directly.
# --------------------------------------------------------------------------- #
def dependency_packages(distro: Distro, arch: str) -> tuple[list[str], ...]:
    """The dependency package lists bash installs, one per install command, in
    order; () if bash has no branch for this distro (it then installs nothing
    and carries on).

    Bash's tests: contains "ubuntu"; starts with "almalinux"; exactly
    "azurelinux3.0", split on whether ARCHITECTURE is aarch64.
    """
    if distro.is_ubuntu:
        return (["libboost-program-options-dev"],)
    if distro.is_almalinux:
        return (["boost-devel"],)
    if distro.name == _AZURE_LINUX:
        if arch == "aarch64":
            return (["boost-devel", "boost-program-options"], ["cmake"])
        return (["boost-devel"],)
    return ()


def configured_commit(config: dict) -> str:
    """bash: $(jq -r '.commit // empty' <<< $nvbandwidth_metadata); "" if unset."""
    value = config.get("commit")
    return "" if value is None or value is False else str(value)


# --------------------------------------------------------------------------- #
# The steps.
# --------------------------------------------------------------------------- #
def _install_dependencies(env: dict[str, str], distro: Distro, arch: str) -> bool:
    """Bash's per-distro dependency installs, stopping at the first failure."""
    deps = dependency_packages(distro, arch)
    if not deps:
        # Kept bash quirk: no branch for this distro, so no Boost; the build
        # may then fail.
        log_warn(_OP, f"install_nvbandwidth_tool.sh installs no dependencies on "
                      f"{distro.name!r}/{arch!r}; building anyway (bash behaviour, "
                      "kept)", distro=distro.name, arch=arch)
        return True
    log_info(_OP, f"Installing the build dependencies for {distro}/{arch}",
             distro=distro.name, arch=arch,
             packages=" ".join(package for packages in deps for package in packages))
    return all(install_packages(packages, env, _OP) for packages in deps)


def _clone(url: str, ref: str, env: dict[str, str]) -> str | None:
    """Clone nvbandwidth at `ref` into a fresh _WORK_DIR; the source directory, or None."""
    if not fresh_dir(_WORK_DIR, _OP):
        return None
    source_dir = git_clone(url, _OP, ref=ref, cwd=_WORK_DIR, env=env)
    # Bash's `pushd nvbandwidth` fails if the URL named another repo.
    if source_dir is not None and os.path.basename(source_dir) != _SOURCE_DIR_NAME:
        log_error(_OP, f"git clone did not create {_SOURCE_DIR_NAME}", path=source_dir)
        return None
    return source_dir


def _checkout_and_record(source_dir: str, commit: str, ref: str,
                         env: dict[str, str]) -> bool:
    """git checkout <commit> (when pinned), then write_repo_source_info (bash's order)."""
    if not commit:
        log_warn(_OP, "versions.json pins no nvbandwidth commit: building the tip of "
                      f"{ref} (bash runs a bare `git checkout`, which changes nothing)",
                 ref=ref)
    elif not run(["git", "checkout", commit], _OP, cwd=source_dir, env=env):
        return False
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
    return recorded


def install(env: dict[str, str]) -> int:
    """Build nvbandwidth from source, install it, record its version. 0 or 3."""
    cfg = require_config("nvbandwidth", env, _OP, "version", "url")
    if cfg is None:
        return 3
    version, url = str(cfg["version"]), str(cfg["url"])
    commit = configured_commit(cfg)
    ref = configured_ref(cfg, version)
    distro = Distro.from_env(env)
    arch = env.get("ARCHITECTURE", "")
    log_info(_OP, f"Installing nvbandwidth {version} on {distro}",
             version=version, source=url, ref=ref, commit=commit or "(not pinned)",
             distro=distro.name, arch=arch, node_type=env.get("TARGET_NODE_TYPE", ""),
             dest=_DEST_DIR, work_dir=_WORK_DIR)

    if not make_dirs(_DEST_DIR, op=_OP) or not _install_dependencies(env, distro, arch):
        return 3
    source_dir = _clone(url, ref, env)
    if source_dir is None or not _checkout_and_record(source_dir, commit, ref, env):
        return 3
    if not run_script(_BUILD_LINES, _OP, modules=[_HPCX_MODULE], cwd=source_dir,
                      env=env):
        return 3
    if not move_file(os.path.join(source_dir, _BINARY), _DEST_DIR, _OP):
        return 3
    remove_paths(_WORK_DIR, op=_OP)          # bash: rm -rf ./nvbandwidth
    if not record_version("NVBANDWIDTH", version, _OP):
        return 3
    log_info(_OP, f"nvbandwidth {version} installed to {_DEST_DIR}", version=version,
             commit=commit or "(not pinned)", ref=ref, dest=_DEST_DIR)
    return 0
