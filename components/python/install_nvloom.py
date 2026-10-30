"""install_nvloom.py — Python port of components/install_nvloom.sh.

What this step does
-------------------
nvloom is NVIDIA's benchmark for multi-node NVLink (MNNVL) bandwidth. It only
makes sense on SKUs with rack-scale NVLink, so the build plan runs this step
for GB200 images only (Ubuntu 24.04 and Azure Linux 3.0, aarch64).

  1. Install the build dependencies (compiler toolchain, Boost
     program_options, cmake) with the distro's package manager.
  2. Clone the nvloom source at the configured ref (versions.json "ref",
     else "branch", else the tag v<version>), check out the pinned "commit",
     and record NVLOOM_REPO_COMMIT / NVLOOM_REPO_REF (write_repo_source_info;
     upstream #650, "build evidence").
  3. On Azure Linux, apply the repo's azurelinux_nvloom.patch, which stops
     nvloom linking Boost statically (that fails on Azure Linux).
  4. With the HPC-X MPI module loaded, build with cmake + make for Blackwell
     (CUDA arch 100), in one shell: `module load` only affects the shell it
     runs in (run_script).
  5. Move the built `nvloom_cli` into /opt/nvidia/nvloom and record NVLOOM.

Bash runs the same commands on Azure Linux 3.0 and Ubuntu, except for the
package names and the patch; see components/install_nvloom.sh.

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets install() as the "install-nvloom" Step's
`action=`; deleting `action=` runs the bash script again, which is why it is
kept.

Deliberate differences from bash
--------------------------------
  * The dependency installs go through install_packages(), which normalises
    `apt install -y` to `apt-get install -y -o Dpkg::Use-Pty=0`.
  * git_clone() puts --branch before the url (`git clone --branch <ref>
    <url>`); bash has `git clone <url> --branch <ref>`, which git treats the
    same.
  * Patch path: bash applies the patch from /home/hpcuser/azhpc-images/...,
    which only exists when the repo is checked out in that one user's home.
    We use <TOP_DIR>/distros/azurelinux3.0/azurelinux_nvloom.patch, and fail
    with 3 if TOP_DIR is unset or the patch file is missing.
  * Unsupported distros: bash skips the build but still records NVLOOM, so the
    image would claim a tool it doesn't have. We fail with 3.
  * Checks first: the config (version, url, commit), distro and patch are
    checked before anything is installed. Bash finds these problems later
    (e.g. `git clone --branch vnull`, or `git checkout null` after the
    dependencies are installed and the source is cloned).
  * write_repo_source_info gets the checkout's absolute path instead of "."
    and always both commit and ref, so it doesn't ask git.
  * Work directory: bash clones into its current directory (the distro build
    dir). We clone into _WORK_DIR, delete any copy left by an earlier failed
    run first (in bash, that leftover would make `git clone` fail), and delete
    it after a successful install (bash's `rm -rf ./nvloom`, via
    remove_paths, so a failure to delete doesn't fail the step).
  * Module load timing: bash loads mpi/hpcx at the very top; here it is
    loaded just before cmake, the only thing that needs it, so a broken
    module setup is found after the dependencies are installed.
"""

from __future__ import annotations

import os

from utils.component_config import record_version, require_config
from utils.distro import Distro
from utils.files import fresh_dir, make_dirs, move_file, remove_paths
from utils.logger import log_error, log_info
from utils.package_installer import install_packages
from utils.process import run, run_script
from utils.repo_source import configured_ref, git_clone, write_repo_source_info

_OP = "install-nvloom"

# Where the nvloom_cli binary is installed (bash's dest_dir).
_DEST_DIR = "/opt/nvidia/nvloom"

# Build area used instead of bash's current directory. `git clone` creates
# the source checkout inside it, in a folder named after the repo.
_WORK_DIR = "/tmp/nvloom-build"
_SOURCE_DIR_NAME = "nvloom"
_BINARY = "nvloom_cli"

_AZURE_LINUX = "azurelinux3.0"
_PATCH_IN_REPO = os.path.join("distros", "azurelinux3.0", "azurelinux_nvloom.patch")

# Dependency installs: one install_packages() call per bash command, in order.
_AZURE_LINUX_DEPS = (
    ["build-essential"],
    ["boost-devel", "boost-program-options"],
    ["cmake"],
)
_UBUNTU_DEPS = (
    ["build-essential"],
    ["libboost-program-options-dev"],
    ["cmake"],
)

# Run in the source directory with mpi/hpcx loaded (bash's trailing
# `module unload mpi/hpcx` isn't needed: the shell exits).
_HPCX_MODULE = "mpi/hpcx"
_BUILD_LINES = [
    'cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="100" .',
    "make -j $(nproc)",
]


# --------------------------------------------------------------------------- #
# Pure helpers: no side effects, so tests call them directly.
# --------------------------------------------------------------------------- #
def dependency_packages(distro: Distro) -> tuple[list[str], ...] | None:
    """The dependency package lists (one per bash install command), or None if
    bash has no branch for this distro.

    Bash's tests: exactly "azurelinux3.0", or containing "ubuntu".
    """
    if distro.name == _AZURE_LINUX:
        return _AZURE_LINUX_DEPS
    if distro.is_ubuntu:
        return _UBUNTU_DEPS
    return None


def patch_path(top_dir: str) -> str:
    """The Azure Linux nvloom patch in this repo checkout (bash hardcodes
    /home/hpcuser/azhpc-images/...)."""
    return os.path.join(top_dir, _PATCH_IN_REPO)


# --------------------------------------------------------------------------- #
# The steps.
# --------------------------------------------------------------------------- #
def _distro_plan(env: dict[str, str], distro: Distro):
    """(dependency installs, patch path or None) for this distro, or None (logged)."""
    deps = dependency_packages(distro)
    if deps is None:
        # Bash would skip the build but still record NVLOOM.
        log_error(_OP, f"nvloom build is not supported on {distro.name!r}",
                  distro=distro.name)
        return None
    if distro.name != _AZURE_LINUX:
        return deps, None
    patch = _azure_linux_patch(env)
    return None if patch is None else (deps, patch)


def _azure_linux_patch(env: dict[str, str]) -> str | None:
    """The patch's path under TOP_DIR, or None (logged) if it can't be found."""
    top_dir = env.get("TOP_DIR", "")
    if not top_dir:
        log_error(_OP, "TOP_DIR is not set; cannot find the nvloom patch")
        return None
    patch = patch_path(top_dir)
    if not os.path.isfile(patch):
        log_error(_OP, f"nvloom patch not found: {patch}", path=patch)
        return None
    return patch


def _clone(url: str, ref: str, env: dict[str, str]) -> str | None:
    """Clone nvloom at `ref` into a fresh _WORK_DIR; the source directory, or None."""
    if not fresh_dir(_WORK_DIR, _OP):
        return None
    source_dir = git_clone(url, _OP, ref=ref, cwd=_WORK_DIR, env=env)
    # Bash's `pushd nvloom` fails if the URL named another repo.
    if source_dir is not None and os.path.basename(source_dir) != _SOURCE_DIR_NAME:
        log_error(_OP, f"git clone did not create {_SOURCE_DIR_NAME}", path=source_dir)
        return None
    return source_dir


def _checkout_and_record(source_dir: str, commit: str, ref: str,
                         env: dict[str, str]) -> bool:
    """git checkout <commit>, then write_repo_source_info (bash's order)."""
    if not run(["git", "checkout", commit], _OP, cwd=source_dir, env=env):
        return False
    try:
        recorded = write_repo_source_info("NVLOOM", source_dir, _OP,
                                          commit=commit, ref=ref, env=env)
    except OSError as exc:
        log_error(_OP, f"could not record the nvloom source: {exc}",
                  commit=commit, ref=ref)
        return False
    if not recorded:
        log_error(_OP, "could not record the nvloom source", commit=commit, ref=ref)
    return recorded


def _build(source_dir: str, patch: str | None, env: dict[str, str]) -> bool:
    """Apply the Azure Linux patch (if any), then cmake + make with HPC-X loaded."""
    if patch is not None:
        if not run(["git", "apply", patch], _OP, cwd=source_dir, env=env):
            return False
        log_info(_OP, f"Applied {os.path.basename(patch)} (Azure Linux: don't link "
                      "Boost statically)", patch=patch)
    return run_script(_BUILD_LINES, _OP, modules=[_HPCX_MODULE], cwd=source_dir,
                      env=env)


def install(env: dict[str, str]) -> int:
    """Build nvloom from source, install nvloom_cli, record its version. 0 or 3."""
    cfg = require_config("nvloom", env, _OP, "version", "url", "commit")
    if cfg is None:
        return 3
    version, url, commit = str(cfg["version"]), str(cfg["url"]), str(cfg["commit"])
    ref = configured_ref(cfg, version)

    distro = Distro.from_env(env)
    plan = _distro_plan(env, distro)
    if plan is None:
        return 3
    deps, patch = plan

    log_info(_OP, f"Installing nvloom {version} on {distro}", version=version,
             source=url, ref=ref, commit=commit, distro=distro.name,
             node_type=env.get("TARGET_NODE_TYPE", ""), dest=_DEST_DIR,
             patch=patch or "(none)", work_dir=_WORK_DIR)
    if not make_dirs(_DEST_DIR, op=_OP):
        return 3
    if not all(install_packages(packages, env, _OP) for packages in deps):
        return 3
    source_dir = _clone(url, ref, env)
    if source_dir is None or not _checkout_and_record(source_dir, commit, ref, env):
        return 3
    if not _build(source_dir, patch, env):
        return 3
    if not move_file(os.path.join(source_dir, _BINARY), _DEST_DIR, _OP):
        return 3
    remove_paths(_WORK_DIR, op=_OP)          # bash: rm -rf ./nvloom
    if not record_version("NVLOOM", version, _OP):
        return 3
    log_info(_OP, f"nvloom {version} installed to {_DEST_DIR}", version=version,
             commit=commit, ref=ref, dest=_DEST_DIR)
    return 0
