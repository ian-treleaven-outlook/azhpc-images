"""install_mpifileutils.py — Python port of components/install_mpifileutils.sh.

Builds mpifileutils (MPI-based parallel file tools: dbcast, dcp, drm, dsync,
dfind, dwalk, dcmp, dtar) from source against HPC-X's MPI, installs it into
/opt/mpifileutils and records MPIFILEUTILS in the component-versions file.
Lustre support is compiled in only if the Lustre client headers are installed
(the Lustre client step runs just before this one).

utils/plans/steps.py sets install() as the `action=` of the
"install-mpifileutils" Step; deleting `action=` runs the bash script again.

Kept from bash
--------------
  * The same commands, arguments, working directories and order: the
    distro's dependency install, the tarball unpacked in the source dir
    (its name comes from the version, not the URL), the build in the build
    dir, removal of both dirs, and the final `ls -la /opt/mpifileutils/bin/`,
    whose failure (no bin dir) stops the step before the version is recorded.
  * cmake, make and make install run in one shell after
    `module load mpi/hpcx` (run_script), because `module load` only affects
    the shell it runs in.

Deliberate differences
----------------------
  * The dependency install goes through install_packages(), which spells it
    `apt-get install -y -o Dpkg::Use-Pty=0` / `dnf install -y`. A
    DISTRIBUTION it has no manager for fails there (3); bash's else branch
    would run dnf anyway.
  * A missing version, url or sha256 in versions.json fails up front (3),
    before anything is installed; bash carries on with empty values.
  * A source folder that tar didn't create fails with a clear message
    before cmake (bash's cmake would fail on the missing directory).
  * Toolkit spellings: `tar -xf` (bash `tar -xzf`), no trailing
    `module unload` (the shell exits), and the `rm -rf` of the build and
    source dirs is remove_paths(), which like `rm -rf` never fails the step.
"""

from __future__ import annotations

import os

from utils.component_config import require_config, write_component_version
from utils.distro import Distro
from utils.download import fetch_verified
from utils.files import extract_tarball, remove_paths
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import run, run_script

_OP = "install-mpifileutils"

_AZURE_LINUX = "azurelinux3.0"

# bash's INSTALL_PREFIX, BUILD_DIR and SRC_DIR. Module level so tests can
# point them at a temporary directory.
_INSTALL_PREFIX = "/opt/mpifileutils"
_BUILD_DIR = "/tmp/mpifileutils-build"
_SRC_DIR = "/tmp/mpifileutils-src"
# Building with Lustre support needs this header from the Lustre client.
_LUSTRE_USER_H = "/usr/include/lustre/lustre_user.h"

# The lists differ deliberately (libbz2-dev is Ubuntu's name for bzip2-devel).
_UBUNTU_DEPS = ["libbz2-dev", "libattr1-dev", "libarchive-dev", "libssl-dev", "libcap-dev"]
_RPM_DEPS = ["bzip2-devel", "libattr-devel", "libarchive-devel"]


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
def lustre_cmake_flag(header: str) -> str:
    """bash's LUSTRE_CMAKE_FLAG: "ON" if `header` is a file (`[[ -f ]]`), else "OFF"."""
    return "ON" if os.path.isfile(header) else "OFF"


def deps_packages(distro: Distro) -> tuple[list[str], str]:
    """bash's dependency packages for this distro, and the name of its branch.

    bash: *"ubuntu"* -> the Ubuntu list; "azurelinux3.0" and anything else
    (the RHEL family) -> the same RPM list.
    """
    if distro.is_ubuntu:
        return list(_UBUNTU_DEPS), "ubuntu"
    if distro.name == _AZURE_LINUX:
        return list(_RPM_DEPS), "azurelinux"
    return list(_RPM_DEPS), "rhel-family"


def tarball_name(version: str) -> str:
    """bash's TARBALL="mpifileutils-v${MPIFILEUTILS_VERSION}.tgz"."""
    return f"mpifileutils-v{version}.tgz"


def build_lines(version: str, lustre: str) -> list[str]:
    """bash's build lines, run after `module load mpi/hpcx`."""
    return [
        f'cmake "{_SRC_DIR}/mpifileutils-v{version}" '
        f'-DCMAKE_INSTALL_PREFIX="{_INSTALL_PREFIX}" '
        "-DENABLE_XATTRS=ON "
        "-DENABLE_LIBARCHIVE=ON "
        f'-DENABLE_LUSTRE="{lustre}" '
        "-DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "make -j$(nproc)",
        "make install",
    ]


# --------------------------------------------------------------------------- #
# Steps.
# --------------------------------------------------------------------------- #
def install_deps(env: dict[str, str]) -> int:
    """Install the build dependencies for this distro. Returns 0 or 3."""
    distro = Distro.from_env(env)
    packages, branch = deps_packages(distro)
    log_info(_OP, f"Installing mpifileutils build dependencies ({branch} branch of "
                  "install_mpifileutils.sh)", distro=str(distro), branch=branch,
             packages=packages)
    return 0 if install_packages(packages, env, _OP) else 3


def _unpack_source(env: dict[str, str], version: str, url: str, sha256: str) -> bool:
    """Download and unpack the source into the source dir (bash: cd $SRC_DIR)."""
    downloaded = fetch_verified(url, sha256, _OP, _SRC_DIR)
    if downloaded is None:
        return False
    tarball = tarball_name(version)
    if downloaded.name != tarball:
        # bash extracts TARBALL (built from the version), not the file it
        # downloaded; tar then fails on the missing file, here too.
        log_warn(_OP, f"downloaded {downloaded.name} but bash extracts {tarball}; "
                      "tar will fail if that file is missing",
                 downloaded=str(downloaded), tarball=tarball)
    if not extract_tarball(tarball, _OP, cwd=_SRC_DIR, env=env):
        return False
    source_dir = f"{_SRC_DIR}/mpifileutils-v{version}"
    if not os.path.isdir(source_dir):
        log_error(_OP, f"mpifileutils source folder {source_dir} was not created by "
                       f"extracting {tarball}", path=source_dir, tarball=tarball)
        return False
    return True


def _build_and_install(env: dict[str, str], version: str, lustre: str) -> bool:
    """cmake + make + make install with HPC-X's MPI, then remove the build dirs."""
    log_info(_OP, f"Building mpifileutils {version} with HPC-X (module load mpi/hpcx)",
             build_dir=_BUILD_DIR, lustre=lustre)
    if not run_script(build_lines(version, lustre), _OP, modules=["mpi/hpcx"],
                      cwd=_BUILD_DIR, env=env):
        return False
    remove_paths(_BUILD_DIR, _SRC_DIR, op=_OP)
    log_info(_OP, f"=== mpifileutils installed to {_INSTALL_PREFIX} ===",
             path=_INSTALL_PREFIX)
    # bash lists the installed tools; no bin dir means a broken install.
    return run(["ls", "-la", f"{_INSTALL_PREFIX}/bin/"], _OP, env=env)


def install(env: dict[str, str]) -> int:
    """Build and install mpifileutils, in bash's order. Returns 0 or 3."""
    cfg = require_config("mpifileutils", env, _OP, "version", "url", "sha256")
    if cfg is None:
        return 3
    version, url, sha256 = str(cfg["version"]), str(cfg["url"]), str(cfg["sha256"])
    lustre = lustre_cmake_flag(_LUSTRE_USER_H)
    log_info(_OP, f"=== Installing mpifileutils {version} ===", version=version, url=url,
             lustre=lustre, prefix=_INSTALL_PREFIX)
    log_info(_OP, f"Lustre support {lustre}: {_LUSTRE_USER_H} "
                  f"{'found' if lustre == 'ON' else 'not found'}",
             lustre=lustre, header=_LUSTRE_USER_H)

    if install_deps(env) != 0:
        return 3
    try:
        for directory in (_INSTALL_PREFIX, _BUILD_DIR, _SRC_DIR):
            os.makedirs(directory, exist_ok=True)       # bash: mkdir -p
        if not (_unpack_source(env, version, url, sha256)
                and _build_and_install(env, version, lustre)):
            return 3
        write_component_version("MPIFILEUTILS", version)
    except OSError as exc:
        # mkdir -p of the three dirs, or the versions file.
        log_error(_OP, f"file operation failed: {exc}", path=getattr(exc, "filename", None))
        return 3

    log_info(_OP, f"Recorded MPIFILEUTILS={version}", component="MPIFILEUTILS",
             version=version)
    log_info(_OP, f"mpifileutils {version} installed to {_INSTALL_PREFIX}",
             version=version, path=_INSTALL_PREFIX, lustre=lustre)
    return 0
