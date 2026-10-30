"""install_mpifileutils.py — Python port of components/install_mpifileutils.sh.

What this step does
-------------------
Builds mpifileutils (MPI-based parallel file tools: dbcast, dcp, drm, dsync,
dfind, dwalk, dcmp, dtar) from source against HPC-X's MPI and installs it into
/opt/mpifileutils, then records MPIFILEUTILS in the component-versions file.
Lustre support is compiled in only if the Lustre client headers are installed
(the Lustre client step runs just before this one).

The original bash (upstream 0c2b8f9), for comparison:

    LUSTRE_CMAKE_FLAG="OFF"
    if [[ -f /usr/include/lustre/lustre_user.h ]]; then LUSTRE_CMAKE_FLAG="ON"; fi
    echo "=== Installing mpifileutils ${MPIFILEUTILS_VERSION} ==="
    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        apt-get install -y libbz2-dev libattr1-dev libarchive-dev libssl-dev libcap-dev
    elif [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
        dnf install -y bzip2-devel libattr-devel libarchive-devel
    else
        dnf install -y bzip2-devel libattr-devel libarchive-devel
    fi
    mkdir -p "${INSTALL_PREFIX}"; mkdir -p "${BUILD_DIR}"; mkdir -p "${SRC_DIR}"
    cd "${SRC_DIR}"
    TARBALL="mpifileutils-v${MPIFILEUTILS_VERSION}.tgz"
    download_and_verify "${MPIFILEUTILS_URL}" "${MPIFILEUTILS_SHA256}"
    tar -xzf "${TARBALL}"
    source /etc/profile.d/modules.sh
    module load mpi/hpcx
    cd "${BUILD_DIR}"
    cmake "${SRC_DIR}/mpifileutils-v${MPIFILEUTILS_VERSION}" \\
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" -DENABLE_XATTRS=ON \\
        -DENABLE_LIBARCHIVE=ON -DENABLE_LUSTRE="${LUSTRE_CMAKE_FLAG}" \\
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    make -j$(nproc)
    make install
    module unload mpi/hpcx
    rm -rf "${BUILD_DIR}" "${SRC_DIR}"
    echo "=== mpifileutils installed to ${INSTALL_PREFIX} ==="
    ls -la "${INSTALL_PREFIX}/bin/"
    write_component_version "MPIFILEUTILS" ${MPIFILEUTILS_VERSION}

(The fork's components/install_mpifileutils.sh dropped the dependency block,
saying an orchestrator step installs them; no such step exists, so this port
runs upstream's commands itself.)

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets this module's install() as the `action=` of the
"install-mpifileutils" Step. install() receives the build environment and
returns 0 for success or 3 for failure; it logs problems and never lets an
exception escape. Each command stops the step if it fails, like `set -e`.

Behaviour kept from bash
------------------------
  * The same commands, arguments, working directories and order: the
    distro's dependency install, `tar -xzf mpifileutils-v<version>.tgz` in
    the source dir (the tarball name comes from the version, not the URL, as
    in bash), the build, `rm -rf` of the build and source dirs, and the final
    `ls -la /opt/mpifileutils/bin/`, whose failure (no bin dir) stops the
    step before the version is recorded.
  * `source /etc/profile.d/modules.sh`, `module load mpi/hpcx`, cmake, make,
    make install and `module unload mpi/hpcx` run in ONE `bash -c` script
    with `set -e`, in the build dir, because `module load` only affects the
    shell it runs in.

Deliberate differences from bash
--------------------------------
  * The dependency install goes through install_packages()
    (utils/package_installer.py), which runs `apt-get install -y -o
    Dpkg::Use-Pty=0` / `dnf install -y` (bash: `apt-get install -y` /
    `dnf install -y`). A DISTRIBUTION it has no manager for fails there (3);
    bash's else branch would run dnf anyway.
  * A missing version, url or sha256 in versions.json fails up front (3),
    before anything is installed. Bash carries on with empty values and fails
    later (or builds "mpifileutils-v").
  * A source folder that `tar` didn't create fails with a clear message
    before cmake (bash's cmake would fail on the missing directory).
  * `mkdir -p` is os.makedirs(exist_ok=True) (same semantics, including
    failing when a path exists as a file).
"""

from __future__ import annotations

import os

from utils.component_config import config_for, write_component_version
from utils.download import download_and_verify
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import exec_program

# The Step's `op` label in plans/steps.py; used on every log line.
_OP = "install-mpifileutils"

_AZURE_LINUX = "azurelinux3.0"

# Install/build locations (bash: INSTALL_PREFIX, BUILD_DIR, SRC_DIR). Module
# level so tests can point them at a temporary directory.
_INSTALL_PREFIX = "/opt/mpifileutils"
_BUILD_DIR = "/tmp/mpifileutils-build"
_SRC_DIR = "/tmp/mpifileutils-src"
# Building with Lustre support needs this header from the Lustre client.
_LUSTRE_USER_H = "/usr/include/lustre/lustre_user.h"

# bash's build dependencies, one list per distro branch. The lists differ
# deliberately (libbz2-dev is Ubuntu's name for bzip2-devel).
_UBUNTU_DEPS = ["libbz2-dev", "libattr1-dev", "libarchive-dev", "libssl-dev", "libcap-dev"]
_RPM_DEPS = ["bzip2-devel", "libattr-devel", "libarchive-devel"]


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
def lustre_cmake_flag(header: str) -> str:
    """bash's LUSTRE_CMAKE_FLAG: "ON" if `header` is a file (`[[ -f ]]`), else "OFF"."""
    return "ON" if os.path.isfile(header) else "OFF"


def deps_packages(distribution: str) -> tuple[list[str], str]:
    """bash's dependency packages for this distro, and the branch name.

    bash: *"ubuntu"* -> the Ubuntu list; "azurelinux3.0" -> the RPM list;
    anything else (RHEL family) -> the same RPM list.
    """
    if "ubuntu" in distribution:
        return list(_UBUNTU_DEPS), "ubuntu"
    if distribution == _AZURE_LINUX:
        return list(_RPM_DEPS), "azurelinux"
    return list(_RPM_DEPS), "rhel-family"


def tarball_name(version: str) -> str:
    """bash's TARBALL="mpifileutils-v${MPIFILEUTILS_VERSION}.tgz"."""
    return f"mpifileutils-v{version}.tgz"


def build_script(version: str, lustre: str) -> str:
    """The `bash -c` script for bash's module load ... module unload lines."""
    return (
        "set -e\n"
        "source /etc/profile.d/modules.sh\n"
        "module load mpi/hpcx\n"
        f'cmake "{_SRC_DIR}/mpifileutils-v{version}" '
        f'-DCMAKE_INSTALL_PREFIX="{_INSTALL_PREFIX}" '
        "-DENABLE_XATTRS=ON "
        "-DENABLE_LIBARCHIVE=ON "
        f'-DENABLE_LUSTRE="{lustre}" '
        "-DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5\n"
        "make -j$(nproc)\n"
        "make install\n"
        "module unload mpi/hpcx\n"
    )


# --------------------------------------------------------------------------- #
# Steps.
# --------------------------------------------------------------------------- #
def _run(command: list[str], what: str, env: dict[str, str], cwd: str | None = None) -> bool:
    """Run one command; log the consequence if it fails (bash `set -e`)."""
    rc = exec_program(command, _OP, cwd=cwd, env=env)
    if rc != 0:
        log_error(_OP, f"{what} failed; stopping, as bash's set -e would",
                  command=command[0], exit_code=rc)
        return False
    return True


def install_deps(env: dict[str, str]) -> int:
    """Install mpifileutils' build dependencies: bash's list for this distro,
    in one install_packages() call (which logs the command and any failure).

    Returns 0 on success, 3 on failure.
    """
    distribution = env.get("DISTRIBUTION", "")
    packages, branch = deps_packages(distribution)
    log_info(_OP, f"Installing mpifileutils build dependencies ({branch} branch of "
                  "install_mpifileutils.sh)", distro=distribution, branch=branch,
             packages=packages)
    return 0 if install_packages(packages, env, _OP) else 3


def install(env: dict[str, str]) -> int:
    """Full mpifileutils install, in bash's order. Returns 0 on success, 3 on failure."""
    cfg = config_for("mpifileutils", env)
    if not isinstance(cfg, dict) or not cfg.get("version"):
        log_error(_OP, "could not resolve mpifileutils version from versions.json")
        return 3
    missing = [key for key in ("url", "sha256") if not cfg.get(key)]
    if missing:
        log_error(_OP, f"mpifileutils config in versions.json is missing "
                       f"{', '.join(missing)}", missing=missing)
        return 3
    version, url, sha256 = str(cfg["version"]), str(cfg["url"]), str(cfg["sha256"])
    tarball = tarball_name(version)
    source_dir = f"{_SRC_DIR}/mpifileutils-v{version}"

    lustre = lustre_cmake_flag(_LUSTRE_USER_H)
    log_info(_OP, f"=== Installing mpifileutils {version} ===", version=version, url=url,
             tarball=tarball, lustre=lustre, prefix=_INSTALL_PREFIX)
    log_info(_OP, f"Lustre support {lustre}: {_LUSTRE_USER_H} "
                  f"{'found' if lustre == 'ON' else 'not found'}",
             lustre=lustre, header=_LUSTRE_USER_H)

    if install_deps(env) != 0:
        return 3

    try:
        for directory in (_INSTALL_PREFIX, _BUILD_DIR, _SRC_DIR):
            os.makedirs(directory, exist_ok=True)

        try:
            downloaded = download_and_verify(url, sha256, dest_dir=_SRC_DIR)
        except Exception as exc:  # network errors, checksum mismatch, ...
            log_error(_OP, f"download/verify failed: {exc}", url=url)
            return 3
        if os.path.basename(str(downloaded)) != tarball:
            # bash extracts TARBALL (built from the version), not the file it
            # downloaded; tar then fails on the missing file, here too.
            log_warn(_OP, f"downloaded {os.path.basename(str(downloaded))} but bash "
                          f"extracts {tarball}; tar will fail if that file is missing",
                     downloaded=str(downloaded), tarball=tarball)

        if not _run(["tar", "-xzf", tarball], f"extracting {tarball}", env, _SRC_DIR):
            return 3
        if not os.path.isdir(source_dir):
            log_error(_OP, f"mpifileutils source folder {source_dir} was not created by "
                           f"extracting {tarball}", path=source_dir, tarball=tarball)
            return 3

        log_info(_OP, f"Building mpifileutils {version} with HPC-X (module load mpi/hpcx)",
                 source_dir=source_dir, build_dir=_BUILD_DIR, lustre=lustre)
        if not _run(["bash", "-c", build_script(version, lustre)],
                    "the mpifileutils build (module load, cmake, make, make install)",
                    env, _BUILD_DIR):
            return 3

        # Cleanup build artifacts.
        if not _run(["rm", "-rf", _BUILD_DIR, _SRC_DIR], "removing the build and source dirs",
                    env):
            return 3

        log_info(_OP, f"=== mpifileutils installed to {_INSTALL_PREFIX} ===",
                 path=_INSTALL_PREFIX)
        # bash lists the installed tools; no bin dir means a broken install.
        if not _run(["ls", "-la", f"{_INSTALL_PREFIX}/bin/"],
                    f"listing {_INSTALL_PREFIX}/bin/ (no installed tools?)", env):
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
