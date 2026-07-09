"""install_mpifileutils.py — Python port of components/install_mpifileutils.sh.

Ports the whole install flow to Python: resolve the version from versions.json,
install build dependencies, download + verify the source, extract it, build it,
and clean up. Only the build step itself (module load + cmake + make) stays in
Bash, because `module load` needs everything to run in one shell (see install()).
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile

from utils.package_installer import PackageInstaller
from utils.component_config import get_component_config, write_component_version
from utils.download import download_and_verify
from utils.process import exec_program
from utils.logger import log_info, log_error

# Install/build locations (mirrors the paths in install_mpifileutils.sh).
_INSTALL_PREFIX = "/opt/mpifileutils"
_BUILD_DIR = "/tmp/mpifileutils-build"
_SRC_DIR = "/tmp/mpifileutils-src"

def get_config(env):
    """Resolve this component's version/url/sha256 from versions.json."""
    versions = json.loads(env.get("COMPONENT_VERSIONS", "{}"))
    return get_component_config(
        "mpifileutils", versions,
        distribution=env.get("DISTRIBUTION", ""),
        architecture=env.get("ARCHITECTURE", ""),
        gpu=env.get("GPU"),
        sku=env.get("SKU"),
        node_type=env.get("NODE_TYPE", "azure-vm"),
    )

# Build dependencies, keyed by package manager.
_DEPS = {
    "apt-get": ["libbz2-dev", "libattr1-dev", "libarchive-dev", "libssl-dev", "libcap-dev"],
    "apt":     ["libbz2-dev", "libattr1-dev", "libarchive-dev", "libssl-dev", "libcap-dev"],
    "tdnf":    ["bzip2-devel", "libattr-devel", "libarchive-devel"],
    "yum":     ["bzip2-devel", "libattr-devel", "libarchive-devel"],
    "dnf":     ["bzip2-devel", "libattr-devel", "libarchive-devel"],
}

def install_deps(env: dict[str, str]) -> int:
    """Install mpifileutils build dependencies via PackageInstaller.

    Returns 0 on success, 3 if no package manager is found or an install fails.
    """
    cfg = get_config(env)
    if cfg and cfg.get("version"):
        log_info("install-mpifileutils", f"Preparing mpifileutils {cfg['version']}")
    installer = PackageInstaller()
    if installer.manager is None:
        return 3
    deps = _DEPS.get(installer.manager.name, [])
    return 0 if installer.install_package(deps) else 3


def install(env: dict[str, str]) -> int:
    """Full mpifileutils install: deps -> download -> extract -> build -> cleanup.

    Returns 0 on success, 3 on any failure.
    """
    cfg = get_config(env)
    if not cfg or not cfg.get("version"):
        log_error("install-mpifileutils",
                  "could not resolve mpifileutils version from versions.json")
        return 3
    version = cfg["version"]
    url = cfg.get("url", "")
    sha256 = cfg.get("sha256", "")

    log_info("install-mpifileutils", f"Installing mpifileutils {version}")

    # 1. build dependencies
    if install_deps(env) != 0:
        return 3

    # 2. directories
    for directory in (_INSTALL_PREFIX, _BUILD_DIR, _SRC_DIR):
        os.makedirs(directory, exist_ok=True)

    # 3. download + verify the source tarball
    try:
        tarball = download_and_verify(url, sha256, dest_dir=_SRC_DIR)
    except Exception as exc:
        log_error("install-mpifileutils", f"download/verify failed: {exc}")
        return 3

    # 4. extract (filter='data' is the safe extraction mode)
    with tarfile.open(tarball) as archive:
        archive.extractall(_SRC_DIR, filter="data")

    # 5. build — module load + cmake + make must share one shell (see module note)
    source_dir = f"{_SRC_DIR}/mpifileutils-v{version}"
    build_script = (
        "set -e\n"
        "source /etc/profile.d/modules.sh\n"
        "module load mpi/hpcx\n"
        f'cmake "{source_dir}" '
        f'-DCMAKE_INSTALL_PREFIX="{_INSTALL_PREFIX}" '
        "-DENABLE_XATTRS=ON -DENABLE_LIBARCHIVE=ON -DENABLE_LUSTRE=OFF "
        "-DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5\n"
        'make -j"$(nproc)"\n'
        "make install\n"
        "module unload mpi/hpcx\n"
    )
    rc = exec_program(["bash", "-c", build_script], "install-mpifileutils",
                      cwd=_BUILD_DIR, env=env)
    if rc != 0:
        log_error("install-mpifileutils", f"build failed with exit code {rc}")
        return 3

    # 6. cleanup
    shutil.rmtree(_BUILD_DIR, ignore_errors=True)
    shutil.rmtree(_SRC_DIR, ignore_errors=True)

    # 7. record the installed version
    write_component_version("MPIFILEUTILS", version)

    log_info("install-mpifileutils",
             f"mpifileutils {version} installed to {_INSTALL_PREFIX}")
    return 0
