"""install_mpifileutils.py — mpifileutils build dependencies.

The Python counterpart to the package-install slice of
components/install_mpifileutils.sh. The old if/elif/else distro branching is now
a table keyed by package manager; detection is handled by
detect_package_manager(), so the only per-distro knowledge left here is the
package names themselves.
"""

from __future__ import annotations

import json
from utils.package_installer import PackageInstaller
from utils.component_config import get_component_config
from utils.logger import log_info

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
