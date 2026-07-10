"""install_nvbandwidth_tool.py — Python port of components/install_nvbandwidth_tool.sh.

Installs the NVIDIA nvbandwidth benchmark: install boost deps, clone the source
at the pinned version tag, build it with cmake/nvcc, move the built binary into
place, and record the installed version.
"""

from __future__ import annotations

import os
import shutil

from utils.package_installer import PackageInstaller
from utils.component_config import config_for, write_component_version
from utils.process import exec_program
from utils.logger import log_info, log_error

_DEST_DIR = "/opt/nvidia/nvbandwidth"
_SRC_DIR = "/tmp/nvbandwidth-src"

# Boost dependencies, keyed by package manager.
_DEPS = {
    "apt-get": ["libboost-program-options-dev"],
    "apt":     ["libboost-program-options-dev"],
    "dnf":     ["boost-devel"],
    "yum":     ["boost-devel"],
    "tdnf":    ["boost-devel", "boost-program-options"],
}


def get_config(env):
    """Resolve nvbandwidth's version/url from versions.json."""
    return config_for("nvbandwidth", env)


def install_deps(env):
    """Install boost (and cmake on azurelinux/aarch64). Returns 0 or 3."""
    installer = PackageInstaller()
    if installer.manager is None:
        return 3
    deps = list(_DEPS.get(installer.manager.name, []))
    if installer.manager.name == "tdnf" and env.get("ARCHITECTURE") == "aarch64":
        deps.append("cmake")
    return 0 if installer.install_package(deps) else 3


def install(env):
    """Deps -> clone -> cmake/make -> install binary -> record version. 0 or 3."""
    cfg = get_config(env)
    if not cfg or not cfg.get("version"):
        log_error("install-nvbandwidth",
                  "could not resolve nvbandwidth version from versions.json")
        return 3
    version = cfg["version"]
    url = cfg.get("url", "")

    log_info("install-nvbandwidth", f"Installing nvbandwidth {version}")

    # 1. dependencies
    if install_deps(env) != 0:
        return 3

    # 2. clone the source at the version tag
    os.makedirs(_DEST_DIR, exist_ok=True)
    shutil.rmtree(_SRC_DIR, ignore_errors=True)
    rc = exec_program(["git", "clone", "--branch", f"v{version}", url, _SRC_DIR],
                      "install-nvbandwidth", env=env)
    if rc != 0:
        log_error("install-nvbandwidth", f"git clone failed with exit code {rc}")
        return 3

    # 3. build (cmake + make)
    steps = [
        ["cmake", "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
         "-DCMAKE_CUDA_ARCHITECTURES=100", "."],
        ["make", "-j", str(os.cpu_count() or 1)],
    ]
    for step in steps:
        rc = exec_program(step, "install-nvbandwidth", cwd=_SRC_DIR, env=env)
        if rc != 0:
            log_error("install-nvbandwidth",
                      f"build step failed: {' '.join(step)} (exit code {rc})")
            return 3

    # 4. install the built binary
    dest = os.path.join(_DEST_DIR, "nvbandwidth")
    if os.path.exists(dest):
        os.remove(dest)
    shutil.move(os.path.join(_SRC_DIR, "nvbandwidth"), dest)

    # 5. cleanup + record the installed version
    shutil.rmtree(_SRC_DIR, ignore_errors=True)
    write_component_version("NVBANDWIDTH", version)

    log_info("install-nvbandwidth", f"nvbandwidth {version} installed to {_DEST_DIR}")
    return 0
