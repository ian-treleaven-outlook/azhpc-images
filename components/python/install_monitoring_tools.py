"""install_monitoring_tools.py — Python port of components/install_monitoring_tools.sh.

Installs Moneo (Azure's HPC monitoring tool): resolve the version from
versions.json, make sure pip is available, download + verify the release
tarball, extract it, run Moneo's service-configuration script, register the
`moneo` shell alias, and record the installed version.

Two helpers are pulled out as testable pure functions:
  - extract_stripped(): tar --strip-components=1 in native Python.
  - ensure_line():      the idempotent `grep -qxF || echo >>` alias guard.
"""

from __future__ import annotations

import os
import tarfile

from utils.package_installer import PackageInstaller
from utils.component_config import config_for, write_component_version
from utils.download import download_and_verify
from utils.process import exec_program
from utils.logger import log_info, log_error

_MONITOR_DIR = "/opt/azurehpc/tools"
_MONEO_DIR = f"{_MONITOR_DIR}/Moneo"
_BASHRC = "/etc/bash.bashrc"
_ALIAS = "alias moneo='python3 /opt/azurehpc/tools/Moneo/moneo.py'"


def extract_stripped(tarball, dest_dir):
    """Extract `tarball` into `dest_dir`, dropping the leading path component.

    Equivalent to `tar --strip-components=1 -C dest_dir`: the archive's single
    top-level directory (e.g. Moneo-1.2.3/) is stripped so its contents land
    directly in dest_dir.
    """
    with tarfile.open(tarball) as archive:
        members = []
        for member in archive.getmembers():
            parts = member.name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue  # the top-level directory entry itself
            member.name = parts[1]
            members.append(member)
        archive.extractall(dest_dir, members=members, filter="data")


def ensure_line(path, line):
    """Append `line` to `path` unless an identical line is already present.

    Idempotent: mirrors the `grep -qxF ... || echo >>` guard in the bash script.
    Returns True if the line was added, False if it already existed.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            if any(existing.rstrip("\n") == line for existing in handle):
                return False
    except FileNotFoundError:
        pass
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return True


def install_deps(env):
    """Make sure pip is available before installing Moneo.

    azurelinux/ubuntu install the distro's python3-pip package; other distros
    upgrade pip in place. Returns 0 on success, 3 on failure.
    """
    distribution = env.get("DISTRIBUTION", "")
    if distribution in ("azurelinux3.0", "ubuntu24.04"):
        installer = PackageInstaller()
        if installer.manager is None:
            return 3
        return 0 if installer.install_package(["python3-pip"]) else 3
    return exec_program(
        ["python3", "-m", "pip", "install", "--upgrade", "pip"],
        "install-monitoring-tools", env=env,
    )


def install(env):
    """Full Moneo install: deps -> download -> extract -> configure -> alias.

    Returns 0 on success, 3 on any failure.
    """
    cfg = config_for("moneo", env)
    if not cfg or not cfg.get("version"):
        log_error("install-monitoring-tools",
                  "could not resolve moneo version from versions.json")
        return 3
    version = cfg["version"]
    sha256 = cfg.get("sha256", "")
    url = f"https://github.com/Azure/Moneo/archive/refs/tags/v{version}.tar.gz"

    log_info("install-monitoring-tools", f"Installing Moneo {version}")

    # 1. dependencies (pip)
    if install_deps(env) != 0:
        return 3

    # 2. download + verify the release tarball
    os.makedirs(_MONITOR_DIR, exist_ok=True)
    try:
        tarball = download_and_verify(url, sha256, dest_dir=_MONITOR_DIR)
    except Exception as exc:
        log_error("install-monitoring-tools", f"download/verify failed: {exc}")
        return 3

    # 3. extract into Moneo/, stripping the top-level Moneo-<version>/ dir
    os.makedirs(_MONEO_DIR, exist_ok=True)
    extract_stripped(tarball, _MONEO_DIR)
    os.chmod(_MONEO_DIR, 0o777)

    # 4. run Moneo's service-configuration script
    service_dir = f"{_MONEO_DIR}/linux_service"
    rc = exec_program(["bash", "configure_service.sh"],
                      "install-monitoring-tools", cwd=service_dir, env=env)
    if rc != 0:
        log_error("install-monitoring-tools",
                  f"configure_service.sh failed with exit code {rc}")
        return 3

    # 5. register the `moneo` shell alias (idempotent)
    ensure_line(_BASHRC, _ALIAS)

    # 6. cleanup + record the installed version
    try:
        os.remove(tarball)
    except OSError:
        pass
    write_component_version("MONEO", version)

    log_info("install-monitoring-tools", f"Moneo {version} installed to {_MONEO_DIR}")
    return 0
