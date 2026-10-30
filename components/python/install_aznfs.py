"""install_aznfs.py — Python port of components/install_aznfs.sh.

Installs the AZNFS mount helper (Azure Blob NFS mounts). Where Microsoft's
package repository (PMC) publishes an aznfs package it comes from there
(install_packages: `apt-get install -y aznfs` on Ubuntu, `dnf install -y
aznfs` on AlmaLinux, Rocky and RHEL). Any other distro (Azure Linux 3.0 today, bash's `else`) runs the
upstream AZNFS-mount installer: download the release's aznfs_install.sh
(SHA256 checked) into the current directory, `sed -i 's/yum/dnf/'` it, run it
with bash, then `rm -f` it. AZNFS_NONINTERACTIVE_INSTALL=1 is set for every
command so the package's setup never stops at a TTY prompt during a Packer
build. No version is recorded (bash records none).

Kept bash quirks:
  * sed's `s/yum/dnf/` has no /g: only the first "yum" on each line changes,
    including inside longer words ("yummy" -> "dnfmy"). sed_yum_to_dnf() was
    compared with real `sed -i` (see its tests).
  * A failed installer leaves aznfs_install.sh in place (`set -e` stops before
    the `rm -f`); the error names its path.

Deliberate differences:
  * Bash's `apt install -y aznfs` runs as install_packages()'s
    `apt-get install -y aznfs` (same package, apt-get's stable CLI).
  * The distro is matched with Distro (a "rhel" family name starts with
    almalinux/rocky/rhel) rather than bash's substring tests; the same for
    every distribution this project builds.
  * A missing aznfs version or sha256 fails before downloading (bash would
    fetch ".../download/null/aznfs_install.sh").
  * AZNFS_NONINTERACTIVE_INSTALL=1 is passed in this step's command
    environment rather than exported (bash's export also only reaches this
    script and its children).
  * A leftover aznfs_install.sh from an earlier failed run is overwritten,
    with a warning. Bash's wget would save aznfs_install.sh.1 and then check
    the old, already-patched file, failing the checksum.
"""

from __future__ import annotations

import os
from pathlib import Path

from utils.component_config import require_config
from utils.distro import Distro
from utils.download import fetch_verified
from utils.files import edit_file, remove_paths
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import run

_OP = "install-aznfs"

# Distro families that install the PMC aznfs package; others run the
# upstream installer.
_PMC_FAMILIES = ("ubuntu", "rhel")

# The distro the installer branch was written for; others reaching it are warned about.
_INSTALLER_DISTRO = "azurelinux"

_INSTALLER_NAME = "aznfs_install.sh"
_INSTALLER_URL = ("https://github.com/Azure/AZNFS-mount/releases/download/"
                  "{version}/" + _INSTALLER_NAME)

# Bash's download_and_verify has no folder argument here: the current directory.
_DOWNLOAD_DIR = "."

_NONINTERACTIVE = {"AZNFS_NONINTERACTIVE_INSTALL": "1"}


def install_branch(distribution: str) -> str:
    """Which branch of bash's if/elif/else runs: "pmc" or "installer"."""
    return "pmc" if Distro(distribution).family in _PMC_FAMILIES else "installer"


def sed_yum_to_dnf(text: str) -> str:
    """`sed 's/yum/dnf/'` on the file's text: the first "yum" on each line.

    A missing final newline stays missing, as with sed.
    """
    return "\n".join(line.replace("yum", "dnf", 1) for line in text.split("\n"))


def _install_from_pmc(run_env: dict[str, str]) -> int:
    """`apt-get install -y aznfs` or `dnf install -y aznfs`."""
    if not install_packages(["aznfs"], run_env, _OP):
        return 3
    log_info(_OP, "AZNFS installed", branch="pmc", package="aznfs")
    return 0


def _download_installer(url: str, sha256: str) -> Path | None:
    leftover = os.path.join(_DOWNLOAD_DIR, _INSTALLER_NAME)
    if os.path.exists(leftover):
        log_warn(_OP, f"{leftover} already exists (left by an earlier failed run?); "
                      "overwriting it (bash's wget would save aznfs_install.sh.1 "
                      "and check the old file)", path=leftover)
    script = fetch_verified(url, sha256, _OP, _DOWNLOAD_DIR)
    return None if script is None else Path(script)


def _patch_installer(script: Path) -> bool:
    """bash `sed -i 's/yum/dnf/' aznfs_install.sh`."""
    def patch(text: str) -> str:
        changed = sum(1 for line in text.split("\n") if "yum" in line)
        log_info(_OP, f"Patched {script}: first 'yum' on each line -> 'dnf' "
                      "(sed -i 's/yum/dnf/')", path=str(script), lines_changed=changed)
        if changed == 0:
            log_warn(_OP, f"{script} has no 'yum' to replace; running it unchanged",
                     path=str(script))
        patched = sed_yum_to_dnf(text)
        log_debug(_OP, f"Patched installer {script}:\n{patched}", path=str(script))
        return patched
    return edit_file(script, patch, _OP)


def _install_from_upstream_installer(env: dict[str, str], run_env: dict[str, str],
                                     distro: Distro) -> int:
    """Download aznfs_install.sh, patch yum -> dnf, run it, delete it."""
    cfg = require_config("aznfs", env, _OP, "version", "sha256")
    if cfg is None:
        return 3
    version = str(cfg["version"])
    url = _INSTALLER_URL.format(version=version)
    if _INSTALLER_DISTRO not in distro.name:
        log_warn(_OP, f"{distro.name!r} has no PMC aznfs package in bash's list; "
                      "using the upstream installer (bash's `else` branch), which "
                      "detects the distro itself", distro=distro.name)
    log_info(_OP, f"Installing AZNFS {version} with the upstream installer",
             branch="installer", version=version, source=url)

    script = _download_installer(url, cfg["sha256"])
    if script is None or not _patch_installer(script):
        return 3
    if not run(["bash", str(script)], _OP, env=run_env):
        log_error(_OP, f"the AZNFS installer failed; {script} is left in place, "
                       "as in bash", path=str(script), version=version)
        return 3
    remove_paths(script, op=_OP)                    # bash `rm -f aznfs_install.sh`
    log_info(_OP, "AZNFS installed", branch="installer", version=version)
    return 0


def install(env: dict[str, str]) -> int:
    """Install AZNFS from PMC or the upstream installer. Returns 0 or 3."""
    distro = Distro.from_env(env)
    branch = install_branch(distro.name)
    run_env = {**env, **_NONINTERACTIVE}            # bash: export AZNFS_NONINTERACTIVE_INSTALL=1
    log_info(_OP, f"Installing AZNFS for {distro}", distro=distro.name,
             branch=branch, noninteractive=True)
    if branch == "installer":
        return _install_from_upstream_installer(env, run_env, distro)
    return _install_from_pmc(run_env)
