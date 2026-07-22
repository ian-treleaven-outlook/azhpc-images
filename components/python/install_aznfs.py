"""install_aznfs.py — Python port of components/install_aznfs.sh.

Installs the AZNFS mount helper. On Ubuntu and RHEL-family distros it comes from
the distro package feed (PMC); on Azure Linux it's a downloaded installer script
(with its yum calls patched to tdnf). Non-interactive install is forced so it
never prompts during a Packer build.
"""

from __future__ import annotations

import os
from pathlib import Path

from utils.package_installer import PackageInstaller
from utils.component_config import config_for
from utils.download import download_and_verify
from utils.process import exec_program
from utils.logger import log_info, log_error

_PACKAGE_DISTROS = ("ubuntu", "almalinux", "rocky", "rhel")


def _install_from_package_feed():
    """Ubuntu / RHEL-family: install aznfs straight from the distro package feed."""
    installer = PackageInstaller()
    if installer.manager is None:
        return 3
    return 0 if installer.install_package(["aznfs"]) else 3


def _install_from_azurelinux_installer(env):
    """Azure Linux: download the vendor installer, patch yum->tdnf, and run it."""
    cfg = config_for("aznfs", env)
    if not cfg or not cfg.get("version"):
        log_error("install-aznfs",
                  "could not resolve aznfs version from versions.json")
        return 3

    version = cfg["version"]
    sha256 = cfg.get("sha256", "")
    url = (f"https://github.com/Azure/AZNFS-mount/releases/download/"
           f"{version}/aznfs_install.sh")

    try:
        script = download_and_verify(url, sha256, dest_dir=".")
    except Exception as exc:
        log_error("install-aznfs", f"download/verify failed: {exc}")
        return 3

    # The installer script defaults to yum; Azure Linux uses tdnf.
    path = Path(script)
    patched = path.read_text(encoding="utf-8").replace("yum", "tdnf")
    path.write_text(patched, encoding="utf-8")

    run_env = {**env, "AZNFS_NONINTERACTIVE_INSTALL": "1"}
    return exec_program(["bash", str(path)], "install-aznfs", env=run_env)


def install(env):
    """Install AZNFS, dispatching by distro.

    Ubuntu / RHEL-family install from the distro package feed; Azure Linux runs
    a downloaded vendor installer. Returns 0 on success, 3 on failure.
    """
    distribution = env.get("DISTRIBUTION", "")
    # Force non-interactive install (no TTY prompts) for Packer builds.
    os.environ["AZNFS_NONINTERACTIVE_INSTALL"] = "1"
    log_info("install-aznfs", f"Installing AZNFS for {distribution}")

    if any(name in distribution for name in _PACKAGE_DISTROS):
        return _install_from_package_feed()
    elif "azurelinux" in distribution:
        return _install_from_azurelinux_installer(env)
    else:
        log_error("install-aznfs", f"unsupported distribution: {distribution}")
        return 3

