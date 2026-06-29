"""package_manager.py — detect the system package manager and build its
non-interactive install command.

Scope note: the project only needs Ubuntu (apt) for now, but detecting the
manager keeps the rest of the code distro-agnostic. `apt-get` is preferred over
`apt` because it has a stable CLI meant for scripting. `tdnf` matters for Azure
HPC / marketplace (Azure Linux) images.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

from utils.logger import log_debug, log_error

# Preference order. apt-get before apt (stable scripting interface).
_MANAGER_ORDER = ["apt-get", "apt", "dnf", "tdnf", "yum", "zypper", "pacman", "apk"]

# How each manager installs a package non-interactively.
_INSTALL_TEMPLATES: dict[str, list[str]] = {
    "apt-get": ["apt-get", "install", "-y"],
    "apt":     ["apt", "install", "-y"],
    "dnf":     ["dnf", "install", "-y"],
    "tdnf":    ["tdnf", "install", "-y"],
    "yum":     ["yum", "install", "-y"],
    "zypper":  ["zypper", "--non-interactive", "install"],
    "pacman":  ["pacman", "-S", "--noconfirm"],
    "apk":     ["apk", "add"],
}


@dataclass(frozen=True)
class PackageManager:
    """A detected package manager and the install command it maps to."""

    name: str
    path: str

    def install_command(self, package: str) -> list[str]:
        """The full argv to install `package` non-interactively."""
        return [*_INSTALL_TEMPLATES[self.name], package]


def detect_package_manager() -> PackageManager | None:
    """Return the first supported package manager found on PATH, else None."""
    for name in _MANAGER_ORDER:
        path = shutil.which(name)
        if path:
            log_debug("detect-pkg-manager", f"Using '{name}' at {path}")
            return PackageManager(name=name, path=path)
    log_error("detect-pkg-manager", "No supported package manager found on PATH")
    return None
