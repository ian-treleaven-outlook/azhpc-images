"""package_manager.py — pick the system package manager and build its
non-interactive install command.

Two ways to pick one:

* `manager_for_distribution("ubuntu24.04")` follows what the bash scripts do
  on each distro: apt-get on Ubuntu, dnf on Azure Linux and the RHEL family
  (AlmaLinux, Rocky, RHEL). The image build uses this, so its commands match
  bash's.
* `detect_package_manager()` takes the first manager found on PATH. The
  `azhpc install` command uses it, so it works on any system.

`apt-get` is used rather than `apt` because it has a stable CLI meant for
scripting; for installs the two behave the same. `tdnf` matters for Azure
HPC / marketplace (Azure Linux) images, though the bash scripts use dnf there.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

from utils.logger import log_debug, log_error

# Preference order. apt-get before apt (stable scripting interface).
_MANAGER_ORDER = ["apt-get", "apt", "dnf", "tdnf", "yum", "zypper", "pacman", "apk"]

# How each manager installs packages non-interactively.
_INSTALL_TEMPLATES: dict[str, list[str]] = {
    # Dpkg::Use-Pty=0 stops dpkg from allocating a PTY, which otherwise
    # redraws progress with carriage returns and garbles captured output.
    "apt-get": ["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0"],
    "apt":     ["apt", "install", "-y", "-o", "Dpkg::Use-Pty=0"],
    "dnf":     ["dnf", "install", "-y"],
    "tdnf":    ["tdnf", "install", "-y"],
    "yum":     ["yum", "install", "-y"],
    "zypper":  ["zypper", "--non-interactive", "install"],
    "pacman":  ["pacman", "-S", "--noconfirm"],
    "apk":     ["apk", "add"],
}

# DISTRIBUTION prefixes (os-release ID) and the manager bash uses on them.
_DISTRIBUTION_MANAGERS = [
    ("ubuntu", "apt-get"),
    ("azurelinux", "dnf"),
    ("almalinux", "dnf"),
    ("rocky", "dnf"),
    ("rhel", "dnf"),
]


@dataclass(frozen=True)
class PackageManager:
    """A package manager and the install command it maps to."""

    name: str
    path: str

    def install_command(self, package: str) -> list[str]:
        """The full argv to install one `package` non-interactively."""
        return self.install_packages_command([package])

    def install_packages_command(self, packages, options=()) -> list[str]:
        """The full argv to install all `packages` in one command.

        `options` go between the template and the packages, e.g.
        ["--disableexcludes=main", "--refresh"] gives
        dnf install -y --disableexcludes=main --refresh <packages>.
        """
        return [*_INSTALL_TEMPLATES[self.name], *options, *packages]


def manager_for_distribution(distribution: str) -> PackageManager | None:
    """The manager the bash scripts use on `distribution` (e.g. "rocky9.7").

    None for a distribution this project doesn't build. `path` is the bare
    name: the build runs it from PATH, as bash does.
    """
    for prefix, name in _DISTRIBUTION_MANAGERS:
        if distribution.startswith(prefix):
            return PackageManager(name=name, path=name)
    return None


def detect_package_manager() -> PackageManager | None:
    """Return the first supported package manager found on PATH, else None."""
    for name in _MANAGER_ORDER:
        path = shutil.which(name)
        if path:
            log_debug("detect-pkg-manager", f"Using '{name}' at {path}")
            return PackageManager(name=name, path=path)
    log_error("detect-pkg-manager", "No supported package manager found on PATH")
    return None
