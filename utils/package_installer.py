"""package_installer.py — install one or more packages through the detected
package manager, logging the story of each install.

Mirrors the design Ian sketched:

    PackageInstaller().install_package(["net-tools", "tree"])
"""

from __future__ import annotations

from utils.logger import log_error, log_info, log_warn
from utils.package_manager import PackageManager, detect_package_manager
from utils.process import exec_program


class PackageInstaller:
    """Install packages via the system package manager."""

    def __init__(self, manager: PackageManager | None = None):
        # Detection is deferred until first use so constructing the object
        # never fails just because no manager is present.
        self._manager = manager

    @property
    def manager(self) -> PackageManager | None:
        if self._manager is None:
            self._manager = detect_package_manager()
        return self._manager

    def install_package(self, packages) -> bool:
        """Install each package in turn. Returns True only if all succeeded.

        Accepts a single package name or an iterable of names.
        """
        manager = self.manager
        if manager is None:
            log_error("install-package",
                      "Cannot install packages: no package manager detected")
            return False

        if isinstance(packages, str):
            packages = [packages]

        all_succeeded = True
        for package in packages:
            log_info("install-package",
                     f"Installing '{package}' using {manager.name}")
            rc = exec_program(manager.install_command(package), "install-package")
            if rc == 0:
                log_info("install-package", f"Successfully installed '{package}'")
            else:
                log_warn("install-package",
                         f"Failed to install '{package}' (exit code {rc})")
                all_succeeded = False

        return all_succeeded
