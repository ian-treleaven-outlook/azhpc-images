"""package_installer.py — install packages through the package manager, logging
the story of each install.

Two entry points:

* `install_packages(packages, env, op)` — what the image build's component
  ports use. It installs all `packages` in ONE command with the manager bash
  uses on the build's distro (env["DISTRIBUTION"]), e.g.

      install_packages(["libbz2-dev", "libattr1-dev"], env, "install-mpifileutils")
      # ubuntu24.04 -> apt-get install -y -o Dpkg::Use-Pty=0 libbz2-dev libattr1-dev
      # rocky9.7    -> dnf install -y libbz2-dev libattr1-dev

  A port says *what* to install; this module decides *how*. One command for
  the whole list, like bash's `apt-get install -y a b c`, so dependency
  resolution and failure behaviour match (all or nothing). Make one call per
  bash command to keep bash's grouping.

* `PackageInstaller().install_package([...])` — the `azhpc install` command:
  detects whatever manager is on PATH and installs each package separately,
  carrying on past failures.

Normalised spellings (deliberate, behaviour-neutral differences from bash):
`apt install` / `apt-get install` -> `apt-get install`; `dnf -y install` /
`--assumeyes` -> `dnf install -y`; `-y` placed after the subcommand. One
side effect: `apt` deletes downloaded .deb files after installing and
`apt-get` keeps them in /var/cache/apt/archives, so a build can leave
.deb files that the bash build wouldn't (the cleanup step removes
/var/cache/* for every install before it).
"""

from __future__ import annotations

from utils.logger import log_error, log_info, log_warn
from utils.package_manager import (PackageManager, detect_package_manager,
                                   manager_for_distribution)
from utils.process import exec_program


def install_packages(packages, env: dict[str, str], op: str, *, options=(),
                     cwd: str | None = None) -> bool:
    """Install `packages` in one command with the distro's manager.

    `packages` is a name or a list of names; it may include version pins
    (`pkg=1.2-3` on apt, `pkg-1.2` on dnf) and local package files
    (`./foo.deb`, `/path/foo.rpm`), exactly as bash passes them. `options`
    are extra manager flags placed before the packages, such as
    `["--only-upgrade"]` (apt-get) or `["--disableexcludes=main", "--refresh"]`
    (dnf); the caller knows which family it is in. `cwd` is where the command
    runs (for relative package files). `op` is the calling step's log label.

    Returns True on success. On failure (unknown distro, or the command
    fails) it logs an ERROR and returns False; the caller returns 3.
    """
    if isinstance(packages, str):
        packages = [packages]
    packages = list(packages)
    distribution = env.get("DISTRIBUTION", "")
    manager = manager_for_distribution(distribution)
    if manager is None:
        log_error(op, f"Cannot install packages: no package manager known for "
                      f"distribution '{distribution}'",
                  distribution=distribution, packages=" ".join(packages))
        return False

    command = manager.install_packages_command(packages, options)
    log_info(op, f"Installing {len(packages)} package(s) with {manager.name}",
             package_manager=manager.name, packages=" ".join(packages),
             options=" ".join(options))
    rc = exec_program(command, op, cwd=cwd, env=env)
    if rc != 0:
        log_error(op, f"{manager.name} install failed with exit code {rc}",
                  package_manager=manager.name, packages=" ".join(packages),
                  exit_code=rc)
        return False
    return True


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
