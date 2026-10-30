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
from utils.process import exec_program, run_capture


def install_packages(packages, env: dict[str, str], op: str, *, options=(),
                     cwd: str | None = None, assume_yes: bool = True) -> bool:
    """Install `packages` in one command with the distro's manager.

    `packages` is a name or a list of names; it may include version pins
    (`pkg=1.2-3` on apt, `pkg-1.2` on dnf) and local package files
    (`./foo.deb`, `/path/foo.rpm`), exactly as bash passes them. `options`
    are extra manager flags placed before the packages, such as
    `["--only-upgrade"]` (apt-get) or `["--disableexcludes=main", "--refresh"]`
    (dnf); the caller knows which family it is in. `cwd` is where the command
    runs (for relative package files). `op` is the calling step's log label.
    assume_yes=False leaves out "-y", only to mirror a bash command that does
    (the manager aborts if it needs to ask; the caller should log_warn why).

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

    command = manager.install_packages_command(packages, options, assume_yes)
    log_info(op, f"Installing {len(packages)} package(s) with {manager.name}",
             package_manager=manager.name, packages=" ".join(packages),
             options=" ".join(options), assume_yes=assume_yes)
    rc = exec_program(command, op, cwd=cwd, env=env)
    if rc != 0:
        log_error(op, f"{manager.name} install failed with exit code {rc}",
                  package_manager=manager.name, packages=" ".join(packages),
                  exit_code=rc)
        return False
    return True


def _manager(env: dict[str, str], op: str, what: str) -> str | None:
    """"apt-get" or "dnf" for the build's distro, or None (logged)."""
    distribution = env.get("DISTRIBUTION", "")
    manager = manager_for_distribution(distribution)
    if manager is None:
        log_error(op, f"Cannot {what}: no package manager known for "
                      f"distribution '{distribution}'", distribution=distribution)
        return None
    return manager.name


def refresh_package_index(env: dict[str, str], op: str) -> bool:
    """Refresh the package lists: `apt-get update` / `dnf makecache --refresh`.

    Bash spells this `apt update`, `apt-get update` or `dnf repolist --refresh`;
    all three just download fresh repository metadata.
    """
    manager = _manager(env, op, "refresh the package index")
    if manager is None:
        return False
    command = ["apt-get", "update"] if manager == "apt-get" else ["dnf", "makecache", "--refresh"]
    return exec_program(command, op, env=env) == 0


def remove_packages(packages, env: dict[str, str], op: str, *, purge: bool = False) -> bool:
    """`apt-get remove|purge -y <packages>` / `dnf remove -y <packages>`.

    purge=True also deletes the packages' configuration files (apt-get purge;
    dnf has no separate purge). All packages in one command, like install.
    """
    packages = [packages] if isinstance(packages, str) else list(packages)
    manager = _manager(env, op, "remove packages")
    if manager is None:
        return False
    verb = ("purge" if purge else "remove") if manager == "apt-get" else "remove"
    log_info(op, f"Removing {len(packages)} package(s) with {manager}",
             package_manager=manager, packages=" ".join(packages), purge=purge)
    return exec_program([manager, verb, "-y", *packages], op, env=env) == 0


def update_packages(packages, env: dict[str, str], op: str) -> bool:
    """Upgrade already-installed `packages` to the newest version.

    `dnf update -y <packages>` / `apt-get install -y --only-upgrade <packages>`.
    """
    packages = [packages] if isinstance(packages, str) else list(packages)
    manager = _manager(env, op, "update packages")
    if manager is None:
        return False
    if manager == "apt-get":
        return install_packages(packages, env, op, options=["--only-upgrade"])
    log_info(op, f"Updating {len(packages)} package(s) with dnf",
             package_manager="dnf", packages=" ".join(packages))
    return exec_program(["dnf", "update", "-y", *packages], op, env=env) == 0


def installed_package_version(package: str, env: dict[str, str], op: str, *,
                              rpm_format: str = "%{VERSION}-%{RELEASE}") -> str | None:
    """The installed version of `package`, or None if it isn't installed (logged).

    Ubuntu: `dpkg-query -W -f '${Version}' <package>`; RPM distros:
    `rpm -q --queryformat <rpm_format> <package>`. Pass rpm_format="%{VERSION}"
    where bash records the version without the release.
    """
    manager = _manager(env, op, "query a package version")
    if manager is None:
        return None
    if manager == "apt-get":
        command = ["dpkg-query", "-W", "-f", "${Version}", package]
    else:
        command = ["rpm", "-q", "--queryformat", rpm_format, package]
    rc, out = run_capture(command, op, env=env)
    version = out.strip()
    if rc != 0 or not version:
        log_error(op, f"{package} is not installed (no version from {command[0]})",
                  package=package)
        return None
    return version


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
