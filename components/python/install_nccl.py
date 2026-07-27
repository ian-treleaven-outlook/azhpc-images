"""install_nccl.py — NCCL build dependencies.

STAGED, NOT WIRED. This is only the package-install slice of
components/install_nccl.sh (its opening if/elif/else). The tarball build,
make pkg.debian.build / pkg.redhat.build, the dpkg/rpm installs, apt-mark hold,
the /etc/dnf/dnf.conf edit and the nccl-rdma-sharp-plugins clone are all still
bash-only, so install-nccl runs as a bash Step. Wiring this in as-is would just
install the deps twice. Kept and tested so it doesn't rot before the rest lands.

The part that is ported: the old if/elif/else distro branching is now a table
keyed by package manager; detection is handled by detect_package_manager(), so
the only per-distro knowledge left here is the package names themselves.
"""

from __future__ import annotations

from utils.package_installer import PackageInstaller

# Build dependencies, keyed by package manager.
_DEPS = {
    "apt-get": ["build-essential", "devscripts", "debhelper", "fakeroot",
                "zlib1g-dev", "libibverbs-dev"],
    "apt":     ["build-essential", "devscripts", "debhelper", "fakeroot",
                "zlib1g-dev", "libibverbs-dev"],
    "tdnf":    ["rpm-build", "rpmdevtools", "autoconf", "automake", "git", "libtool"],
    "yum":     ["rpm-build", "rpmdevtools"],
    "dnf":     ["rpm-build", "rpmdevtools"],
}


def install_deps(env: dict[str, str]) -> int:
    """Install NCCL build dependencies via PackageInstaller.

    Returns 0 on success, 3 if no package manager is found or an install fails.
    """
    installer = PackageInstaller()
    if installer.manager is None:
        return 3
    deps = _DEPS.get(installer.manager.name, [])
    return 0 if installer.install_package(deps) else 3
