"""remove_unused_packages.py — Python port of
distros/ubuntu22.04/remove_unused_packages.sh.

Removes graphviz from the Ubuntu 22.04 image: its security updates on Jammy
need Ubuntu Pro, so it would leave unfixable CVEs in a published image. The
bash script is one line:

    apt-get remove --auto-remove -y graphviz

distros/ubuntu22.04/install.sh runs it before install_utils.sh, so the
Ubuntu 22.04 plan lists it as a pre-bootstrap step (utils/plans/ubuntu22_04.py).
No other distro folder has the script.

Kept apt behaviour:
  * `--auto-remove` also removes every automatically installed package that
    nothing needs any more, not only graphviz's own dependencies.
  * If graphviz is not installed, apt-get says so and exits 0, so the step
    succeeds without removing anything.
"""

from __future__ import annotations

from utils.logger import log_info
from utils.process import run

_OP = "remove-unused-packages"

# Packages whose security fixes on Ubuntu 22.04 are only available with Ubuntu Pro.
_PRO_ONLY_PACKAGES = ["graphviz"]


# TODO(toolkit): remove_packages() in utils/package_installer.py has no
# --auto-remove option; use it here once it does.
def _remove_with_unused_dependencies(packages: list[str], env: dict[str, str]) -> bool:
    """`apt-get remove --auto-remove -y <packages>`, as the bash script runs it."""
    log_info(_OP, f"Removing {len(packages)} package(s) and dependencies nothing "
                  "else needs (apt-get --auto-remove)",
             package_manager="apt-get", packages=" ".join(packages))
    return run(["apt-get", "remove", "--auto-remove", "-y", *packages], _OP, env=env)


def install(env: dict[str, str]) -> int:
    """Remove the packages that need Ubuntu Pro for security updates."""
    if not _remove_with_unused_dependencies(_PRO_ONLY_PACKAGES, env):
        return 3
    log_info(_OP, "Removed packages that need Ubuntu Pro for security updates",
             packages=" ".join(_PRO_ONLY_PACKAGES))
    return 0
