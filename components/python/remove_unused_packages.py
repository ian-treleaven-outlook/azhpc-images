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
from utils.package_installer import remove_packages

_OP = "remove-unused-packages"

# Packages whose security fixes on Ubuntu 22.04 are only available with Ubuntu Pro.
_PRO_ONLY_PACKAGES = ["graphviz"]


def install(env: dict[str, str]) -> int:
    """Remove the packages that need Ubuntu Pro for security updates."""
    # --auto-remove: also remove dependencies nothing else needs, as bash does.
    if not remove_packages(_PRO_ONLY_PACKAGES, env, _OP, options=["--auto-remove"]):
        return 3
    log_info(_OP, "Removed packages that need Ubuntu Pro for security updates",
             packages=" ".join(_PRO_ONLY_PACKAGES))
    return 0
