"""step.py — one step of a build plan (see utils/plans/__init__.py).

A Step is one line of a distro's install.sh: usually a component script
(`$COMPONENT_DIR/install_doca.sh`), sometimes a script in the distro folder
(`./network-config.sh`) or inline shell with no script of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:                     # only for type hints; avoids an import cycle
    from utils.build_config import BuildConfig


@dataclass(frozen=True)
class Step:
    op: str                       # log label, e.g. "install-rocm"
    script: str = ""              # canonical bash script name (parity + bash exec)
    action: Callable[[dict[str, str]], int] | None = None  # optional python action
    args: tuple[str, ...] = ()
    base: str = "component"       # "component" (components/) or "distro" (build_dir)
    when: Callable[[BuildConfig], bool] = lambda cfg: True  # condition
    # Why the step is skipped when `when` is false, if that skip deserves a
    # WARNING rather than the usual INFO line — e.g. a distro's install.sh never
    # runs this script, which may be an upstream gap someone should look at.
    skip_warning: str = ""
    # Set when the Python port is behind its bash script, e.g. upstream changed
    # the script after it was ported. The value names the ticket that will
    # update the port, e.g. "ADO 987". While it is set, build() runs the bash
    # script instead of `action` and logs a WARNING, so a build never runs a
    # stale port. Remove it once the port matches the bash again.
    resync: str = ""

    @property
    def runner(self) -> str:
        """How build() runs this step: "python" (its action) or "bash".

        A step runs as bash when it has no Python action, or when its port is
        marked resync (behind its bash script). build() and `azhpc.py
        --dry-run` both use this, so the dry run shows what a build would do.
        """
        return "python" if self.action is not None and not self.resync else "bash"
