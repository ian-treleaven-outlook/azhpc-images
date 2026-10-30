"""step.py — one step of a build plan (see utils/plans/__init__.py).

A Step is one line of a distro's install.sh: usually a component script
(`$COMPONENT_DIR/install_doca.sh`), sometimes a script in the distro folder
(`./network-config.sh`) or in utils/ (`$UTILS_DIR/clear_history.sh`), or
inline shell with no script of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:                     # only for type hints; avoids an import cycle
    from utils.build_config import BuildConfig

# Where a step's script is (Step.base), as install.sh calls it:
#   "component"  $COMPONENT_DIR/x.sh   components/
#   "distro"     ./x.sh                the distro folder, e.g. distros/rocky9.x
#   "utils"      $UTILS_DIR/x.sh       utils/
_BASE_FOLDERS = {"component": "components", "utils": "utils"}
BASES = (*_BASE_FOLDERS, "distro")


@dataclass(frozen=True)
class Step:
    op: str                       # log label, e.g. "install-rocm"
    script: str = ""              # canonical bash script name (parity + bash exec)
    action: Callable[[dict[str, str]], int] | None = None  # optional python action
    args: tuple[str, ...] = ()
    base: str = "component"       # where the script is: one of BASES (see above)
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

    def __post_init__(self):
        # dataclasses call __post_init__ after filling in the fields. Checking
        # here makes a misspelt base fail when the plan is built, not mid-build.
        if self.base not in BASES:
            raise ValueError(f"step {self.op}: base must be one of {BASES}, not {self.base!r}")

    def script_path(self, repo_root: Path, distro_dir: str) -> Path:
        """Where the bash script is, under `repo_root`.

        `distro_dir` is the build's distro folder (BuildConfig.distro_dir,
        e.g. "distros/rocky9.x"), used for base="distro".
        """
        folder = distro_dir if self.base == "distro" else _BASE_FOLDERS[self.base]
        return Path(repo_root) / folder / self.script

    @property
    def runner(self) -> str:
        """How build() runs this step: "python" (its action) or "bash".

        A step runs as bash when it has no Python action, or when its port is
        marked resync (behind its bash script). build() and `azhpc.py
        --dry-run` both use this, so the dry run shows what a build would do.
        """
        return "python" if self.action is not None and not self.resync else "bash"
