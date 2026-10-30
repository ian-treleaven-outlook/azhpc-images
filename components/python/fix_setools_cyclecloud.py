"""fix_setools_cyclecloud.py — Python port of components/fix_setools_cyclecloud.sh.

AlmaLinux, Rocky and RHEL 8/9 run it right after install_utils.sh. Their
SETools (the python3-setools package, which `semanage` imports) looks up its
own version with pkg_resources:

    __version__ = pkg_resources.get_distribution("setools").version

When the Python package metadata is missing, that import fails, and with it
every `semanage` call, including CycleCloud's. This step replaces the lookup
with the RPM's version as a plain string, then checks that `semanage --help`
works. SETools 4.5.0 dropped the lookup upstream, so newer SETools is left
alone.

As in bash, the step succeeds without changing anything when python3-setools
isn't installed or doesn't use the legacy lookup (both logged at INFO). It
fails when it can't find the installed setools/__init__.py, or when semanage
still fails after the patch.

Differences from bash (no change to the result):
  * One `rpm -q --queryformat %{VERSION}` answers both "is it installed?" and
    "which version?"; bash runs `rpm -q` first and asks for the version again
    just before its sed.
  * When `rpm -ql` lists no setools/__init__.py, bash's `set -e` stops on the
    failing grep without a message; this logs the ERROR bash prints when the
    path it found isn't a file.
"""

from __future__ import annotations

from pathlib import Path

from utils.files import edit_file
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import installed_package_version
from utils.process import run, run_capture

_OP = "fix-setools-cyclecloud"

_PACKAGE = "python3-setools"
_SEMANAGE = "/usr/sbin/semanage"
# The line bash looks for (grep -F) and rewrites (sed).
_LEGACY_LOOKUP = '__version__ = pkg_resources.get_distribution("setools").version'


def pin_version(text: str, version: str) -> str:
    """`text` with the legacy lookup replaced by `__version__ = "<version>"`.

    Like bash's sed without the /g flag, only the first match on each line
    is replaced. sed's lines are split on "\\n" only, so this splits the same
    way (str.splitlines() would also split on other characters such as "\\r").
    """
    pinned = f'__version__ = "{version}"'
    lines = text.split("\n")
    return "\n".join(line.replace(_LEGACY_LOOKUP, pinned, 1) for line in lines)


def find_setools_init(env: dict[str, str]) -> Path | None:
    """The installed setools/__init__.py, from the package's file list, or None (logged).

    bash: `rpm -ql python3-setools | grep '/setools/__init__\\.py$'`, then
    `[[ -f ... ]]`. Several matches fail too: bash would test them as one
    multi-line path.
    """
    # Only the file list matters: bash's pipeline (no pipefail) ignores rpm's exit code.
    _, file_list = run_capture(["rpm", "-ql", _PACKAGE], _OP, env=env)
    matches = [line for line in file_list.splitlines()
               if line.endswith("/setools/__init__.py")]
    if len(matches) != 1 or not Path(matches[0]).is_file():
        log_error(_OP, "unable to locate the installed setools/__init__.py",
                  package=_PACKAGE, matches=" ".join(matches))
        return None
    return Path(matches[0])


def uses_legacy_lookup(init_file: Path) -> bool:
    """bash `grep -Fq <legacy lookup> <file>`: does the file contain it?

    An unreadable file counts as "no", as grep's error exit does in bash's `if`.
    """
    try:
        text = init_file.read_text(encoding="utf-8", errors="surrogateescape")
    except OSError as exc:
        log_warn(_OP, f"could not read {init_file}: {exc}", path=str(init_file))
        return False
    return _LEGACY_LOOKUP in text


def install(env: dict[str, str]) -> int:
    """Pin SETools' version so semanage works without pkg_resources metadata."""
    # missing_ok: "not installed" is a reason to skip, not an error.
    version = installed_package_version(_PACKAGE, env, _OP, rpm_format="%{VERSION}",
                                        missing_ok=True)
    if version is None:
        log_info(_OP, f"{_PACKAGE} is not installed, skipping patch", package=_PACKAGE)
        return 0

    init_file = find_setools_init(env)
    if init_file is None:
        return 3
    if not uses_legacy_lookup(init_file):
        log_info(_OP, "setools does not use the legacy version lookup, skipping patch",
                 path=str(init_file))
        return 0

    # edit_file reads the file, passes its text to the lambda (a small unnamed
    # function), and writes back what the lambda returns, like `sed -i`.
    if not edit_file(init_file, lambda text: pin_version(text, version), _OP):
        return 3
    if not run([_SEMANAGE, "--help"], _OP, env=env):
        log_error(_OP, "semanage still fails after patching setools",
                  path=str(init_file), version=version)
        return 3
    log_info(_OP, f"Verified: semanage works with setools {version}",
             path=str(init_file), version=version)
    return 0
