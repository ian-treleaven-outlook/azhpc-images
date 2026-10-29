"""package_pinning.py — Python port of dnf_pin_packages in utils/utilities.sh.

Several components stop later `dnf` operations from upgrading packages they
installed or built, by "pinning" (version-locking) them. The bash helper is:

    function dnf_pin_packages {
        command -v dnf >/dev/null 2>&1 || return 0
        local pkgs=()
        for arg in "$@"; do
            [[ -n "$arg" ]] && pkgs+=("$arg")
        done
        if [[ ${#pkgs[@]} -eq 0 ]]; then
            echo "dnf_pin_packages: no non-empty package arguments provided ..." >&2
            return 1
        fi
        if ! dnf versionlock add "${pkgs[@]}"; then
            echo "dnf_pin_packages: failed to pin package(s): ${pkgs[*]}" >&2
            return 1
        fi
    }

So: no dnf on the machine (Ubuntu) means nothing to do, and success; empty
arguments (usually an unset variable in the caller) are an error; otherwise run
`dnf versionlock add <packages...>` once and report whether it worked.

Package names are passed through exactly as given, including wildcards such as
"libnccl*" (dnf expands those itself, not the shell) and any spaces inside one
argument (bash's quoted "$VAR" passes a space-separated list as ONE argument).
"""

from __future__ import annotations

import shutil

from utils.logger import log_error, log_info
from utils.process import exec_program


def dnf_pin_packages(packages: list[str], op: str,
                     env: dict[str, str] | None = None) -> bool:
    """Version-lock `packages` with dnf so later updates leave them alone.

    `op` is the calling step's log label. Returns True on success (or when dnf
    isn't installed, as in bash), False on failure — after logging why.
    """
    # shutil.which() is Python's `command -v`. Look on the build env's PATH
    # when one is given, since that's the PATH exec_program will use.
    path = env.get("PATH") if env else None
    if shutil.which("dnf", path=path) is None:
        log_info(op, "dnf not found; no packages to pin (as in bash)",
                 packages=" ".join(packages))
        return True

    # Like bash's `[[ -n "$arg" ]]`: drop empty arguments, keep everything else.
    pkgs = [p for p in packages if p]
    if not pkgs:
        log_error(op, "dnf_pin_packages: no non-empty package arguments provided "
                      "(likely an unset variable or empty list)")
        return False

    if exec_program(["dnf", "versionlock", "add", *pkgs], op, env=env) != 0:
        log_error(op, f"dnf_pin_packages: failed to pin package(s): {' '.join(pkgs)}",
                  packages=" ".join(pkgs))
        return False

    log_info(op, f"Pinned package(s) against dnf updates: {' '.join(pkgs)}",
             packages=" ".join(pkgs))
    return True
