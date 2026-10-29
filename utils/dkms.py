"""dkms.py — Python port of check_dkms_status in utils/utilities.sh.

Some vendor packages hide DKMS build failures inside their package scripts:
`apt-get install` / `dnf install` succeeds, but the kernel module was never
built. So, straight after installing such a package, components check that
DKMS really built and installed the module for the running kernel. The bash
helper is:

    check_dkms_status() {
        command -v dkms >/dev/null 2>&1 || return 0

        local kernel_version=$(uname -r)
        local module status found current_installed
        local log_file

        for module in "$@"; do
            found=0
            current_installed=0
            while IFS= read -r status; do
                found=1
                echo "${status}"
                if grep -q "${kernel_version}.*: installed" <<< "${status}"; then
                    current_installed=1
                fi
            done < <(dkms status -m "${module}" 2>/dev/null || true)

            if [[ ${found} -eq 0 ]]; then
                echo "ERROR: DKMS module ${module} has no status entry" >&2
                exit 1
            fi
            if [[ ${current_installed} -ne 1 ]]; then
                echo "ERROR: DKMS module ${module} is not installed for ${kernel_version}" >&2
                for log_file in /var/lib/dkms/${module}/*/build/make.log; do
                    [[ -f "${log_file}" ]] || continue
                    echo "--- ${log_file} ---" >&2
                    tail -n 80 "${log_file}" >&2 || true
                done
                exit 1
            fi
        done
    }

So: no dkms on the machine means nothing to check (success). Otherwise, for
each module, every line of `dkms status -m <module>` is printed; there must be
at least one line, and one of them must say the module is installed for the
running kernel (e.g. "lustre-client/2.15.8, 5.14.0-611.el9.x86_64, x86_64:
installed"). If not, the last 80 lines of each DKMS make.log for the module
are printed and the build stops.

Here the status lines are logged at INFO, a failure is a log_error (with the
make.log tails as its detail block), and the result is returned as a bool
instead of `exit 1`.
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil

from utils.logger import log_error, log_error_detail, log_info, log_warn
from utils.process import run_capture

# Where DKMS keeps its build trees (bash: /var/lib/dkms/${module}/*/build/make.log).
# A module constant so tests can point it at a temporary directory.
_DKMS_TREE = "/var/lib/dkms"

# How many lines of each make.log bash prints (`tail -n 80`).
_MAKE_LOG_TAIL_LINES = 80


def _command_exists(name: str, env: dict[str, str] | None) -> bool:
    """`command -v <name>`: shutil.which() on the build env's PATH (the PATH
    run_capture will use), or ours when no env is given."""
    path = env.get("PATH") if env else None
    return shutil.which(name, path=path) is not None


def running_kernel() -> str:
    """The running kernel's release, like `uname -r`."""
    return platform.release()


def status_lines(output: str) -> list[str]:
    """The lines bash's `while IFS= read -r status` loop sees in `output`.

    `read` only returns success for newline-terminated lines, so a final line
    with no newline is NOT seen by the loop (dkms always ends its output with a
    newline, so this does not matter in practice). Empty lines still count.

        "a\\nb\\n" -> ["a", "b"]      "a\\nb" -> ["a"]      "" -> []
    """
    # split("\n") leaves whatever follows the last newline as the final item:
    # "" for newline-terminated output, or the unterminated line. Drop it.
    return output.split("\n")[:-1]


def installed_pattern(kernel: str) -> re.Pattern[str]:
    """The regex bash greps each status line with: "${kernel_version}.*: installed".

    grep reads it as a basic regular expression (BRE). Kernel releases only
    contain letters, digits and . _ - + ~; of those only "." is special in a
    BRE (any character), and "+" is literal there but special in Python. So
    escape everything, then give "." its regex meaning back, as grep does.
    """
    kernel_regex = re.escape(kernel).replace(r"\.", ".")
    return re.compile(kernel_regex + ".*: installed")


def make_log_tails(module: str) -> str:
    """The make.log tails bash prints when `module` is not installed.

    One "--- <path> ---" header plus the last 80 lines per make.log under
    /var/lib/dkms/<module>/*/build/. "" if there are none. Bash expands the glob
    in the locale's sort order; sorted() is plain character order, which is the
    same for these paths in practice.
    """
    pattern = os.path.join(_DKMS_TREE, module, "*", "build", "make.log")
    parts = []
    for log_file in sorted(glob.glob(pattern)):
        if not os.path.isfile(log_file):        # [[ -f "${log_file}" ]] || continue
            continue
        parts.append(f"--- {log_file} ---")
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace",
                      newline="") as handle:
                text = handle.read()
        except OSError as exc:                  # bash: `tail ... || true`
            parts.append(f"(could not read {log_file}: {exc})")
            continue
        # tail counts "\n"-separated lines; a final line without a newline still
        # counts. (splitlines() would also split on "\r", which tail doesn't.)
        lines = text.split("\n")
        if lines[-1] == "":                     # the newline ending the last line
            lines.pop()
        parts.extend(lines[-_MAKE_LOG_TAIL_LINES:])
    return "\n".join(parts)


def check_dkms_status(modules: list[str], op: str, env: dict[str, str] | None = None,
                      *, kernel: str | None = None) -> bool:
    """Check each DKMS module in `modules` is installed for the running kernel.

    `op` is the calling step's log label; `kernel` defaults to `uname -r` (tests
    pass their own). Returns True when every module is installed (or when dkms
    isn't on the machine, as in bash), False after logging the failure.
    """
    if not _command_exists("dkms", env):
        log_warn(op, "dkms not found; skipping the DKMS status check, as bash's "
                     "check_dkms_status does", modules=" ".join(modules))
        return True

    kernel = kernel or running_kernel()
    pattern = installed_pattern(kernel)

    for module in modules:
        # Bash discards stderr and ignores the exit code (`2>/dev/null || true`);
        # run_capture returns only stdout, and a non-zero exit is just an answer.
        _rc, output = run_capture(["dkms", "status", "-m", module], op, env=env,
                                  failure_is_error=False)
        lines = status_lines(output)
        for line in lines:
            log_info(op, f"DKMS status: {line}", module=module)

        if not lines:
            log_error(op, f"DKMS module {module} has no status entry",
                      module=module, kernel=kernel)
            return False
        if not any(pattern.search(line) for line in lines):
            tails = make_log_tails(module)
            log_error_detail(op, f"DKMS module {module} is not installed for {kernel}",
                             tails or f"(no make.log found under {_DKMS_TREE}/{module})",
                             module=module, kernel=kernel)
            return False
        log_info(op, f"DKMS module {module} is installed for kernel {kernel}",
                 module=module, kernel=kernel, dkms_status="installed")
    return True
