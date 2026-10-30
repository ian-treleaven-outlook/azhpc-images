"""install_azsecpack_prereqs.py — Python port of components/install_azsecpack_prereqs.sh.

What this step does
-------------------
AzSecPack (Azure's security monitoring agent, installed on the VM later)
ships its own copy of nmap. That copy does not work on kernels with 64 KB
memory pages (common on aarch64, e.g. GB200). So on a 64 KB-page kernel this
step installs nmap from the distro's package repositories instead:

  * page size is not 64 KB         ->  nothing to do
  * 64 KB and nmap already on PATH ->  nothing to do (log its version)
  * 64 KB, no nmap                 ->  install_packages(["nmap"]):
                                       Ubuntu: apt-get, Azure Linux and the
                                       RHEL family: dnf;
                                       then nmap must be on PATH

The original bash, for comparison (added upstream in #598; #614 only removed
trailing whitespace):

    set -ex

    install_nmap() {
        if command -v nmap >/dev/null 2>&1; then
            echo "nmap already installed: $(nmap --version 2>/dev/null | head -1 || true)"
            return 0
        fi

        if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
            apt-get install -y nmap
        else
            # AzureLinux and RHEL-family(AlmaLinux, Rocky Linux, RHEL, etc.)
            dnf install -y nmap
        fi

        if ! command -v nmap >/dev/null 2>&1; then
            echo "[ERROR] nmap package installed but nmap binary is not available"
            exit 1
        fi
        echo "OK - nmap installed: $(nmap --version 2>/dev/null | head -1 || true)"
    }

    # nmap shipped with azsecpack doesn't work on 64KB page size
    # Install nmap from packeger manager when kernel page size is 64KB (65536 bytes)
    PAGE_SIZE=$(getconf PAGE_SIZE)
    if [ "$PAGE_SIZE" = "65536" ]; then
        install_nmap
    fi

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets this module's install() as the `action=` of the
"install-azsecpack-prereqs" Step, which runs on every distro, as in bash.
Deleting `action=` from that Step makes the build run the bash script again.
That is why the .sh file is kept.

Like every action, install() receives the build environment (a dict of
environment variables) and returns 0 for success or 3 for failure. It logs
problems and returns 3; it does not let exceptions escape to the build loop.

Page size: os.sysconf instead of `getconf PAGE_SIZE`
---------------------------------------------------
`getconf PAGE_SIZE` is a small program that calls the C library function
sysconf(_SC_PAGESIZE) and prints the number. Python's os.sysconf("SC_PAGE_SIZE")
calls that same function directly, so the value is the same (checked in WSL:
both report 4096), without starting a process or parsing its output. A test
also compares the two on the machine running the tests.

Deliberate differences
----------------------
  * If the page size can't be read, Python logs an error and returns 3. Bash
    also stops there (with `set -e`, a failing `PAGE_SIZE=$(getconf ...)`
    ends the script), just with less explanation.
  * `command -v nmap` is done with shutil.which("nmap", path=env["PATH"]): the
    same PATH search bash does (bash's `command -v` would also find a shell
    function or alias named nmap; a non-interactive build script has none).
  * Exit codes: 3 instead of bash's 1 (from `set -e` or `exit 1`), as for every
    action.
  * nmap is installed with install_packages(), which picks the manager from
    DISTRIBUTION (`apt-get install -y -o Dpkg::Use-Pty=0 nmap` on Ubuntu,
    `dnf install -y nmap` otherwise). Bash's else branch would also try dnf
    on a distribution it doesn't know (or an empty one); install_packages()
    logs an error and the step returns 3 instead.
"""

from __future__ import annotations

import os
import shutil

from utils.logger import log_error, log_info
from utils.package_installer import install_packages
from utils.process import run_capture

# The Step's `op` label in plans/steps.py. Every log line from this module uses it.
_OP = "install-azsecpack-prereqs"

# bash: `if [ "$PAGE_SIZE" = "65536" ]`. 64 KB pages, the only size on which
# AzSecPack's own nmap fails.
_PAGE_SIZE_64K = 65536

# The command and the package have the same name on every distro. Bash runs
# no `apt-get update` first: earlier build steps have refreshed the lists.
_NMAP = "nmap"


def page_size() -> int:
    """The kernel's memory page size in bytes, like `getconf PAGE_SIZE`.

    Raises OSError or ValueError if the system can't report it; install()
    turns that into a logged error.
    """
    return os.sysconf("SC_PAGE_SIZE")


def first_line(output: str) -> str:
    """The first line of `output`, like `| head -1` inside bash's $(...).

    "Nmap version 7.94 ( https://nmap.org )\\nPlatform: ...\\n"
        -> "Nmap version 7.94 ( https://nmap.org )"
    ""  -> ""
    """
    # split("\n", 1) cuts at the first newline only; [0] is the text before it.
    return output.split("\n", 1)[0]


def _nmap_path(env: dict[str, str]) -> str | None:
    """Where `command -v nmap` would find nmap on env's PATH, or None."""
    # path=None (no PATH in env) makes shutil.which use this process's PATH.
    return shutil.which(_NMAP, path=env.get("PATH"))


def _nmap_version(env: dict[str, str]) -> str:
    """bash: `$(nmap --version 2>/dev/null | head -1 || true)`.

    Never fails: like `|| true`, a failing nmap just gives what it printed
    (often nothing). failure_is_error=False logs a non-zero exit as INFO.
    """
    _, output = run_capture([_NMAP, "--version"], _OP, env=env,
                            failure_is_error=False)
    return first_line(output)


def install(env: dict[str, str]) -> int:
    """On a 64 KB-page kernel, make sure the distro's nmap is installed.

    `env` is the build environment made by component_env() in build_config.py.
    DISTRIBUTION picks apt-get or dnf (in install_packages()); PATH is
    searched for nmap.

    Returns 0 on success and 3 on failure.
    """
    # ---- 1. Page size (bash: PAGE_SIZE=$(getconf PAGE_SIZE)) ---------------
    try:
        size = page_size()
    except (OSError, ValueError) as exc:
        log_error(_OP, f"could not read the kernel page size: {exc}")
        return 3

    if size != _PAGE_SIZE_64K:
        log_info(_OP, "Page size is not 64 KB; AzSecPack's own nmap works, "
                      "nothing to install", page_size=size,
                 required_page_size=_PAGE_SIZE_64K)
        return 0
    log_info(_OP, "Page size is 64 KB; AzSecPack's own nmap does not work, "
                  "so the distro's nmap is needed", page_size=size)

    # ---- 2. Already installed? (bash: command -v nmap) ---------------------
    path = _nmap_path(env)
    if path is not None:
        version = _nmap_version(env)
        log_info(_OP, f"nmap already installed: {version}", path=path,
                 version=version)
        return 0

    # ---- 3. Install from the distro ----------------------------------------
    distribution = env.get("DISTRIBUTION", "")
    log_info(_OP, "nmap not found on PATH; installing it from the distro",
             distribution=distribution)
    if not install_packages([_NMAP], env, _OP):
        # install_packages has already logged the command and its exit code;
        # say what the failure means.
        log_error(_OP, "could not install nmap; AzSecPack would be left without "
                       "a working nmap on this 64 KB-page kernel",
                  package=_NMAP, distribution=distribution)
        return 3

    # ---- 4. Check the install worked (bash: second command -v nmap) --------
    path = _nmap_path(env)
    if path is None:
        log_error(_OP, "nmap package installed but nmap binary is not available",
                  package=_NMAP, search_path=env.get("PATH"))
        return 3

    version = _nmap_version(env)
    log_info(_OP, f"OK - nmap installed: {version}", path=path, version=version)
    return 0
