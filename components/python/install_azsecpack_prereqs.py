"""install_azsecpack_prereqs.py — Python port of components/install_azsecpack_prereqs.sh.

AzSecPack (Azure's security monitoring agent, installed on the VM later)
ships its own nmap, which does not work on kernels with 64 KB memory pages
(common on aarch64, e.g. GB200). So on a 64 KB-page kernel this step makes
sure the distro's nmap is installed (bash's install_nmap):

  * page size is not 64 KB          ->  nothing to do
  * 64 KB and nmap already on PATH  ->  nothing to do (log its version)
  * 64 KB, no nmap                  ->  install_packages(["nmap"]), then nmap
                                        must be on PATH

Kept bash behaviour:
  * The version logged is `nmap --version | head -1 || true`: a failing nmap
    just gives an empty version, never a failure.
  * No `apt-get update` first; earlier build steps refreshed the lists.

Deliberate differences:
  * The page size comes from os.sysconf("SC_PAGE_SIZE"), the C call
    `getconf PAGE_SIZE` makes and prints (a test compares the two). If it
    can't be read the step fails, as bash's `set -e` does.
  * `command -v nmap` is command_exists(): the same PATH search (bash would
    also find a function or alias, which a build script has none of).
  * install_packages() picks the manager from DISTRIBUTION, as bash's
    `== *"ubuntu"*` test does (`apt-get install -y -o Dpkg::Use-Pty=0 nmap`
    on Ubuntu, `dnf install -y nmap` on the RPM distros). On a distribution
    it doesn't know (or none) the step fails, where bash's `else` would try
    dnf.
"""

from __future__ import annotations

import os

from utils.logger import log_error, log_info
from utils.package_installer import install_packages
from utils.process import command_exists, run_capture

_OP = "install-azsecpack-prereqs"

# The only page size on which AzSecPack's own nmap fails (bash: "65536").
_PAGE_SIZE_64K = 65536

# The command and the package have the same name on every distro.
_NMAP = "nmap"


def page_size() -> int:
    """The kernel's memory page size in bytes, like `getconf PAGE_SIZE`.

    Raises OSError or ValueError if the system can't report it.
    """
    return os.sysconf("SC_PAGE_SIZE")


def first_line(output: str) -> str:
    """The first line of `output`, like `| head -1` inside bash's $(...)."""
    return output.split("\n", 1)[0]


def _read_page_size() -> int | None:
    try:
        return page_size()
    except (OSError, ValueError) as exc:
        log_error(_OP, f"could not read the kernel page size: {exc}")
        return None


def _nmap_version(env: dict[str, str]) -> str:
    """bash `$(nmap --version 2>/dev/null | head -1 || true)`: never fails."""
    _, output = run_capture([_NMAP, "--version"], _OP, env=env,
                            failure_is_error=False)
    return first_line(output)


def _ensure_nmap(env: dict[str, str]) -> int:
    """bash install_nmap(): install the distro's nmap unless it is on PATH."""
    if command_exists(_NMAP, env):
        version = _nmap_version(env)
        log_info(_OP, f"nmap already installed: {version}", version=version)
        return 0

    distribution = env.get("DISTRIBUTION", "")
    log_info(_OP, "nmap not found on PATH; installing it from the distro",
             distribution=distribution)
    if not install_packages([_NMAP], env, _OP):
        log_error(_OP, "could not install nmap; AzSecPack would be left without "
                       "a working nmap on this 64 KB-page kernel",
                  package=_NMAP, distribution=distribution)
        return 3
    if not command_exists(_NMAP, env):
        log_error(_OP, "nmap package installed but nmap binary is not available",
                  package=_NMAP, search_path=env.get("PATH"))
        return 3

    version = _nmap_version(env)
    log_info(_OP, f"OK - nmap installed: {version}", version=version)
    return 0


def install(env: dict[str, str]) -> int:
    """On a 64 KB-page kernel, make sure the distro's nmap is installed."""
    size = _read_page_size()
    if size is None:
        return 3
    if size != _PAGE_SIZE_64K:
        log_info(_OP, "Page size is not 64 KB; AzSecPack's own nmap works, "
                      "nothing to install", page_size=size,
                 required_page_size=_PAGE_SIZE_64K)
        return 0
    log_info(_OP, "Page size is 64 KB; AzSecPack's own nmap does not work, "
                  "so the distro's nmap is needed", page_size=size)
    return _ensure_nmap(env)
