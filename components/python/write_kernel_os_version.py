"""write_kernel_os_version.py — Python port of components/write_kernel_os_version.sh.

Near the end of the build, records in /opt/azurehpc/component_versions.txt
which kernel the image was built with and, on Azure Linux, which OS release
package:

  * every distro:  KERNEL = `uname -r`                 e.g. "6.8.0-1015-azure"
  * azurelinux3.0: OS     = `rpm -qf /etc/os-release` minus its last 12
                            characters (bash `${os_version::-12}`)
                                                       e.g. "azurelinux-release-3.0-25"

Kept bash behaviour: the Azure Linux test is an exact match on
"azurelinux3.0", so a future release won't silently reuse the 12-character
suffix rule. The kernel is the running one, as with `uname -r`
(platform.release()).

Deliberate differences: an empty kernel version, a failing rpm, or rpm output
too short to drop 12 characters from fails the step, where bash would record
an empty or truncated value.
"""

from __future__ import annotations

import platform

from utils.component_config import record_version
from utils.logger import log_error
from utils.process import run_capture

_OP = "write-kernel-os-version"

_AZURE_LINUX = "azurelinux3.0"
_OS_RELEASE_FILE = "/etc/os-release"

# What `${os_version::-12}` drops: ".azl3.noarch" on Azure Linux 3.0.
_RPM_SUFFIX_LENGTH = 12


def azure_linux_os_version(rpm_output: str) -> str | None:
    """`rpm -qf /etc/os-release` output minus its last 12 characters.

        "azurelinux-release-3.0-25.azl3.noarch\\n"  ->  "azurelinux-release-3.0-25"

    None if nothing would be left.
    """
    package = rpm_output.strip()
    if len(package) <= _RPM_SUFFIX_LENGTH:
        return None
    return package[:-_RPM_SUFFIX_LENGTH]


def _record_kernel() -> bool:
    kernel = platform.release()                     # bash `uname -r`
    if not kernel:
        log_error(_OP, "could not determine the running kernel version")
        return False
    return record_version("KERNEL", kernel, _OP)


def _record_os(env: dict[str, str]) -> bool:
    """Azure Linux: record the OS release package (see the module docstring)."""
    rc, output = run_capture(["rpm", "-qf", _OS_RELEASE_FILE], _OP, env=env)
    if rc != 0:
        log_error(_OP, f"could not find the package that owns {_OS_RELEASE_FILE}")
        return False
    os_version = azure_linux_os_version(output)
    if os_version is None:
        log_error(_OP, f"unexpected rpm output for {_OS_RELEASE_FILE}: "
                       f"{output.strip()!r}")
        return False
    return record_version("OS", os_version, _OP)


def install(env: dict[str, str]) -> int:
    """Record the kernel version, and on Azure Linux the OS version."""
    if not _record_kernel():
        return 3
    if env.get("DISTRIBUTION", "") == _AZURE_LINUX and not _record_os(env):
        return 3
    return 0
