"""write_kernel_os_version.py — Python port of components/write_kernel_os_version.sh.

What this step does
-------------------
Near the end of an image build we record which kernel the image was built
with, and on Azure Linux, which OS release package. The values go into the
shared component-versions file (/opt/azurehpc/component_versions.txt) that
other install steps also write to. Someone inspecting a finished image can
then see what it contains.

  * Every distro:  KERNEL = output of `uname -r`        e.g. "6.8.0-1015-azure"
  * Azure Linux:   OS     = the RPM package that owns /etc/os-release, minus
                            its last 12 characters      e.g. "azurelinux-release-3.0-25"

Every supported distro runs this step and records KERNEL. Azure Linux is the only
one that also records OS, exactly as in bash. Most of the code below is for that
Azure Linux part, because it's the only part that runs a command and has to clean
up the command's output. The KERNEL part is one line on every distro.

The original bash, for comparison:

    if [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
       write_component_version "KERNEL" $(uname -r)
       os_version=$(rpm -qf /etc/os-release)
       write_component_version "OS" ${os_version::-12}
    else
       write_component_version "KERNEL" $(uname -r)
    fi

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"write-kernel-os-version" Step. When a Step has an action, the build calls the
Python function instead of running the bash script. Deleting `action=` from that
Step makes the build run the bash script again. That is why the .sh file is kept.

Like every action, install() receives the build environment (a dict of
environment variables) and returns 0 for success or 3 for failure. It logs
problems and returns 3; it does not let exceptions escape to the build loop.
"""

# Lets type hints such as `str | None` work on Python versions older than 3.10.
# Type hints document expected types; they don't change how the code runs.
from __future__ import annotations

import platform

from utils.component_config import write_component_version
from utils.logger import log_error, log_info
from utils.process import run_capture

# The Step's `op` label in build_config.py. Every log line from this module uses
# it, so this step's output can be found in a build log.
_OP = "write-kernel-os-version"

# The only distro on which bash also records an OS version. This is an exact
# match, as in bash. A future Azure Linux release won't silently reuse the
# 12-character suffix rule below; it would need a deliberate update.
_AZURE_LINUX = "azurelinux3.0"

# The file whose owning RPM package identifies the OS release.
_OS_RELEASE_FILE = "/etc/os-release"

# How many characters bash's `${os_version::-12}` removes from the end of the
# package name. On Azure Linux 3.0 that is the ".azl3.noarch" suffix.
_RPM_SUFFIX_LENGTH = 12


def azure_linux_os_version(rpm_output: str) -> str | None:
    """Turn `rpm -qf /etc/os-release` output into the OS version we record.

    Mirrors bash's `${os_version::-12}`, which drops the last 12 characters:

        "azurelinux-release-3.0-25.azl3.noarch"  ->  "azurelinux-release-3.0-25"
                                  ^^^^^^^^^^^^
                                  12 characters removed (".azl3.noarch")

    This is a "pure" function: it only transforms its input and has no side
    effects, so tests can call it directly without mocking anything.

    Returns None when the output can't produce a version: it's empty, or it's
    too short to remove 12 characters and leave anything. Bash would fail or
    record an empty value there. Treating it as an error gives a clear message.
    """
    # rpm ends its output with a newline. Bash's $(...) removes trailing
    # newlines automatically; in Python, .strip() removes surrounding whitespace.
    package = rpm_output.strip()

    # Without this check, a short string would produce an empty or truncated
    # value below instead of an obvious error.
    if len(package) <= _RPM_SUFFIX_LENGTH:
        return None

    # A slice like package[:-12] means "everything except the last 12
    # characters". It is the Python equivalent of bash's ${package::-12}.
    return package[:-_RPM_SUFFIX_LENGTH]


def _record(component: str, version: str) -> bool:
    """Write one entry to the component-versions file.

    Returns True if it was written, or False after logging why it could not be.
    """
    # write_component_version() writes to a file under /opt. That can fail (for
    # example, without root permissions). File and system errors raise OSError.
    # The "OS" in its name is unrelated to the OS version recorded here.
    # Catching it lets install() keep its contract: log and return 3, no traceback.
    try:
        write_component_version(component, version)
    except OSError as exc:
        log_error(_OP, f"could not record {component}={version}: {exc}")
        return False
    return True


def install(env: dict[str, str]) -> int:
    """Record the kernel version, and on Azure Linux the OS version.

    `env` is the build environment made by component_env() in build_config.py.
    Only DISTRIBUTION is used here, for example "ubuntu24.04" or "azurelinux3.0".

    Returns 0 on success and 3 on failure. These are the build's documented
    exit codes.
    """
    # ---- 1. Kernel version (every distro) --------------------------------
    # platform.release() is Python's equivalent of `uname -r`. Like bash, it
    # reports the kernel that is running now.
    kernel = platform.release()
    if not kernel:
        # An empty string counts as "false" in an `if` test, so this catches "".
        log_error(_OP, "could not determine the running kernel version")
        return 3

    if not _record("KERNEL", kernel):
        return 3
    log_info(_OP, f"Recorded kernel version {kernel}")

    # ---- 2. OS version (Azure Linux only) --------------------------------
    # env.get(key, default) returns the default instead of raising a KeyError
    # when the key is missing. Without a DISTRIBUTION, record the kernel only.
    if env.get("DISTRIBUTION", "") != _AZURE_LINUX:
        return 0

    # run_capture() runs a command and returns two values: the exit code and
    # what the command printed to stdout. `rc, output = ...` unpacks them into
    # two variables. run_capture() already logs the command and any failure code.
    rc, output = run_capture(["rpm", "-qf", _OS_RELEASE_FILE], _OP, env=env)
    if rc != 0:
        log_error(_OP, f"could not find the package that owns {_OS_RELEASE_FILE}")
        return 3

    os_version = azure_linux_os_version(output)
    if os_version is None:
        # !r adds quotes around the value. That shows empty or odd output
        # clearly in the log.
        log_error(_OP, f"unexpected rpm output for {_OS_RELEASE_FILE}: "
                       f"{output.strip()!r}")
        return 3

    if not _record("OS", os_version):
        return 3
    log_info(_OP, f"Recorded OS version {os_version}")
    return 0
