"""install_mofed.py — Python port of components/install_mofed.sh.

Azure Linux 3.0 installs the InfiniBand / RDMA driver stack (Mellanox OFED,
"MOFED") from its own repos, where the other distros use DOCA-OFED
(install_doca.py). This step:

  1. On azurelinux3.0, installs MOFED with three `dnf install` commands, as
     bash does: the build dependencies, the kernel packages (the "-hwe"
     variants on aarch64, e.g. GB200), then the user-space stack (rdma-core,
     UCX, perftest, ...). On any other distribution bash installs nothing,
     and neither does this port.
  2. Logs the MOFED release version (the mlnx-ofa_kernel(-hwe)-devel RPM
     version in `dnf list installed`, e.g. "24.10-20") and records OFED as
     `ofed_info` reports it (e.g. "24.10-0.7.0"). The image tests compare
     that against `ofed_info`, so the source version is the one recorded.
  3. Adds a systemd drop-in so openibd starts after udev settles and
     restarts on failure, then reloads systemd and enables openibd.

MOFED has no entry in versions.json; dnf picks the Azure Linux repo version.

Kept bash behaviour:
  * The release version is informational only: bash just echoes it, so an
    empty one is logged as a warning and never fails the step.

Deliberate differences:
  * Installs go through install_packages() (`dnf install -y`), systemctl
    through utils.services.
  * `sudo dnf list installed` runs without sudo (the build already runs as
    root).
  * An unreadable `ofed_info` version fails the step, as for DOCA (see
    utils/ofed.py). Bash would record "" or e.g. "0.7.0-" and carry on.
"""

from __future__ import annotations

import re

from utils.files import write_file
from utils.logger import log_info, log_warn
from utils.ofed import record_ofed_version
from utils.package_installer import install_packages
from utils.process import run_capture
from utils.services import systemctl

_OP = "install-mofed"

# The only distribution install_mofed.sh installs packages on.
_AZURE_LINUX = "azurelinux3.0"

# "Packages for MOFED": what the MOFED packages need to build and run.
_BUILD_DEPENDENCIES = [
    "iptables-devel", "libdb-devel", "libmnl-devel", "libgudev", "fuse-devel",
    "libgudev-devel", "pciutils-devel", "libusb", "openssl-devel", "libusb-devel",
    "flex", "lsof", "automake", "autoconf",
]

# The kernel modules. aarch64 (GB200) runs the HWE kernel, so it gets the
# "-hwe" builds; bash's `else` gives every other architecture the x86_64 list.
_KERNEL_PACKAGES = {
    "aarch64": [
        "mlnx-ofa_kernel", "mlnx-ofa_kernel-hwe-modules", "mlnx-ofa_kernel-hwe-devel",
        "mlnx-ofa_kernel-source", "mft_kernel-hwe", "iser-hwe", "isert-hwe",
        "mlnx-nfsrdma-hwe", "srp-hwe", "xpmem", "xpmem-hwe-modules",
    ],
    "x86_64": [
        "mlnx-ofa_kernel", "mlnx-ofa_kernel-modules", "mlnx-ofa_kernel-devel",
        "mlnx-ofa_kernel-source", "mft_kernel", "iser", "isert",
        "mlnx-nfsrdma", "srp", "xpmem", "xpmem-modules",
    ],
}

_USER_SPACE_PACKAGES = [
    "libibumad", "infiniband-diags", "libibverbs", "libibverbs-utils", "ofed-scripts",
    "mlnx-tools", "librdmacm", "librdmacm-utils", "rdma-core", "rdma-core-devel",
    "mstflint", "ibacm", "ibarr", "ibsim", "perftest", "libxpmem", "libxpmem-devel",
    "mlnx-ethtool", "mlnx-iproute2", "multiperf", "srp_daemon", "ucx", "ucx-cma",
    "ucx-devel", "ucx-ib", "ucx-ib-mlx5", "ucx-rdmacm", "ucx-static", "ucx-xpmem",
    "libunwind", "libunwind-devel",
]

# The package whose RPM version is the MOFED release version.
_RELEASE_PACKAGE = {"aarch64": "mlnx-ofa_kernel-hwe-devel", "x86_64": "mlnx-ofa_kernel-devel"}

_OPENIBD_DROP_IN = "/etc/systemd/system/openibd.service.d/override.conf"
_OPENIBD_OVERRIDE = """\
[Unit]
After=systemd-udev-settle.service
Wants=systemd-udev-settle.service

[Service]
Restart=on-failure
RestartSec=5
"""

# sed 's/.*\s\+\([0-9.]\+-[0-9]\+\)_.*/\1/' in Python's regex syntax.
_RELEASE_SED = re.compile(r".*[ \t\n\r\f\v]+([0-9.]+-[0-9]+)_.*")


def architecture_key(env: dict[str, str]) -> str:
    """The package tables' key: aarch64, or x86_64 for anything else (bash's `else`)."""
    return "aarch64" if env.get("ARCHITECTURE") == "aarch64" else "x86_64"


def release_version(dnf_list_output: str, package: str) -> str:
    """MOFED's release version from `dnf list installed` output, or "".

    Mirrors `dnf list installed | grep <package> | sed 's/.*\\s\\+\\([0-9.]\\+-[0-9]\\+\\)_.*/\\1/'`:

        "mlnx-ofa_kernel-devel.x86_64   24.10-20_6.6.96.2.azl3   @azurelinux"
                                                        ->  "24.10-20"

    Like grep, the package name matches anywhere in a line; like sed, a line
    whose version doesn't fit the pattern is kept whole.
    """
    lines = [line for line in dnf_list_output.split("\n") if package in line]
    return "\n".join(_RELEASE_SED.sub(r"\1", line, count=1) for line in lines)


def _install_mofed(env: dict[str, str]) -> bool:
    """The three dnf installs, on Azure Linux 3.0 only, as in bash."""
    distribution = env.get("DISTRIBUTION", "")
    if distribution != _AZURE_LINUX:
        log_warn(_OP, f"Not {_AZURE_LINUX}: installing no MOFED packages, as in bash",
                 distribution=distribution)
        return True
    arch = architecture_key(env)
    log_info(_OP, "Installing MOFED from the Azure Linux repos", architecture=arch)
    return all(install_packages(packages, env, _OP) for packages in
               (_BUILD_DEPENDENCIES, _KERNEL_PACKAGES[arch], _USER_SPACE_PACKAGES))


def _log_release_version(env: dict[str, str]) -> None:
    """Log the release version bash echoes; never a failure."""
    package = _RELEASE_PACKAGE[architecture_key(env)]
    _, listing = run_capture(["dnf", "list", "installed"], _OP, env=env,
                             failure_is_error=False)
    version = release_version(listing, package)
    if version:
        log_info(_OP, f"MOFED release version {version}", release_version=version,
                 package=package)
    else:
        log_warn(_OP, f"could not read the MOFED release version ({package}) from "
                      "`dnf list installed`; bash only echoes it, so carrying on",
                 package=package)


def _configure_openibd(env: dict[str, str]) -> bool:
    """Start openibd after udev settles, restart it on failure, and enable it."""
    return (write_file(_OPENIBD_DROP_IN, _OPENIBD_OVERRIDE, _OP)
            and systemctl("daemon-reload", op=_OP, env=env)
            and systemctl("enable", "openibd", op=_OP, env=env))


def install(env: dict[str, str]) -> int:
    """Install MOFED (Azure Linux), record OFED and enable openibd. Returns 0 or 3."""
    if not _install_mofed(env):
        return 3
    _log_release_version(env)
    if not record_ofed_version(env, _OP):
        return 3
    if not _configure_openibd(env):
        return 3
    log_info(_OP, "MOFED installed; openibd enabled")
    return 0
