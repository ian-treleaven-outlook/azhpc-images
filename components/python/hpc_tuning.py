"""hpc_tuning.py — Python port of components/hpc-tuning.sh.

Applies the image's HPC system tuning, in bash's order:

  1. Distro clean-up (_DISTRO_TUNING): Ubuntu installs netifaces/PyYAML and
     disables ufw; the RHEL family disables firewalld (AlmaLinux 8.10 and
     RHEL 8 only: Rocky 8 has no firewalld) and removes azsec-monitor (auoms,
     which uses a lot of CPU); Azure Linux 3.0 only removes azsec-monitor.
  2. NVLINK_RACKSCALE=true: larger TCP buffers in sysctl.conf and NUMA
     binding in systemd's system.conf.
  3. Raised memlock/nofile/stack limits; on the RHEL family also
     DefaultLimitMEMLOCK=infinity for systemd, then `systemctl daemon-reexec`.
  4. Kernel settings in sysctl.conf.
  5. The sunrpc_tcp_settings boot service (loads sunrpc, re-applies
     sysctl.conf), enabled, started, and required to be active.
  6. A udev rule setting NFS read-ahead to 15380 KB, then `udevadm control
     --reload`.

The step has no versions.json entry and records no version.

Kept bash quirks
----------------
  * Distro matches are bash's: exact names for Ubuntu and "azurelinux3.0",
    prefix matches (`almalinux*`) for the RHEL family.
  * `${NVLINK_RACKSCALE,,}`: "True" and "TRUE" count as true.
  * Every `>>` / `tee -a` appends, so a rerun adds the lines again. That is
    kept; a warning is logged when the text is already there.
  * The udev heredoc is unquoted, so bash joins its backslash-continued lines
    into ONE line and turns `\\$kernel` / `\\$\\$4` into `$kernel` / `$$4`
    (udev's escape for a literal `$`, #651). _NFS_READAHEAD_RULE is that
    output, byte for byte.
  * `set -e`: the first failing command stops the step. Bash's "service
    inactive!" message is never reached (`is-active` already stopped the
    script); here the inactive service is logged and the step returns 3. The
    azsec-monitor `dnf list` probe is an `if` condition, so its failure only
    means "not installed".

Deliberate differences
----------------------
None in behaviour. `sudo` is dropped (the build runs as root); each file's
appended lines are written in one go rather than one echo at a time (same
result); bash's `apt install -y <pkg>` runs as install_packages()'s
`apt-get install -y` (still one command per package).
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.distro import Distro
from utils.files import write_file
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import run, run_all, run_capture
from utils.services import service_is_active, systemctl

_OP = "hpc-tuning"

# Files this step changes (tests point them into a temporary directory).
_SYSCTL_CONF = "/etc/sysctl.conf"
_SYSTEMD_SYSTEM_CONF = "/etc/systemd/system.conf"
_SYSTEMD_USER_CONF = "/etc/systemd/user.conf"
_LIMITS_CONF = "/etc/security/limits.conf"
_SUNRPC_SCRIPT = "/usr/sbin/sunrpc_tcp_settings.sh"
_SUNRPC_UNIT_FILE = "/etc/systemd/system/sunrpc_tcp_settings.service"
_NFS_READAHEAD_RULES = "/etc/udev/rules.d/90-nfs-readahead.rules"

_SUNRPC_SERVICE = "sunrpc_tcp_settings"
_AZSEC_PACKAGE = "azsec-monitor"


@dataclass(frozen=True)
class DistroTuning:
    """What step 1 does on one distro, in this order."""

    branch: str
    pip_commands: tuple[tuple[str, ...], ...] = ()
    apt_packages: tuple[str, ...] = ()       # one install command per package, like bash
    disable_services: tuple[str, ...] = ()
    remove_azsec_monitor: bool = False


_UBUNTU_2204 = DistroTuning("ubuntu22.04",
                            pip_commands=(("pip3", "install", "-U", "netifaces"),
                                          ("pip3", "install", "-U", "PyYAML")),
                            disable_services=("ufw",))
_UBUNTU_2404 = DistroTuning("ubuntu24.04",
                            apt_packages=("python3-netifaces", "python3-yaml"),
                            disable_services=("ufw",))
_RHEL_FAMILY = DistroTuning("rhel-family", remove_azsec_monitor=True)
_RHEL_FAMILY_FIREWALLD = DistroTuning("rhel-family", disable_services=("firewalld",),
                                      remove_azsec_monitor=True)
_AZURELINUX = DistroTuning("azurelinux3.0", remove_azsec_monitor=True)
_NO_TUNING = DistroTuning("none")

# Bash's exact-name branches. Ubuntu 26.04 shares the 24.04 branch.
_DISTRO_TUNING = {
    "ubuntu22.04": _UBUNTU_2204,
    "ubuntu24.04": _UBUNTU_2404,
    "ubuntu26.04": _UBUNTU_2404,
    "azurelinux3.0": _AZURELINUX,
}

# ---- Text appended or written (exactly as bash produces it) -----------------
_RACKSCALE_SYSCTL_LINES = (
    "net.core.rmem_max = 2147483647\n"
    "net.core.wmem_max = 2147483647\n"
    "net.ipv4.tcp_rmem = 4096 67108864 1073741824\n"
    "net.ipv4.tcp_wmem = 4096 67108864 1073741824\n"
)
_RACKSCALE_SYSTEMD_LINES = "NUMAPolicy=bind\nNUMAMask=0-1\n"

# Columns are padded with spaces (no tabs), as in the bash heredoc.
_LIMITS_BLOCK = """\
*               hard    memlock         unlimited
*               soft    memlock         unlimited
*               hard    nofile          1048576
*               soft    nofile          1048576
*               hard    stack           unlimited
*               soft    stack           unlimited
"""

_MEMLOCK_LINE = "DefaultLimitMEMLOCK=infinity\n"

_SYSCTL_LINES = (
    "vm.zone_reclaim_mode = 1\n"
    "net.ipv4.neigh.default.gc_thresh1 = 4096\n"
    "net.ipv4.neigh.default.gc_thresh2 = 8192\n"
    "net.ipv4.neigh.default.gc_thresh3 = 16384\n"
    "sunrpc.tcp_max_slot_table_entries = 128\n"
)

_SUNRPC_SCRIPT_TEXT = """\
#!/bin/bash

modprobe sunrpc
sysctl -p
"""

_SUNRPC_UNIT_TEXT = """\
[Unit]
Description=Set sunrpc tcp settings

[Service]
User=root
Type=oneshot
ExecStart=/usr/sbin/sunrpc_tcp_settings.sh
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
"""

# One line: adjacent literals are joined, there is no newline until the end.
_NFS_READAHEAD_RULE = (
    'SUBSYSTEM=="bdi", '
    'ACTION=="add", '
    'PROGRAM="/usr/bin/awk -v bdi=$kernel \'BEGIN{ret=1} '
    '{if ($$4 == bdi) {ret=0}} END{exit ret}\' /proc/fs/nfsfs/volumes", '
    'ATTR{read_ahead_kb}="15380"\n'
)


# ---- Pure helpers -------------------------------------------------------------
def disables_firewalld(distribution: str) -> bool:
    """True for almalinux8.10 and rhel8* (not Rocky 8, which has no firewalld)."""
    return distribution == "almalinux8.10" or distribution.startswith("rhel8")


def distro_tuning(distro: Distro) -> DistroTuning:
    """Which branch of bash's first if/elif chain runs on `distro`."""
    if distro.name in _DISTRO_TUNING:
        return _DISTRO_TUNING[distro.name]
    if distro.is_rhel_family:
        return _RHEL_FAMILY_FIREWALLD if disables_firewalld(distro.name) else _RHEL_FAMILY
    return _NO_TUNING


def is_rackscale(env: dict[str, str]) -> bool:
    """True when NVLINK_RACKSCALE is "true" in any case, like ${NVLINK_RACKSCALE,,}."""
    return env.get("NVLINK_RACKSCALE", "").lower() == "true"


# ---- Steps ----------------------------------------------------------------------
def _append(path: str, text: str) -> bool:
    """bash `>>` / `tee -a`: appends again on a rerun, as bash does, with a warning."""
    try:
        with open(path, encoding="utf-8", errors="surrogateescape") as f:
            already_there = text in f.read()
    except OSError:
        already_there = False       # missing (or unreadable: write_file reports it)
    if already_there:
        log_warn(_OP, f"{path} already contains these lines (rerun?); appending "
                      "them again, as bash does", path=path)
    return write_file(path, text, _OP, append=True)


# TODO(ADO 1001): toolkit candidate: remove_installed_packages(packages, env, op) -> bool
# (probe `dnf list installed` / `dpkg -s`, then `dnf remove -y` / `apt-get remove -y`).
def _remove_azsec_monitor(env: dict[str, str]) -> bool:
    """`if dnf list installed azsec-monitor >/dev/null 2>&1; then dnf remove -y ...; fi`."""
    # A non-zero probe is the normal "not installed" answer, not an ERROR.
    rc, _ = run_capture(["dnf", "list", "installed", _AZSEC_PACKAGE], _OP, env=env,
                        failure_is_error=False)
    if rc != 0:
        log_info(_OP, f"{_AZSEC_PACKAGE} is not installed; nothing to remove",
                 package=_AZSEC_PACKAGE, installed=False)
        return True
    log_info(_OP, f"Removing {_AZSEC_PACKAGE} (auoms) to avoid its CPU usage",
             package=_AZSEC_PACKAGE, installed=True)
    return run(["dnf", "remove", "-y", _AZSEC_PACKAGE], _OP, env=env)


def _distro_cleanup(tuning: DistroTuning, env: dict[str, str]) -> bool:
    """Step 1: packages, services and azsec-monitor for this distro."""
    if tuning is _NO_TUNING:
        log_info(_OP, "No distro-specific tuning for this distribution",
                 distro=env.get("DISTRIBUTION", ""), branch=tuning.branch)
        return True
    if not run_all([list(c) for c in tuning.pip_commands], _OP, env=env):
        return False
    for package in tuning.apt_packages:
        if not install_packages([package], env, _OP):
            return False
    for service in tuning.disable_services:
        if not systemctl("disable", service, op=_OP, env=env):
            return False
        log_info(_OP, f"Disabled the {service} service", service=service)
    return not tuning.remove_azsec_monitor or _remove_azsec_monitor(env)


def _rackscale_settings(env: dict[str, str]) -> bool:
    """Step 2: TCP buffer sysctls and NUMA binding for NVLink rack-scale systems."""
    if not is_rackscale(env):
        log_info(_OP, "NVLINK_RACKSCALE is not true: skipping rack-scale settings",
                 nvlink_rackscale=False,
                 nvlink_rackscale_value=env.get("NVLINK_RACKSCALE", ""))
        return True
    log_info(_OP, "NVLINK_RACKSCALE is true: adding TCP buffer sysctls and NUMA "
                  "binding", nvlink_rackscale=True)
    return (_append(_SYSCTL_CONF, _RACKSCALE_SYSCTL_LINES)
            and _append(_SYSTEMD_SYSTEM_CONF, _RACKSCALE_SYSTEMD_LINES))


def _raise_limits(distro: Distro, env: dict[str, str]) -> bool:
    """Step 3: limits.conf, plus systemd's DefaultLimitMEMLOCK on the RHEL family."""
    if not _append(_LIMITS_CONF, _LIMITS_BLOCK):
        return False
    if not distro.is_rhel_family:
        return True
    log_info(_OP, "RHEL family: setting DefaultLimitMEMLOCK=infinity for systemd",
             distro=distro.name)
    return (_append(_SYSTEMD_SYSTEM_CONF, _MEMLOCK_LINE)
            and _append(_SYSTEMD_USER_CONF, _MEMLOCK_LINE)
            and systemctl("daemon-reexec", op=_OP, env=env))


def _start_sunrpc_service(env: dict[str, str]) -> bool:
    """Step 5: write, enable and start sunrpc_tcp_settings; it must be active."""
    if not (write_file(_SUNRPC_SCRIPT, _SUNRPC_SCRIPT_TEXT, _OP, mode=0o755)
            and write_file(_SUNRPC_UNIT_FILE, _SUNRPC_UNIT_TEXT, _OP)
            and systemctl("enable", _SUNRPC_SERVICE, op=_OP, env=env)
            and systemctl("start", _SUNRPC_SERVICE, op=_OP, env=env)):
        return False
    if not service_is_active(_SUNRPC_SERVICE, op=_OP, env=env):
        log_error(_OP, f"{_SUNRPC_SERVICE} service is not active after starting it",
                  service=_SUNRPC_SERVICE)
        return False
    log_info(_OP, f"{_SUNRPC_SERVICE} is enabled and active", service=_SUNRPC_SERVICE)
    return True


def _install_nfs_readahead_rule(env: dict[str, str]) -> bool:
    """Step 6: the NFS read-ahead udev rule, then reload udev."""
    return (write_file(_NFS_READAHEAD_RULES, _NFS_READAHEAD_RULE, _OP)
            and run(["udevadm", "control", "--reload"], _OP, env=env))


def install(env: dict[str, str]) -> int:
    """Apply the HPC system tuning (steps in the module docstring); 0 or 3."""
    distro = Distro.from_env(env)
    tuning = distro_tuning(distro)
    log_info(_OP, "Applying HPC tuning", distro=distro.name, branch=tuning.branch,
             nvlink_rackscale=is_rackscale(env))

    if not _distro_cleanup(tuning, env):
        return 3
    if not _rackscale_settings(env):
        return 3
    if not _raise_limits(distro, env):
        return 3
    if not _append(_SYSCTL_CONF, _SYSCTL_LINES):
        return 3
    if not _start_sunrpc_service(env):
        return 3
    if not _install_nfs_readahead_rule(env):
        return 3

    log_info(_OP, "HPC tuning applied", distro=distro.name, branch=tuning.branch,
             nvlink_rackscale=is_rackscale(env), systemd_memlock=distro.is_rhel_family)
    return 0
