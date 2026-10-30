"""hpc_tuning.py — Python port of components/hpc-tuning.sh.

What this step does
-------------------
Applies the image's HPC system tuning. In order:

  1. Distro-specific clean-up:
       * ubuntu22.04: `pip3 install -U netifaces`, `pip3 install -U PyYAML`,
         `systemctl disable ufw`
       * ubuntu24.04 and ubuntu26.04: `apt install -y python3-netifaces`,
         `apt install -y python3-yaml`, `systemctl disable ufw`
       * almalinux*/rocky*/rhel*: `systemctl disable firewalld` on almalinux8.10
         and rhel8* only (Rocky 8 has no firewalld), then remove the
         azsec-monitor package (auoms, which uses a lot of CPU) if installed
       * azurelinux3.0: remove azsec-monitor if installed
       * any other distro: nothing
  2. NVLINK_RACKSCALE=true (any case): larger TCP buffers in /etc/sysctl.conf
     and NUMA binding (NUMAPolicy=bind, NUMAMask=0-1) in
     /etc/systemd/system.conf.
  3. Raise memlock/nofile/stack limits in /etc/security/limits.conf; on the
     RHEL family also DefaultLimitMEMLOCK=infinity for systemd, then
     `systemctl daemon-reexec`.
  4. Kernel settings in /etc/sysctl.conf (zone reclaim, neighbour table sizes,
     sunrpc TCP slot table).
  5. A boot-time service, sunrpc_tcp_settings, that loads the sunrpc module and
     re-applies sysctl.conf (the sunrpc setting needs the module loaded).
     It is enabled, started, and must be active.
  6. A udev rule that sets read-ahead to 15380 KB on NFS mounts, then
     `udevadm control --reload`.

The original bash, for comparison (heredoc bodies shortened; the exact text
is in the constants below):

    set -ex

    if [[ $DISTRIBUTION == "ubuntu22.04" ]]; then
        pip3 install -U netifaces
        pip3 install -U PyYAML
        systemctl disable ufw
    elif [[ $DISTRIBUTION == "ubuntu24.04" ]] || [[ $DISTRIBUTION == "ubuntu26.04" ]]; then
        apt install -y python3-netifaces
        apt install -y python3-yaml
        systemctl disable ufw
    elif [[ $DISTRIBUTION == almalinux* ]] || [[ $DISTRIBUTION == rocky* ]] || [[ $DISTRIBUTION == rhel* ]]; then
        if [[ $DISTRIBUTION == almalinux8.10 ]] || [[ $DISTRIBUTION == rhel8* ]]; then
            systemctl disable firewalld
        fi
        if dnf list installed azsec-monitor >/dev/null 2>&1; then dnf remove -y azsec-monitor; fi
    elif [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
        if dnf list installed azsec-monitor >/dev/null 2>&1; then dnf remove -y azsec-monitor; fi
    fi

    if [[ "${NVLINK_RACKSCALE,,}" == "true" ]]; then
        echo "net.core.rmem_max = 2147483647" >> /etc/sysctl.conf
        ... (3 more net.* lines)
        echo "NUMAPolicy=bind" | tee -a /etc/systemd/system.conf
        echo "NUMAMask=0-1" | tee -a /etc/systemd/system.conf
    fi

    cat << EOF >> /etc/security/limits.conf
    *               hard    memlock         unlimited
    ... (6 lines)
    EOF

    if [[ $DISTRIBUTION == almalinux* ]] || [[ $DISTRIBUTION == rocky* ]] || [[ $DISTRIBUTION == rhel* ]]; then
        echo "DefaultLimitMEMLOCK=infinity" | sudo tee -a /etc/systemd/system.conf
        echo "DefaultLimitMEMLOCK=infinity" | sudo tee -a /etc/systemd/user.conf
        sudo systemctl daemon-reexec
    fi

    echo "vm.zone_reclaim_mode = 1" >> /etc/sysctl.conf
    ... (4 more lines)

    cat <<EOF >/usr/sbin/sunrpc_tcp_settings.sh
    ... (modprobe sunrpc; sysctl -p)
    EOF
    chmod 755 /usr/sbin/sunrpc_tcp_settings.sh
    cat <<EOF >/etc/systemd/system/sunrpc_tcp_settings.service
    ... (oneshot unit running the script)
    EOF
    systemctl enable sunrpc_tcp_settings
    systemctl start sunrpc_tcp_settings
    systemctl is-active --quiet sunrpc_tcp_settings
    ... (exit if inactive)

    cat > /etc/udev/rules.d/90-nfs-readahead.rules << EOM
    SUBSYSTEM=="bdi", \\
    ACTION=="add", \\
    PROGRAM="/usr/bin/awk -v bdi=\\$kernel 'BEGIN{ret=1} {if (\\$\\$4 == bdi) {ret=0}} END{exit ret}' /proc/fs/nfsfs/volumes", \\
    ATTR{read_ahead_kb}="15380"
    EOM

    udevadm control --reload

Some bash details that matter for exact parity:

  * `[[ $DISTRIBUTION == almalinux* ]]` is a *prefix* match (unlike
    disable_cloudinit.sh, which uses substring matches).
  * `${NVLINK_RACKSCALE,,}` lower-cases the value, so "True" and "TRUE" count.
  * `>>` and `tee -a` append; `>` replaces.
  * The udev rule heredoc is unquoted, so bash turns `\\$kernel` into `$kernel`
    and `\\$\\$4` into `$$4`, AND joins lines ending in a backslash: the rule is
    written as ONE line. (udev would accept either form.) udev itself then
    substitutes `$kernel` with the device name and reads `$$` as a literal
    `$`, so awk sees `$4` (#651; before it the file held a bare `$4`, which
    udev tried to treat as one of its own substitutions).
  * `set -e`: the first failing command stops the script. The "service
    inactive" message after `systemctl is-active` is never reached, because
    `set -e` already stopped the script when is-active returned non-zero. The
    azsec-monitor `dnf list` probe is the exception: it's an `if` condition,
    so its failure just means "not installed".

All of this was checked by running the bash script and this module against
the same scratch directory tree in WSL and comparing every file byte for
byte (contents and modes) and the commands run.

Reruns
------
Every `>>`/`tee -a` is an append, so running the step twice adds the lines
twice, as in bash. That's kept; a warning is logged when a file already
contains the text being appended.

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets this module's install() as the `action=` of the
"hpc-tuning" Step. When a Step has an action, the build calls the Python
function instead of running the bash script. Deleting `action=` from that Step
makes the build run the bash script again. That is why the .sh file is kept.

This step has no entry in versions.json and records no component version, so
it doesn't call config_for() or write_component_version().

Deliberate differences
----------------------
None in behaviour. Mechanical details:

  * Files are written with Python instead of echo/cat/tee, and `sudo` is
    dropped from `sudo tee` and `sudo systemctl daemon-reexec` (the build
    already runs as root, where sudo is a no-op). Each file's appended lines
    are written in one go rather than one echo at a time; the result is
    identical.
  * Package installs go through install_packages(), so bash's
    `apt install -y <pkg>` runs as `apt-get install -y -o Dpkg::Use-Pty=0 <pkg>`
    (the documented normalisation; still one command per package).
"""

from __future__ import annotations

import os

from utils.logger import log_debug, log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import exec_program, run_capture

# The Step's `op` label in plans/steps.py. Every log line from this module uses
# it, so this step's output can be found in a build log.
_OP = "hpc-tuning"

# ---- Files this step changes -------------------------------------------------
# Module-level "constants" (capital names, not changed while the program runs)
# so tests can point them at a temporary directory instead of the real /etc.
_SYSCTL_CONF = "/etc/sysctl.conf"
_SYSTEMD_SYSTEM_CONF = "/etc/systemd/system.conf"
_SYSTEMD_USER_CONF = "/etc/systemd/user.conf"
_LIMITS_CONF = "/etc/security/limits.conf"
_SUNRPC_SCRIPT = "/usr/sbin/sunrpc_tcp_settings.sh"
_SUNRPC_UNIT_FILE = "/etc/systemd/system/sunrpc_tcp_settings.service"
_NFS_READAHEAD_RULES = "/etc/udev/rules.d/90-nfs-readahead.rules"

# The unit name as bash passes it to systemctl (without ".service").
_SUNRPC_SERVICE = "sunrpc_tcp_settings"

# ---- Distro handling ---------------------------------------------------------
# Prefix matches, as in bash: [[ $DISTRIBUTION == almalinux* ]].
_RHEL_FAMILY_PREFIXES = ("almalinux", "rocky", "rhel")

# What the two Ubuntu branches install before `systemctl disable ufw`.
# 22.04 runs pip3 commands as they are; 24.04 installs apt packages, one
# install_packages() call per package, like bash's one `apt install` per line.
_UBUNTU_PIP_COMMANDS = {
    "ubuntu22.04": [["pip3", "install", "-U", "netifaces"],
                    ["pip3", "install", "-U", "PyYAML"]],
    "ubuntu24.04": [],
}
_UBUNTU_APT_PACKAGES = {
    "ubuntu22.04": [],
    "ubuntu24.04": ["python3-netifaces", "python3-yaml"],
}

# Which Ubuntu branch each distribution takes (exact matches, as in bash).
# Ubuntu 26.04 shares the 24.04 branch:
#   elif [[ $DISTRIBUTION == "ubuntu24.04" ]] || [[ $DISTRIBUTION == "ubuntu26.04" ]]
_UBUNTU_BRANCH = {
    "ubuntu22.04": "ubuntu22.04",
    "ubuntu24.04": "ubuntu24.04",
    "ubuntu26.04": "ubuntu24.04",
}

_AZSEC_PACKAGE = "azsec-monitor"

# ---- Text appended or written (exactly as bash produces it) -----------------
# Each echo adds its text plus a newline.
_RACKSCALE_SYSCTL_LINES = (
    "net.core.rmem_max = 2147483647\n"
    "net.core.wmem_max = 2147483647\n"
    "net.ipv4.tcp_rmem = 4096 67108864 1073741824\n"
    "net.ipv4.tcp_wmem = 4096 67108864 1073741824\n"
)
_RACKSCALE_SYSTEMD_LINES = "NUMAPolicy=bind\nNUMAMask=0-1\n"

# The limits heredoc: columns are padded with spaces (no tabs), as in bash.
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

# What bash's unquoted heredoc produces: the backslash-newline pairs are
# removed (so it's one line), and `\$` becomes `$`. `$$4` is udev's escape for
# a literal `$`, so awk gets `$4`; `$kernel` is a udev substitution (the device
# name). The string is split over several source lines only for readability;
# Python joins adjacent string literals, so there are no newlines until the
# final "\n". It was compared byte for byte with bash's output in WSL.
_NFS_READAHEAD_RULE = (
    'SUBSYSTEM=="bdi", '
    'ACTION=="add", '
    'PROGRAM="/usr/bin/awk -v bdi=$kernel \'BEGIN{ret=1} '
    '{if ($$4 == bdi) {ret=0}} END{exit ret}\' /proc/fs/nfsfs/volumes", '
    'ATTR{read_ahead_kb}="15380"\n'
)


# --------------------------------------------------------------------------- #
# Pure helpers: they only compute values, so tests call them directly.
# --------------------------------------------------------------------------- #
def is_rhel_family(distribution: str) -> bool:
    """True for almalinux*, rocky* and rhel* (prefix match, as in bash).

    str.startswith() accepts a tuple and is True if any prefix matches.
    """
    return distribution.startswith(_RHEL_FAMILY_PREFIXES)


def distro_branch(distribution: str) -> str | None:
    """Which branch of bash's first if/elif chain runs, or None for none.

    Returns "ubuntu22.04", "ubuntu24.04" (also for ubuntu26.04, which shares
    that branch), "rhel-family" or "azurelinux3.0". The checks run in bash's
    order.
    """
    if distribution in _UBUNTU_BRANCH:
        return _UBUNTU_BRANCH[distribution]
    if is_rhel_family(distribution):
        return "rhel-family"
    if distribution == "azurelinux3.0":
        return "azurelinux3.0"
    return None


def disables_firewalld(distribution: str) -> bool:
    """True for almalinux8.10 and rhel8* (not Rocky 8, which has no firewalld)."""
    return distribution == "almalinux8.10" or distribution.startswith("rhel8")


def is_rackscale(env: dict[str, str]) -> bool:
    """True when NVLINK_RACKSCALE is "true" in any case, like ${NVLINK_RACKSCALE,,}."""
    return env.get("NVLINK_RACKSCALE", "").lower() == "true"


# --------------------------------------------------------------------------- #
# File and command operations. Each returns True on success, or False after
# logging why it failed. A failed open/write/chmod raises OSError (for example
# PermissionError without root); catching it lets install() keep its contract:
# log and return 3, no traceback.
# --------------------------------------------------------------------------- #
def _append(path: str, text: str) -> bool:
    """Append `text` to `path`, like `>>` / `tee -a` (creates a missing file).

    Not idempotent, exactly like bash: a rerun appends the same lines again.
    We warn when the text is already there, then append anyway.
    """
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="surrogateescape") as f:
                if text in f.read():
                    log_warn(_OP, f"{path} already contains these lines (rerun?); "
                                  "appending them again, as bash does", path=path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
    except OSError as exc:
        log_error(_OP, f"could not append to {path}: {exc}", path=path)
        return False
    lines_added = text.count("\n")
    # splitlines() gives the lines without their "\n"; joining them with " | "
    # puts the added settings on one readable log field.
    log_info(_OP, f"Appended {lines_added} line(s) to {path}", path=path,
             action="appended", lines_added=lines_added,
             content=" | ".join(text.splitlines()))
    return True


def _write(path: str, text: str, mode: int | None = None) -> bool:
    """Replace `path` with `text` (like `cat > path`), then chmod if `mode` is given."""
    existed = os.path.exists(path)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        if mode is not None:
            os.chmod(path, mode)
    except OSError as exc:
        log_error(_OP, f"could not write {path}: {exc}", path=path)
        return False
    action = "replaced" if existed else "created"
    fields = {"path": path, "action": action, "lines": text.count("\n")}
    if mode is not None:
        # oct(0o755) is "0o755"; [2:] drops the "0o" so the log shows "755".
        fields["mode"] = oct(mode)[2:]
    log_info(_OP, f"Wrote {path} ({action})", **fields)
    log_debug(_OP, f"Contents of {path}: {text!r}")
    return True


def _run(command: list[str], env: dict[str, str]) -> bool:
    """Run one command; False (after logging the consequence) if it fails.

    exec_program() already logs the command, its exit code, and on failure its
    last output lines, so here we only add what the failure means.
    """
    rc = exec_program(command, _OP, env=env)
    if rc != 0:
        log_error(_OP, f"'{' '.join(command)}' failed; HPC tuning is incomplete",
                  command=" ".join(command), exit_code=rc)
        return False
    return True


def _remove_azsec_monitor(env: dict[str, str]) -> bool:
    """`if dnf list installed azsec-monitor ...; then dnf remove -y azsec-monitor; fi`."""
    # Bash hides this probe's output (`>/dev/null 2>&1`). run_capture() keeps
    # it out of the INFO log too (output goes to DEBUG) and returns the exit
    # code. A non-zero exit is the normal "not installed" answer, so
    # failure_is_error=False stops it being logged as a failed command.
    rc, _ = run_capture(["dnf", "list", "installed", _AZSEC_PACKAGE], _OP, env=env,
                        failure_is_error=False)
    if rc != 0:
        log_info(_OP, f"{_AZSEC_PACKAGE} is not installed; nothing to remove",
                 package=_AZSEC_PACKAGE, installed=False)
        return True
    log_info(_OP, f"{_AZSEC_PACKAGE} (auoms) is installed; removing it to "
                  "avoid its CPU usage", package=_AZSEC_PACKAGE, installed=True)
    if not _run(["dnf", "remove", "-y", _AZSEC_PACKAGE], env):
        return False
    log_info(_OP, f"Removed {_AZSEC_PACKAGE}", package=_AZSEC_PACKAGE)
    return True


def _distro_cleanup(env: dict[str, str], distribution: str, branch: str | None) -> bool:
    """Step 1: the distro-specific if/elif chain."""
    if branch is None:
        log_info(_OP, "No distro-specific tuning for this distribution",
                 distro=distribution, branch="none")
        return True

    if branch in _UBUNTU_PIP_COMMANDS:
        log_info(_OP, "Ubuntu: installing netifaces/PyYAML and disabling ufw",
                 distro=distribution, branch=branch)
        for command in _UBUNTU_PIP_COMMANDS[branch]:
            if not _run(command, env):
                return False
        for package in _UBUNTU_APT_PACKAGES[branch]:
            if not install_packages([package], env, _OP):
                return False
        if not _run(["systemctl", "disable", "ufw"], env):
            return False
        log_info(_OP, "Disabled the ufw service", service="ufw")
        return True

    if branch == "rhel-family":
        firewalld = disables_firewalld(distribution)
        log_info(_OP, "RHEL family: " + ("disabling firewalld, then " if firewalld
                                         else "firewalld left alone, ")
                 + f"removing {_AZSEC_PACKAGE} if installed",
                 distro=distribution, branch=branch, disable_firewalld=firewalld)
        if firewalld:
            if not _run(["systemctl", "disable", "firewalld"], env):
                return False
            log_info(_OP, "Disabled the firewalld service", service="firewalld")
        return _remove_azsec_monitor(env)

    # azurelinux3.0
    log_info(_OP, f"Azure Linux: removing {_AZSEC_PACKAGE} if installed",
             distro=distribution, branch=branch)
    return _remove_azsec_monitor(env)


def install(env: dict[str, str]) -> int:
    """Apply the HPC system tuning (see the module docstring for the steps).

    `env` is the build environment made by component_env() in build_config.py
    plus the caller's environment. DISTRIBUTION and NVLINK_RACKSCALE are read;
    env is also passed on to every command.

    Returns 0 on success and 3 on failure.
    """
    distribution = env.get("DISTRIBUTION", "")
    branch = distro_branch(distribution)
    rackscale = is_rackscale(env)
    rhel = is_rhel_family(distribution)

    log_info(_OP, "Applying HPC tuning", distro=distribution,
             branch=branch or "none", nvlink_rackscale=rackscale)

    # ---- 1. distro-specific packages/services ------------------------------
    if not _distro_cleanup(env, distribution, branch):
        return 3

    # ---- 2. NVLink rack-scale (e.g. GB200) network + NUMA settings ---------
    if rackscale:
        log_info(_OP, "NVLINK_RACKSCALE is true: adding TCP buffer sysctls and "
                      "NUMA binding", nvlink_rackscale=True)
        if not _append(_SYSCTL_CONF, _RACKSCALE_SYSCTL_LINES):
            return 3
        if not _append(_SYSTEMD_SYSTEM_CONF, _RACKSCALE_SYSTEMD_LINES):
            return 3
    else:
        log_info(_OP, "NVLINK_RACKSCALE is not true: skipping rack-scale settings",
                 nvlink_rackscale=False,
                 nvlink_rackscale_value=env.get("NVLINK_RACKSCALE", ""))

    # ---- 3. resource limits --------------------------------------------------
    if not _append(_LIMITS_CONF, _LIMITS_BLOCK):
        return 3
    if rhel:
        log_info(_OP, "RHEL family: setting DefaultLimitMEMLOCK=infinity for "
                      "systemd", distro=distribution)
        if not _append(_SYSTEMD_SYSTEM_CONF, _MEMLOCK_LINE):
            return 3
        if not _append(_SYSTEMD_USER_CONF, _MEMLOCK_LINE):
            return 3
        if not _run(["systemctl", "daemon-reexec"], env):
            return 3

    # ---- 4. kernel settings ------------------------------------------------
    if not _append(_SYSCTL_CONF, _SYSCTL_LINES):
        return 3

    # ---- 5. sunrpc_tcp_settings service --------------------------------------
    if not _write(_SUNRPC_SCRIPT, _SUNRPC_SCRIPT_TEXT, mode=0o755):
        return 3
    if not _write(_SUNRPC_UNIT_FILE, _SUNRPC_UNIT_TEXT):
        return 3
    for command in (["systemctl", "enable", _SUNRPC_SERVICE],
                    ["systemctl", "start", _SUNRPC_SERVICE]):
        if not _run(command, env):
            return 3
    # `is-active --quiet` prints nothing; its exit code says whether the unit
    # is active (0) or not. Bash stops here under `set -e` if it isn't.
    rc = exec_program(["systemctl", "is-active", "--quiet", _SUNRPC_SERVICE],
                      _OP, env=env)
    if rc != 0:
        log_error(_OP, f"{_SUNRPC_SERVICE} service is not active after starting it",
                  service=_SUNRPC_SERVICE, exit_code=rc)
        return 3
    log_info(_OP, f"Enabled and started {_SUNRPC_SERVICE}; it is active",
             service=_SUNRPC_SERVICE)

    # ---- 6. NFS read-ahead udev rule -----------------------------------------
    if not _write(_NFS_READAHEAD_RULES, _NFS_READAHEAD_RULE):
        return 3
    if not _run(["udevadm", "control", "--reload"], env):
        return 3
    log_info(_OP, "Reloaded udev rules")

    log_info(_OP, "HPC tuning applied", distro=distribution,
             branch=branch or "none", nvlink_rackscale=rackscale,
             systemd_memlock=rhel)
    return 0
