"""disable_cloudinit.py — Python port of components/disable_cloudinit.sh.

What this step does
-------------------
Near the end of an image build we stop cloud-init from managing the network
configuration, and replace it with a fixed configuration. This changes how the
finished image brings up its network, so every file and service change is
logged.

There are two branches, chosen by DISTRIBUTION (and, on Ubuntu, by
TARGET_NODE_TYPE):

  * RHEL family (DISTRIBUTION contains "almalinux", "rocky" or "rhel"):
      1. Append `network: {config: disabled}` to
         /etc/cloud/cloud.cfg.d/99-custom-networking.cfg.
      2. Edit /etc/sysconfig/network-scripts/ifcfg-eth0 in place, deleting the
         lines that start with HWADDR=, DHCP_HOSTNAME= and IPV6INIT=, so the
         image doesn't carry the build VM's MAC address and DHCP host name.
      3. EL8 only (almalinux8.10, rocky8.10, rhel8*): cloud-init keeps putting
         those lines back, so also install a boot-time service that deletes
         them again and restarts NetworkManager:
           /usr/sbin/disable_cloudinit.sh                (mode 755)
           /etc/systemd/system/disable_cloudinit.service
         then `systemctl daemon-reload`, `enable`, `start`, and check that the
         service is active.

  * Everything else (Ubuntu, and also Azure Linux — see below):
      1. Write (replace) /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg
         with `network: {config: disabled}`.
      2. TARGET_NODE_TYPE == baremetal_1p: write /etc/netplan/net.yaml (DHCP on
         the MANA NICs and eth0), chmod 600, delete
         /etc/netplan/50-cloud-init.yaml if it exists, and create
         /etc/cloud/cloud-init.disabled (which turns cloud-init off entirely).
         Any other node type (default azure_vm_regular): replace
         /etc/netplan/50-cloud-init.yaml with a plain "DHCP on eth0" config and
         run `netplan apply`.

The original bash, for comparison (heredoc bodies shortened; the exact text
is in the constants and helpers below):

    set -ex

    if [[ $DISTRIBUTION == *"almalinux"* ]] || [[ $DISTRIBUTION == *"rocky"* ]] || [[ $DISTRIBUTION == *"rhel"* ]]; then
        cat << EOF >> /etc/cloud/cloud.cfg.d/99-custom-networking.cfg
    network: {config: disabled}
    EOF
        sed -i '/^HWADDR=.*$/d' /etc/sysconfig/network-scripts/ifcfg-eth0
        sed -i '/^DHCP_HOSTNAME=.*$/d' /etc/sysconfig/network-scripts/ifcfg-eth0
        sed -i '/^IPV6INIT=.*$/d' /etc/sysconfig/network-scripts/ifcfg-eth0

        if [[ $DISTRIBUTION == "almalinux8.10" ]] || [[ $DISTRIBUTION == "rocky8.10" ]] || [[ $DISTRIBUTION == rhel8* ]]; then
            SCRIPT_PATH="/usr/sbin/disable_cloudinit.sh"
            SERVICE_PATH="/etc/systemd/system/disable_cloudinit.service"
            IFCFG_FILE="/etc/sysconfig/network-scripts/ifcfg-eth0"
            cat << EOF > "$SCRIPT_PATH"
    ... (the three sed lines on "$IFCFG_FILE", then systemctl restart NetworkManager)
    EOF
            chmod 755 "$SCRIPT_PATH"
            cat << EOF > "$SERVICE_PATH"
    ... ([Unit] / [Service] ExecStart=$SCRIPT_PATH / [Install])
    EOF
            systemctl daemon-reload
            systemctl enable disable_cloudinit.service
            systemctl start disable_cloudinit.service
            systemctl is-active --quiet disable_cloudinit.service
            ... (exit if inactive)
        fi
    else
        echo network: {config: disabled} | sudo tee /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg
        if [[ "${TARGET_NODE_TYPE:-azure_vm_regular}" == "baremetal_1p" ]]; then
            bash -c "cat > /etc/netplan/net.yaml" <<'EOF'
    ... (mana-nics + eth0, DHCP)
    EOF
            chmod 600 /etc/netplan/net.yaml
            if [ -f /etc/netplan/50-cloud-init.yaml ]; then
                rm /etc/netplan/50-cloud-init.yaml
            fi
            touch /etc/cloud/cloud-init.disabled
        else
            bash -c "cat > /etc/netplan/50-cloud-init.yaml" <<'EOF'
    ... (eth0, DHCP, version 2)
    EOF
            netplan apply
        fi
    fi

Some bash details that matter for exact parity:

  * `<< EOF` (unquoted delimiter) expands `$VARIABLES` inside the text, so
    "$IFCFG_FILE" and $SCRIPT_PATH are replaced by their values when the files
    are generated. `$/` (in the sed patterns) is not a variable, so it stays as
    written. `<<'EOF'` (quoted delimiter) copies the text literally.
  * `echo network: {config: disabled}` is not brace expansion (there is no
    comma inside the braces), so echo prints the three words separated by
    single spaces: "network: {config: disabled}" plus a newline.
  * `>>` appends; `>` and `tee` (without -a) replace the file.
  * `set -e`: the first failing command stops the script. A missing
    ifcfg-eth0 makes sed fail, so the step fails there. The "service inactive"
    message after `systemctl is-active` is never reached, because `set -e`
    already stopped the script when is-active returned non-zero.

All of this was checked by running the bash script and this module against
the same scratch directory tree in WSL and comparing every file byte for
byte (contents and modes) and the commands run.

Azure Linux: skipped, as in bash
--------------------------------
distros/azurelinux3.0/install.sh does NOT run disable_cloudinit.sh, so
build_plan skips this step on Azure Linux and logs a warning saying so. That
matches bash, but it may be an upstream gap worth a closer look: Azure Linux
images keep cloud-init's network configuration.

If install() is called on Azure Linux anyway (outside build_plan), it behaves
like the bash script would: "azurelinux3.0" contains none of
almalinux/rocky/rhel, so it takes the Ubuntu/netplan branch. Azure Linux uses
systemd-networkd, not netplan, so writing /etc/netplan/50-cloud-init.yaml fails
and the step returns 3, after logging a warning.

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"disable-cloudinit" Step. When a Step has an action, the build calls the
Python function instead of running the bash script. Deleting `action=` from
that Step makes the build run the bash script again. That is why the .sh file
is kept.

This step has no entry in versions.json and records no component version, so
it doesn't call config_for() or write_component_version().

Deliberate differences
----------------------
None in behaviour. Mechanical details:
  * Files are written with Python instead of cat/tee/echo, and without `sudo`
    (the build already runs as root; bash's `sudo tee` is a no-op there).
  * The three `sed -i` deletions are applied to the file contents in one
    read-modify-write instead of three; the result is identical. Python
    rewrites the file in place, while `sed -i` writes a new file and renames
    it over the old one (if ifcfg-eth0 were a symlink, sed would replace the
    link with a regular file; Python writes through it).
"""

from __future__ import annotations

import os
import re

from utils.component_config import DEFAULT_NODE_TYPE
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.process import exec_program

# The Step's `op` label in build_config.py. Every log line from this module uses
# it, so this step's output can be found in a build log.
_OP = "disable-cloudinit"

# ---- RHEL-family paths ------------------------------------------------------
# Module-level "constants" (capital names, not changed while the program runs)
# so tests can point them at a temporary directory instead of the real /etc.
_CUSTOM_NETWORKING_CFG = "/etc/cloud/cloud.cfg.d/99-custom-networking.cfg"
_IFCFG_FILE = "/etc/sysconfig/network-scripts/ifcfg-eth0"
_SCRIPT_PATH = "/usr/sbin/disable_cloudinit.sh"
_SERVICE_PATH = "/etc/systemd/system/disable_cloudinit.service"
_SERVICE = "disable_cloudinit.service"

# ---- Ubuntu (netplan) paths -------------------------------------------------
_DISABLE_NETWORK_CFG = "/etc/cloud/cloud.cfg.d/99-disable-network-config.cfg"
_NETPLAN_BAREMETAL = "/etc/netplan/net.yaml"
_NETPLAN_CLOUD_INIT = "/etc/netplan/50-cloud-init.yaml"
_CLOUD_INIT_DISABLED = "/etc/cloud/cloud-init.disabled"

# Substrings (not prefixes) bash looks for: [[ $DISTRIBUTION == *"rocky"* ]].
_RHEL_FAMILY_MARKERS = ("almalinux", "rocky", "rhel")

# The node type that gets the bare-metal netplan layout. Out-of-repo bare-metal
# overlays rely on this check, so it is kept even though nothing in this repo
# sets TARGET_NODE_TYPE to it.
_BAREMETAL_1P = "baremetal_1p"

# The ifcfg-eth0 keys bash deletes, in bash's order. Each sed command is
# '/^KEY=.*$/d': delete every line that starts with KEY=.
_IFCFG_KEYS_TO_DELETE = ("HWADDR=", "DHCP_HOSTNAME=", "IPV6INIT=")

# What both `cat << EOF` (RHEL) and `echo network: {config: disabled}`
# (Ubuntu) produce: one line plus its newline.
_NETWORK_CONFIG_DISABLED = "network: {config: disabled}\n"

# The two netplan files come from quoted heredocs (<<'EOF'), so they are
# copied literally. The backslash after the opening quotes stops the string
# from starting with a newline. Indentation is exactly as in bash (2 spaces in
# net.yaml, 4 spaces in 50-cloud-init.yaml).
_NETPLAN_BAREMETAL_TEXT = """\
network:
  ethernets:
    mana-nics:
      dhcp4: true
      match:
        driver: mana
        name: e*
    eth0:
      dhcp4: true
"""

_NETPLAN_CLOUD_INIT_TEXT = """\
network:
    ethernets:
        eth0:
            dhcp4: true
    version: 2
"""


# --------------------------------------------------------------------------- #
# Pure helpers: they only compute values, so tests call them directly.
# --------------------------------------------------------------------------- #
def is_rhel_family(distribution: str) -> bool:
    """True when bash would take the RHEL branch.

    Mirrors `[[ $DISTRIBUTION == *"almalinux"* ]] || ... *"rocky"* || ...
    *"rhel"*`: a substring test, so "almalinux9.7" and "rhel8.10" match, and
    "azurelinux3.0" and "ubuntu24.04" don't.
    """
    # any() is True if at least one item in the sequence is true.
    return any(marker in distribution for marker in _RHEL_FAMILY_MARKERS)


def needs_ifcfg_cleanup_service(distribution: str) -> bool:
    """True for the EL8 distros that also get the boot-time cleanup service.

    Mirrors `[[ $DISTRIBUTION == "almalinux8.10" ]] || [[ $DISTRIBUTION ==
    "rocky8.10" ]] || [[ $DISTRIBUTION == rhel8* ]]`: two exact names plus a
    prefix match for RHEL 8.
    """
    return (distribution in ("almalinux8.10", "rocky8.10")
            or distribution.startswith("rhel8"))


def node_type(env: dict[str, str]) -> str:
    """TARGET_NODE_TYPE, defaulting like bash's ${TARGET_NODE_TYPE:-azure_vm_regular}.

    `:-` uses the default when the variable is unset *or empty*. `x or y` in
    Python gives y when x is empty or None, which matches that.
    """
    return env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE


def delete_lines_starting_with(text: str, prefix: str) -> tuple[str, int]:
    """Remove every line that starts with `prefix`, like sed '/^PREFIX.*$/d'.

    Returns the new text and how many lines were removed, for the log.

        delete_lines_starting_with("DEVICE=eth0\\nHWADDR=00:0d\\n", "HWADDR=")
            -> ("DEVICE=eth0\\n", 1)

    sed works line by line, and only "\\n" ends a line. The regular expression
    below splits the text into lines that *keep* their "\\n" (`[^\\n]*\\n`),
    plus a last line without one if the file doesn't end with a newline
    (`[^\\n]+`). Keeping each line's own ending means deleting a line never
    changes the lines around it, exactly as with sed.
    """
    lines = re.findall(r"[^\n]*\n|[^\n]+", text)
    kept = [line for line in lines if not line.startswith(prefix)]
    return "".join(kept), len(lines) - len(kept)


def cleanup_script_text(ifcfg_file: str) -> str:
    """The text of /usr/sbin/disable_cloudinit.sh, as bash's heredoc makes it.

    The heredoc is unquoted, so bash replaces "$IFCFG_FILE" with its value
    (the double quotes stay). In this f-string, {ifcfg_file} is replaced the
    same way. The `$/` in each sed pattern is literal, as in bash.
    """
    return f"""\
#!/bin/bash

# Remove Hardware Mac Address and DHCP Name
sed -i '/^HWADDR=.*$/d' "{ifcfg_file}"
sed -i '/^DHCP_HOSTNAME=.*$/d' "{ifcfg_file}"
sed -i '/^IPV6INIT=.*$/d' "{ifcfg_file}"

# Restart NetworkManager to apply changes
systemctl restart NetworkManager
"""


def cleanup_service_text(script_path: str) -> str:
    """The text of disable_cloudinit.service; $SCRIPT_PATH expanded as in bash."""
    return f"""\
[Unit]
Description=Clean network config after boot
After=network.target NetworkManager.service
Wants=NetworkManager.service

[Service]
Type=oneshot
ExecStart={script_path}
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"""


# --------------------------------------------------------------------------- #
# File operations. Each returns True on success, or False after logging why it
# failed. A failed open/read/write/chmod raises OSError (for example
# FileNotFoundError or PermissionError); catching it lets install() keep its
# contract: log and return 3, no traceback.
# --------------------------------------------------------------------------- #
def _append(path: str, text: str) -> bool:
    """Append `text` to `path`, like `cat << EOF >> path`.

    Appending is not idempotent: rerunning the step adds the text again, as
    bash does. If the text is already there we say so (a rerun is the usual
    cause) but still append, for parity.
    """
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="surrogateescape") as f:
                if text in f.read():
                    log_warn(_OP, f"{path} already contains this text (rerun?); "
                                  "appending it again, as bash does", path=path)
        # Mode "a" creates a missing file and otherwise adds to its end. Like
        # bash, it doesn't create a missing parent directory.
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
    except OSError as exc:
        log_error(_OP, f"could not append to {path}: {exc}", path=path)
        return False
    log_info(_OP, f"Appended to {path}", path=path, action="appended",
             lines_added=text.count("\n"))
    log_debug(_OP, f"Text appended to {path}: {text!r}")
    return True


def _write(path: str, text: str, mode: int | None = None) -> bool:
    """Replace `path` with `text` (like `cat > path`), then chmod if `mode` is given.

    Mode "w" creates a missing file or empties an existing one before writing.
    A new file gets the usual default permissions (0666 minus the umask), as
    with bash's `>`; an existing file keeps its permissions until chmod.
    """
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
    # oct(0o755) is "0o755"; [2:] drops the "0o" so the log shows "755".
    fields = {"path": path, "action": action, "lines": text.count("\n")}
    if mode is not None:
        fields["mode"] = oct(mode)[2:]
    log_info(_OP, f"Wrote {path} ({action})", **fields)
    log_debug(_OP, f"Contents of {path}: {text!r}")
    return True


def _remove_ifcfg_lines(path: str) -> bool:
    """Apply the three sed deletions to ifcfg-eth0, in bash's order.

    sed -i fails when the file is missing, which stops the bash script, so a
    missing file is a failure here too.
    """
    # newline="" turns off Python's newline translation, so "\r\n" stays as is,
    # as sed would leave it. errors="surrogateescape" passes bytes that aren't
    # valid UTF-8 through unchanged; sed doesn't care about the encoding either.
    try:
        with open(path, encoding="utf-8", errors="surrogateescape",
                  newline="") as f:
            text = f.read()
    except OSError as exc:
        log_error(_OP, f"could not read {path} (bash's sed -i fails here too): "
                       f"{exc}", path=path)
        return False

    removed: dict[str, int] = {}
    for key in _IFCFG_KEYS_TO_DELETE:
        text, removed[key] = delete_lines_starting_with(text, key)

    try:
        with open(path, "w", encoding="utf-8", errors="surrogateescape",
                  newline="") as f:
            f.write(text)
    except OSError as exc:
        log_error(_OP, f"could not rewrite {path}: {exc}", path=path)
        return False

    # Field names can't contain "=", so "HWADDR=" becomes "hwaddr_removed".
    counts = {f"{key.rstrip('=').lower()}_removed": n for key, n in removed.items()}
    log_info(_OP, f"Edited {path} in place: deleted HWADDR=, DHCP_HOSTNAME= "
                  "and IPV6INIT= lines", path=path, action="edited", **counts)
    return True


def _run(command: list[str], env: dict[str, str]) -> bool:
    """Run one command; False (after logging the consequence) if it fails.

    exec_program() already logs the command, its exit code, and on failure its
    last output lines, so here we only add what the failure means.
    """
    rc = exec_program(command, _OP, env=env)
    if rc != 0:
        log_error(_OP, f"'{' '.join(command)}' failed; network configuration "
                       "is incomplete", command=" ".join(command), exit_code=rc)
        return False
    return True


# --------------------------------------------------------------------------- #
# The two branches.
# --------------------------------------------------------------------------- #
def _disable_rhel(env: dict[str, str], distribution: str) -> bool:
    """RHEL family: cloud-init config, ifcfg-eth0 cleanup, EL8 service."""
    if not _append(_CUSTOM_NETWORKING_CFG, _NETWORK_CONFIG_DISABLED):
        return False
    if not _remove_ifcfg_lines(_IFCFG_FILE):
        return False

    if not needs_ifcfg_cleanup_service(distribution):
        log_info(_OP, "No boot-time ifcfg cleanup service needed (only "
                      "almalinux8.10, rocky8.10 and rhel8* get one)",
                 distro=distribution, el8_service=False)
        return True

    log_info(_OP, f"Installing {_SERVICE}: on EL8 cloud-init restores the "
                  "deleted ifcfg-eth0 lines at boot",
             distro=distribution, el8_service=True)
    if not _write(_SCRIPT_PATH, cleanup_script_text(_IFCFG_FILE), mode=0o755):
        return False
    if not _write(_SERVICE_PATH, cleanup_service_text(_SCRIPT_PATH)):
        return False

    # Same order as bash. Starting the service runs the cleanup script once now
    # (which also restarts NetworkManager).
    for command in (["systemctl", "daemon-reload"],
                    ["systemctl", "enable", _SERVICE],
                    ["systemctl", "start", _SERVICE]):
        if not _run(command, env):
            return False
    log_info(_OP, f"Enabled and started {_SERVICE}", service=_SERVICE)

    # `is-active --quiet` prints nothing; its exit code says whether the unit
    # is active (0) or not. Bash stops here under `set -e` if it isn't.
    rc = exec_program(["systemctl", "is-active", "--quiet", _SERVICE], _OP,
                      env=env)
    if rc != 0:
        log_error(_OP, f"{_SERVICE} is not active after starting it",
                  service=_SERVICE, exit_code=rc)
        return False
    log_info(_OP, f"{_SERVICE} is active", service=_SERVICE)
    return True


def _disable_netplan(env: dict[str, str], distribution: str,
                     target_node_type: str) -> bool:
    """Ubuntu (and, by bash's branching, Azure Linux): cloud-init + netplan."""
    if distribution.startswith("azurelinux"):
        # Flagged, behaviour kept: see "Azure Linux" in the module docstring.
        log_warn(_OP, "Azure Linux takes the netplan branch, as in bash; it has "
                      "no netplan, so writing /etc/netplan will likely fail. "
                      "distros/azurelinux3.0/install.sh doesn't run this step",
                 distro=distribution)

    # `echo ... | sudo tee FILE` replaces FILE (tee without -a).
    if not _write(_DISABLE_NETWORK_CFG, _NETWORK_CONFIG_DISABLED):
        return False

    if target_node_type == _BAREMETAL_1P:
        log_info(_OP, "Bare-metal 1P node: writing net.yaml and disabling "
                      "cloud-init entirely", node_type=target_node_type)
        if not _write(_NETPLAN_BAREMETAL, _NETPLAN_BAREMETAL_TEXT, mode=0o600):
            return False
        # os.path.isfile() is bash's `[ -f FILE ]`: an existing regular file
        # (following symlinks).
        if os.path.isfile(_NETPLAN_CLOUD_INIT):
            try:
                os.remove(_NETPLAN_CLOUD_INIT)
            except OSError as exc:
                log_error(_OP, f"could not remove {_NETPLAN_CLOUD_INIT}: {exc}",
                          path=_NETPLAN_CLOUD_INIT)
                return False
            log_info(_OP, f"Removed {_NETPLAN_CLOUD_INIT}",
                     path=_NETPLAN_CLOUD_INIT, action="removed")
        else:
            log_info(_OP, f"{_NETPLAN_CLOUD_INIT} not present; nothing to remove",
                     path=_NETPLAN_CLOUD_INIT)
        # Like `touch`: create an empty file, or just update its timestamps.
        existed = os.path.exists(_CLOUD_INIT_DISABLED)
        try:
            with open(_CLOUD_INIT_DISABLED, "a", encoding="utf-8"):
                pass
            os.utime(_CLOUD_INIT_DISABLED)
        except OSError as exc:
            log_error(_OP, f"could not touch {_CLOUD_INIT_DISABLED}: {exc}",
                      path=_CLOUD_INIT_DISABLED)
            return False
        log_info(_OP, f"Touched {_CLOUD_INIT_DISABLED} (cloud-init disabled)",
                 path=_CLOUD_INIT_DISABLED,
                 action="touched" if existed else "created")
        return True

    log_info(_OP, "Regular node: replacing 50-cloud-init.yaml with DHCP on "
                  "eth0 and applying it", node_type=target_node_type)
    if not _write(_NETPLAN_CLOUD_INIT, _NETPLAN_CLOUD_INIT_TEXT):
        return False
    return _run(["netplan", "apply"], env)


def install(env: dict[str, str]) -> int:
    """Disable cloud-init's network management and pin a fixed configuration.

    `env` is the build environment made by component_env() in build_config.py.
    DISTRIBUTION and TARGET_NODE_TYPE choose the branch; env is also passed on
    to systemctl and netplan.

    Returns 0 on success and 3 on failure.
    """
    distribution = env.get("DISTRIBUTION", "")
    target_node_type = node_type(env)
    rhel = is_rhel_family(distribution)
    branch = "rhel-ifcfg" if rhel else "netplan"

    log_info(_OP, "Disabling cloud-init network configuration",
             distro=distribution, node_type=target_node_type, branch=branch)

    if rhel:
        log_info(_OP, "RHEL family: disabling cloud-init networking and "
                      "cleaning ifcfg-eth0", distro=distribution)
        ok = _disable_rhel(env, distribution)
    else:
        log_info(_OP, "Not RHEL family: disabling cloud-init networking and "
                      "writing a netplan config", distro=distribution)
        ok = _disable_netplan(env, distribution, target_node_type)

    if not ok:
        return 3
    log_info(_OP, "Cloud-init network configuration disabled",
             distro=distribution, node_type=target_node_type, branch=branch)
    return 0
