"""disable_cloudinit.py — Python port of components/disable_cloudinit.sh.

Near the end of the build, stop cloud-init managing the network and pin a
fixed configuration instead. Two branches:

  * RHEL family: append `network: {config: disabled}` to
    /etc/cloud/cloud.cfg.d/99-custom-networking.cfg, and delete the HWADDR=,
    DHCP_HOSTNAME= and IPV6INIT= lines from ifcfg-eth0 so the image doesn't
    carry the build VM's MAC and DHCP name. On EL8 cloud-init puts those
    lines back at boot, so a boot-time service (/usr/sbin/disable_cloudinit.sh
    + disable_cloudinit.service) deletes them again; it is daemon-reloaded,
    enabled, started, and must be active.
  * Everything else: replace 99-disable-network-config.cfg with the same
    line. Then on a baremetal_1p node write /etc/netplan/net.yaml (mode 600),
    delete 50-cloud-init.yaml and touch /etc/cloud/cloud-init.disabled; on
    any other node replace 50-cloud-init.yaml with DHCP on eth0 and run
    `netplan apply`.

The generated texts are the constants and *_text() helpers below; the EL8
ones come from unquoted heredocs, so $IFCFG_FILE / $SCRIPT_PATH are expanded
(the `$/` in the sed patterns is literal).

utils/plans/steps.py makes install() the action of the "disable-cloudinit"
Step; deleting `action=` runs the bash again. The step records no version.

Azure Linux: distros/azurelinux3.0/install.sh doesn't run this step, so
build_plan skips it with a warning. Called anyway, it takes the netplan
branch as bash would (azurelinux contains none of almalinux/rocky/rhel) and
fails, as Azure Linux has no netplan; a warning says so first.

Deliberate differences from bash
--------------------------------
  * Files are written with write_file (no `sudo tee`: the build runs as
    root), which creates missing parent folders. So on Azure Linux the step
    now fails at `netplan apply`, not when writing /etc/netplan.
  * The three `sed -i` deletions are one read-modify-write; the result is the
    same. sed replaces the file, Python writes through a symlink.
  * Distro tests use Distro (bash's `*"rhel"*` contains-tests and `rhel8*`
    become is_rhel_family and major "8"; the same for every distro built).

Kept bash quirks
----------------
  * Appending to 99-custom-networking.cfg isn't idempotent: a rerun appends
    again (warned).
  * EL8 is almalinux8.10 and rocky8.10 exactly, plus any RHEL 8.
  * A missing ifcfg-eth0 fails the step, as `sed -i` does under `set -e`.
"""

from __future__ import annotations

import os
import re

from utils.component_config import DEFAULT_NODE_TYPE
from utils.distro import Distro
from utils.files import write_file
from utils.logger import log_error, log_info, log_warn
from utils.process import run
from utils.services import service_is_active, systemctl

_OP = "disable-cloudinit"

# Module-level so the tests can point them at a temporary folder.
# RHEL family:
_CUSTOM_NETWORKING_CFG = "/etc/cloud/cloud.cfg.d/99-custom-networking.cfg"
_IFCFG_FILE = "/etc/sysconfig/network-scripts/ifcfg-eth0"
_SCRIPT_PATH = "/usr/sbin/disable_cloudinit.sh"
_SERVICE_PATH = "/etc/systemd/system/disable_cloudinit.service"
_SERVICE = "disable_cloudinit.service"
# netplan:
_DISABLE_NETWORK_CFG = "/etc/cloud/cloud.cfg.d/99-disable-network-config.cfg"
_NETPLAN_BAREMETAL = "/etc/netplan/net.yaml"
_NETPLAN_CLOUD_INIT = "/etc/netplan/50-cloud-init.yaml"
_CLOUD_INIT_DISABLED = "/etc/cloud/cloud-init.disabled"

# Out-of-repo bare-metal overlays set this node type, so the check stays even
# though nothing in this repo does.
_BAREMETAL_1P = "baremetal_1p"

# bash: sed -i '/^KEY=.*$/d' for each, in this order.
_IFCFG_KEYS_TO_DELETE = ("HWADDR=", "DHCP_HOSTNAME=", "IPV6INIT=")

# Both `cat << EOF` and `echo network: {config: disabled}` (no comma, so no
# brace expansion) give this line.
_NETWORK_CONFIG_DISABLED = "network: {config: disabled}\n"

# Quoted heredocs (<<'EOF'): literal, indentation exactly as in bash.
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
# Pure helpers
# --------------------------------------------------------------------------- #
def needs_ifcfg_cleanup_service(distro: Distro) -> bool:
    """bash: `== "almalinux8.10" || == "rocky8.10" || == rhel8*`."""
    return distro.name in ("almalinux8.10", "rocky8.10") or (
        distro.is_rhel and distro.major == "8")


def node_type(env: dict[str, str]) -> str:
    """bash's ${TARGET_NODE_TYPE:-azure_vm_regular} (unset or empty -> default)."""
    return env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE


def delete_lines_starting_with(text: str, prefix: str) -> tuple[str, int]:
    """sed '/^PREFIX.*$/d': (text without lines starting with `prefix`, count).

    Each line keeps its own "\\n" (and a last line may have none), so deleting
    a line never changes its neighbours, as with sed.
    """
    lines = re.findall(r"[^\n]*\n|[^\n]+", text)
    kept = [line for line in lines if not line.startswith(prefix)]
    return "".join(kept), len(lines) - len(kept)


def cleanup_script_text(ifcfg_file: str) -> str:
    """/usr/sbin/disable_cloudinit.sh, with "$IFCFG_FILE" expanded as in bash."""
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
    """disable_cloudinit.service, with $SCRIPT_PATH expanded as in bash."""
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
# Steps: each returns True, or False after the failure was logged.
# --------------------------------------------------------------------------- #
def _append_config_disabled(path: str) -> bool:
    """bash's `cat << EOF >> path`; a rerun appends again, so warn when it does."""
    try:
        with open(path, encoding="utf-8", errors="surrogateescape") as handle:
            if _NETWORK_CONFIG_DISABLED in handle.read():
                log_warn(_OP, f"{path} already contains this text (rerun?); "
                              "appending it again, as bash does", path=path)
    except OSError:
        pass       # missing or unreadable: write_file creates it or logs the error
    return write_file(path, _NETWORK_CONFIG_DISABLED, _OP, append=True)


# TODO(ADO 1001): toolkit candidate: edit_file(path, edit, op) -> bool — `sed -i`
# style read/modify/write that round-trips the bytes and fails on a missing file.
def _remove_ifcfg_lines(path: str) -> bool:
    """The three sed deletions on ifcfg-eth0, in bash's order.

    newline="" and surrogateescape keep "\\r\\n" and non-UTF-8 bytes as sed would.
    """
    try:
        with open(path, encoding="utf-8", errors="surrogateescape", newline="") as handle:
            text = handle.read()
        removed = {}
        for key in _IFCFG_KEYS_TO_DELETE:
            text, removed[key] = delete_lines_starting_with(text, key)
        with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as handle:
            handle.write(text)
    except OSError as exc:
        log_error(_OP, f"could not edit {path} (bash's sed -i fails here too): {exc}",
                  path=path)
        return False
    counts = {f"{key.rstrip('=').lower()}_removed": n for key, n in removed.items()}
    log_info(_OP, f"Edited {path} in place: deleted HWADDR=, DHCP_HOSTNAME= "
                  "and IPV6INIT= lines", path=path, action="edited", **counts)
    return True


def _install_cleanup_service(env: dict[str, str]) -> bool:
    """EL8: the boot-time service that deletes the ifcfg-eth0 lines again."""
    if not (write_file(_SCRIPT_PATH, cleanup_script_text(_IFCFG_FILE), _OP, mode=0o755)
            and write_file(_SERVICE_PATH, cleanup_service_text(_SCRIPT_PATH), _OP)
            and systemctl("daemon-reload", op=_OP, env=env)
            and systemctl("enable", _SERVICE, op=_OP, env=env)
            # Starting it runs the cleanup now (and restarts NetworkManager).
            and systemctl("start", _SERVICE, op=_OP, env=env)):
        return False
    if not service_is_active(_SERVICE, op=_OP, env=env):
        log_error(_OP, f"{_SERVICE} is not active after starting it", service=_SERVICE)
        return False
    log_info(_OP, f"Enabled and started {_SERVICE}; it is active", service=_SERVICE)
    return True


def _disable_rhel(env: dict[str, str], distro: Distro) -> bool:
    """RHEL family: cloud-init config, ifcfg-eth0 cleanup, EL8 service."""
    if not (_append_config_disabled(_CUSTOM_NETWORKING_CFG)
            and _remove_ifcfg_lines(_IFCFG_FILE)):
        return False
    if not needs_ifcfg_cleanup_service(distro):
        log_info(_OP, "No boot-time ifcfg cleanup service needed (only "
                      "almalinux8.10, rocky8.10 and rhel8* get one)",
                 distro=str(distro), el8_service=False)
        return True
    log_info(_OP, f"Installing {_SERVICE}: on EL8 cloud-init restores the "
                  "deleted ifcfg-eth0 lines at boot", distro=str(distro), el8_service=True)
    return _install_cleanup_service(env)


def _remove_cloud_init_netplan() -> bool:
    """bash: `if [ -f 50-cloud-init.yaml ]; then rm ...; fi`.

    Plain `rm` (not `rm -rf`) fails the step under `set -e`, so this isn't
    remove_paths, which never fails.
    """
    if not os.path.isfile(_NETPLAN_CLOUD_INIT):
        log_info(_OP, f"{_NETPLAN_CLOUD_INIT} not present; nothing to remove",
                 path=_NETPLAN_CLOUD_INIT)
        return True
    try:
        os.remove(_NETPLAN_CLOUD_INIT)
    except OSError as exc:
        log_error(_OP, f"could not remove {_NETPLAN_CLOUD_INIT}: {exc}",
                  path=_NETPLAN_CLOUD_INIT)
        return False
    log_info(_OP, f"Removed {_NETPLAN_CLOUD_INIT}", path=_NETPLAN_CLOUD_INIT,
             action="removed")
    return True


# TODO(ADO 1001): toolkit candidate: touch(path, op) -> bool — bash `touch`.
def _touch(path: str) -> bool:
    """bash `touch`: create an empty file, or update its timestamps."""
    existed = os.path.exists(path)
    try:
        with open(path, "a", encoding="utf-8"):
            pass
        os.utime(path)
    except OSError as exc:
        log_error(_OP, f"could not touch {path}: {exc}", path=path)
        return False
    log_info(_OP, f"Touched {path}", path=path, action="touched" if existed else "created")
    return True


def _disable_netplan(env: dict[str, str], distro: Distro, target: str) -> bool:
    """Ubuntu (and, by bash's branching, Azure Linux): cloud-init + netplan."""
    if distro.is_azurelinux:
        log_warn(_OP, "Azure Linux takes the netplan branch, as in bash; it has no "
                      "netplan, so `netplan apply` will likely fail. "
                      "distros/azurelinux3.0/install.sh doesn't run this step",
                 distro=str(distro))
    # `echo ... | sudo tee FILE` replaces FILE (tee without -a).
    if not write_file(_DISABLE_NETWORK_CFG, _NETWORK_CONFIG_DISABLED, _OP):
        return False
    if target == _BAREMETAL_1P:
        log_info(_OP, "Bare-metal 1P node: writing net.yaml and disabling "
                      "cloud-init entirely", node_type=target)
        return (write_file(_NETPLAN_BAREMETAL, _NETPLAN_BAREMETAL_TEXT, _OP, mode=0o600)
                and _remove_cloud_init_netplan()
                and _touch(_CLOUD_INIT_DISABLED))
    log_info(_OP, "Regular node: replacing 50-cloud-init.yaml with DHCP on "
                  "eth0 and applying it", node_type=target)
    return (write_file(_NETPLAN_CLOUD_INIT, _NETPLAN_CLOUD_INIT_TEXT, _OP)
            and run(["netplan", "apply"], _OP, env=env))


def install(env: dict[str, str]) -> int:
    """Disable cloud-init's network management and pin a fixed configuration."""
    distro = Distro.from_env(env)
    target = node_type(env)
    # bash: [[ $DISTRIBUTION == *"almalinux"* || *"rocky"* || *"rhel"* ]]
    branch = "rhel-ifcfg" if distro.is_rhel_family else "netplan"
    log_info(_OP, "Disabling cloud-init network configuration",
             distro=str(distro), node_type=target, branch=branch)

    if distro.is_rhel_family:
        ok = _disable_rhel(env, distro)
    else:
        ok = _disable_netplan(env, distro, target)
    if not ok:
        return 3
    log_info(_OP, "Cloud-init network configuration disabled",
             distro=str(distro), node_type=target, branch=branch)
    return 0
