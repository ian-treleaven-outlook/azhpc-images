"""azurelinux_network_config.py — Python port of distros/azurelinux3.0/network-config.sh.

The last step of the Azure Linux build (install.sh runs it as
./network-config.sh). It:

  1. Writes /etc/systemd/networkd.conf so systemd-networkd uses the DNS
     domain from DHCP as the search domain (UseDomains=yes for DHCPv4 and
     DHCPv6; see systemd/systemd#32194 and microsoft/azurelinux#8741), then
     restarts systemd-networkd and checks that it is active.
  2. Resets the kernel firewall: Azure Linux's iptables service loads rules
     from /etc/systemd/scripts/ip4save and ip6save at boot, so it is stopped
     and disabled, and the INPUT, OUTPUT and FORWARD policies set to ACCEPT.

AlmaLinux, Rocky and RHEL have a different network-config.sh (NetworkManager
and cloud-init settings), ported in rhel_family_network_config.py; this
module is only the Azure Linux one.

Deliberate difference: bash's `systemctl is-active systemd-networkd` prints
the state; this uses `systemctl is-active --quiet` (service_is_active) and
logs an error if it is not active. Both stop the build there.
"""

from __future__ import annotations

from utils.files import write_file
from utils.logger import log_error, log_info
from utils.process import run_all
from utils.services import service_is_active, systemctl

_OP = "network-config"

_NETWORKD_CONF = "/etc/systemd/networkd.conf"
# The bash heredoc, word for word (the commented-out lines are systemd's defaults).
_NETWORKD_SETTINGS = """\
[Network]
#SpeedMeter=no
#SpeedMeterIntervalSec=10sec
#ManageForeignRoutingPolicyRules=yes
#ManageForeignRoutes=yes
#RouteTable=

[DHCPv4]
#DUIDType=vendor
#DUIDRawData=
UseDomains=yes

[DHCPv6]
#DUIDType=vendor
#DUIDRawData=
UseDomains=yes
"""

_NETWORKD = "systemd-networkd"
_IPTABLES_SERVICE = "iptables"
_FIREWALL_CHAINS = ("INPUT", "OUTPUT", "FORWARD")


def accept_policy_commands() -> list[list[str]]:
    """`iptables -P <chain> ACCEPT` for each built-in chain, in bash's order."""
    return [["iptables", "-P", chain, "ACCEPT"] for chain in _FIREWALL_CHAINS]


def _use_dhcp_search_domains(env: dict[str, str]) -> bool:
    """Write networkd.conf, restart systemd-networkd, and check it came back."""
    if not (write_file(_NETWORKD_CONF, _NETWORKD_SETTINGS, _OP)
            and systemctl("restart", _NETWORKD, op=_OP, env=env)):
        return False
    if not service_is_active(_NETWORKD, op=_OP, env=env):
        log_error(_OP, f"{_NETWORKD} is not active after the restart", unit=_NETWORKD)
        return False
    log_info(_OP, "systemd-networkd uses the DHCP domain as DNS search domain",
             path=_NETWORKD_CONF)
    return True


def _reset_firewall(env: dict[str, str]) -> bool:
    """Stop and disable the iptables service, then accept all traffic."""
    if not (systemctl("stop", _IPTABLES_SERVICE, op=_OP, env=env)
            and systemctl("disable", _IPTABLES_SERVICE, op=_OP, env=env)
            and run_all(accept_policy_commands(), _OP, env=env)):
        return False
    log_info(_OP, "iptables service disabled; firewall policies set to ACCEPT",
             chains=" ".join(_FIREWALL_CHAINS))
    return True


def install(env: dict[str, str]) -> int:
    """Use DHCP search domains and reset the firewall. Returns 0 or 3."""
    if not _use_dhcp_search_domains(env):
        return 3
    if not _reset_firewall(env):
        return 3
    return 0
