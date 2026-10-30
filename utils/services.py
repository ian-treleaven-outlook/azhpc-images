"""services.py — systemd, the way the bash scripts use it.

Bash calls `systemctl` directly, about 40 times across the components. A port
says what it wants instead:

    if not systemctl("enable", "nvidia-dcgm", op=_OP, env=env):
        return 3
    if not service_is_active("sunrpc_tcp_settings", op=_OP, env=env):
        ...
"""

from __future__ import annotations

from utils.process import exec_program, run


def systemctl(verb: str, *units: str, op: str, env=None) -> bool:
    """bash `systemctl <verb> <units...>`; True on success.

    e.g. systemctl("daemon-reload", op=_OP), systemctl("enable", "--now", "x.service", op=_OP).
    """
    return run(["systemctl", verb, *units], op, env=env)


def service_is_active(unit: str, *, op: str, env=None) -> bool:
    """bash `systemctl is-active --quiet <unit>`: a probe, so "inactive" isn't an ERROR."""
    return exec_program(["systemctl", "is-active", "--quiet", unit], op, env=env,
                        failure_is_error=False) == 0
