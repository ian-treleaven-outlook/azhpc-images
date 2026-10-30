"""disable_auto_upgrade.py — Python port of distros/ubuntu26.04/disable_auto_upgrade.sh.

Turns off Ubuntu's background "unattended-upgrades", which on a build VM (and
a finished HPC image) can hold the apt/dpkg lock or change pinned packages:

  1. Sets `APT::Periodic::Unattended-Upgrade "0";` in
     /etc/apt/apt.conf.d/20auto-upgrades, with bash's
         sed -i 's/APT::Periodic::Unattended-Upgrade ".*/APT::Periodic::Unattended-Upgrade "0";/' /etc/apt/apt.conf.d/20auto-upgrades
  2. `systemctl stop`, `disable`, then `mask` unattended-upgrades.service.

Not in any build plan today: upstream #623 deleted the ubuntu22.04/24.04
copies (Packer's prerequisites.sh now turns background apt off earlier). Only
distros/ubuntu26.04/install.sh still runs the script; the module is kept for
the Ubuntu 26.04 plan (ADO 993). On a distro without the apt file it fails
before running systemctl.

Kept sed behaviour: `.*` runs to the end of the line, so everything after the
opening quote (comments, a "\\r", a second setting) is replaced; text before
the match is kept; lines without a match are unchanged.
"""

from __future__ import annotations

import re

from utils.logger import log_error, log_info
from utils.services import systemctl

_OP = "disable-auto-upgrade"

_AUTO_UPGRADES_FILE = "/etc/apt/apt.conf.d/20auto-upgrades"
_SERVICE = "unattended-upgrades.service"

# sed's pattern; `.` and `*` are its only special characters, as in Python.
_SETTING_PATTERN = re.compile(r'APT::Periodic::Unattended-Upgrade ".*')
_SETTING_OFF = 'APT::Periodic::Unattended-Upgrade "0";'


def set_unattended_upgrade_off(text: str) -> str:
    """The sed substitution on `text`, line by line (first match per line).

    Splits on "\\n" only, as sed does, so a missing final newline stays missing.
    """
    return "\n".join(_SETTING_PATTERN.sub(_SETTING_OFF, line, count=1)
                     for line in text.split("\n"))


def _rewrite_auto_upgrades_file() -> bool:
    """bash `sed -i` on the apt file.

    newline="" keeps "\\r\\n" as it is and surrogateescape passes non-UTF-8
    bytes through, as sed does; write_file() would do neither.
    """
    # TODO(ADO 1001): toolkit candidate: byte-preserving `sed -i`, e.g.
    # edit_file(path, transform: Callable[[str], str], op) -> (old, new) | None
    try:
        with open(_AUTO_UPGRADES_FILE, encoding="utf-8",
                  errors="surrogateescape", newline="") as f:
            original = f.read()
        with open(_AUTO_UPGRADES_FILE, "w", encoding="utf-8",
                  errors="surrogateescape", newline="") as f:
            f.write(set_unattended_upgrade_off(original))
    except OSError as exc:
        log_error(_OP, f"could not update {_AUTO_UPGRADES_FILE}: {exc}",
                  path=_AUTO_UPGRADES_FILE)
        return False
    log_info(_OP, f"Set APT::Periodic::Unattended-Upgrade to \"0\" in "
                  f"{_AUTO_UPGRADES_FILE}", path=_AUTO_UPGRADES_FILE)
    return True


def install(env: dict[str, str]) -> int:
    """Turn off Ubuntu's unattended upgrades."""
    if not _rewrite_auto_upgrades_file():
        return 3
    for verb in ("stop", "disable", "mask"):
        if not systemctl(verb, _SERVICE, op=_OP, env=env):
            return 3
    log_info(_OP, f"Stopped, disabled and masked {_SERVICE}")
    return 0
