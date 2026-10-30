"""rhel_family_network_config.py — Python port of network-config.sh in the
AlmaLinux, Rocky and RHEL distro folders.

All six copies (almalinux8.10, almalinux9.x, rocky8.10, rocky9.x, rhel8.10,
rhel9.x) are the same script, so one port serves them; a test checks they
stay identical. install.sh runs it near the end, as ./network-config.sh.
Azure Linux's network-config.sh is different (azurelinux_network_config.py).

It makes two changes:

  1. NetworkManager: adds `no-auto-default=*` after the `[main]` line of
     /etc/NetworkManager/NetworkManager.conf, so NetworkManager doesn't
     create an automatic "Wired connection" profile for every interface
     without one. bash: `sed -i '/\\[main\\]/a no-auto-default=*' <conf>`.
  2. cloud-init: a drop-in for cloud-init-local.service that deletes
     cloud-init's cached instance object (obj.pkl) before the service
     starts, so each VM made from the image works out its network
     configuration afresh instead of reusing the build VM's.

Kept bash quirks (each logged as a WARNING):
  * sed appends after EVERY line containing "[main]" (the pattern is
    unanchored, so a comment such as "# see [main]" matches too).
  * Run twice, sed adds a second `no-auto-default=*` line.
  * With no "[main]" line, sed changes nothing and still succeeds.

A missing NetworkManager.conf fails the step, as the sed does under `set -e`.
"""

from __future__ import annotations

import re

from utils.files import edit_file, file_has_line, write_file
from utils.logger import log_info, log_warn

_OP = "network-config"

_NETWORKMANAGER_CONF = "/etc/NetworkManager/NetworkManager.conf"
_NO_AUTO_DEFAULT = "no-auto-default=*"
# A whole line containing "[main]" (sed's /\[main\]/), and its newline if it
# has one. re.MULTILINE makes ^ and $ match at the start and end of each line.
_MAIN_LINE = re.compile(r"^(?P<line>.*\[main\].*)$\n?", re.MULTILINE)

_CLOUD_INIT_DROP_IN = ("/lib/systemd/system/cloud-init-local.service.d/"
                       "50-azure-clear-persistent-obj-pkl.conf")
# The bash heredoc, word for word. (Its EOF is unquoted, but the text has no
# $ or backquote, so bash writes it unchanged.)
_CLOUD_INIT_SETTINGS = """\
[Service]
ExecStartPre=-/bin/sh -xc 'if [ -e /var/lib/cloud/instance/obj.pkl ]; then echo "cleaning persistent cloud-init object"; rm /var/lib/cloud/instance/obj.pkl; fi; exit 0'
"""


def add_no_auto_default(text: str) -> str:
    """bash `sed '/\\[main\\]/a no-auto-default=*'` on `text`.

    After every line containing "[main]", insert `no-auto-default=*`. Like
    GNU sed's `a`, a matching last line without a newline gets one first.
    """
    # re.sub calls this for each match and puts what it returns in its place.
    def with_setting(match: re.Match) -> str:
        return f"{match['line']}\n{_NO_AUTO_DEFAULT}\n"

    return _MAIN_LINE.sub(with_setting, text)


def disable_auto_default_connections() -> bool:
    """Add no-auto-default=* under [main] in NetworkManager.conf."""
    if file_has_line(_NETWORKMANAGER_CONF, _NO_AUTO_DEFAULT):
        log_warn(_OP, f"{_NETWORKMANAGER_CONF} already has {_NO_AUTO_DEFAULT}; adding it "
                      "again, as network-config.sh's sed does on a rerun",
                 path=_NETWORKMANAGER_CONF)

    main_lines: list[str] = []

    # edit_file calls this with the file's text. Besides editing it, note the
    # [main] lines, so they can be checked below.
    def add_after_main(text: str) -> str:
        main_lines.extend(_MAIN_LINE.findall(text))
        return add_no_auto_default(text)

    if not edit_file(_NETWORKMANAGER_CONF, add_after_main, _OP):
        return False
    if not main_lines:
        log_warn(_OP, f"{_NETWORKMANAGER_CONF} has no [main] line, so {_NO_AUTO_DEFAULT} "
                      "was not added (bash's sed succeeds the same way)",
                 path=_NETWORKMANAGER_CONF)
    elif len(main_lines) > 1:
        log_warn(_OP, f"{_NETWORKMANAGER_CONF} has {len(main_lines)} lines containing "
                      f"[main]; {_NO_AUTO_DEFAULT} was added after each, as bash's sed does",
                 path=_NETWORKMANAGER_CONF, main_lines=len(main_lines))
    return True


def install(env: dict[str, str]) -> int:
    """Stop NetworkManager's automatic profiles; make cloud-init forget the build VM."""
    if not disable_auto_default_connections():
        return 3
    # write_file creates the drop-in folder first, like bash's `mkdir -p`.
    if not write_file(_CLOUD_INIT_DROP_IN, _CLOUD_INIT_SETTINGS, _OP):
        return 3
    log_info(_OP, "NetworkManager makes no automatic profiles; cloud-init clears its "
                  "cached instance object at boot",
             networkmanager_conf=_NETWORKMANAGER_CONF, drop_in=_CLOUD_INIT_DROP_IN)
    return 0
