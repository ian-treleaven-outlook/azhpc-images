"""disable_auto_upgrade.py — Python port of distros/ubuntu26.04/disable_auto_upgrade.sh.

What this step does
-------------------
Ubuntu installs security updates in the background with the
"unattended-upgrades" service. On a build VM (and on a finished HPC image) that
is unwanted: it can hold the apt/dpkg lock, or change packages we pinned. This
step turns it off in two ways:

  1. In /etc/apt/apt.conf.d/20auto-upgrades it sets
         APT::Periodic::Unattended-Upgrade "0";
     so apt's periodic job no longer starts unattended upgrades.
  2. It stops, disables, and masks unattended-upgrades.service, so the
     service is not running now and cannot be started later.

The original bash, for comparison (distros/ubuntu26.04/disable_auto_upgrade.sh;
it is byte-identical to the ubuntu22.04 and ubuntu24.04 copies that upstream
deleted in #623, which this module was first ported from):

    set -e

    sed -i 's/APT::Periodic::Unattended-Upgrade ".*/APT::Periodic::Unattended-Upgrade "0";/' /etc/apt/apt.conf.d/20auto-upgrades

    systemctl stop unattended-upgrades.service
    systemctl disable unattended-upgrades.service
    systemctl mask unattended-upgrades.service

The sed command reads the file line by line. On each line it finds the first
text matching `APT::Periodic::Unattended-Upgrade "` followed by anything (`.*`
runs to the end of the line) and replaces that with
`APT::Periodic::Unattended-Upgrade "0";`. Anything before the match on the line
(such as indentation) is kept. Lines without a match are unchanged.
set_unattended_upgrade_off() below does the same text transformation in Python.

How it fits into the Python build
---------------------------------
Currently this module is NOT wired into any build plan. Upstream PR #623
deleted disable_auto_upgrade.sh from distros/ubuntu22.04 and
distros/ubuntu24.04 and removed the calls from their install.sh scripts: the
Packer prerequisites script (packer/scripts/prerequisites.sh) now turns off
background apt activity earlier in the build. So the Ubuntu 22.04 and 24.04
build plans no longer have a "disable-auto-upgrade" step.

Ubuntu 26.04 still runs it: distros/ubuntu26.04/install.sh calls
./disable_auto_upgrade.sh near the end (after trivy_scan.sh). The module is
kept for the future Ubuntu 26.04 build plan (ADO 993), which will add a
"disable-auto-upgrade" Step with this module's install() as its `action=`.
When a Step has an action, the build calls the Python function instead of
running the bash script; deleting `action=` makes the build run the bash
script again.

Like every action, install() receives the build environment (a dict of
environment variables) and returns 0 for success or 3 for failure.

Ubuntu only, as in bash
-----------------------
The bash script now exists only in distros/ubuntu26.04, and only that
install.sh runs it.

If install() is called on an RPM distro anyway (outside a build plan), it fails
for the closest equivalent reason to the missing bash script: the apt file
/etc/apt/apt.conf.d/20auto-upgrades does not exist, so it logs an error and
returns 3 before running systemctl.

Deliberate differences
----------------------
None in behavior. Small mechanical details: Python rewrites the file in place,
while `sed -i` writes a new file and renames it over the old one. The resulting
contents are identical.
"""

from __future__ import annotations

import re

from utils.logger import log_error, log_info
from utils.process import exec_program

# The Step's `op` label in build_config.py. Every log line from this module uses it.
_OP = "disable-auto-upgrade"

# The apt configuration file that bash edits with sed.
_AUTO_UPGRADES_FILE = "/etc/apt/apt.conf.d/20auto-upgrades"

# The systemd unit that bash stops, disables, and masks.
_SERVICE = "unattended-upgrades.service"

# The same pattern as in the sed command. re.compile() turns the text into a
# reusable regular-expression object. The r"..." prefix makes a "raw" string,
# where backslashes are not special (there are none here, but it is the usual
# habit for regex text). In this pattern only `.` (any character) and `*` (repeat
# zero or more times) are special, exactly as in sed; `:` and `-` are literal.
_SETTING_PATTERN = re.compile(r'APT::Periodic::Unattended-Upgrade ".*')
_SETTING_OFF = 'APT::Periodic::Unattended-Upgrade "0";'


def set_unattended_upgrade_off(text: str) -> str:
    """Return `text` with the Unattended-Upgrade setting changed to "0".

    Mirrors the sed command, line by line:

        'APT::Periodic::Unattended-Upgrade "1";'
            -> 'APT::Periodic::Unattended-Upgrade "0";'
        '  APT::Periodic::Unattended-Upgrade "1"; // note'
            -> '  APT::Periodic::Unattended-Upgrade "0";'   (rest of line replaced)
        'APT::Periodic::Update-Package-Lists "1";'
            -> unchanged

    This is a "pure" function: it only transforms its input, so tests can call
    it directly with real file contents.
    """
    # sed works on one line at a time, and only "\n" ends a line. We split on
    # "\n" only (str.splitlines() would also split on "\r" and other characters,
    # which sed does not). If the text ends with "\n", the last piece is "" and
    # join() puts the final newline back; if not, none is added, just like sed.
    lines = text.split("\n")
    # count=1 replaces only the first match on each line, like sed's s/// without
    # the `g` flag. In Python's re, `.` does not match "\n", but each piece has no
    # "\n" anyway, so `.*` stops at the end of the line, as in sed.
    return "\n".join(_SETTING_PATTERN.sub(_SETTING_OFF, line, count=1)
                     for line in lines)


def _rewrite_auto_upgrades_file() -> bool:
    """Apply set_unattended_upgrade_off() to the apt file, like `sed -i`.

    Returns True on success, or False after logging why it failed (for example,
    the file does not exist, which also makes `sed -i` fail).
    """
    # newline="" turns off Python's newline translation, so "\r\n" stays as it
    # is in the file, as sed would leave it. errors="surrogateescape" lets any
    # bytes that aren't valid UTF-8 pass through unchanged, since sed doesn't
    # care about the encoding either.
    try:
        with open(_AUTO_UPGRADES_FILE, encoding="utf-8",
                  errors="surrogateescape", newline="") as f:
            original = f.read()
        updated = set_unattended_upgrade_off(original)
        with open(_AUTO_UPGRADES_FILE, "w", encoding="utf-8",
                  errors="surrogateescape", newline="") as f:
            f.write(updated)
    except OSError as exc:
        log_error(_OP, f"could not update {_AUTO_UPGRADES_FILE}: {exc}")
        return False
    return True


def install(env: dict[str, str]) -> int:
    """Turn off Ubuntu's unattended upgrades.

    `env` is the build environment; it is passed on to systemctl.
    Returns 0 on success and 3 on failure.
    """
    # ---- 1. apt setting (the sed line) -------------------------------------
    if not _rewrite_auto_upgrades_file():
        return 3
    log_info(_OP, f"Set APT::Periodic::Unattended-Upgrade to \"0\" in "
                  f"{_AUTO_UPGRADES_FILE}")

    # ---- 2. systemd unit: stop, disable, mask (same order as bash) ---------
    # Like `set -e`, the first failing command stops the step.
    for verb in ("stop", "disable", "mask"):
        if exec_program(["systemctl", verb, _SERVICE], _OP, env=env) != 0:
            log_error(_OP, f"systemctl {verb} {_SERVICE} failed")
            return 3

    log_info(_OP, f"Stopped, disabled and masked {_SERVICE}")
    return 0
