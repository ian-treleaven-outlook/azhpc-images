"""disable_predictive_interface_renaming.py — Python port of
distros/<distro>/disable_predictive_interface_renaming.sh (ubuntu22.04 and
ubuntu24.04 carry the same script).

Adds the kernel parameter `net.ifnames=0` (classic eth0, eth1, ... interface
names, which the HPC image and its tooling expect, instead of "predictable"
names such as enP1s0) to the Ubuntu cloud-image GRUB settings file
/etc/default/grub.d/50-cloudimg-settings.cfg, then runs `update-grub` to
regenerate /boot/grub/grub.cfg. build_plan runs it on Ubuntu only; elsewhere
the settings file is missing and the step fails before update-grub.

The bash edit, for comparison:

    GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX console=tty1 console=ttyS0 earlyprintk=ttyS0"
    sed -i "s/${GRUB_CMDLINE_LINUX}/${GRUB_CMDLINE_LINUX} ${KERNEL_PARAMETER}/" /etc/default/grub.d/50-cloudimg-settings.cfg

Kept bash quirks:
  * GRUB_CMDLINE_LINUX is normally unset, so the search text starts with a
    space: " console=tty1 ...". It therefore only matches when something
    other than the opening quote comes before "console=tty1" (checked with
    GNU sed):
        GRUB_CMDLINE_LINUX_DEFAULT="console=tty1 console=ttyS0 earlyprintk=ttyS0"
            -> unchanged
        GRUB_CMDLINE_LINUX="quiet console=tty1 console=ttyS0 earlyprintk=ttyS0"
            -> ...earlyprintk=ttyS0 net.ifnames=0"
    When nothing matches the step still succeeds, as sed does; install()
    logs a warning so a build log shows net.ifnames=0 was not added.
  * Only the first match on each line changes; a rerun adds the parameter
    again.

Deliberate difference: sed treats the search text as a regular expression,
this port matches it literally. They agree for the normal value (letters,
digits, `=` and spaces); a GRUB_CMDLINE_LINUX containing regex characters,
`/` or `&` could behave differently in sed.
"""

from __future__ import annotations

from utils.logger import log_error, log_info, log_warn
from utils.process import run

_OP = "disable-predictive-interface-renaming"

_GRUB_SETTINGS_FILE = "/etc/default/grub.d/50-cloudimg-settings.cfg"
_KERNEL_PARAMETER = "net.ifnames=0"
_CONSOLE_PARAMETERS = "console=tty1 console=ttyS0 earlyprintk=ttyS0"


def search_text(env: dict[str, str]) -> str:
    """The text bash's sed searches for: "$GRUB_CMDLINE_LINUX console=tty1 ...".

    Unset expands to "", so the usual result starts with a space.
    """
    return f"{env.get('GRUB_CMDLINE_LINUX', '')} {_CONSOLE_PARAMETERS}"


def add_kernel_parameter(text: str, search: str, parameter: str) -> str:
    """`sed "s/<search>/<search> <parameter>/"` on `text`, matching literally.

    Only the first occurrence on each line changes; lines without it, and the
    file's final newline (or lack of one), are left as they are.
    """
    replacement = f"{search} {parameter}"
    return "\n".join(line.replace(search, replacement, 1)
                     for line in text.split("\n"))


def _edit_settings_file(search: str) -> tuple[str, str] | None:
    """bash `sed -i` on the settings file: (original, updated) text, or None.

    newline="" keeps "\\r\\n" as it is and surrogateescape passes non-UTF-8
    bytes through, as sed does; write_file() would do neither.
    """
    # TODO(ADO 1001): toolkit candidate: byte-preserving `sed -i`, e.g.
    # edit_file(path, transform: Callable[[str], str], op) -> (old, new) | None
    try:
        with open(_GRUB_SETTINGS_FILE, encoding="utf-8",
                  errors="surrogateescape", newline="") as f:
            original = f.read()
        updated = add_kernel_parameter(original, search, _KERNEL_PARAMETER)
        with open(_GRUB_SETTINGS_FILE, "w", encoding="utf-8",
                  errors="surrogateescape", newline="") as f:
            f.write(updated)
    except OSError as exc:
        log_error(_OP, f"could not update {_GRUB_SETTINGS_FILE}: {exc}",
                  path=_GRUB_SETTINGS_FILE)
        return None
    return original, updated


def _log_edit(original: str, updated: str, search: str) -> None:
    if updated == original:
        log_warn(_OP, f"No line in {_GRUB_SETTINGS_FILE} contains the search "
                      f"text, so {_KERNEL_PARAMETER} was not added (bash's sed "
                      "succeeds silently here too)",
                 path=_GRUB_SETTINGS_FILE, search=search, lines_changed=0)
        return
    changed = sum(1 for line in original.split("\n") if search in line)
    log_info(_OP, f"Added {_KERNEL_PARAMETER} in {_GRUB_SETTINGS_FILE}",
             path=_GRUB_SETTINGS_FILE, lines_changed=changed)


def install(env: dict[str, str]) -> int:
    """Add net.ifnames=0 to the GRUB settings and regenerate the GRUB config."""
    search = search_text(env)
    edit = _edit_settings_file(search)
    if edit is None:
        return 3
    _log_edit(*edit, search)
    if not run(["update-grub"], _OP, env=env):
        return 3
    log_info(_OP, "Regenerated the GRUB configuration")
    return 0
