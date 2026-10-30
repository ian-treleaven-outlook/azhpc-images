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

from utils.files import edit_file
from utils.logger import log_info, log_warn
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


def _parameter_adder(search: str):
    """The edit_file transform: add_kernel_parameter, logging how many lines match.

    bash's sed succeeds silently when nothing matches; the warning makes a
    build log show that net.ifnames=0 was not added.
    """
    def transform(text: str) -> str:
        matching = sum(1 for line in text.split("\n") if search in line)
        if matching:
            log_info(_OP, f"Adding {_KERNEL_PARAMETER} in {_GRUB_SETTINGS_FILE}",
                     path=_GRUB_SETTINGS_FILE, lines_changed=matching)
        else:
            log_warn(_OP, f"No line in {_GRUB_SETTINGS_FILE} contains the search "
                          f"text, so {_KERNEL_PARAMETER} was not added (bash's sed "
                          "succeeds silently here too)",
                     path=_GRUB_SETTINGS_FILE, search=search, lines_changed=0)
        return add_kernel_parameter(text, search, _KERNEL_PARAMETER)
    return transform


def install(env: dict[str, str]) -> int:
    """Add net.ifnames=0 to the GRUB settings and regenerate the GRUB config."""
    if not edit_file(_GRUB_SETTINGS_FILE, _parameter_adder(search_text(env)), _OP):
        return 3
    if not run(["update-grub"], _OP, env=env):
        return 3
    log_info(_OP, "Regenerated the GRUB configuration")
    return 0
