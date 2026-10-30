"""add_udev_rules.py — Python port of components/add-udev-rules.sh.

What this step does
-------------------
Adds two udev rules files' worth of text, then makes udev pick them up:

  * /etc/udev/rules.d/60-rdma-persistent-naming.rules gets a rule that keeps
    InfiniBand device names exactly as the kernel provides them (mlx5_0, ...).
    It overrides the rdma-core package's rule, which would rename them to
    stable names such as ibp0s12f4. Most of the appended text is a comment
    block copied from rdma-core explaining the naming modes.
  * /etc/udev/rules.d/99-issm.rules gets a rule that places the issm* devices
    under /dev/infiniband/.
  * `udevadm control --reload-rules` makes udev re-read its rules files, and
    `udevadm trigger` replays device events so the rules apply to devices that
    already exist.

The original bash, for comparison (rules text shortened here; the full text is
in _RDMA_NAMING_RULES below):

    set -e

    cat << EOF >> /etc/udev/rules.d/60-rdma-persistent-naming.rules
    # SPDX-License-Identifier: (GPL-2.0 OR Linux-OpenIB)
    ...
    ACTION=="add", SUBSYSTEM=="infiniband", PROGRAM="rdma_rename %k NAME_KERNEL"
    EOF

    cat << EOF >> /etc/udev/rules.d/99-issm.rules
    KERNEL=="issm*", NAME="infiniband/%k"
    EOF

    udevadm control --reload-rules
    udevadm trigger

`cat << EOF >> file` appends the lines between `<< EOF` and `EOF` to the file,
creating it if it does not exist. The heredoc delimiter is unquoted, so bash
would expand `$`, backticks and backslashes in the text; this text contains
none of them, so it is written exactly as it appears. Each block ends with a
newline after its last line.

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets this module's install() as the `action=` of the
"add-udev-rules" Step. When a Step has an action, the build calls the Python
function instead of running the bash script. Deleting `action=` from that Step
makes the build run the bash script again. That is why the .sh file is kept.

Like every action, install() receives the build environment (a dict of
environment variables) and returns 0 for success or 3 for failure.

Behaviour kept from bash
------------------------
  * The text is appended, not written over. Running the step twice adds the
    blocks twice, as bash does (the duplicate rules are harmless).
  * Order: both appends, then reload, then trigger. The first failure stops
    the step (bash's `set -e`), and later steps don't run.

There are no deliberate differences from bash.
"""

from __future__ import annotations

from utils.logger import log_error, log_info
from utils.process import exec_program

# The Step's `op` label in plans/steps.py. Every log line from this module uses
# it, so this step's output can be found in a build log.
_OP = "add-udev-rules"

# The two files bash appends to. They are module-level "constants" (by
# convention, names in capitals are not changed while the program runs) so the
# tests can point them at a temporary directory instead of the real /etc.
_RDMA_NAMING_RULES_FILE = "/etc/udev/rules.d/60-rdma-persistent-naming.rules"
_ISSM_RULES_FILE = "/etc/udev/rules.d/99-issm.rules"

# The exact text of the first heredoc, byte for byte, including its final
# newline. A triple-quoted string ("""...""") can span several lines. The
# backslash right after the opening quotes stops the string from starting with
# a newline, so the text begins at "# SPDX".
_RDMA_NAMING_RULES = """\
# SPDX-License-Identifier: (GPL-2.0 OR Linux-OpenIB)
# Copyright (c) 2019, Mellanox Technologies. All rights reserved. See COPYING file
#
# Rename modes:
# NAME_FALLBACK - Try to name devices in the following order:
#                 by-pci -> by-guid -> kernel
# NAME_KERNEL - leave name as kernel provided
# NAME_PCI - based on PCI/slot/function location
# NAME_GUID - based on system image GUID
#
# The stable names are combination of device type technology and rename mode.
# Infiniband - ib*
# RoCE - roce*
# iWARP - iw*
# OPA - opa*
# Default (unknown protocol) - rdma*
#
# Example:
# * NAME_PCI
#   pci = 0000:00:0c.4
#   Device type = IB
#   mlx5_0 -> ibp0s12f4
# * NAME_GUID
#   GUID = 5254:00c0:fe12:3455
#   Device type = RoCE
#   mlx5_0 -> rocex525400c0fe123455

# Suppressing/overwriting the udev rule rdma_core package. NAME_KERNEL leaves the IB devices names as provided by the kernel.
ACTION=="add", SUBSYSTEM=="infiniband", PROGRAM="rdma_rename %k NAME_KERNEL"
"""

# The exact text of the second heredoc: one line plus its newline.
_ISSM_RULES = 'KERNEL=="issm*", NAME="infiniband/%k"\n'

# The udev commands, run in this order after both files are written.
_UDEVADM_COMMANDS = (
    ["udevadm", "control", "--reload-rules"],
    ["udevadm", "trigger"],
)


def _append(path: str, text: str) -> bool:
    """Append `text` to the file at `path`, like bash's `cat << EOF >> path`.

    Mode "a" (append) creates the file if it is missing and otherwise adds to
    its end, leaving the existing content in place. Like bash, it does not
    create a missing parent directory.

    Returns True on success, or False after logging why the write failed.
    """
    # `with open(...) as f:` closes the file when the block ends, even if an
    # error happens inside it. A failed open or write raises OSError (for
    # example PermissionError without root); catching it lets install() keep
    # its contract: log and return 3, no traceback.
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
    except OSError as exc:
        log_error(_OP, f"could not append udev rules to {path}: {exc}")
        return False
    log_info(_OP, f"Appended udev rules to {path}")
    return True


def install(env: dict[str, str]) -> int:
    """Append the RDMA naming and issm udev rules, then reload and trigger udev.

    `env` is the build environment made by component_env() in build_config.py.
    It is passed on to udevadm unchanged; nothing in it is read here.

    Returns 0 on success and 3 on failure.
    """
    # ---- 1. Append both rules blocks, in bash's order ---------------------
    if not _append(_RDMA_NAMING_RULES_FILE, _RDMA_NAMING_RULES):
        return 3
    if not _append(_ISSM_RULES_FILE, _ISSM_RULES):
        return 3

    # ---- 2. Make udev load the new rules and apply them -------------------
    # exec_program() runs the command, logs its output, and returns its exit
    # code (127 if udevadm could not be started at all).
    for command in _UDEVADM_COMMANDS:
        rc = exec_program(command, _OP, env=env)
        if rc != 0:
            log_error(_OP, f"'{' '.join(command)}' failed with exit code {rc}")
            return 3

    return 0
