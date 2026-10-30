"""add_udev_rules.py — Python port of components/add-udev-rules.sh.

Appends two udev rules blocks, then makes udev apply them:

  * /etc/udev/rules.d/60-rdma-persistent-naming.rules: a rule that keeps
    InfiniBand device names as the kernel provides them (mlx5_0, ...),
    overriding rdma-core's rule that would rename them (e.g. ibp0s12f4). Most
    of the text is rdma-core's comment block on the naming modes.
  * /etc/udev/rules.d/99-issm.rules: places the issm* devices under
    /dev/infiniband/.
  * `udevadm control --reload-rules` then `udevadm trigger`, so the rules
    apply to devices that already exist.

Bash writes each block with `cat << EOF >> file`. The delimiter is unquoted,
but the text has no `$`, backticks or backslashes, so it is written as it
appears, each block ending with a newline.

Kept bash behaviour: the text is appended, so a rerun adds the blocks again
(duplicate rules are harmless). The first failure stops the step.

Deliberate difference: write_file() creates a missing /etc/udev/rules.d,
where bash's `>>` would fail.
"""

from __future__ import annotations

from utils.files import write_file
from utils.process import run_all

_OP = "add-udev-rules"

_RDMA_NAMING_RULES_FILE = "/etc/udev/rules.d/60-rdma-persistent-naming.rules"
_ISSM_RULES_FILE = "/etc/udev/rules.d/99-issm.rules"

# The first heredoc, byte for byte, with its final newline.
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

# The second heredoc: one line plus its newline.
_ISSM_RULES = 'KERNEL=="issm*", NAME="infiniband/%k"\n'

# Run after both files are written.
_UDEVADM_COMMANDS = [
    ["udevadm", "control", "--reload-rules"],
    ["udevadm", "trigger"],
]


def install(env: dict[str, str]) -> int:
    """Append the RDMA naming and issm udev rules, then reload and trigger udev."""
    if not write_file(_RDMA_NAMING_RULES_FILE, _RDMA_NAMING_RULES, _OP, append=True):
        return 3
    if not write_file(_ISSM_RULES_FILE, _ISSM_RULES, _OP, append=True):
        return 3
    if not run_all(_UDEVADM_COMMANDS, _OP, env=env):
        return 3
    return 0