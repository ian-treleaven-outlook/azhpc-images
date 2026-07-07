"""install_doca.py — DOCA/OFED helpers.

A partial Python counterpart to components/install_doca.sh. Only the piece with
real, testable logic is ported here: parsing the Open MPI version out of
`apt-cache show openmpi` output. The heavier shell-specific work (dpkg/rpm
installs, equivs marker building, dnf.conf edits) still lives in the .sh script.
"""

from __future__ import annotations


def parse_openmpi_version(apt_cache_output: str) -> str | None:
    """Extract the Open MPI version from `apt-cache show openmpi` output.

    Mirrors the shell one-liner:
        apt-cache show openmpi | awk '/^Version:/ {print $2; exit}'

    Returns the first ``Version:`` value found, or None if there isn't one
    (the shell script treats a missing version as a fatal error).
    """
    for line in apt_cache_output.splitlines():
        if line.startswith("Version:"):
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()
    return None
