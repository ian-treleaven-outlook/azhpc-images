"""ofed — the OFED version `ofed_info` reports.

Both InfiniBand driver steps end the same way: DOCA-OFED (install_doca.py,
every distro but Azure Linux) and MOFED (install_mofed.py, Azure Linux)
record OFED as `ofed_info` reports it, which is what the image tests check.
Bash does it with the same pipeline in both scripts:

    ofed_info | sed -n '1,1p' | awk -F'-' 'OFS="-" {print $3,$4}' | tr -d ':'

Deliberate difference: a failing `ofed_info`, or a first line with fewer
than four '-'-separated fields, fails the step. Bash would record "" (or
e.g. "0.7.0-") and carry on.
"""

from __future__ import annotations

from utils.component_config import record_version
from utils.logger import log_error
from utils.process import run_capture


def parse_ofed_version(ofed_info_output: str) -> str | None:
    """The OFED version in `ofed_info` output, e.g. "25.10-1.7.1".

    The first line split on '-', its 3rd and 4th fields joined with '-',
    without ':' (bash's pipeline above). None if the first line has too few
    fields.
    """
    lines = ofed_info_output.splitlines()
    if not lines:
        return None
    fields = lines[0].split("-")
    if len(fields) < 4:
        return None
    return f"{fields[2]}-{fields[3]}".replace(":", "") or None


def record_ofed_version(env: dict[str, str], op: str) -> bool:
    """Record OFED as `ofed_info` reports it; False (logged) if it can't be read."""
    rc, output = run_capture(["ofed_info"], op, env=env)
    version = parse_ofed_version(output) if rc == 0 else None
    if not version:
        log_error(op, "could not read the OFED version from ofed_info",
                  exit_code=rc, first_line=output.split("\n", 1)[0])
        return False
    return record_version("OFED", version, op)
