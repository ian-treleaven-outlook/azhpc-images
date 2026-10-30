"""install_intel_libs.py — Python port of components/install_intel_libs.sh.

Installs Intel oneAPI MKL: downloads the offline installer named by
versions.json (`intel_one_mkl`: version, url, sha256; SHA256 checked) into
/tmp, runs it non-interactively (`sh <installer> -s -a -s --eula accept`),
records INTEL_ONE_MKL and removes the installer (bash `rm -f`).

Deliberate differences: a missing version, url or sha256 fails before
downloading; the installer is downloaded to and run from /tmp rather than the
current directory; and the version is recorded once, after the installer
succeeds, where bash records it both before downloading and at the end.
"""

from __future__ import annotations

from utils.component_config import require_config, write_component_version
from utils.download import fetch_verified
from utils.files import remove_paths
from utils.logger import log_info
from utils.process import run

_OP = "install-intel-libs"
_WORK_DIR = "/tmp"


def install(env: dict[str, str]) -> int:
    """Download and run the Intel oneAPI MKL offline installer. Returns 0 or 3."""
    cfg = require_config("intel_one_mkl", env, _OP, "version", "url", "sha256")
    if cfg is None:
        return 3
    version = cfg["version"]
    log_info(_OP, "Installing Intel oneAPI MKL", version=version, source=cfg["url"])

    installer = fetch_verified(cfg["url"], cfg["sha256"], _OP, _WORK_DIR)
    if installer is None:
        return 3
    if not run(["sh", str(installer), "-s", "-a", "-s", "--eula", "accept"],
               _OP, env=env):
        return 3
    write_component_version("INTEL_ONE_MKL", version)
    remove_paths(installer, op=_OP)

    log_info(_OP, "Intel oneAPI MKL installed", version=version)
    return 0
