"""install_intel_libs.py — Python port of components/install_intel_libs.sh.

Resolve the version, download + verify the Intel oneAPI MKL offline installer,
run it non-interactively, record the version, and clean up.
"""

from __future__ import annotations

from pathlib import Path

from utils.component_config import config_for, write_component_version
from utils.download import download_and_verify
from utils.process import exec_program
from utils.logger import log_info, log_error

_WORK_DIR = "/tmp"


def get_config(env):
    """Resolve Intel oneAPI MKL's version/url/sha256 from versions.json."""
    return config_for("intel_one_mkl", env)


def install(env: dict[str, str]) -> int:
    """Download and run the Intel oneAPI MKL offline installer.

    Returns 0 on success, 3 on failure.
    """
    cfg = get_config(env)
    if not cfg or not cfg.get("version"):
        log_error("install-intel-libs",
                  "could not resolve intel_one_mkl version from versions.json")
        return 3
    version = cfg["version"]
    url = cfg.get("url", "")
    sha256 = cfg.get("sha256", "")

    log_info("install-intel-libs", f"Installing Intel oneAPI MKL {version}")

    # 1. download + verify the offline installer
    try:
        installer = download_and_verify(url, sha256, dest_dir=_WORK_DIR)
    except Exception as exc:
        log_error("install-intel-libs", f"download/verify failed: {exc}")
        return 3

    # 2. run the vendor's self-extracting installer non-interactively
    rc = exec_program(
        ["sh", str(installer), "-s", "-a", "-s", "--eula", "accept"],
        "install-intel-libs", env=env)
    if rc != 0:
        log_error("install-intel-libs", f"installer failed with exit code {rc}")
        return 3

    # 3. record the installed version
    write_component_version("INTEL_ONE_MKL", version)

    # 4. cleanup
    try:
        Path(installer).unlink()
    except OSError:
        pass

    log_info("install-intel-libs", f"Intel oneAPI MKL {version} installed")
    return 0
