"""install_hpcdiag.py — Python port of components/install_hpcdiag.sh.

Fetch the latest azhpc-diagnostics release from the GitHub API, download and
extract its source tarball, and install the diagnostics script. The GitHub API
response is parsed natively with json (replacing `curl | grep | cut`), and there
is no checksum (GitHub publishes none for source tarballs).
"""

from __future__ import annotations

import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

from utils.download import download
from utils.logger import log_info, log_error

_API_URL = "https://api.github.com/repos/Azure/azhpc-diagnostics/releases/latest"
_DEST_DIR = "/opt/azurehpc/diagnostics"
_WORK_DIR = "/tmp"
_DIAG_SCRIPT = "Linux/src/gather_azhpc_vm_diagnostics.sh"


def latest_tarball_url(api_response: str):
    """Return the source tarball URL from a GitHub 'latest release' response.

    Replaces the shell `curl ... | grep tarball_url | cut ...` with a plain
    JSON lookup. Returns None if the field is absent.
    """
    data = json.loads(api_response)
    return data.get("tarball_url") or None


def install(env: dict[str, str]) -> int:
    """Install the azhpc diagnostics script. Returns 0 on success, 3 on failure."""
    log_info("install-hpcdiag", "Installing azhpc diagnostics")

    # 1. find the latest release's source tarball (native JSON, no curl|grep|cut)
    try:
        with urllib.request.urlopen(_API_URL) as resp:
            url = latest_tarball_url(resp.read().decode("utf-8"))
    except OSError as exc:
        log_error("install-hpcdiag", f"failed to query GitHub API: {exc}")
        return 3
    if not url:
        log_error("install-hpcdiag", "no tarball_url in GitHub API response")
        return 3

    # 2. download (GitHub publishes no checksum for source tarballs)
    dest = Path(_WORK_DIR) / Path(url).name
    try:
        download(url, dest)
    except OSError as exc:
        log_error("install-hpcdiag", f"download failed: {exc}")
        return 3

    # 3. extract and locate the unpacked top-level directory
    with tarfile.open(dest) as archive:
        top = archive.getnames()[0].split("/")[0]
        archive.extractall(_WORK_DIR, filter="data")
    unpacked = Path(_WORK_DIR) / top

    # 4. install the diagnostics script
    Path(_DEST_DIR).mkdir(parents=True, exist_ok=True)
    shutil.copy(unpacked / _DIAG_SCRIPT, _DEST_DIR)

    # 5. cleanup
    shutil.rmtree(unpacked, ignore_errors=True)
    try:
        dest.unlink()
    except OSError:
        pass

    log_info("install-hpcdiag", f"diagnostics installed to {_DEST_DIR}")
    return 0
