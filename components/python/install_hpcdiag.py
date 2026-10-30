"""install_hpcdiag.py — Python port of components/install_hpcdiag.sh.

Installs the azhpc-diagnostics script: asks the GitHub API for the latest
release's source tarball, downloads and unpacks it in /tmp, copies
Linux/src/gather_azhpc_vm_diagnostics.sh into /opt/azurehpc/diagnostics, then
removes the tarball and the unpacked tree (bash `rm -r`).

Kept bash behaviour: no checksum (GitHub publishes none for source
tarballs), and the unpacked directory is the archive's first entry (bash:
`tar -xzvf ... | head -1`).

Deliberate differences: the API response is parsed as JSON instead of
`curl | grep tarball_url | cut -d\\" -f4`, and the work happens in /tmp
rather than the current directory.
"""

from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

from utils.download import fetch
from utils.files import copy_file, extract_tarball, make_dirs, remove_paths
from utils.logger import log_error, log_info

_OP = "install-hpcdiag"

_API_URL = "https://api.github.com/repos/Azure/azhpc-diagnostics/releases/latest"
_DEST_DIR = "/opt/azurehpc/diagnostics"
_WORK_DIR = "/tmp"
_DIAG_SCRIPT = "Linux/src/gather_azhpc_vm_diagnostics.sh"


def latest_tarball_url(api_response: str):
    """The source tarball URL from a GitHub 'latest release' response, or None."""
    data = json.loads(api_response)
    return data.get("tarball_url") or None


def tarball_top_dir(tarball) -> str:
    """The archive's first entry's top-level directory (bash `| head -1`)."""
    with tarfile.open(tarball) as archive:
        return archive.getnames()[0].split("/")[0]


def _latest_release_url() -> str | None:
    try:
        with urllib.request.urlopen(_API_URL) as resp:
            url = latest_tarball_url(resp.read().decode("utf-8"))
    except OSError as exc:
        log_error(_OP, f"failed to query GitHub API: {exc}", url=_API_URL)
        return None
    if not url:
        log_error(_OP, "no tarball_url in GitHub API response "
                       "(possible rate limiting)", url=_API_URL)
    return url


def _unpack(tarball: Path, env: dict[str, str]) -> Path | None:
    """Unpack in /tmp; the unpacked directory, or None."""
    try:
        top = tarball_top_dir(tarball)
    except (OSError, tarfile.TarError, IndexError) as exc:
        log_error(_OP, f"could not read {tarball}: {exc}", path=str(tarball))
        return None
    if not extract_tarball(tarball, _OP, cwd=_WORK_DIR, env=env):
        return None
    return Path(_WORK_DIR) / top


def _install_script(unpacked: Path) -> bool:
    """bash `mkdir -p $DESTINATION_DIR` and `cp .../gather_azhpc_vm_diagnostics.sh`."""
    return (make_dirs(_DEST_DIR, op=_OP)
            and copy_file(unpacked / _DIAG_SCRIPT, _DEST_DIR, _OP))


def install(env: dict[str, str]) -> int:
    """Install the azhpc diagnostics script. Returns 0 on success, 3 on failure."""
    log_info(_OP, "Installing azhpc diagnostics")
    url = _latest_release_url()
    if url is None:
        return 3
    tarball = fetch(url, _OP, _WORK_DIR)          # bash `wget`: GitHub publishes no checksum
    if tarball is None:
        return 3
    unpacked = _unpack(tarball, env)
    if unpacked is None:
        return 3
    if not _install_script(unpacked):
        return 3
    remove_paths(tarball, unpacked, op=_OP)
    log_info(_OP, f"diagnostics installed to {_DEST_DIR}", source=url)
    return 0
