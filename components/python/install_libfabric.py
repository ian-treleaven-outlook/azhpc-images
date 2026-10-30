"""install_libfabric.py — Python port of components/install_libfabric.sh.

On non-InfiniBand SKUs (NCv6) libfabric replaces UCX as MPI's network layer.
This step resolves libfabric's version, url and sha256 from versions.json,
downloads and verifies the source tarball, builds it into /opt/libfabric with
configure/make, and records LIBFABRIC's version. No `module load` is needed,
so each build command runs on its own rather than in one shell.

Deliberate differences from the bash
------------------------------------
  * The tarball is downloaded and unpacked in /tmp (bash uses its cwd), and
    the source tree and tarball are removed afterwards (bash leaves them).
  * `tar -xvf` becomes `tar -xf` (extract_tarball; -v only printed names).
  * A missing version, url or sha256 fails the step up front; bash's `jq -r`
    would carry on with "null".
"""

from __future__ import annotations

import os
from pathlib import Path

from utils.component_config import require_config, write_component_version
from utils.download import fetch_verified
from utils.files import extract_tarball, remove_paths
from utils.logger import log_info
from utils.process import run_all

_OP = "install-libfabric"
_WORK_DIR = "/tmp"
_INSTALL_PREFIX = "/opt/libfabric"


def build_commands(source_dir: Path, jobs: int) -> list[list[str]]:
    """configure/make/make install, with psm3 disabled: it hangs on MANA-only systems."""
    return [
        [str(source_dir / "configure"), f"--prefix={_INSTALL_PREFIX}", "--disable-psm3"],
        ["make", "-j", str(jobs)],
        ["make", "install"],
    ]


def install(env: dict[str, str]) -> int:
    """Download, build and install libfabric into /opt/libfabric. 0, or 3 on failure."""
    cfg = require_config("libfabric", env, _OP, "version", "url", "sha256")
    if cfg is None:
        return 3
    version = cfg["version"]
    log_info(_OP, f"Installing libfabric {version}", version=version, source=cfg["url"])

    tarball = fetch_verified(cfg["url"], cfg["sha256"], _OP, dest_dir=_WORK_DIR)
    if tarball is None or not extract_tarball(tarball, _OP, cwd=_WORK_DIR):
        return 3
    # bash: LIBFABRIC_FOLDER=$(basename $URL .tar.bz2)
    source_dir = Path(_WORK_DIR) / Path(tarball).name.removesuffix(".tar.bz2")
    if not run_all(build_commands(source_dir, os.cpu_count() or 1), _OP,
                   cwd=str(source_dir), env=env):
        return 3

    write_component_version("LIBFABRIC", version)
    remove_paths(source_dir, tarball, op=_OP)
    log_info(_OP, f"libfabric {version} installed to {_INSTALL_PREFIX}", version=version)
    return 0