"""install_libfabric.py — Python port of components/install_libfabric.sh.

Resolve the version, download + verify the source, extract it, build it with
configure/make, and record the version. Unlike mpifileutils, libfabric needs no
`module load`, so the build steps run as individual commands via exec_program —
no bash subprocess required.
"""

from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path

from utils.component_config import config_for, write_component_version
from utils.download import download_and_verify
from utils.process import exec_program
from utils.logger import log_info, log_error

_WORK_DIR = "/tmp"
_INSTALL_PREFIX = "/opt/libfabric"


def install(env: dict[str, str]) -> int:
    """Download, build, and install libfabric into /opt/libfabric.

    Returns 0 on success, 3 on failure.
    """
    cfg = config_for("libfabric", env)
    if not cfg or not cfg.get("version"):
        log_error("install-libfabric",
                  "could not resolve libfabric version from versions.json")
        return 3
    version = cfg["version"]
    url = cfg.get("url", "")
    sha256 = cfg.get("sha256", "")

    log_info("install-libfabric", f"Installing libfabric {version}")

    # 1. download + verify
    try:
        tarball = download_and_verify(url, sha256, dest_dir=_WORK_DIR)
    except Exception as exc:
        log_error("install-libfabric", f"download/verify failed: {exc}")
        return 3

    # 2. extract (tarfile auto-detects the .tar.bz2 compression)
    with tarfile.open(tarball) as archive:
        archive.extractall(_WORK_DIR, filter="data")
    folder = Path(_WORK_DIR) / Path(tarball).name.removesuffix(".tar.bz2")

    # 3. build — tcp/verbs/shm providers; disable psm3 (hangs on MANA-only
    #    systems). No module load is needed, so each step is its own command.
    build_steps = [
        [str(folder / "configure"), f"--prefix={_INSTALL_PREFIX}", "--disable-psm3"],
        ["make", "-j", str(os.cpu_count() or 1)],
        ["make", "install"],
    ]
    for cmd in build_steps:
        rc = exec_program(cmd, "install-libfabric", cwd=str(folder), env=env)
        if rc != 0:
            log_error("install-libfabric",
                      f"build step failed: {' '.join(cmd)} (exit code {rc})")
            return 3

    # 4. record the installed version
    write_component_version("LIBFABRIC", version)

    # 5. cleanup
    shutil.rmtree(folder, ignore_errors=True)
    try:
        Path(tarball).unlink()
    except OSError:
        pass

    log_info("install-libfabric", f"libfabric {version} installed to {_INSTALL_PREFIX}")
    return 0
