"""install_cmake.py — Python port of components/install_cmake.sh.

A fully-native port: download the CMake release, extract it, copy the binaries
and share files into /usr/local, record the version, and clean up. No Bash is
involved — CMake ships prebuilt, so there is no compile step (unlike
mpifileutils, which delegates its build to a bash subprocess).
"""

from __future__ import annotations

import glob
import shutil
import tarfile
from pathlib import Path

from utils.component_config import config_for, write_component_version
from utils.download import download_and_verify
from utils.logger import log_info, log_error

_WORK_DIR = "/tmp"
_BIN_DIR = "/usr/local/bin"
_SHARE_DIR = "/usr/local/share"
_TOOLS = ("ccmake", "cmake", "cpack", "ctest")


def install(env: dict[str, str]) -> int:
    """Download and install CMake into /usr/local.

    Returns 0 on success, 3 on failure.
    """
    cfg = config_for("cmake", env)                                                       # parse versions.json ONCE
    if not cfg or not cfg.get("version"):                                       # real error handling 
        log_error("install-cmake", "could not resolve cmake version from versions.json")
        return 3
    
    version = cfg["version"]
    url = cfg.get("url", "")
    sha256 = cfg.get("sha256", "")

    log_info("install-cmake", f"Installing CMake {version}")

    # 1. download + verify
    try:
        tarball = download_and_verify(url, sha256, dest_dir=_WORK_DIR)          # urllib + hashlib
    except Exception as exc:
        log_error("install-cmake", f"download/verify failed: {exc}")
        return 3

    # 2. extract (the tarball unpacks to a dir named like the tarball stem)
    with tarfile.open(tarball) as archive:
        archive.extractall(_WORK_DIR, filter="data")                            # safe extraction
    extracted = Path(_WORK_DIR) / Path(tarball).name.removesuffix(".tar.gz")

    # 3. copy the CMake binaries into /usr/local/bin
    for tool in _TOOLS:
        shutil.copy(extracted / "bin" / tool, _BIN_DIR)

    # 4. copy share/cmake-* into /usr/local/share
    for share in glob.glob(str(extracted / "share" / "cmake-*")):
        shutil.copytree(share, Path(_SHARE_DIR) / Path(share).name,
                        dirs_exist_ok=True)

    # 5. record the installed version
    write_component_version("CMAKE", version)

    # 6. cleanup
    shutil.rmtree(extracted, ignore_errors=True)
    try:
        Path(tarball).unlink()
    except OSError:
        pass

    log_info("install-cmake", f"CMake {version} installed to {_BIN_DIR}")
    return 0
