"""install_cuda_samples.py — Python port of components/install_cuda_samples.sh.

Downloads the NVIDIA cuda-samples source matching the pinned version, builds it
with cmake/nvcc, and installs the built Samples tree into the CUDA toolkit's
samples directory. (The bash script records no component version, so neither
does this port.)
"""

from __future__ import annotations

import os
import shutil
import tarfile

from utils.component_config import config_for
from utils.download import download_and_verify
from utils.process import exec_program
from utils.logger import log_info, log_error

_WORK_DIR = "/tmp/cuda-samples-build"


def get_config(env):
    """Resolve the cuda component config (driver + samples) from versions.json."""
    return config_for("cuda", env)


def install(env):
    """Download -> extract -> cmake/make -> install Samples. Returns 0 or 3."""
    cfg = get_config(env) or {}
    samples = cfg.get("samples") or {}
    driver = cfg.get("driver") or {}
    version = samples.get("version")
    sha256 = samples.get("sha256", "")
    driver_version = driver.get("version")
    if not version or not driver_version:
        log_error("install-cuda-samples",
                  "could not resolve cuda samples/driver version from versions.json")
        return 3

    url = f"https://github.com/NVIDIA/cuda-samples/archive/refs/tags/v{version}.tar.gz"
    log_info("install-cuda-samples", f"Installing CUDA samples {version}")

    os.makedirs(_WORK_DIR, exist_ok=True)

    # 1. download + verify the source tarball
    try:
        tarball = download_and_verify(url, sha256, dest_dir=_WORK_DIR)
    except Exception as exc:
        log_error("install-cuda-samples", f"download/verify failed: {exc}")
        return 3

    # 2. extract
    with tarfile.open(tarball) as archive:
        archive.extractall(_WORK_DIR, filter="data")

    # 3. build (cmake + make)
    source_dir = os.path.join(_WORK_DIR, f"cuda-samples-{version}")
    build_dir = os.path.join(source_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    steps = [
        ["cmake", "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc", ".."],
        ["make", "-j", str(os.cpu_count() or 1)],
    ]
    for step in steps:
        rc = exec_program(step, "install-cuda-samples", cwd=build_dir, env=env)
        if rc != 0:
            log_error("install-cuda-samples",
                      f"build step failed: {' '.join(step)} (exit code {rc})")
            return 3

    # 4. install the built Samples into the CUDA toolkit (mv -vT semantics)
    built_samples = os.path.join(build_dir, "Samples")
    dest = f"/usr/local/cuda-{driver_version}/samples"
    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)
    shutil.move(built_samples, dest)

    # 5. cleanup
    shutil.rmtree(_WORK_DIR, ignore_errors=True)

    log_info("install-cuda-samples", f"CUDA samples {version} installed to {dest}")
    return 0
