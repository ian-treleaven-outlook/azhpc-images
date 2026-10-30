"""cuda.py — the CUDA toolkit pieces both NVIDIA driver ports share.

install_nvidiagpudriver.sh and install_nvidiagriddriver.sh finish the same
way: install cuda-toolkit-<X-Y>, write /etc/profile.d/cuda.sh, and record the
CUDA version nvcc reports. Their ports (install_nvidiagpudriver,
install_nvidiagriddriver) take those pieces from here so the two can't drift:

    package = cuda_toolkit_package(cuda_version)        # "13.0" -> cuda-toolkit-13-0
    write_file(CUDA_PROFILE, CUDA_PROFILE_TEXT, op, mode=0o644)
    rc, out = run_capture(list(NVCC_VERSION_COMMAND), op, env=env)
    version = nvcc_cuda_version(out)                    # "13.0.88"

On Ubuntu both rely on NVIDIA's CUDA apt repo, which the bootstrap
(install_cuda_repo_and_dkms.sh, upstream #643) adds; check_cuda_apt_source()
logs whether it is there. Bash doesn't check, so neither port fails on it.
"""

from __future__ import annotations

import glob
import os
import re

from utils.logger import log_info, log_warn

CUDA_PROFILE = "/etc/profile.d/cuda.sh"
# What bash's two `echo '...' | tee [-a] /etc/profile.d/cuda.sh` lines write.
# Single-quoted in bash, so the ${...} are literal; a login shell expands them.
CUDA_PROFILE_TEXT = (
    'export PATH="${PATH:+$PATH:}/usr/local/cuda/bin"\n'
    'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/cuda/lib64"\n'
)

# bash: cuda_version=$(source /etc/profile; nvcc --version | ...). A bash of
# its own, so /etc/profile (and so cuda.sh) can put nvcc on PATH; the rest of
# the pipeline is nvcc_cuda_version().
NVCC_VERSION_COMMAND = ("bash", "-c", "source /etc/profile; nvcc --version")

# Where the bootstrap's cuda-keyring install puts NVIDIA's CUDA apt source.
APT_SOURCES_DIR = "/etc/apt/sources.list.d"
_CUDA_APT_SOURCE_GLOB = "cuda-*"

# awk field splitting / bash word splitting: spaces, tabs and newlines.
_BLANKS = re.compile(r"[ \t\n]+")


def cuda_toolkit_package(cuda_version: str) -> str:
    """cuda-toolkit-${CUDA_DRIVER_VERSION//./-}: "13.0" -> "cuda-toolkit-13-0"."""
    return "cuda-toolkit-" + cuda_version.replace(".", "-")


def nvcc_cuda_version(nvcc_output: str) -> str | None:
    """The CUDA version from `nvcc --version` output, or None.

    Mirrors  nvcc --version | grep release | awk '{print $6}' | cut -c2-
    and the unquoted ${cuda_version} (only its first word is recorded):

        "Cuda compilation tools, release 13.0, V13.0.88"  ->  "13.0.88"

    A matching line with fewer than 6 fields gives "", which word splitting
    drops. None when nothing is left. Checked against grep/awk/cut in WSL.
    """
    values = []
    for line in nvcc_output.split("\n"):
        if "release" in line:
            fields = [field for field in _BLANKS.split(line) if field]
            values.append(fields[5][1:] if len(fields) >= 6 else "")
    words = [word for word in _BLANKS.split("\n".join(values)) if word]
    return words[0] if words else None


def check_cuda_apt_source(op: str, sources_dir: str = APT_SOURCES_DIR) -> bool:
    """Log whether NVIDIA's CUDA apt source (cuda-*) is in `sources_dir`.

    Returns True if it is. Only logs: bash doesn't check, and its apt install
    of the toolkit is what fails without the repo.
    """
    sources = sorted(glob.glob(os.path.join(sources_dir, _CUDA_APT_SOURCE_GLOB)))
    if sources:
        log_info(op, "NVIDIA CUDA apt source present (added by "
                     "install_cuda_repo_and_dkms.sh in the bootstrap)",
                 cuda_apt_sources=", ".join(sources))
        return True
    log_warn(op, f"no CUDA apt source ({_CUDA_APT_SOURCE_GLOB}) in {sources_dir}: "
                 "the bootstrap's install_cuda_repo_and_dkms.sh should have added "
                 "it; continuing, as bash does (its apt install then fails)",
             path=sources_dir)
    return False
