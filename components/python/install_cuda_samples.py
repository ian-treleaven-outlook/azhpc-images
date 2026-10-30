"""install_cuda_samples.py — Python port of components/install_cuda_samples.sh.

What this does
--------------
Downloads NVIDIA's cuda-samples source for the pinned version, builds it with
cmake/make against the installed CUDA toolkit, and installs the built samples
in /usr/local/cuda-<cuda driver version>/samples. Like bash, it records no
component version.

How the samples are installed depends on the cuda-samples release (changed
upstream in #614, Ubuntu 26.04 support):
  * Newer releases ship cmake/InstallSamples.cmake. Then cmake is told the
    install directory (-DCUDA_SAMPLES_INSTALL_DIR=...) and, after make,
    `cmake --install .` copies the samples there.
  * Older releases don't. Then the built Samples tree is moved there with
    `mv -vT`, as before.

The original bash, for comparison:

    cuda_metadata=$(get_component_config "cuda")
    CUDA_DRIVER_VERSION=$(jq -r '.driver.version' <<< $cuda_metadata)
    CUDA_SAMPLES_VERSION=$(jq -r '.samples.version' <<< $cuda_metadata)
    CUDA_SAMPLES_SHA256=$(jq -r '.samples.sha256' <<< $cuda_metadata)

    TARBALL="v${CUDA_SAMPLES_VERSION}.tar.gz"
    CUDA_SAMPLES_DOWNLOAD_URL=https://github.com/NVIDIA/cuda-samples/archive/refs/tags/${TARBALL}
    download_and_verify ${CUDA_SAMPLES_DOWNLOAD_URL} ${CUDA_SAMPLES_SHA256}
    tar -xvf ${TARBALL}
    pushd ./cuda-samples-${CUDA_SAMPLES_VERSION}
    mkdir build && cd build
    CMAKE_OPTIONS=(-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc)
    if [[ -f ../cmake/InstallSamples.cmake ]]; then
        CMAKE_OPTIONS+=(-DCUDA_SAMPLES_INSTALL_DIR=/usr/local/cuda-${CUDA_DRIVER_VERSION}/samples)
    fi
    cmake "${CMAKE_OPTIONS[@]}" ..
    make -j $(nproc)
    if [[ -f ../cmake/InstallSamples.cmake ]]; then
        cmake --install .
    else
        mv -vT ./Samples /usr/local/cuda-${CUDA_DRIVER_VERSION}/samples
    fi
    popd

How it fits into the Python build
---------------------------------
This is not a build_plan Step: install.sh never runs install_cuda_samples.sh
itself. The driver scripts run it as a nested script, right after they install
the CUDA toolkit (so nvcc exists): install_nvidiagpudriver.sh and
install_nvidiagriddriver.sh. Their Python ports call install(env) at that
point, and treat a non-zero result as a failure of their own step. Keep the
`install(env) -> int` signature: both driver ports depend on it.

Log lines use the op "install-cuda-samples", so the samples build can be told
apart from the driver step that called it (which logs before and after).

Commands that still run as programs
-----------------------------------
tar, cmake, nproc, make, `cmake --install .` and `mv -vT`, exactly as bash
runs them. Using the real `mv -T` keeps its behaviour: an existing, non-empty
samples directory makes the move (and so the step) fail, as in bash.
`cmake --install` overwrites files in place, so it has no such problem.

The InstallSamples.cmake check is made once, before cmake; bash makes the same
check again after make, but nothing in the build creates or removes that file
(it comes from the source tarball), so the answer can't change.

Deliberate differences from bash
--------------------------------
  * The work happens in _WORK_DIR, emptied first and removed afterwards (also
    on failure). Bash works in the caller's current directory (the distro build
    dir) and leaves the tarball and source tree for the driver script's final
    `rm -rf *.tar.gz` / `rm -rf -- */` cleanup.
  * A missing samples.version, samples.sha256 or driver.version fails before
    downloading anything. Bash would try ".../vnull.tar.gz" (and install the
    result to /usr/local/cuda-null/samples).

Bash quirk, kept and warned about
---------------------------------
On the `mv -vT` path (older samples without InstallSamples.cmake), rerunning
on the same image fails: `mv -T` refuses to replace the existing, non-empty
samples directory. A warning is logged when the destination already exists,
before the move.
"""

from __future__ import annotations

import os
import shutil

from utils.component_config import config_for
from utils.download import download_and_verify
from utils.logger import log_error, log_info, log_warn
from utils.process import exec_program, run_capture

# The log label for this nested component (it has no Step of its own).
_OP = "install-cuda-samples"

# Our own download/build directory (see "Deliberate differences" above).
_WORK_DIR = "/tmp/cuda-samples-build"

# Where the CUDA toolkits live: the samples go to <here>/cuda-<version>/samples.
# Module-level so the tests can point it at a temporary directory.
_CUDA_PREFIX = "/usr/local"

# The nvcc that cmake is told to use (bash's -DCMAKE_CUDA_COMPILER value).
_NVCC = "/usr/local/cuda/bin/nvcc"

_URL_TEMPLATE = "https://github.com/NVIDIA/cuda-samples/archive/refs/tags/{tarball}"

# Shipped by newer cuda-samples releases (relative to the source directory).
# When present, the samples are installed with `cmake --install` instead of mv.
_INSTALL_SAMPLES_CMAKE = os.path.join("cmake", "InstallSamples.cmake")


def samples_url(version: str) -> str:
    """The cuda-samples tarball URL for `version` (bash's CUDA_SAMPLES_DOWNLOAD_URL)."""
    return _URL_TEMPLATE.format(tarball=f"v{version}.tar.gz")


def samples_destination(cuda_version: str) -> str:
    """/usr/local/cuda-<cuda driver version>/samples, where the samples go."""
    return f"{_CUDA_PREFIX}/cuda-{cuda_version}/samples"


def cmake_options(dest: str, cmake_install: bool) -> list[str]:
    """bash's CMAKE_OPTIONS array: the nvcc to use, plus the install directory
    when the release supports `cmake --install` (InstallSamples.cmake)."""
    options = [f"-DCMAKE_CUDA_COMPILER={_NVCC}"]
    if cmake_install:
        options.append(f"-DCUDA_SAMPLES_INSTALL_DIR={dest}")
    return options


def _resolve(env: dict[str, str]) -> tuple[str, str, str] | None:
    """(samples version, samples sha256, cuda driver version), or None (logged)."""
    cfg = config_for("cuda", env)
    samples = cfg.get("samples") if isinstance(cfg, dict) else None
    driver = cfg.get("driver") if isinstance(cfg, dict) else None
    samples = samples if isinstance(samples, dict) else {}
    driver = driver if isinstance(driver, dict) else {}
    values = {
        "samples.version": samples.get("version"),
        "samples.sha256": samples.get("sha256"),
        "driver.version": driver.get("version"),
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        log_error(_OP, f"could not resolve cuda {', '.join(missing)} from versions.json",
                  distro=env.get("DISTRIBUTION", ""), sku=env.get("SKU", ""))
        return None
    return (str(values["samples.version"]), str(values["samples.sha256"]),
            str(values["driver.version"]))


def _build_and_install(env: dict[str, str], version: str, sha256: str,
                       dest: str) -> bool:
    """Download, extract, build and install the samples. False (logged) on failure."""
    url = samples_url(version)
    try:
        tarball = download_and_verify(url, sha256, dest_dir=_WORK_DIR)
    except Exception as exc:
        log_error(_OP, f"download/verify of the cuda-samples source failed: {exc}",
                  url=url)
        return False

    if exec_program(["tar", "-xvf", os.path.basename(str(tarball))], _OP,
                    cwd=_WORK_DIR, env=env) != 0:
        log_error(_OP, "could not extract the cuda-samples source", tarball=str(tarball))
        return False

    source_dir = os.path.join(_WORK_DIR, f"cuda-samples-{version}")
    build_dir = os.path.join(source_dir, "build")
    try:
        os.mkdir(build_dir)            # bash: mkdir build (no -p)
    except OSError as exc:
        log_error(_OP, f"could not create {build_dir}: {exc}", path=build_dir)
        return False

    # bash: [[ -f ../cmake/InstallSamples.cmake ]], from the build dir.
    marker = os.path.join(source_dir, _INSTALL_SAMPLES_CMAKE)
    cmake_install = os.path.isfile(marker)
    if cmake_install:
        log_info(_OP, f"{_INSTALL_SAMPLES_CMAKE} found: cmake will install the samples "
                      f"to {dest} (cmake --install)", marker=marker,
                 install_method="cmake --install", dest=dest)
    else:
        log_info(_OP, f"{_INSTALL_SAMPLES_CMAKE} not found (older samples release): "
                      f"the built Samples tree will be moved to {dest} (mv -vT)",
                 marker=marker, install_method="mv -vT", dest=dest)

    if exec_program(["cmake", *cmake_options(dest, cmake_install), ".."], _OP,
                    cwd=build_dir, env=env) != 0:
        log_error(_OP, "cmake configuration of the CUDA samples failed", path=build_dir)
        return False

    # $(nproc): ask nproc itself, so OMP_NUM_THREADS etc. count as in bash.
    rc, output = run_capture(["nproc"], _OP, env=env)
    jobs = output.strip()
    if rc != 0 or not jobs.isdigit():
        log_error(_OP, f"could not read the CPU count from nproc: {jobs!r}", exit_code=rc)
        return False
    if exec_program(["make", "-j", jobs], _OP, cwd=build_dir, env=env) != 0:
        log_error(_OP, "building the CUDA samples failed", path=build_dir, jobs=jobs)
        return False

    if cmake_install:
        if exec_program(["cmake", "--install", "."], _OP, cwd=build_dir, env=env) != 0:
            log_error(_OP, f"cmake --install of the CUDA samples to {dest} failed",
                      path=dest)
            return False
        log_info(_OP, f"Installed the built CUDA samples to {dest} (cmake --install)",
                 path=dest, action="installed")
        return True

    if os.path.lexists(dest):
        # BASH QUIRK (kept): `mv -T` will not replace an existing, non-empty
        # directory, so a rerun on the same image fails here, as in bash.
        log_warn(_OP, f"{dest} already exists; `mv -vT` fails if it is not an "
                      "empty directory (bash behaviour, kept)", path=dest)
    if exec_program(["mv", "-vT", "./Samples", dest], _OP, cwd=build_dir, env=env) != 0:
        log_error(_OP, f"could not move the built samples to {dest}", path=dest)
        return False
    log_info(_OP, f"Moved the built CUDA samples to {dest}", path=dest, action="moved")
    return True


def install(env: dict[str, str]) -> int:
    """Download, build and install the CUDA samples. Returns 0 or 3."""
    resolved = _resolve(env)
    if resolved is None:
        return 3
    version, sha256, cuda_version = resolved
    dest = samples_destination(cuda_version)
    log_info(_OP, f"Installing CUDA samples {version} for CUDA {cuda_version}",
             samples_version=version, cuda_version=cuda_version,
             source=samples_url(version), dest=dest, work_dir=_WORK_DIR)

    try:
        shutil.rmtree(_WORK_DIR, ignore_errors=True)
        os.makedirs(_WORK_DIR, exist_ok=True)
    except OSError as exc:
        log_error(_OP, f"could not create {_WORK_DIR}: {exc}", path=_WORK_DIR)
        return 3

    try:
        ok = _build_and_install(env, version, sha256, dest)
    finally:
        shutil.rmtree(_WORK_DIR, ignore_errors=True)
        log_info(_OP, f"Removed work dir {_WORK_DIR}", path=_WORK_DIR)
    if not ok:
        log_error(_OP, "CUDA samples installation failed", samples_version=version)
        return 3

    log_info(_OP, f"CUDA samples {version} installed", samples_version=version,
             cuda_version=cuda_version, dest=dest)
    return 0
