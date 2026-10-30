"""install_cuda_samples.py — Python port of components/install_cuda_samples.sh.

What this does
--------------
Downloads NVIDIA's cuda-samples source for the pinned version (cuda
samples.version / samples.sha256 in versions.json), builds it with cmake and
make against the installed CUDA toolkit, and installs the samples in
/usr/local/cuda-<cuda driver.version>/samples. Like bash, it records no
component version.

How the samples are installed depends on the release (upstream #614):
  * newer releases ship cmake/InstallSamples.cmake: cmake is given the install
    directory (-DCUDA_SAMPLES_INSTALL_DIR=...) and `cmake --install .` copies
    the samples there after make;
  * older releases don't: the built Samples tree is moved with `mv -vT`.

How it fits into the Python build
---------------------------------
Not a build_plan Step: the driver scripts run install_cuda_samples.sh right
after installing the CUDA toolkit (so nvcc exists). Their ports,
install_nvidiagpudriver and install_nvidiagriddriver, call install(env) at
that point and treat a non-zero result as their own failure, so keep the
`install(env) -> int` signature. Logs use the op "install-cuda-samples" so
the samples build can be told apart from the driver step around it.

tar, cmake, nproc, make, `cmake --install .` and `mv -vT` run as programs,
as in bash. The real `mv -T` keeps its behaviour: an existing, non-empty
samples directory makes the move (and the step) fail.

Deliberate differences from bash
--------------------------------
  * The work happens in _WORK_DIR, emptied first and removed afterwards (also
    on failure). Bash works in the caller's directory (the distro build dir)
    and leaves the tarball and source tree to the driver script's final
    `rm -rf *.tar.gz` / `rm -rf -- */`.
  * A missing samples.version, samples.sha256 or driver.version fails before
    anything is downloaded. Bash would fetch ".../vnull.tar.gz" (and install
    to /usr/local/cuda-null/samples).
  * tar runs as `tar -xf` (bash: `tar -xvf`; -v only listed the files).
  * The InstallSamples.cmake check is made once, before cmake. Bash repeats
    it after make, but the file comes from the tarball and nothing in the
    build creates or removes it, so the answer can't change.

Bash quirk, kept and warned about
---------------------------------
On the `mv -vT` path, rerunning on the same image fails: `mv -T` refuses to
replace the existing, non-empty samples directory. A warning is logged before
the move when the destination already exists.
"""

from __future__ import annotations

import os

from utils.component_config import require_config
from utils.download import fetch_verified
from utils.files import extract_tarball, remove_paths
from utils.logger import log_error, log_info, log_warn
from utils.process import run, run_capture

# The log label for this nested component (it has no Step of its own).
_OP = "install-cuda-samples"

# Our own download/build directory (see "Deliberate differences" above).
_WORK_DIR = "/tmp/cuda-samples-build"

# The samples go to <here>/cuda-<version>/samples. Module-level so the tests
# can point it at a temporary directory.
_CUDA_PREFIX = "/usr/local"

# The nvcc cmake is told to use (bash's -DCMAKE_CUDA_COMPILER value).
_NVCC = "/usr/local/cuda/bin/nvcc"

_URL_TEMPLATE = "https://github.com/NVIDIA/cuda-samples/archive/refs/tags/{tarball}"

# Shipped by newer cuda-samples releases (relative to the source directory).
_INSTALL_SAMPLES_CMAKE = os.path.join("cmake", "InstallSamples.cmake")


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Steps. Each returns a falsy value after its failure has been logged.
# --------------------------------------------------------------------------- #
def _fresh_work_dir() -> bool:
    # TODO(ADO 1001): toolkit candidate: fresh_dir(path, op) -> bool
    # (rm -rf, then mkdir -p; three driver/samples ports do this).
    remove_paths(_WORK_DIR, op=_OP)
    try:
        os.makedirs(_WORK_DIR, exist_ok=True)
    except OSError as exc:
        log_error(_OP, f"could not create {_WORK_DIR}: {exc}", path=_WORK_DIR)
        return False
    return True


def _unpack_source(env: dict[str, str], version: str, sha256: str) -> str | None:
    """Download, verify and extract the source; return the source directory."""
    tarball = fetch_verified(samples_url(version), sha256, _OP, _WORK_DIR)
    if tarball is None or not extract_tarball(tarball.name, _OP, cwd=_WORK_DIR, env=env):
        return None
    return os.path.join(_WORK_DIR, f"cuda-samples-{version}")


def _make_build_dir(source_dir: str) -> str | None:
    build_dir = os.path.join(source_dir, "build")
    try:
        os.mkdir(build_dir)            # bash: mkdir build (no -p)
    except OSError as exc:
        log_error(_OP, f"could not create {build_dir}: {exc}", path=build_dir)
        return None
    return build_dir


def _uses_cmake_install(source_dir: str, dest: str) -> bool:
    """bash: [[ -f ../cmake/InstallSamples.cmake ]], from the build dir."""
    marker = os.path.join(source_dir, _INSTALL_SAMPLES_CMAKE)
    cmake_install = os.path.isfile(marker)
    method = "cmake --install" if cmake_install else "mv -vT"
    log_info(_OP, f"{_INSTALL_SAMPLES_CMAKE} {'found' if cmake_install else 'not found'}: "
                  f"installing the samples to {dest} with {method}",
             marker=marker, install_method=method, dest=dest)
    return cmake_install


def _cpu_count(env: dict[str, str]) -> str | None:
    """$(nproc): ask nproc itself, so OMP_NUM_THREADS etc. count as in bash."""
    rc, output = run_capture(["nproc"], _OP, env=env)
    jobs = output.strip()
    if rc != 0 or not jobs.isdigit():
        log_error(_OP, f"could not read the CPU count from nproc: {jobs!r}", exit_code=rc)
        return None
    return jobs


def _build(env: dict[str, str], build_dir: str, dest: str, cmake_install: bool) -> bool:
    """cmake <options> .. && make -j $(nproc)"""
    if not run(["cmake", *cmake_options(dest, cmake_install), ".."], _OP,
               cwd=build_dir, env=env):
        return False
    jobs = _cpu_count(env)
    return jobs is not None and run(["make", "-j", jobs], _OP, cwd=build_dir, env=env)


def _install_samples(env: dict[str, str], build_dir: str, dest: str,
                     cmake_install: bool) -> bool:
    if cmake_install:
        return run(["cmake", "--install", "."], _OP, cwd=build_dir, env=env)
    if os.path.lexists(dest):
        # BASH QUIRK (kept): `mv -T` won't replace an existing, non-empty
        # directory, so a rerun on the same image fails here, as in bash.
        log_warn(_OP, f"{dest} already exists; `mv -vT` fails if it is not an "
                      "empty directory (bash behaviour, kept)", path=dest)
    return run(["mv", "-vT", "./Samples", dest], _OP, cwd=build_dir, env=env)


def _build_and_install(env: dict[str, str], version: str, sha256: str,
                       dest: str) -> bool:
    source_dir = _unpack_source(env, version, sha256)
    if source_dir is None:
        return False
    build_dir = _make_build_dir(source_dir)
    if build_dir is None:
        return False
    cmake_install = _uses_cmake_install(source_dir, dest)
    return (_build(env, build_dir, dest, cmake_install)
            and _install_samples(env, build_dir, dest, cmake_install))


def install(env: dict[str, str]) -> int:
    """Download, build and install the CUDA samples. Returns 0 or 3."""
    cfg = require_config("cuda", env, _OP,
                         "samples.version", "samples.sha256", "driver.version")
    if cfg is None:
        return 3
    version = str(cfg["samples"]["version"])
    cuda_version = str(cfg["driver"]["version"])
    dest = samples_destination(cuda_version)
    log_info(_OP, f"Installing CUDA samples {version} for CUDA {cuda_version}",
             samples_version=version, cuda_version=cuda_version,
             source=samples_url(version), dest=dest, work_dir=_WORK_DIR)

    if not _fresh_work_dir():
        return 3
    try:
        ok = _build_and_install(env, version, str(cfg["samples"]["sha256"]), dest)
    finally:
        remove_paths(_WORK_DIR, op=_OP)
    if not ok:
        log_error(_OP, "CUDA samples installation failed", samples_version=version,
                  dest=dest)
        return 3

    log_info(_OP, f"CUDA samples {version} installed", samples_version=version,
             cuda_version=cuda_version, dest=dest)
    return 0