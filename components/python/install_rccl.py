"""install_rccl.py — Python port of components/install_rccl.sh.

What this step does
-------------------
RCCL (ROCm Communication Collectives Library) is AMD's equivalent of NVIDIA's
NCCL. On AMD GPU images this step:

  1. Installs RCCL, one branch per distro (see _BRANCHES):
       * Ubuntu 26.04  RCCL came with the ROCm 10 SDK (install-rocm); its
                       version is read from rccl-config-version.cmake and must
                       look like X.Y.Z.
       * Azure Linux   the rccl, rccl-devel and rccl-unittests packages from
                       the ROCm repo install-rocm set up; the version is what
                       rpm reports.
       * other distros builds RCCL from source into /opt/rccl for gfx90a
                       (MI200) and gfx942 (MI300X) only, to save compile time.
                       The source is a git clone of a ref + commit when
                       versions.json gives a ref (or, for older entries, a
                       branch; needed because of ROCm/rccl#1877), else the
                       release tarball. CMakeLists.txt is edited first so the
                       compile uses every CPU and the memory-hungry link ~1 job
                       per 16 GB of RAM. The version is versions.json's.
  2. Sets NUMA auto-balancing off and a larger vm.max_map_count: applied now
     on Ubuntu (sysctl) and appended to /etc/sysctl.conf everywhere.
  3. Builds rccl-tests (the *_perf benchmarks) from a sparse checkout of
     ROCm/rocm-systems against RCCL + HPC-X MPI, into /opt/rccl-tests. On
     Ubuntu 26.04 the checkout is the commit ROCm's TheRock repo pins for this
     ROCm release (branch therock-<rocm version>), with extra cmake arguments
     (see rccl_tests_cmake_args).
  4. Builds linux-rdma perftest with ROCm support into /opt/rocm-perftest.
  Commits/refs of the RCCL clone, rocm-systems and perftest are recorded
  (RCCL_REPO_*, RCCL_TESTS_REPO_*, PERFTEST_REPO_*).

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets install() as the `action=` of the "install-rccl"
Step (AMD images only, after install-rocm); deleting `action=` runs the bash
script again. The rccl-tests build runs as one run_script() after `module
load mpi/hpcx`, because `module` (and the $HPCX_MPI_DIR it sets) only
affects the shell it runs in. install() returns 0 or 3.

Behaviour kept from bash
------------------------
  * Commands, arguments, order and stop-on-failure points are bash's.
  * The ref is chosen like `jq -r '.ref // .branch // empty'`; "" and "null"
    mean "use the tarball" (checked against real jq).
  * Ubuntu 26.04's version is read like `awk -F '"' '/^set\\(PACKAGE_VERSION
    "/ {print $2; exit}'` and must match ^[0-9]+\\.[0-9]+\\.[0-9]+$, else the
    step stops, as `[[ ... ]]` under `set -e` does.
  * Ubuntu 26.04's rccl-tests cmake gets -DROCM_PATH twice (/opt/rocm, then
    /opt/rocm/core; CMake keeps the later one) until AMD ships the missing
    /opt/rocm/.info link (bash TODO), and `git sparse-checkout set
    projects/rccl-tests` runs twice (harmless).
  * The CMakeLists.txt edits behave like bash's `sed -i -E 's/.../.../'`: per
    line, first match only; no match is not an error (we warn). The link-job
    maths is bash's: (MemTotal GB + 15) / 16, clamped to 1..16; no MemTotal
    line counts as 0 GB (we warn). Both checked against GNU sed/awk.
  * /etc/sysctl.conf is appended to, so a rerun adds the lines again (we warn
    when they are already there).
  * Azure Linux and Ubuntu 26.04 don't read versions.json's "rccl" entry.
  * Repo source info is recorded where bash does: RCCL (git clone only)
    after `make install`, before the source is removed; RCCL_TESTS right
    after the checkout; PERFTEST right after the clone.

Deliberate differences from bash
--------------------------------
  * Installs go through install_packages(): `apt-get install -y -o
    Dpkg::Use-Pty=0` / `dnf install -y` (bash: `apt install -y` / `dnf
    install -y`).
  * Work happens in _WORK_DIR (emptied first, removed at the end) instead of
    the distro build directory.
  * Missing config fails up front (require_config): on the source build
    version and url always (the url names the source folder), commit when a
    ref is given (bash would `git checkout null`), sha256 for the tarball; on
    Ubuntu 26.04 the ROCm version (bash would clone "therock-null").
  * Azure Linux: a failing or empty `rpm -q` fails the step (bash would
    record rpm's "package rccl is not installed"). Ubuntu 26.04: an empty
    TheRock `rev-parse` fails (bash would fetch an empty commit). A source
    folder the clone/tarball didn't create fails with a clear message.
  * `nproc` is read once for the sed edit and `make -j`; the rccl-tests
    script still calls $(nproc) itself. A non-numeric result fails the step.
  * `mkdir -p /opt/rccl-tests`, the rocm-systems checkout and its repo info
    run before `module load mpi/hpcx` (bash loads it first; git doesn't need
    HPC-X).
  * Toolkit normalisations: the tarball is unpacked with `tar -xf` (bash
    `tar -xzf`); `rm -rf`s go through remove_paths(), which never fails the
    build; the rccl-tests script drops bash's trailing `module unload` (the
    shell exits).
"""

from __future__ import annotations

import glob
import json
import os
import re
import shlex
from dataclasses import dataclass
from typing import Callable

from utils.component_config import require_config, write_component_version
from utils.distro import Distro
from utils.download import fetch_verified
from utils.files import extract_tarball, remove_paths, write_file
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import run, run_all, run_capture, run_script
from utils.repo_source import write_repo_source_info

_OP = "install-rccl"

# ---- Paths (module-level so tests can point them at a temporary directory) --
_WORK_DIR = "/tmp/azhpc-rccl"
_MEMINFO = "/proc/meminfo"
_SYSCTL_CONF = "/etc/sysctl.conf"
_RCCL_TESTS_DIR = "/opt/rccl-tests"
_PERFTEST_PREFIX_DIR = "/opt/rocm-perftest"
# Ubuntu 26.04: the CMake package-version file of the packaged RCCL.
_RCCL_CMAKE_VERSION_FILE = "/opt/rocm/lib/cmake/rccl/rccl-config-version.cmake"

_RCCL_GIT_URL = "https://github.com/ROCm/rccl.git"
_ROCM_SYSTEMS_GIT_URL = "https://github.com/ROCm/rocm-systems.git"
_THEROCK_GIT_URL = "https://github.com/ROCm/TheRock.git"
_PERFTEST_GIT_URL = "https://github.com/linux-rdma/perftest.git"

# Where RCCL ends up: the ROCm packages install into /opt/rocm; the source
# build into /opt/rccl.
_PACKAGED_RCCL_PREFIX = "/opt/rocm"
_SOURCE_RCCL_PREFIX = "/opt/rccl"

# Package installs, one list per bash command.
_AZURE_LINUX_RCCL_PACKAGES = ["rccl", "rccl-devel", "rccl-unittests"]
_PERFTEST_UBUNTU_PACKAGES = ["libpci-dev"]
_PERFTEST_UBUNTU2604_PACKAGES = ["libibumad-dev"]
# Installed after _PERFTEST_UBUNTU_PACKAGES.
_PERFTEST_EXTRA_PACKAGES = {"ubuntu26.04": _PERFTEST_UBUNTU2604_PACKAGES}

# Bash's AMD_GPU_TARGETS: Azure's AMD GPUs are MI300X (gfx942); gfx90a (MI200)
# is kept for compatibility. Building only these instead of upstream's 10-13
# default architectures cuts RCCL/rccl-tests compile time.
_AMD_GPU_TARGETS = "gfx90a;gfx942"

# The two kernel settings, in bash's order (sysctl argument = sysctl.conf line).
_SYSCTL_SETTINGS = ("kernel.numa_balancing=0", "vm.max_map_count=1048576")

# Link-time parallelism limits (see linker_jobs()).
_GB_PER_LINKER_JOB = 16
_MAX_LINKER_JOBS = 16

# The two sed patterns, from GNU sed -E: its `\s` is [[:space:]], spelled out
# because Python's \s also matches non-ASCII spaces; `[^)]*` can't cross a
# line because sed (and _sed_first_per_line) work line by line.
_WS = r"[ \t\n\r\f\v]"
_COMPILE_JOBS_RE = re.compile(
    rf"(target_compile_options\({_WS}*rccl{_WS}+PRIVATE[^)]*-parallel-jobs=)12")
_LINK_JOBS_RE = re.compile(
    rf"(target_link_options\({_WS}*rccl{_WS}+PRIVATE[^)]*-parallel-jobs=)[0-9]+")

# bash: [[ "$rccl_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] (used with fullmatch).
_RELEASE_VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
# The awk pattern /^set\(PACKAGE_VERSION "/ as a plain prefix.
_PACKAGE_VERSION_PREFIX = 'set(PACKAGE_VERSION "'


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
def tarball_name(url: str) -> str:
    """bash's TARBALL=$(basename ${rccl_url})."""
    return url.rstrip("/").rsplit("/", 1)[-1]


def source_folder_name(tarball: str) -> str:
    """bash's rccl_folder=rccl-$(basename $TARBALL .tar.gz).

    e.g. "rocm-6.4.4.tar.gz" -> "rccl-rocm-6.4.4", the archive's top
    directory. basename only strips the suffix when something is left.
    """
    suffix = ".tar.gz"
    if tarball.endswith(suffix) and tarball != suffix:
        tarball = tarball[:-len(suffix)]
    return f"rccl-{tarball}"


def rccl_ref(config) -> str:
    """bash: rccl_ref=$(jq -r '.ref // .branch // empty' <<< $rccl_metadata).

    `a // b` gives b when a is missing, null or false; `empty` prints nothing;
    -r prints other non-strings as JSON (1, true). Checked against real jq.
    """
    if isinstance(config, dict):
        for key in ("ref", "branch"):
            value = config.get(key)
            if value is not None and value is not False:
                return value if isinstance(value, str) else json.dumps(value)
    return ""


def uses_git_clone(ref) -> bool:
    """bash: [[ $rccl_ref != "" && $rccl_ref != "null" ]]."""
    return ref is not None and str(ref) not in ("", "null")


def rccl_version_from_cmake(text: str) -> str:
    """awk -F '"' '/^set\\(PACKAGE_VERSION "/ {print $2; exit}' over `text`:
    the second '"'-field of the first matching line, "" if none matches."""
    for line in text.split("\n"):
        if line.startswith(_PACKAGE_VERSION_PREFIX):
            return line.split('"')[1]
    return ""


def is_release_version(version: str) -> bool:
    """bash: [[ "$rccl_version" =~ ^[0-9]+\\.[0-9]+\\.[0-9]+$ ]]."""
    return _RELEASE_VERSION_RE.fullmatch(version) is not None


def therock_branch(rocm_version: str) -> str:
    """bash: RCCL_TEST_GIT_ARGS=(--branch "therock-${rocm_version}")."""
    return f"therock-{rocm_version}"


def rccl_tests_cmake_args(distribution: str) -> list[str]:
    """bash's RCCL_TEST_CMAKE_ARGS for this distro. (bash's quotes around
    -DGPU_TARGETS="..." only stop the `;` ending the command.)"""
    if distribution == "ubuntu26.04":
        return ["-DROCM_PATH=/opt/rocm/core", "-DGPU_TARGETS=gfx90a;gfx942;gfx1250",
                "-DCMAKE_INSTALL_RPATH=/opt/rocm/lib",
                "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"]
    return [f"-DGPU_TARGETS={_AMD_GPU_TARGETS}"]


def rccl_source_cmake_command() -> list[str]:
    """The RCCL source build's cmake (run in <source>/build). Bash's comment:
    AMDGPU_TARGETS is what RCCL 6.2.x reads, GPU_TARGETS 6.4+."""
    return ["cmake", "-DCMAKE_PREFIX_PATH=/opt/rocm/",
            f"-DCMAKE_INSTALL_PREFIX={_SOURCE_RCCL_PREFIX}",
            f"-DGPU_TARGETS={_AMD_GPU_TARGETS}", f"-DAMDGPU_TARGETS={_AMD_GPU_TARGETS}",
            ".."]


def _sed_first_per_line(pattern: re.Pattern, value: str, text: str) -> tuple[str, int]:
    """`sed -E 's/(<prefix>)<old>/\\1<value>/'` over `text`: per line, first
    match only. Returns (new text, lines changed). Line endings (a missing
    final newline, "\\r\\n") are kept exactly."""
    lines = text.split("\n")
    changed = 0
    for index, line in enumerate(lines):
        # A function replacement avoids backslash-escape surprises.
        lines[index], count = pattern.subn(lambda m: m.group(1) + value, line, count=1)
        changed += count
    return "\n".join(lines), changed


def set_compile_jobs(text: str, nproc: int) -> tuple[str, int]:
    """First sed: compile `-parallel-jobs=12` -> `-parallel-jobs=<nproc>`."""
    return _sed_first_per_line(_COMPILE_JOBS_RE, str(nproc), text)


def set_link_jobs(text: str, jobs: int) -> tuple[str, int]:
    """Second sed: link `-parallel-jobs=<any number>` -> `-parallel-jobs=<jobs>`."""
    return _sed_first_per_line(_LINK_JOBS_RE, str(jobs), text)


def mem_gb_from_meminfo(meminfo: str) -> int | None:
    """awk '/^MemTotal:/ {printf "%d", $2/1024/1024}': total RAM in whole GB,
    or None without a MemTotal line (awk prints nothing)."""
    for line in meminfo.split("\n"):
        if line.startswith("MemTotal:"):
            fields = line.split()
            # awk turns a non-numeric field into 0.
            match = re.match(r"[0-9]+", fields[1]) if len(fields) > 1 else None
            kb = int(match.group(0)) if match else 0
            return kb // (1024 * 1024)
    return None


def linker_jobs(mem_gb: int) -> int:
    """bash: (mem_gb + 15) / 16, clamped to 1..16 — one link job per started
    16 GB (amdgcn-link is memory-hungry; upstream's 16 OOM-kills small builders)."""
    jobs = (mem_gb + _GB_PER_LINKER_JOB - 1) // _GB_PER_LINKER_JOB
    return max(1, min(_MAX_LINKER_JOBS, jobs))


def azure_linux_rccl_version(rpm_output: str) -> str | None:
    """`rpm -q --queryformat '%{VERSION}-%{RELEASE}' rccl` output -> version."""
    return rpm_output.strip() or None


def rccl_tests_script(rccl_prefix: str, cmake_args: list[str]) -> list[str]:
    """The lines that build rccl-tests and copy *_perf to /opt/rccl-tests.

    run_script() runs them in the work directory after `module load
    mpi/hpcx`; $PATH, $HPCX_MPI_DIR and $(nproc) are left for that shell.
    The extra cmake arguments (bash's "${RCCL_TEST_CMAKE_ARGS[@]}") are
    shell-quoted so each stays one argument.
    """
    extra = " ".join(shlex.quote(arg) for arg in cmake_args)
    return [
        "pushd ./rocm-systems",
        "git sparse-checkout set projects/rccl-tests",
        "pushd projects/rccl-tests",
        "mkdir build",
        "pushd build",
        "PATH=/opt/rocm/bin:$PATH cmake \\",
        "    -DCMAKE_BUILD_TYPE=Release \\",
        f'    -DCMAKE_PREFIX_PATH="{rccl_prefix};/opt/rocm;$HPCX_MPI_DIR" \\',
        "    -DROCM_PATH=/opt/rocm \\",
        "    -DUSE_MPI=ON \\",
        f"    {extra} \\",
        "    ..",
        "make -j$(nproc)",
        f"cp ./*_perf {_RCCL_TESTS_DIR}/",
        "popd",
        "popd",
        "popd",
        "rm -rf rocm-systems",
    ]


def perftest_package_lists(distro: Distro) -> list[list[str]]:
    """The perftest prerequisites, one list per bash `apt install`."""
    lists = [_PERFTEST_UBUNTU_PACKAGES] if distro.is_ubuntu else []
    extra = _PERFTEST_EXTRA_PACKAGES.get(distro.name)
    return lists + [extra] if extra else lists


# --------------------------------------------------------------------------- #
# Small helpers.
# --------------------------------------------------------------------------- #
# TODO(ADO 1001): toolkit candidate: file_has_line(path, line) -> bool (for the
# "already there; appending again, as bash does" warning before `>>`).
def _has_line(path: str, line: str) -> bool:
    """True if `path` already has `line`; a missing/unreadable file is "no"."""
    try:
        with open(path, encoding="utf-8") as handle:
            return line in handle.read().splitlines()
    except OSError:
        return False


# TODO(ADO 1001): toolkit candidate: cpu_count(op, env=) -> int | None
# (bash `$(nproc)`, checked to be a positive number).
def _nproc(env: dict[str, str]) -> int | None:
    rc, output = run_capture(["nproc"], _OP, env=env)
    if rc != 0:
        return None                     # run_capture has logged the failure
    value = int(output.strip()) if output.strip().isdigit() else 0
    if value < 1:
        log_error(_OP, f"unexpected nproc output: {output.strip()!r}")
        return None
    return value


# TODO(ADO 1001): toolkit candidate: edit_file(path, transform, op) -> bool
# (bash `sed -i`, byte-preserving: surrogateescape + newline="").
def _edit_cmakelists(path: str, nproc: int, jobs: int) -> None:
    """Apply the two sed edits to RCCL's CMakeLists.txt in place. Raises
    OSError; install() catches it."""
    with open(path, encoding="utf-8", errors="surrogateescape", newline="") as handle:
        text = handle.read()
    text, compile_changes = set_compile_jobs(text, nproc)
    text, link_changes = set_link_jobs(text, jobs)
    with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as handle:
        handle.write(text)
    log_info(_OP, f"Edited {path} in place: compile -parallel-jobs={nproc}, "
                  f"link -parallel-jobs={jobs}", path=path,
             compile_lines_changed=compile_changes, link_lines_changed=link_changes)
    for kind, count in (("compile", compile_changes), ("link", link_changes)):
        if count == 0:
            # sed silently changes nothing; the build then uses upstream values.
            log_warn(_OP, f"no target_{kind}_options(rccl PRIVATE ... -parallel-jobs=) "
                          f"line matched in {path}; left unchanged, as sed would",
                     path=path)


# --------------------------------------------------------------------------- #
# What install() works from, resolved and checked before any change.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Inputs:
    distro: Distro
    branch: _Branch
    config: dict | None         # versions.json "rccl" (source build only)
    rocm_version: str | None    # Ubuntu 26.04: names TheRock's branch
    nproc: int

    @property
    def cmake_args(self) -> list[str]:
        return rccl_tests_cmake_args(self.distro.name)

    def start_fields(self) -> dict:
        """The fields every branch's opening log line carries."""
        return {"distro": self.distro.name, "rccl_prefix": self.branch.rccl_prefix,
                "rccl_tests_cmake_args": self.cmake_args}


# --------------------------------------------------------------------------- #
# 1. RCCL. Each branch returns the version to record, or None (logged).
# --------------------------------------------------------------------------- #
def _ubuntu2604_rccl_version(env: dict[str, str], inputs: _Inputs) -> str | None:
    """RCCL came with the ROCm SDK; read its version."""
    log_info(_OP, f"Using the RCCL from the ROCm {inputs.rocm_version} SDK packages on "
                  f"{inputs.distro} (installed by install-rocm)", source="rocm-sdk",
             rocm_version=inputs.rocm_version, **inputs.start_fields())
    try:
        with open(_RCCL_CMAKE_VERSION_FILE, encoding="utf-8", errors="replace") as handle:
            version = rccl_version_from_cmake(handle.read())
    except OSError as exc:
        log_error(_OP, f"could not read the packaged RCCL version: {exc}",
                  path=_RCCL_CMAKE_VERSION_FILE)
        return None
    if not is_release_version(version):
        log_error(_OP, f"packaged RCCL version {version!r} from {_RCCL_CMAKE_VERSION_FILE} "
                       "is not X.Y.Z (bash's [[ =~ ]] check stops here too)",
                  path=_RCCL_CMAKE_VERSION_FILE, version=version)
        return None
    log_info(_OP, f"Packaged RCCL version is {version}",
             path=_RCCL_CMAKE_VERSION_FILE, version=version)
    return version


# TODO(ADO 1001): toolkit candidate: installed_package_version(package, env, op)
# -> str | None (rpm -q --queryformat / dpkg-query -W -f='${Version}').
def _install_packaged(env: dict[str, str], inputs: _Inputs) -> str | None:
    log_info(_OP, f"Installing RCCL on {inputs.distro} from the ROCm dnf packages "
                  "(Azure Linux ships RCCL packaged)", source="dnf",
             **inputs.start_fields())
    if not install_packages(_AZURE_LINUX_RCCL_PACKAGES, env, _OP):
        return None
    rc, output = run_capture(
        ["rpm", "-q", "--queryformat", "%{VERSION}-%{RELEASE}", "rccl"], _OP, env=env)
    version = azure_linux_rccl_version(output) if rc == 0 else None
    if version is None:
        log_error(_OP, "could not read the installed rccl version from rpm",
                  exit_code=rc, output=output.strip())
    return version


# TODO(ADO 1001): toolkit candidate: git_clone(url, dest, op, *, ref=None,
# commit=None, cwd=, env=) -> bool (git clone --branch <ref> + git checkout).
def _fetch_source(env: dict[str, str], config: dict, folder: str, tarball: str) -> bool:
    """Put the RCCL source in _WORK_DIR/<folder>: git clone, or the tarball."""
    ref = rccl_ref(config)
    if uses_git_clone(ref):
        commit = str(config["commit"])
        log_info(_OP, "RCCL source: git clone of a pinned ref/commit "
                      "(works around ROCm/rccl#1877)",
                 source="git", ref=ref, commit=commit, folder=folder)
        return (run(["git", "clone", "--branch", ref, _RCCL_GIT_URL, folder], _OP,
                    cwd=_WORK_DIR, env=env)
                and run(["git", "checkout", commit], _OP,
                        cwd=os.path.join(_WORK_DIR, folder), env=env))
    log_info(_OP, "RCCL source: release tarball (no ref/branch in versions.json)",
             source="tarball", url=config["url"], folder=folder)
    return (fetch_verified(config["url"], config["sha256"], _OP, _WORK_DIR) is not None
            and extract_tarball(tarball, _OP, cwd=_WORK_DIR, env=env))


def _tune_parallelism(cmakelists: str, nproc: int) -> None:
    """The two sed edits: compile with every CPU, link ~1 job per 16 GB."""
    with open(_MEMINFO, encoding="utf-8") as handle:
        mem_gb = mem_gb_from_meminfo(handle.read())
    if mem_gb is None:
        log_warn(_OP, f"no MemTotal in {_MEMINFO}; treating RAM as 0 GB like bash")
        mem_gb = 0
    jobs = linker_jobs(mem_gb)
    log_info(_OP, f"RCCL link parallelism: detected {mem_gb} GB RAM -> "
                  f"-parallel-jobs={jobs}", mem_gb=mem_gb, linker_jobs=jobs, nproc=nproc)
    _edit_cmakelists(cmakelists, nproc, jobs)


def _build_rccl(env: dict[str, str], build_dir: str, version: str, nproc: int) -> bool:
    # bash's `CXX=... CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake ...` sets them
    # for cmake only.
    cmake_env = {**env, "CXX": "/opt/rocm/bin/hipcc", "CMAKE_POLICY_VERSION_MINIMUM": "3.5"}
    log_info(_OP, f"Building RCCL {version} into {_SOURCE_RCCL_PREFIX} for "
                  f"{_AMD_GPU_TARGETS}", version=version, prefix=_SOURCE_RCCL_PREFIX,
             gpu_targets=_AMD_GPU_TARGETS, phase="cmake")
    if not run(rccl_source_cmake_command(), _OP, cwd=build_dir, env=cmake_env):
        return False
    log_info(_OP, "Compiling and installing RCCL", phase="make", jobs=nproc)
    return run_all([["make", f"-j{nproc}"], ["make", "install"]], _OP,
                   cwd=build_dir, env=env)


def _install_from_source(env: dict[str, str], inputs: _Inputs) -> str | None:
    config = inputs.config
    version, ref = str(config["version"]), rccl_ref(config)
    tarball = tarball_name(str(config["url"]))
    folder = source_folder_name(tarball)
    source_dir = os.path.join(_WORK_DIR, folder)
    log_info(_OP, f"Installing RCCL {version} on {inputs.distro} from source (no "
                  "usable RCCL package on this distro)", version=version,
             source=str(config["url"]), ref=ref, commit=str(config.get("commit")),
             gpu_targets=_AMD_GPU_TARGETS, **inputs.start_fields())
    if not _fetch_source(env, config, folder, tarball):
        return None
    if not os.path.isdir(source_dir):
        log_error(_OP, f"RCCL source folder {source_dir} was not created by the "
                       "clone/extract", path=source_dir)
        return None
    os.mkdir(os.path.join(source_dir, "build"))     # plain mkdir: fails if present
    _tune_parallelism(os.path.join(source_dir, "CMakeLists.txt"), inputs.nproc)
    if not _build_rccl(env, os.path.join(source_dir, "build"), version, inputs.nproc):
        return None
    if not uses_git_clone(ref):
        log_info(_OP, "RCCL built from the release tarball; no repo commit/ref to "
                      "record, as in bash", source="tarball")
    elif not write_repo_source_info("RCCL", source_dir, _OP, commit=str(config["commit"]),
                                    ref=ref, env=env):
        return None
    remove_paths(source_dir, os.path.join(_WORK_DIR, tarball), op=_OP)
    log_info(_OP, f"RCCL {version} installed to {_SOURCE_RCCL_PREFIX}; source removed",
             version=version, prefix=_SOURCE_RCCL_PREFIX)
    return version


# --------------------------------------------------------------------------- #
# 2. Kernel settings.
# --------------------------------------------------------------------------- #
def _configure_sysctl(env: dict[str, str], distro: Distro) -> bool:
    if distro.is_ubuntu:
        if not run_all([["sysctl", setting] for setting in _SYSCTL_SETTINGS], _OP, env=env):
            return False
        log_info(_OP, "Applied kernel settings now with sysctl",
                 settings=list(_SYSCTL_SETTINGS))
    else:
        log_info(_OP, "Not applying sysctl settings now (Ubuntu only in bash); "
                      "they take effect at boot from sysctl.conf")
    # Bash's two `echo ... | tee -a` lines append even on a rerun.
    already = [setting for setting in _SYSCTL_SETTINGS if _has_line(_SYSCTL_CONF, setting)]
    if already:
        log_warn(_OP, f"{_SYSCTL_CONF} already has {', '.join(already)}; "
                      "appending again, as bash does on a rerun",
                 path=_SYSCTL_CONF, settings=already)
    return write_file(_SYSCTL_CONF, "".join(f"{s}\n" for s in _SYSCTL_SETTINGS), _OP,
                      append=True)


# --------------------------------------------------------------------------- #
# 3. rccl-tests. Each checkout leaves _WORK_DIR/rocm-systems sparse-cloned.
# --------------------------------------------------------------------------- #
def _clone_rocm_systems(env: dict[str, str], inputs: _Inputs) -> bool:
    return run(["git", "clone", "--depth=1", "--filter=blob:none", "--sparse",
                _ROCM_SYSTEMS_GIT_URL], _OP, cwd=_WORK_DIR, env=env)


def _therock_rocm_systems_commit(env: dict[str, str], branch: str) -> str | None:
    """Clone TheRock's `branch` and read the rocm-systems commit it pins (a
    git submodule, so `HEAD:rocm-systems` names it)."""
    if not run(["git", "clone", "--depth=1", "--filter=blob:none", "--sparse",
                "--branch", branch, _THEROCK_GIT_URL], _OP, cwd=_WORK_DIR, env=env):
        return None
    rc, output = run_capture(["git", "-C", "TheRock", "rev-parse", "HEAD:rocm-systems"],
                             _OP, cwd=_WORK_DIR, env=env)
    commit = output.strip() if rc == 0 else ""
    if not commit:
        log_error(_OP, "could not read the rocm-systems commit pinned by TheRock",
                  branch=branch, exit_code=rc)
        return None
    log_info(_OP, f"TheRock {branch} pins rocm-systems at {commit}",
             therock_branch=branch, rccl_tests_commit=commit)
    return commit


def _checkout_rocm_systems_therock(env: dict[str, str], inputs: _Inputs) -> bool:
    """Ubuntu 26.04: rocm-systems at the commit this ROCm release was built from."""
    branch = therock_branch(inputs.rocm_version)
    log_info(_OP, "Ubuntu 26.04: rccl-tests from the rocm-systems commit that "
                  f"TheRock pins for ROCm {inputs.rocm_version}",
             therock_branch=branch, phase="rccl-tests")
    commit = _therock_rocm_systems_commit(env, branch)
    if commit is None or not run_all([
        ["git", "init", "rocm-systems"],
        ["git", "-C", "rocm-systems", "remote", "add", "origin", _ROCM_SYSTEMS_GIT_URL],
        ["git", "-C", "rocm-systems", "sparse-checkout", "set", "projects/rccl-tests"],
        ["git", "-C", "rocm-systems", "fetch", "--depth=1", "--filter=blob:none",
         "origin", commit],
        ["git", "-C", "rocm-systems", "checkout", "--detach", "FETCH_HEAD"],
    ], _OP, cwd=_WORK_DIR, env=env):
        return False
    remove_paths(os.path.join(_WORK_DIR, "TheRock"), op=_OP)
    return True


def _build_rccl_tests(env: dict[str, str], inputs: _Inputs) -> bool:
    os.makedirs(_RCCL_TESTS_DIR, exist_ok=True)
    if not (inputs.branch.checkout_rocm_systems(env, inputs)
            and write_repo_source_info("RCCL_TESTS", os.path.join(_WORK_DIR, "rocm-systems"),
                                       _OP, env=env)):
        return False
    log_info(_OP, "Building rccl-tests from ROCm/rocm-systems (sparse clone) with HPC-X",
             rccl_prefix=inputs.branch.rccl_prefix, dest=_RCCL_TESTS_DIR,
             cmake_args=inputs.cmake_args, phase="rccl-tests")
    if not run_script(rccl_tests_script(inputs.branch.rccl_prefix, inputs.cmake_args),
                      _OP, modules=["mpi/hpcx"], cwd=_WORK_DIR, env=env):
        return False
    binaries = sorted(os.path.basename(p)
                      for p in glob.glob(os.path.join(_RCCL_TESTS_DIR, "*_perf")))
    log_info(_OP, f"Installed {len(binaries)} rccl-tests binaries in {_RCCL_TESTS_DIR}",
             dest=_RCCL_TESTS_DIR, binaries=binaries)
    return True


# --------------------------------------------------------------------------- #
# 4. perftest.
# --------------------------------------------------------------------------- #
def _build_perftest(env: dict[str, str], inputs: _Inputs) -> bool:
    extra = _PERFTEST_EXTRA_PACKAGES.get(inputs.distro.name)
    if extra:
        log_info(_OP, f"{inputs.distro}: perftest also needs {' '.join(extra)}",
                 packages=extra)
    if not all(install_packages(packages, env, _OP)
               for packages in perftest_package_lists(inputs.distro)):
        return False
    log_info(_OP, "Building perftest with ROCm support",
             url=_PERFTEST_GIT_URL, prefix=_PERFTEST_PREFIX_DIR, phase="perftest")
    source_dir = os.path.join(_WORK_DIR, "perftest")
    if not (run(["git", "clone", _PERFTEST_GIT_URL], _OP, cwd=_WORK_DIR, env=env)
            and write_repo_source_info("PERFTEST", source_dir, _OP, env=env)):
        return False
    os.makedirs(_PERFTEST_PREFIX_DIR, exist_ok=True)
    if not run_all([
        ["./autogen.sh"],
        ["./configure", "--enable-rocm", "--with-rocm=/opt/rocm",
         f"--prefix={_PERFTEST_PREFIX_DIR}/"],
        ["make", f"-j{inputs.nproc}"],
        ["make", "install"],
    ], _OP, cwd=source_dir, env=env):
        return False
    remove_paths(source_dir, op=_OP)
    log_info(_OP, f"perftest installed to {_PERFTEST_PREFIX_DIR}",
             prefix=_PERFTEST_PREFIX_DIR)
    return True


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Branch:
    """How one of bash's distro branches differs."""
    install_rccl: Callable[[dict, _Inputs], str | None]   # -> the version to record
    rccl_prefix: str
    checkout_rocm_systems: Callable[[dict, _Inputs], bool]
    config_component: str | None = None   # what versions.json entry it reads


_SOURCE_BUILD = _Branch(_install_from_source, _SOURCE_RCCL_PREFIX, _clone_rocm_systems,
                        "rccl")
# Bash's exact-match branches; every other distro builds from source.
_BRANCHES = {
    "ubuntu26.04": _Branch(_ubuntu2604_rccl_version, _PACKAGED_RCCL_PREFIX,
                           _checkout_rocm_systems_therock, "rocm"),
    "azurelinux3.0": _Branch(_install_packaged, _PACKAGED_RCCL_PREFIX,
                             _clone_rocm_systems),
}


def _source_config(env: dict[str, str]) -> dict | None:
    """versions.json "rccl" with the keys the source build reads: commit for a
    git clone, sha256 for the tarball."""
    config = require_config("rccl", env, _OP, "version", "url")
    if config is None:
        return None
    needed = "commit" if uses_git_clone(rccl_ref(config)) else "sha256"
    return require_config("rccl", env, _OP, needed)


def _resolve_inputs(env: dict[str, str], distro: Distro) -> _Inputs | None:
    branch = _BRANCHES.get(distro.name, _SOURCE_BUILD)
    config = rocm_version = None
    if branch.config_component == "rccl":
        config = _source_config(env)
        if config is None:
            return None
    elif branch.config_component == "rocm":
        rocm = require_config("rocm", env, _OP, "version")
        if rocm is None:
            return None
        rocm_version = str(rocm["version"])
    nproc = _nproc(env)
    if nproc is None:
        return None
    return _Inputs(distro, branch, config, rocm_version, nproc)


def _install_all(env: dict[str, str], inputs: _Inputs) -> str | None:
    """Steps 1-4; the RCCL version, or None (logged) at the first failure."""
    version = inputs.branch.install_rccl(env, inputs)
    if version is None:
        return None
    write_component_version("RCCL", version)
    log_info(_OP, f"Recorded RCCL={version}", component="RCCL", version=version)
    if not (_configure_sysctl(env, inputs.distro)
            and _build_rccl_tests(env, inputs)
            and _build_perftest(env, inputs)):
        return None
    return version


def install(env: dict[str, str]) -> int:
    """Install RCCL, rccl-tests and perftest, and set kernel settings. Returns 0 or 3."""
    inputs = _resolve_inputs(env, Distro.from_env(env))
    if inputs is None:
        return 3
    # An empty scratch area, so a failed run's leftovers can't break `git
    # clone` / `mkdir build`; removed whatever happens (our addition).
    remove_paths(_WORK_DIR, op=_OP)
    try:
        os.makedirs(_WORK_DIR, exist_ok=True)
        version = _install_all(env, inputs)
    except OSError as exc:
        # mkdir, CMakeLists edit, meminfo, and the versions file.
        log_error(_OP, f"file operation failed: {exc}",
                  path=getattr(exc, "filename", None))
        return 3
    finally:
        remove_paths(_WORK_DIR, op=_OP)
    if version is None:
        return 3
    log_info(_OP, f"RCCL {version} installed with rccl-tests and perftest",
             version=version, distro=inputs.distro.name,
             rccl_prefix=inputs.branch.rccl_prefix,
             rccl_tests=_RCCL_TESTS_DIR, perftest=_PERFTEST_PREFIX_DIR)
    return 0
