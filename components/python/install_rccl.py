"""install_rccl.py — Python port of components/install_rccl.sh.

What this step does
-------------------
RCCL (ROCm Communication Collectives Library) is AMD's equivalent of NVIDIA's
NCCL: the library multi-GPU / multi-node jobs use for all-reduce and friends.
This step, on AMD GPU images:

  1. Installs RCCL:
       * Ubuntu 26.04  RCCL already came with the ROCm 10 SDK packages
                       (install-rocm). Its version is read from
                       /opt/rocm/lib/cmake/rccl/rccl-config-version.cmake and
                       must look like X.Y.Z.
       * Azure Linux   from the distro packages (rccl, rccl-devel,
                       rccl-unittests) in the ROCm repo set up by install-rocm;
                       the version recorded is what rpm reports.
       * other distros builds RCCL from source into /opt/rccl, for the Azure
                       AMD GPU architectures only (gfx90a = MI200, gfx942 =
                       MI300X) to save compile time. The source is a `git
                       clone` of a ref (tag/branch) + commit when versions.json
                       gives a ref (or, for older entries, a branch; needed
                       because of https://github.com/ROCm/rccl/issues/1877),
                       otherwise the release tarball. Before building,
                       CMakeLists.txt is edited so the compile step uses every
                       CPU and the memory-hungry link step uses ~1 job per
                       16 GB of RAM. For a git clone, the commit and ref are
                       recorded (RCCL_REPO_COMMIT / RCCL_REPO_REF). The version
                       recorded is the one in versions.json.
  2. Sets two kernel settings recommended for RCCL: NUMA auto-balancing off and
     a much larger vm.max_map_count. They are applied now on Ubuntu (sysctl)
     and appended to /etc/sysctl.conf on every distro (applied at boot).
  3. Builds rccl-tests (the *_perf benchmarks) from the ROCm/rocm-systems
     monorepo against RCCL + HPC-X MPI, into /opt/rccl-tests, for gfx90a and
     gfx942. On Ubuntu 26.04 the rocm-systems commit is the one ROCm's TheRock
     repo pins for this ROCm release (branch therock-<rocm version>), it also
     builds for gfx1250, and the ROCm path is /opt/rocm/core. The checkout's
     commit and ref are recorded (RCCL_TESTS_REPO_*).
  4. Builds the upstream linux-rdma perftest tools with ROCm support into
     /opt/rocm-perftest (Ubuntu 26.04 also needs libibumad-dev), recording
     its commit and ref (PERFTEST_REPO_*).

The original bash, for comparison (abridged; see components/install_rccl.sh):

    rccl_metadata=$(get_component_config "rccl")
    AMD_GPU_TARGETS="gfx90a;gfx942"
    RCCL_TEST_CMAKE_ARGS=(-DGPU_TARGETS="${AMD_GPU_TARGETS}")
    RCCL_TEST_GIT_ARGS=()
    if [[ $DISTRIBUTION == "ubuntu26.04" ]]; then
        rocm_version=$(jq -r '.version' <<< "$(get_component_config "rocm")")
        RCCL_TEST_CMAKE_ARGS=(-DROCM_PATH=/opt/rocm/core -DGPU_TARGETS="gfx90a;gfx942;gfx1250" \\
            -DCMAKE_INSTALL_RPATH=/opt/rocm/lib -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON)
        RCCL_TEST_GIT_ARGS=(--branch "therock-${rocm_version}")
    fi

    if [[ $DISTRIBUTION == "ubuntu26.04" ]]; then
        rccl_version=$(awk -F '"' '/^set\\(PACKAGE_VERSION "/ {print $2; exit}' \\
            /opt/rocm/lib/cmake/rccl/rccl-config-version.cmake)
        [[ "$rccl_version" =~ ^[0-9]+\\.[0-9]+\\.[0-9]+$ ]]
        write_component_version "RCCL" "$rccl_version"
    elif [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
        dnf install -y rccl rccl-devel rccl-unittests
        write_component_version "RCCL" "$(rpm -q --queryformat '%{VERSION}-%{RELEASE}' rccl)"
    else
        rccl_ref=$(jq -r '.ref // .branch // empty' <<< $rccl_metadata)
        rccl_commit/version/url/sha256=$(jq -r '.<field>' <<< $rccl_metadata)
        TARBALL=$(basename ${rccl_url})
        rccl_folder=rccl-$(basename $TARBALL .tar.gz)
        if [[ $rccl_ref != "" && $rccl_ref != "null" ]]; then
            git clone --branch ${rccl_ref} https://github.com/ROCm/rccl.git ${rccl_folder}
            pushd ${rccl_folder}; git checkout ${rccl_commit}; popd
        else
            download_and_verify ${rccl_url} ${rccl_sha256}
            tar -xzf ${TARBALL}
        fi
        mkdir ./${rccl_folder}/build
        pushd ./${rccl_folder}/build
        sed -i -E "s/(target_compile_options\\(\\s*rccl\\s+PRIVATE[^)]*-parallel-jobs=)12/\\1$(nproc)/" ../CMakeLists.txt
        mem_gb=$(awk '/^MemTotal:/ {printf "%d", $2/1024/1024}' /proc/meminfo)
        num_linker_jobs=$(( (mem_gb + 15) / 16 ))      # then clamped to 1..16
        sed -i -E "s/(target_link_options\\(\\s*rccl\\s+PRIVATE[^)]*-parallel-jobs=)[0-9]+/\\1${num_linker_jobs}/" ../CMakeLists.txt
        CXX=/opt/rocm/bin/hipcc CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake -DCMAKE_PREFIX_PATH=/opt/rocm/ -DCMAKE_INSTALL_PREFIX=/opt/rccl \\
            -DGPU_TARGETS="${AMD_GPU_TARGETS}" -DAMDGPU_TARGETS="${AMD_GPU_TARGETS}" ..
        make -j$(nproc)
        make install
        popd
        if [[ $rccl_ref != "" && $rccl_ref != "null" ]]; then
            write_repo_source_info "RCCL" ./${rccl_folder} --commit "${rccl_commit}" --ref "${rccl_ref}"
        fi
        rm -rf ${TARBALL} ${rccl_folder}
        write_component_version "RCCL" ${rccl_version}
    fi

    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        sysctl kernel.numa_balancing=0
        sysctl vm.max_map_count=1048576
    fi
    echo "kernel.numa_balancing=0" | tee -a /etc/sysctl.conf
    echo "vm.max_map_count=1048576" | tee -a /etc/sysctl.conf

    source /etc/profile.d/modules.sh
    module load mpi/hpcx
    RCCL_PREFIX = /opt/rocm on azurelinux3.0 and ubuntu26.04, else /opt/rccl
    mkdir -p /opt/rccl-tests
    if [[ $DISTRIBUTION == "ubuntu26.04" ]]; then
        git clone --depth=1 --filter=blob:none --sparse "${RCCL_TEST_GIT_ARGS[@]}" https://github.com/ROCm/TheRock.git
        rccl_tests_commit=$(git -C TheRock rev-parse HEAD:rocm-systems)
        git init rocm-systems
        git -C rocm-systems remote add origin https://github.com/ROCm/rocm-systems.git
        git -C rocm-systems sparse-checkout set projects/rccl-tests
        git -C rocm-systems fetch --depth=1 --filter=blob:none origin "$rccl_tests_commit"
        git -C rocm-systems checkout --detach FETCH_HEAD
        rm -rf TheRock
    else
        git clone --depth=1 --filter=blob:none --sparse https://github.com/ROCm/rocm-systems.git
    fi
    write_repo_source_info "RCCL_TESTS" rocm-systems
    ... sparse-checkout projects/rccl-tests, cmake (with $HPCX_MPI_DIR and
    "${RCCL_TEST_CMAKE_ARGS[@]}"), make, cp ./*_perf /opt/rccl-tests/,
    rm -rf rocm-systems ...
    module unload mpi/hpcx

    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then apt install -y libpci-dev; fi
    if [[ $DISTRIBUTION == "ubuntu26.04" ]]; then apt install -y libibumad-dev; fi
    git clone https://github.com/linux-rdma/perftest.git
    write_repo_source_info "PERFTEST" perftest
    mkdir -p /opt/rocm-perftest
    pushd ./perftest
    ./autogen.sh
    ./configure --enable-rocm --with-rocm=/opt/rocm --prefix=/opt/rocm-perftest/
    make -j$(nproc); make install
    popd; rm -rf perftest

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-rccl" Step (AMD images only, after install-rocm). Deleting `action=`
from that Step makes the build run the bash script again, which is why the .sh
file is kept.

Most of the work is single commands run with exec_program. The rccl-tests
build is the exception: it needs `module load mpi/hpcx` (and the
$HPCX_MPI_DIR variable that sets), and `module` only affects the shell it runs
in, so that part runs as one `bash -c` script with `set -e` (like
install_mpifileutils.py). The git clones before it, the CMakeLists.txt edits,
the memory maths and /etc/sysctl.conf are plain Python / single commands.
write_repo_source_info is the Python port in utils/repo_source.py.

install() returns 0 for success or 3 for failure. It logs problems and returns
3; it never lets an exception escape to the build loop.

Behaviour kept from bash
------------------------
  * Commands, arguments and order are the same (installs spelled the
    install_packages() way, see below), and the first failing command
    stops the step (bash's `set -e`).
  * The ref is chosen like `jq -r '.ref // .branch // empty'`: "ref" unless it
    is missing/null/false, then "branch" likewise, else "" (checked against
    real jq). As before, "" and "null" mean "use the tarball".
  * Ubuntu 26.04's RCCL version is read like `awk -F '"' '/^set\\(PACKAGE_VERSION
    "/ {print $2; exit}'` and must match ^[0-9]+\\.[0-9]+\\.[0-9]+$, else the
    step fails as bash's `[[ ... ]]` under `set -e` does (checked against real
    awk/bash).
  * Ubuntu 26.04's rccl-tests cmake gets -DROCM_PATH twice (/opt/rocm, then
    /opt/rocm/core); the later one wins in CMake. Bash's TODO: AMD needs to
    ship the missing /opt/rocm/.info link before the override can go. Its
    rocm-systems checkout also gets `git sparse-checkout set
    projects/rccl-tests` twice (harmless), as in bash.
  * The CMakeLists.txt edits behave like `sed -i -E 's/.../.../'`: line by
    line, first match per line, a line that doesn't match is left alone, and
    no match at all is not an error (we log a warning). Checked against real
    GNU sed with representative lines.
  * The link-job maths is bash's: mem_gb = MemTotal kB / 1024 / 1024 truncated
    (awk printf "%d"), jobs = (mem_gb + 15) / 16 clamped to 1..16. If
    /proc/meminfo has no MemTotal line, awk prints nothing, bash's arithmetic
    treats it as 0 and uses 1 job; we do the same (with a warning).
  * /etc/sysctl.conf is appended to, so a rerun adds the two lines again
    (with a warning when they are already there).
  * The Azure Linux and Ubuntu 26.04 branches don't use the "rccl" entry of
    versions.json, as in bash.
  * Repo source info is recorded at the same points as bash: RCCL (git clone
    only) after `make install`, before the source is removed and before RCCL
    itself; RCCL_TESTS right after the checkout; PERFTEST right after the
    clone.

Deliberate differences from bash
--------------------------------
  * Package installs go through install_packages() (utils/package_installer.py),
    which spells them `apt-get install -y -o Dpkg::Use-Pty=0` / `dnf install -y`
    (bash: `apt install -y` / `dnf install -y`).
  * Work directory: bash clones/downloads/builds in its current directory (the
    distro build dir). We use _WORK_DIR, emptied first (so leftovers from a
    failed run can't break `git clone` / `mkdir build`) and removed at the end.
  * Missing config fails up front on the source-build path: version and url
    always (the url names the source folder), commit when a ref is given
    (bash would run `git checkout null`), sha256 when downloading the tarball.
    On Ubuntu 26.04 a missing ROCm version fails up front too (bash would
    clone branch "therock-null" and fail there).
  * Azure Linux: a failing or empty `rpm -q` fails the step. Bash, inside
    "$(...)", would record rpm's message ("package rccl is not installed").
  * Ubuntu 26.04: an empty `git rev-parse HEAD:rocm-systems` result fails the
    step (bash would go on to fetch an empty commit).
  * A source folder that the clone/tarball didn't create fails with a clear
    message (bash's `mkdir ./<folder>/build` would fail too, less clearly).
  * nproc is read once with `nproc` (run_capture) for the sed edit and
    `make -j`; the bash -c script still calls $(nproc) itself. A non-numeric
    result fails the step.
  * rccl-tests: `mkdir -p /opt/rccl-tests`, the rocm-systems checkout (and
    TheRock lookup on 26.04) and write_repo_source_info run BEFORE `source
    /etc/profile.d/modules.sh; module load mpi/hpcx`, which is the first line
    of the bash -c build script. Bash loads the module first; git and the
    repo-info lookup don't use anything HPC-X provides.
  * Cleanup of the RCCL source/tarball, TheRock, rocm-systems and perftest
    ignores deletion errors (bash's `rm -rf` practically never fails).
"""

# Lets type hints such as `str | None` work on older Python versions.
from __future__ import annotations

import glob
import json
import os
import re
import shlex
import shutil

from utils.component_config import config_for, write_component_version
from utils.download import download_and_verify
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import exec_program, run_capture
from utils.repo_source import write_repo_source_info

# The Step's `op` label in build_config.py; used on every log line.
_OP = "install-rccl"

# Bash's exact-match distro branches.
_AZURE_LINUX = "azurelinux3.0"
_UBUNTU_2604 = "ubuntu26.04"

# ---- Paths (module-level so tests can point them at a temporary directory) --
# Scratch area for the RCCL source, rocm-systems and perftest checkouts.
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

# Where RCCL ends up: the ROCm packages install into /opt/rocm (Azure Linux,
# Ubuntu 26.04); the source build installs into /opt/rccl (everything else).
_PACKAGED_RCCL_PREFIX = "/opt/rocm"
_SOURCE_RCCL_PREFIX = "/opt/rccl"

# Package installs, one list per bash command.
_AZURE_LINUX_RCCL_PACKAGES = ["rccl", "rccl-devel", "rccl-unittests"]
_PERFTEST_UBUNTU_PACKAGES = ["libpci-dev"]
_PERFTEST_UBUNTU2604_PACKAGES = ["libibumad-dev"]

# Bash's AMD_GPU_TARGETS: Azure's AMD GPUs are MI300X (gfx942); gfx90a (MI200)
# is kept for compatibility. Building only these instead of upstream's 10-13
# default architectures cuts RCCL/rccl-tests compile time.
_AMD_GPU_TARGETS = "gfx90a;gfx942"

# The two kernel settings, in bash's order: (sysctl argument, sysctl.conf line).
_SYSCTL_SETTINGS = ("kernel.numa_balancing=0", "vm.max_map_count=1048576")

# Link-time parallelism limits (see linker_jobs()).
_GB_PER_LINKER_JOB = 16
_MAX_LINKER_JOBS = 16

# The two sed patterns, translated from GNU sed -E to Python:
#   * `\s` in GNU sed means [[:space:]]; we spell the class out because
#     Python's \s also matches non-ASCII spaces.
#   * `[^)]*` can't cross a line because sed works on one line at a time, and
#     so do we (see _sed_first_per_line).
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
# Pure helpers: no side effects, so tests call them directly.
# --------------------------------------------------------------------------- #
def tarball_name(url: str) -> str:
    """bash's TARBALL=$(basename ${rccl_url}): the URL's last path part."""
    # basename ignores trailing slashes; rsplit("/", 1)[-1] takes the last part.
    return url.rstrip("/").rsplit("/", 1)[-1]


def source_folder_name(tarball: str) -> str:
    """bash's rccl_folder=rccl-$(basename $TARBALL .tar.gz).

    e.g. "rocm-6.4.4.tar.gz" -> "rccl-rocm-6.4.4", which is the top directory
    GitHub puts in the archive. basename only strips the suffix when something
    is left, hence the length check.
    """
    suffix = ".tar.gz"
    if tarball.endswith(suffix) and tarball != suffix:
        tarball = tarball[:-len(suffix)]
    return f"rccl-{tarball}"


def rccl_ref(config) -> str:
    """bash: rccl_ref=$(jq -r '.ref // .branch // empty' <<< $rccl_metadata).

    jq's `a // b` gives b when a is missing, null or false; `empty` prints
    nothing. Any other value (even "") is printed as is; `-r` prints strings
    without quotes and other values as JSON (1, true). Checked against real jq.
    """
    if isinstance(config, dict):
        for key in ("ref", "branch"):
            value = config.get(key)
            if value is not None and value is not False:
                return value if isinstance(value, str) else json.dumps(value)
    return ""


def uses_git_clone(ref) -> bool:
    """bash: [[ $rccl_ref != "" && $rccl_ref != "null" ]].

    jq -r prints "null" for a null value; config_for() gives None instead.
    """
    return ref is not None and str(ref) not in ("", "null")


def rccl_version_from_cmake(text: str) -> str:
    """awk -F '"' '/^set\\(PACKAGE_VERSION "/ {print $2; exit}' over `text`.

    The first line starting with `set(PACKAGE_VERSION "` wins; its second
    '"'-separated field (the text between the first two quotes, or to the end
    of the line) is the version. "" when no line matches (awk prints nothing).
    """
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
    """bash's RCCL_TEST_CMAKE_ARGS array for this distro.

    The quotes in bash (-DGPU_TARGETS="gfx90a;gfx942") only keep the `;` from
    ending the command; cmake receives -DGPU_TARGETS=gfx90a;gfx942.
    """
    if distribution == _UBUNTU_2604:
        # Bash TODO: AMD needs to provide the missing /opt/rocm/.info link;
        # revisit the ROCM_PATH override once it exists.
        return ["-DROCM_PATH=/opt/rocm/core", "-DGPU_TARGETS=gfx90a;gfx942;gfx1250",
                "-DCMAKE_INSTALL_RPATH=/opt/rocm/lib",
                "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"]
    return [f"-DGPU_TARGETS={_AMD_GPU_TARGETS}"]


def rccl_source_cmake_command() -> list[str]:
    """The cmake command for the RCCL source build (run in <source>/build).

    Bash's comment: AMDGPU_TARGETS is what RCCL 6.2.x reads, GPU_TARGETS 6.4+.
    """
    return ["cmake", "-DCMAKE_PREFIX_PATH=/opt/rocm/",
            f"-DCMAKE_INSTALL_PREFIX={_SOURCE_RCCL_PREFIX}",
            f"-DGPU_TARGETS={_AMD_GPU_TARGETS}", f"-DAMDGPU_TARGETS={_AMD_GPU_TARGETS}",
            ".."]


def _sed_first_per_line(pattern: re.Pattern, value: str, text: str) -> tuple[str, int]:
    """Mirror `sed -E 's/(<prefix>)<old>/\\1<value>/'` over `text`.

    sed works line by line and, without the `g` flag, replaces only the first
    match on each line. Returns (new text, number of lines changed).
    split("\\n") + "\\n".join() keeps the line endings exactly, including a
    missing newline at the end and any "\\r" before "\\n".
    """
    lines = text.split("\n")
    changed = 0
    for index, line in enumerate(lines):
        # A function as the replacement avoids backslash-escape surprises.
        # m.group(1) is the text inside the first (...), sed's \1.
        new, count = pattern.subn(lambda m: m.group(1) + value, line, count=1)
        lines[index] = new
        changed += count
    return "\n".join(lines), changed


def set_compile_jobs(text: str, nproc: int) -> tuple[str, int]:
    """First sed: compile `-parallel-jobs=12` -> `-parallel-jobs=<nproc>`."""
    return _sed_first_per_line(_COMPILE_JOBS_RE, str(nproc), text)


def set_link_jobs(text: str, jobs: int) -> tuple[str, int]:
    """Second sed: link `-parallel-jobs=<any number>` -> `-parallel-jobs=<jobs>`."""
    return _sed_first_per_line(_LINK_JOBS_RE, str(jobs), text)


def mem_gb_from_meminfo(meminfo: str) -> int | None:
    """awk '/^MemTotal:/ {printf "%d", $2/1024/1024}' — total RAM in whole GB.

    MemTotal is in kB, so kB / 1024 / 1024 = GB, truncated like printf "%d".
    (For whole, non-negative numbers, // (floor division) gives the same.)
    Returns None when there's no MemTotal line (awk would print nothing).
    """
    for line in meminfo.split("\n"):
        if line.startswith("MemTotal:"):
            fields = line.split()
            # awk turns a non-numeric field into 0; so do we.
            match = re.match(r"[0-9]+", fields[1]) if len(fields) > 1 else None
            kb = int(match.group(0)) if match else 0
            return kb // (1024 * 1024)
    return None


def linker_jobs(mem_gb: int) -> int:
    """bash: num_linker_jobs = (mem_gb + 15) / 16, clamped to 1..16.

    i.e. one link job per started 16 GB of RAM (amdgcn-link is memory-hungry
    and the upstream default of 16 OOM-kills small builders). min()/max() do
    the two clamping `if`s.
    """
    jobs = (mem_gb + _GB_PER_LINKER_JOB - 1) // _GB_PER_LINKER_JOB
    return max(1, min(_MAX_LINKER_JOBS, jobs))


def azure_linux_rccl_version(rpm_output: str) -> str | None:
    """`rpm -q --queryformat '%{VERSION}-%{RELEASE}' rccl` output -> version.

    bash's "$(...)" strips trailing newlines; .strip() does that (and spaces).
    """
    version = rpm_output.strip()
    return version or None


def rccl_tests_script(rccl_prefix: str, cmake_args: list[str]) -> str:
    """The bash script that builds rccl-tests and copies *_perf to /opt/rccl-tests.

    It runs in the work directory, after rocm-systems has been checked out
    there. One shell, because `module load mpi/hpcx` (which also sets
    $HPCX_MPI_DIR) only affects the shell that runs it. `set -e` stops at the
    first failure, as in bash. $PATH, $HPCX_MPI_DIR and $(nproc) are left for
    that shell to expand; rccl_prefix, the extra cmake arguments (bash's
    "${RCCL_TEST_CMAKE_ARGS[@]}", shell-quoted so each stays one argument) and
    the destination are filled in here.
    """
    extra = " ".join(shlex.quote(arg) for arg in cmake_args)
    return (
        "set -e\n"
        "source /etc/profile.d/modules.sh\n"
        "module load mpi/hpcx\n"
        "pushd ./rocm-systems\n"
        "git sparse-checkout set projects/rccl-tests\n"
        "pushd projects/rccl-tests\n"
        "mkdir build\n"
        "pushd build\n"
        "PATH=/opt/rocm/bin:$PATH cmake \\\n"
        "    -DCMAKE_BUILD_TYPE=Release \\\n"
        f'    -DCMAKE_PREFIX_PATH="{rccl_prefix};/opt/rocm;$HPCX_MPI_DIR" \\\n'
        "    -DROCM_PATH=/opt/rocm \\\n"
        "    -DUSE_MPI=ON \\\n"
        f"    {extra} \\\n"
        "    ..\n"
        "make -j$(nproc)\n"
        f"cp ./*_perf {_RCCL_TESTS_DIR}/\n"
        "popd\n"
        "popd\n"
        "popd\n"
        "rm -rf rocm-systems\n"
        "module unload mpi/hpcx\n"
    )


# --------------------------------------------------------------------------- #
# Small helpers with side effects.
# --------------------------------------------------------------------------- #
def _run_all(commands, env: dict[str, str], cwd: str | None = None) -> bool:
    """Run commands in order, stopping at the first failure (like `set -e`).

    exec_program() already logs each command, its exit code, and on failure
    its last output lines; here we only add what the failure means.
    """
    for command in commands:
        rc = exec_program(command, _OP, cwd=cwd, env=env)
        if rc != 0:
            # Long bash -c scripts are summarised by their first word.
            shown = command[0] if command[0] == "bash" else " ".join(command)
            log_error(_OP, f"RCCL install stopped: `{shown}` failed", exit_code=rc)
            return False
    return True


def _record_repo(component: str, repo_dir: str, env: dict[str, str], *,
                 commit: str | None = None, ref: str | None = None) -> bool:
    """write_repo_source_info, plus what a failure means for this step."""
    if write_repo_source_info(component, repo_dir, _OP, commit=commit, ref=ref, env=env):
        return True
    log_error(_OP, f"RCCL install stopped: could not record the {component} "
                   "source commit/ref", component=component, path=repo_dir)
    return False


def _nproc(env: dict[str, str]) -> int | None:
    """Run `nproc` (CPUs this process may use) and return it as a number."""
    rc, output = run_capture(["nproc"], _OP, env=env)
    if rc != 0:
        log_error(_OP, "could not count CPUs with nproc")
        return None
    try:
        value = int(output.strip())
    except ValueError:
        value = 0
    if value < 1:
        log_error(_OP, f"unexpected nproc output: {output.strip()!r}")
        return None
    return value


# --------------------------------------------------------------------------- #
# RCCL itself. Each returns the version to record, or None on failure.
# They may raise OSError from file operations; install() catches it.
# --------------------------------------------------------------------------- #
def _ubuntu2604_rccl_version() -> str | None:
    """Ubuntu 26.04: RCCL came with the ROCm SDK; read its version."""
    try:
        with open(_RCCL_CMAKE_VERSION_FILE, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        log_error(_OP, f"could not read the packaged RCCL version: {exc}",
                  path=_RCCL_CMAKE_VERSION_FILE)
        return None
    version = rccl_version_from_cmake(text)
    if not is_release_version(version):
        log_error(_OP, f"packaged RCCL version {version!r} from {_RCCL_CMAKE_VERSION_FILE} "
                       "is not X.Y.Z (bash's [[ =~ ]] check stops here too)",
                  path=_RCCL_CMAKE_VERSION_FILE, version=version)
        return None
    log_info(_OP, f"Packaged RCCL version is {version}",
             path=_RCCL_CMAKE_VERSION_FILE, version=version)
    return version


def _install_packaged(env: dict[str, str]) -> str | None:
    if not install_packages(_AZURE_LINUX_RCCL_PACKAGES, env, _OP):
        return None
    rc, output = run_capture(
        ["rpm", "-q", "--queryformat", "%{VERSION}-%{RELEASE}", "rccl"], _OP, env=env)
    version = azure_linux_rccl_version(output) if rc == 0 else None
    if version is None:
        log_error(_OP, "could not read the installed rccl version from rpm",
                  exit_code=rc, output=output.strip())
        return None
    return version


def _fetch_source(env: dict[str, str], config: dict, folder: str, tarball: str) -> bool:
    """Put the RCCL source in _WORK_DIR/<folder>: git clone, or tarball."""
    ref = rccl_ref(config)
    if uses_git_clone(ref):
        commit = str(config["commit"])
        log_info(_OP, "RCCL source: git clone of a pinned ref/commit "
                      "(works around ROCm/rccl#1877)",
                 source="git", ref=ref, commit=commit, folder=folder)
        return (_run_all([["git", "clone", "--branch", ref, _RCCL_GIT_URL, folder]],
                         env, cwd=_WORK_DIR)
                and _run_all([["git", "checkout", commit]], env,
                             cwd=os.path.join(_WORK_DIR, folder)))

    url, sha256 = config["url"], config["sha256"]
    log_info(_OP, "RCCL source: release tarball (no ref/branch in versions.json)",
             source="tarball", url=url, folder=folder)
    try:
        download_and_verify(url, sha256, dest_dir=_WORK_DIR)
    except Exception as exc:  # network errors, checksum mismatch, ...
        log_error(_OP, f"download/verify of the RCCL tarball failed: {exc}", url=url)
        return False
    # tar (not Python's tarfile) keeps bash's extraction semantics.
    return _run_all([["tar", "-xzf", tarball]], env, cwd=_WORK_DIR)


def _edit_cmakelists(path: str, nproc: int, jobs: int) -> None:
    """Apply the two sed edits to RCCL's CMakeLists.txt, in place."""
    # newline="" keeps "\r\n" as is, and surrogateescape lets any byte that
    # isn't valid UTF-8 round-trip unchanged - sed doesn't care about encodings.
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


def _install_from_source(env: dict[str, str], config: dict, nproc: int) -> str | None:
    version = str(config["version"])
    tarball = tarball_name(str(config["url"]))
    folder = source_folder_name(tarball)
    source_dir = os.path.join(_WORK_DIR, folder)

    if not _fetch_source(env, config, folder, tarball):
        return None
    if not os.path.isdir(source_dir):
        log_error(_OP, f"RCCL source folder {source_dir} was not created by the "
                       "clone/extract", folder=folder)
        return None

    build_dir = os.path.join(source_dir, "build")
    os.mkdir(build_dir)     # bash's plain `mkdir` (fails if it already exists)

    # ---- Parallelism tuning ------------------------------------------------
    with open(_MEMINFO, encoding="utf-8") as handle:
        mem_gb = mem_gb_from_meminfo(handle.read())
    if mem_gb is None:
        log_warn(_OP, f"no MemTotal in {_MEMINFO}; treating RAM as 0 GB like bash")
        mem_gb = 0
    jobs = linker_jobs(mem_gb)
    log_info(_OP, f"RCCL link parallelism: detected {mem_gb} GB RAM -> "
                  f"-parallel-jobs={jobs}", mem_gb=mem_gb, linker_jobs=jobs,
             nproc=nproc)
    _edit_cmakelists(os.path.join(source_dir, "CMakeLists.txt"), nproc, jobs)

    # ---- Build and install -------------------------------------------------
    # `VAR=value cmake ...` in bash sets VAR for that one command only;
    # {**env, ...} makes a copy of env with the two extra variables.
    cmake_env = {**env, "CXX": "/opt/rocm/bin/hipcc", "CMAKE_POLICY_VERSION_MINIMUM": "3.5"}
    log_info(_OP, f"Building RCCL {version} into {_SOURCE_RCCL_PREFIX} for "
                  f"{_AMD_GPU_TARGETS}", version=version, prefix=_SOURCE_RCCL_PREFIX,
             gpu_targets=_AMD_GPU_TARGETS, phase="cmake")
    if not _run_all([rccl_source_cmake_command()], cmake_env, cwd=build_dir):
        return None
    log_info(_OP, "Compiling and installing RCCL", phase="make", jobs=nproc)
    if not _run_all([["make", f"-j{nproc}"], ["make", "install"]], env, cwd=build_dir):
        return None

    # bash: only a git clone has a commit/ref to record.
    ref = rccl_ref(config)
    if uses_git_clone(ref):
        if not _record_repo("RCCL", source_dir, env, commit=str(config["commit"]), ref=ref):
            return None
    else:
        log_info(_OP, "RCCL built from the release tarball; no repo commit/ref to "
                      "record, as in bash", source="tarball")

    # bash: rm -rf ${TARBALL} ${rccl_folder}
    shutil.rmtree(source_dir, ignore_errors=True)
    tarball_path = os.path.join(_WORK_DIR, tarball)
    if os.path.isfile(tarball_path):
        os.remove(tarball_path)
    log_info(_OP, f"RCCL {version} installed to {_SOURCE_RCCL_PREFIX}; source removed",
             version=version, prefix=_SOURCE_RCCL_PREFIX)
    return version


# --------------------------------------------------------------------------- #
# Kernel settings, rccl-tests, perftest.
# --------------------------------------------------------------------------- #
def _configure_sysctl(env: dict[str, str], is_ubuntu: bool) -> bool:
    if is_ubuntu:
        commands = [["sysctl", setting] for setting in _SYSCTL_SETTINGS]
        if not _run_all(commands, env):
            return False
        log_info(_OP, "Applied kernel settings now with sysctl",
                 settings=list(_SYSCTL_SETTINGS))
    else:
        log_info(_OP, "Not applying sysctl settings now (Ubuntu only in bash); "
                      "they take effect at boot from sysctl.conf")

    # Two `echo ... | tee -a` lines: one append per setting, each with "\n".
    # `tee -a` appends even when the line is already there (e.g. on a rerun),
    # so the file can end up with duplicates; harmless, but worth a warning.
    try:
        with open(_SYSCTL_CONF, encoding="utf-8") as handle:
            existing = handle.read().splitlines()
    except OSError:          # missing (or unreadable): nothing to duplicate
        existing = []
    already = [setting for setting in _SYSCTL_SETTINGS if setting in existing]
    if already:
        log_warn(_OP, f"{_SYSCTL_CONF} already has {', '.join(already)}; "
                      "appending again, as bash does on a rerun",
                 path=_SYSCTL_CONF, settings=already)
    with open(_SYSCTL_CONF, "a", encoding="utf-8") as handle:
        handle.write("".join(f"{setting}\n" for setting in _SYSCTL_SETTINGS))
    log_info(_OP, f"Appended to {_SYSCTL_CONF}", path=_SYSCTL_CONF,
             action="appended", settings=list(_SYSCTL_SETTINGS))
    return True


def _checkout_rocm_systems_therock(env: dict[str, str], rocm_version: str) -> bool:
    """Ubuntu 26.04: check out rocm-systems at the commit TheRock pins.

    TheRock (ROCm's build repo) has rocm-systems as a git submodule; its
    therock-<version> branch records which rocm-systems commit that ROCm
    release was built from. `HEAD:rocm-systems` names that submodule entry.
    """
    branch = therock_branch(rocm_version)
    if not _run_all([["git", "clone", "--depth=1", "--filter=blob:none", "--sparse",
                      "--branch", branch, _THEROCK_GIT_URL]], env, cwd=_WORK_DIR):
        return False
    rc, output = run_capture(["git", "-C", "TheRock", "rev-parse", "HEAD:rocm-systems"],
                             _OP, cwd=_WORK_DIR, env=env)
    commit = output.strip() if rc == 0 else ""
    if not commit:
        log_error(_OP, "could not read the rocm-systems commit pinned by TheRock",
                  branch=branch, exit_code=rc)
        return False
    log_info(_OP, f"TheRock {branch} pins rocm-systems at {commit}",
             therock_branch=branch, rccl_tests_commit=commit)
    if not _run_all([
        ["git", "init", "rocm-systems"],
        ["git", "-C", "rocm-systems", "remote", "add", "origin", _ROCM_SYSTEMS_GIT_URL],
        ["git", "-C", "rocm-systems", "sparse-checkout", "set", "projects/rccl-tests"],
        ["git", "-C", "rocm-systems", "fetch", "--depth=1", "--filter=blob:none",
         "origin", commit],
        ["git", "-C", "rocm-systems", "checkout", "--detach", "FETCH_HEAD"],
    ], env, cwd=_WORK_DIR):
        return False
    shutil.rmtree(os.path.join(_WORK_DIR, "TheRock"), ignore_errors=True)   # rm -rf TheRock
    return True


def _build_rccl_tests(env: dict[str, str], distribution: str, rccl_prefix: str,
                      rocm_version: str | None) -> bool:
    os.makedirs(_RCCL_TESTS_DIR, exist_ok=True)
    cmake_args = rccl_tests_cmake_args(distribution)

    # Sparse-clone only the rccl-tests subproject of rocm-systems.
    if distribution == _UBUNTU_2604:
        log_info(_OP, "Ubuntu 26.04: rccl-tests from the rocm-systems commit that "
                      f"TheRock pins for ROCm {rocm_version}",
                 therock_branch=therock_branch(rocm_version), phase="rccl-tests")
        if not _checkout_rocm_systems_therock(env, rocm_version):
            return False
    elif not _run_all([["git", "clone", "--depth=1", "--filter=blob:none", "--sparse",
                        _ROCM_SYSTEMS_GIT_URL]], env, cwd=_WORK_DIR):
        return False
    if not _record_repo("RCCL_TESTS", os.path.join(_WORK_DIR, "rocm-systems"), env):
        return False

    script = rccl_tests_script(rccl_prefix, cmake_args)
    log_info(_OP, "Building rccl-tests from ROCm/rocm-systems (sparse clone) with HPC-X",
             rccl_prefix=rccl_prefix, dest=_RCCL_TESTS_DIR, cmake_args=cmake_args,
             phase="rccl-tests")
    log_debug(_OP, f"rccl-tests build script:\n{script}")
    if not _run_all([["bash", "-c", script]], env, cwd=_WORK_DIR):
        return False
    # glob() lists the files matching a shell-style pattern.
    binaries = sorted(os.path.basename(p)
                      for p in glob.glob(os.path.join(_RCCL_TESTS_DIR, "*_perf")))
    log_info(_OP, f"Installed {len(binaries)} rccl-tests binaries in {_RCCL_TESTS_DIR}",
             dest=_RCCL_TESTS_DIR, binaries=binaries)
    return True


def _build_perftest(env: dict[str, str], distribution: str, is_ubuntu: bool,
                    nproc: int) -> bool:
    if is_ubuntu and not install_packages(_PERFTEST_UBUNTU_PACKAGES, env, _OP):
        return False
    if distribution == _UBUNTU_2604:
        log_info(_OP, "Ubuntu 26.04: perftest also needs libibumad-dev",
                 package="libibumad-dev")
        if not install_packages(_PERFTEST_UBUNTU2604_PACKAGES, env, _OP):
            return False

    log_info(_OP, "Building perftest with ROCm support",
             url=_PERFTEST_GIT_URL, prefix=_PERFTEST_PREFIX_DIR, phase="perftest")
    if not _run_all([["git", "clone", _PERFTEST_GIT_URL]], env, cwd=_WORK_DIR):
        return False
    source_dir = os.path.join(_WORK_DIR, "perftest")
    if not _record_repo("PERFTEST", source_dir, env):
        return False
    os.makedirs(_PERFTEST_PREFIX_DIR, exist_ok=True)

    # "./autogen.sh" is found relative to cwd, as in bash after `pushd`.
    if not _run_all([
        ["./autogen.sh"],
        ["./configure", "--enable-rocm", "--with-rocm=/opt/rocm",
         f"--prefix={_PERFTEST_PREFIX_DIR}/"],
        ["make", f"-j{nproc}"],
        ["make", "install"],
    ], env, cwd=source_dir):
        return False
    shutil.rmtree(source_dir, ignore_errors=True)    # bash: rm -rf perftest
    log_info(_OP, f"perftest installed to {_PERFTEST_PREFIX_DIR}",
             prefix=_PERFTEST_PREFIX_DIR)
    return True


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def _check_source_config(config) -> str | None:
    """Return why the source-build config is unusable, or None if it's fine."""
    if not isinstance(config, dict) or not config.get("version"):
        return "could not resolve rccl version from versions.json"
    if not config.get("url"):
        return "rccl url missing from versions.json (it names the source folder)"
    if uses_git_clone(rccl_ref(config)):
        if not config.get("commit"):
            return "rccl ref/branch given without a commit in versions.json"
    elif not config.get("sha256"):
        return "rccl sha256 missing from versions.json (needed for the tarball)"
    return None


def install(env: dict[str, str]) -> int:
    """Install RCCL, rccl-tests and perftest, and set kernel settings. Returns 0 or 3.

    `env` is the build environment made by component_env() in build_config.py.
    Uses DISTRIBUTION plus the values config_for() needs to find "rccl" (or
    "rocm" on Ubuntu 26.04).
    """
    distribution = env.get("DISTRIBUTION", "")
    is_ubuntu = "ubuntu" in distribution            # bash: == *"ubuntu"*
    is_2604 = distribution == _UBUNTU_2604
    # RCCL ships via ROCm distro packages and lives in /opt/rocm.
    packaged = distribution in (_AZURE_LINUX, _UBUNTU_2604)

    config = None
    rocm_version = None
    if is_2604:
        # bash: rocm_version=$(jq -r '.version' <<< "$rocm_metadata"), for the
        # TheRock branch name.
        rocm = config_for("rocm", env)
        if not isinstance(rocm, dict) or not rocm.get("version"):
            log_error(_OP, "could not resolve rocm version from versions.json "
                           "(needed for the TheRock branch)", distro=distribution)
            return 3
        rocm_version = str(rocm["version"])
    elif not packaged:
        config = config_for("rccl", env)
        problem = _check_source_config(config)
        if problem:
            log_error(_OP, problem, distro=distribution)
            return 3

    rccl_prefix = _PACKAGED_RCCL_PREFIX if packaged else _SOURCE_RCCL_PREFIX
    cmake_args = rccl_tests_cmake_args(distribution)
    if is_2604:
        log_info(_OP, f"Using the RCCL from the ROCm {rocm_version} SDK packages on "
                      f"{distribution} (installed by install-rocm)",
                 distro=distribution, source="rocm-sdk", rocm_version=rocm_version,
                 rccl_prefix=rccl_prefix, rccl_tests_cmake_args=cmake_args)
    elif packaged:
        log_info(_OP, f"Installing RCCL on {distribution} from the ROCm dnf packages "
                      "(Azure Linux ships RCCL packaged)",
                 distro=distribution, source="dnf", rccl_prefix=rccl_prefix,
                 rccl_tests_cmake_args=cmake_args)
    else:
        log_info(_OP, f"Installing RCCL {config['version']} on {distribution} "
                      "from source (no usable RCCL package on this distro)",
                 distro=distribution, version=str(config["version"]),
                 source=str(config["url"]), ref=rccl_ref(config),
                 commit=str(config.get("commit")), rccl_prefix=rccl_prefix,
                 gpu_targets=_AMD_GPU_TARGETS, rccl_tests_cmake_args=cmake_args)

    nproc = _nproc(env)
    if nproc is None:
        return 3

    try:
        # Start from an empty work directory (see the module docstring).
        shutil.rmtree(_WORK_DIR, ignore_errors=True)
        os.makedirs(_WORK_DIR, exist_ok=True)

        # ---- 1. RCCL ----------------------------------------------------------
        if is_2604:
            version = _ubuntu2604_rccl_version()
        elif packaged:
            version = _install_packaged(env)
        else:
            version = _install_from_source(env, config, nproc)
        if version is None:
            return 3
        write_component_version("RCCL", version)
        log_info(_OP, f"Recorded RCCL={version}", component="RCCL", version=version)

        # ---- 2. Kernel settings ------------------------------------------------
        if not _configure_sysctl(env, is_ubuntu):
            return 3

        # ---- 3. rccl-tests -----------------------------------------------------
        if not _build_rccl_tests(env, distribution, rccl_prefix, rocm_version):
            return 3

        # ---- 4. perftest -------------------------------------------------------
        if not _build_perftest(env, distribution, is_ubuntu, nproc):
            return 3
    except OSError as exc:
        # mkdir/CMakeLists edit/sysctl.conf/meminfo, and the versions file.
        log_error(_OP, f"file operation failed: {exc}",
                  path=getattr(exc, "filename", None))
        return 3
    finally:
        # Our own addition: never leave the scratch area behind.
        shutil.rmtree(_WORK_DIR, ignore_errors=True)

    log_info(_OP, f"RCCL {version} installed with rccl-tests and perftest",
             version=version, distro=distribution, rccl_prefix=rccl_prefix,
             rccl_tests=_RCCL_TESTS_DIR, perftest=_PERFTEST_PREFIX_DIR)
    return 0
