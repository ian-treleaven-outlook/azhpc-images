"""install_rccl.py — Python port of components/install_rccl.sh.

What this step does
-------------------
RCCL (ROCm Communication Collectives Library) is AMD's equivalent of NVIDIA's
NCCL: the library multi-GPU / multi-node jobs use for all-reduce and friends.
This step, on AMD GPU images:

  1. Installs RCCL:
       * Azure Linux   from the distro packages (rccl, rccl-devel,
                       rccl-unittests) in the ROCm repo set up by install-rocm;
                       the version recorded is what rpm reports.
       * other distros builds RCCL from source into /opt/rccl. The source is
                       a `git clone` of a branch + commit when versions.json
                       gives a branch (needed because of
                       https://github.com/ROCm/rccl/issues/1877), otherwise
                       the release tarball. Before building, CMakeLists.txt is
                       edited so the compile step uses every CPU and the
                       memory-hungry link step uses ~1 job per 16 GB of RAM.
                       The version recorded is the one in versions.json.
  2. Sets two kernel settings recommended for RCCL: NUMA auto-balancing off and
     a much larger vm.max_map_count. They are applied now on Ubuntu (sysctl)
     and appended to /etc/sysctl.conf on every distro (applied at boot).
  3. Builds rccl-tests (the *_perf benchmarks) from the ROCm/rocm-systems
     monorepo against RCCL + HPC-X MPI, into /opt/rccl-tests.
  4. Builds the upstream linux-rdma perftest tools with ROCm support into
     /opt/rocm-perftest.

The original bash, for comparison (abridged; see components/install_rccl.sh):

    rccl_metadata=$(get_component_config "rccl")
    if [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
        dnf install -y rccl rccl-devel rccl-unittests
        write_component_version "RCCL" "$(rpm -q --queryformat '%{VERSION}-%{RELEASE}' rccl)"
    else
        rccl_branch/commit/version/url/sha256=$(jq -r '.<field>' <<< $rccl_metadata)
        TARBALL=$(basename ${rccl_url})
        rccl_folder=rccl-$(basename $TARBALL .tar.gz)
        if [[ $rccl_branch != "" && $rccl_branch != "null" ]]; then
            git clone --branch ${rccl_branch} https://github.com/ROCm/rccl.git ${rccl_folder}
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
        CXX=/opt/rocm/bin/hipcc CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake -DCMAKE_PREFIX_PATH=/opt/rocm/ -DCMAKE_INSTALL_PREFIX=/opt/rccl ..
        make -j$(nproc)
        make install
        popd
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
    RCCL_PREFIX = /opt/rocm on azurelinux3.0, else /opt/rccl
    mkdir -p /opt/rccl-tests
    git clone --depth=1 --filter=blob:none --sparse https://github.com/ROCm/rocm-systems.git
    ... sparse-checkout projects/rccl-tests, cmake (with HPC-X), make,
    cp ./*_perf /opt/rccl-tests/, rm -rf rocm-systems ...
    module unload mpi/hpcx

    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then apt install -y libpci-dev; fi
    git clone https://github.com/linux-rdma/perftest.git
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
build is the exception: it needs `module load mpi/hpcx` (and the $HPCX variable
that sets), and `module` only affects the shell it runs in, so that part runs
as one `bash -c` script with `set -e` (like install_mpifileutils.py). The
CMakeLists.txt edits, the memory maths and /etc/sysctl.conf are plain Python.

install() returns 0 for success or 3 for failure. It logs problems and returns
3; it never lets an exception escape to the build loop.

Behaviour kept from bash
------------------------
  * Commands, arguments and order are the same, and the first failing command
    stops the step (bash's `set -e`).
  * The CMakeLists.txt edits behave like `sed -i -E 's/.../.../'`: line by
    line, first match per line, a line that doesn't match is left alone, and
    no match at all is not an error (we log a warning). Checked against real
    GNU sed with representative lines.
  * The link-job maths is bash's: mem_gb = MemTotal kB / 1024 / 1024 truncated
    (awk printf "%d"), jobs = (mem_gb + 15) / 16 clamped to 1..16. If
    /proc/meminfo has no MemTotal line, awk prints nothing, bash's arithmetic
    treats it as 0 and uses 1 job; we do the same (with a warning).
  * /etc/sysctl.conf is appended to, so a rerun adds the two lines again.
  * The Azure Linux branch doesn't read versions.json, as in bash.

Deliberate differences from bash
--------------------------------
  * Work directory: bash clones/downloads/builds in its current directory (the
    distro build dir). We use _WORK_DIR, emptied first (so leftovers from a
    failed run can't break `git clone` / `mkdir build`) and removed at the end.
  * Missing config fails up front on the source-build path: version and url
    always (the url names the source folder), commit when a branch is given
    (bash would run `git checkout null`), sha256 when downloading the tarball.
  * Azure Linux: a failing or empty `rpm -q` fails the step. Bash, inside
    "$(...)", would record rpm's message ("package rccl is not installed").
  * A source folder that the clone/tarball didn't create fails with a clear
    message (bash's `mkdir ./<folder>/build` would fail too, less clearly).
  * nproc is read once with `nproc` (run_capture) for the sed edit and
    `make -j`; the bash -c scripts still call $(nproc) themselves. A
    non-numeric result fails the step.
  * `mkdir -p /opt/rccl-tests` runs just before the rccl-tests script, i.e.
    before `module load` rather than just after it.
  * Cleanup of the RCCL source/tarball, rocm-systems and perftest ignores
    deletion errors (bash's `rm -rf` practically never fails).
"""

# Lets type hints such as `str | None` work on older Python versions.
from __future__ import annotations

import glob
import os
import re
import shutil

from utils.component_config import config_for, write_component_version
from utils.download import download_and_verify
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.process import exec_program, run_capture

# The Step's `op` label in build_config.py; used on every log line.
_OP = "install-rccl"

_AZURE_LINUX = "azurelinux3.0"

# ---- Paths (module-level so tests can point them at a temporary directory) --
# Scratch area for the RCCL source, rocm-systems and perftest checkouts.
_WORK_DIR = "/tmp/azhpc-rccl"
_MEMINFO = "/proc/meminfo"
_SYSCTL_CONF = "/etc/sysctl.conf"
_RCCL_TESTS_DIR = "/opt/rccl-tests"
_PERFTEST_PREFIX_DIR = "/opt/rocm-perftest"

_RCCL_GIT_URL = "https://github.com/ROCm/rccl.git"
_ROCM_SYSTEMS_GIT_URL = "https://github.com/ROCm/rocm-systems.git"
_PERFTEST_GIT_URL = "https://github.com/linux-rdma/perftest.git"

# Where RCCL ends up: the ROCm packages install into /opt/rocm (Azure Linux);
# the source build installs into /opt/rccl (everything else).
_PACKAGED_RCCL_PREFIX = "/opt/rocm"
_SOURCE_RCCL_PREFIX = "/opt/rccl"

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


def uses_git_clone(branch) -> bool:
    """bash: [[ $rccl_branch != "" && $rccl_branch != "null" ]].

    `jq -r` prints "null" for a missing key; config_for() gives None instead.
    """
    return branch is not None and str(branch) not in ("", "null")


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


def rccl_tests_script(rccl_prefix: str) -> str:
    """The bash script that builds rccl-tests and copies *_perf to /opt/rccl-tests.

    One shell, because `module load mpi/hpcx` (which also sets $HPCX) only
    affects the shell that runs it. `set -e` stops at the first failure, as in
    bash. $PATH, $HPCX and $(nproc) are left for that shell to expand;
    rccl_prefix and the destination are filled in here.
    """
    return (
        "set -e\n"
        "source /etc/profile.d/modules.sh\n"
        "module load mpi/hpcx\n"
        f"git clone --depth=1 --filter=blob:none --sparse {_ROCM_SYSTEMS_GIT_URL}\n"
        "pushd ./rocm-systems\n"
        "git sparse-checkout set projects/rccl-tests\n"
        "pushd projects/rccl-tests\n"
        "mkdir build\n"
        "pushd build\n"
        "PATH=/opt/rocm/bin:$PATH cmake \\\n"
        "    -DCMAKE_BUILD_TYPE=Release \\\n"
        f'    -DCMAKE_PREFIX_PATH="{rccl_prefix};/opt/rocm;$HPCX" \\\n'
        "    -DROCM_PATH=/opt/rocm \\\n"
        "    -DUSE_MPI=ON \\\n"
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
def _install_packaged(env: dict[str, str]) -> str | None:
    if not _run_all([["dnf", "install", "-y", "rccl", "rccl-devel", "rccl-unittests"]], env):
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
    branch = config.get("branch")
    if uses_git_clone(branch):
        commit = str(config["commit"])
        log_info(_OP, "RCCL source: git clone of a pinned branch/commit "
                      "(works around ROCm/rccl#1877)",
                 source="git", branch=str(branch), commit=commit, folder=folder)
        return (_run_all([["git", "clone", "--branch", str(branch), _RCCL_GIT_URL, folder]],
                         env, cwd=_WORK_DIR)
                and _run_all([["git", "checkout", commit]], env,
                             cwd=os.path.join(_WORK_DIR, folder)))

    url, sha256 = config["url"], config["sha256"]
    log_info(_OP, "RCCL source: release tarball (no branch in versions.json)",
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
    log_info(_OP, f"Building RCCL {version} into {_SOURCE_RCCL_PREFIX}",
             version=version, prefix=_SOURCE_RCCL_PREFIX, phase="cmake")
    if not _run_all([["cmake", "-DCMAKE_PREFIX_PATH=/opt/rocm/",
                      f"-DCMAKE_INSTALL_PREFIX={_SOURCE_RCCL_PREFIX}", ".."]],
                    cmake_env, cwd=build_dir):
        return None
    log_info(_OP, "Compiling and installing RCCL", phase="make", jobs=nproc)
    if not _run_all([["make", f"-j{nproc}"], ["make", "install"]], env, cwd=build_dir):
        return None

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
    with open(_SYSCTL_CONF, "a", encoding="utf-8") as handle:
        handle.write("".join(f"{setting}\n" for setting in _SYSCTL_SETTINGS))
    log_info(_OP, f"Appended to {_SYSCTL_CONF}", path=_SYSCTL_CONF,
             action="appended", settings=list(_SYSCTL_SETTINGS))
    return True


def _build_rccl_tests(env: dict[str, str], rccl_prefix: str) -> bool:
    os.makedirs(_RCCL_TESTS_DIR, exist_ok=True)
    script = rccl_tests_script(rccl_prefix)
    log_info(_OP, "Building rccl-tests from ROCm/rocm-systems (sparse clone) with HPC-X",
             rccl_prefix=rccl_prefix, dest=_RCCL_TESTS_DIR, phase="rccl-tests")
    log_debug(_OP, f"rccl-tests build script:\n{script}")
    if not _run_all([["bash", "-c", script]], env, cwd=_WORK_DIR):
        return False
    # glob() lists the files matching a shell-style pattern.
    binaries = sorted(os.path.basename(p)
                      for p in glob.glob(os.path.join(_RCCL_TESTS_DIR, "*_perf")))
    log_info(_OP, f"Installed {len(binaries)} rccl-tests binaries in {_RCCL_TESTS_DIR}",
             dest=_RCCL_TESTS_DIR, binaries=binaries)
    return True


def _build_perftest(env: dict[str, str], is_ubuntu: bool, nproc: int) -> bool:
    if is_ubuntu and not _run_all([["apt", "install", "-y", "libpci-dev"]], env):
        return False

    log_info(_OP, "Building perftest with ROCm support",
             url=_PERFTEST_GIT_URL, prefix=_PERFTEST_PREFIX_DIR, phase="perftest")
    if not _run_all([["git", "clone", _PERFTEST_GIT_URL]], env, cwd=_WORK_DIR):
        return False
    os.makedirs(_PERFTEST_PREFIX_DIR, exist_ok=True)

    source_dir = os.path.join(_WORK_DIR, "perftest")
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
    if uses_git_clone(config.get("branch")):
        if not config.get("commit"):
            return "rccl branch given without a commit in versions.json"
    elif not config.get("sha256"):
        return "rccl sha256 missing from versions.json (needed for the tarball)"
    return None


def install(env: dict[str, str]) -> int:
    """Install RCCL, rccl-tests and perftest, and set kernel settings. Returns 0 or 3.

    `env` is the build environment made by component_env() in build_config.py.
    Uses DISTRIBUTION plus the values config_for() needs to find "rccl".
    """
    distribution = env.get("DISTRIBUTION", "")
    is_ubuntu = "ubuntu" in distribution            # bash: == *"ubuntu"*
    packaged = distribution == _AZURE_LINUX

    config = None
    if not packaged:
        config = config_for("rccl", env)
        problem = _check_source_config(config)
        if problem:
            log_error(_OP, problem, distro=distribution)
            return 3

    rccl_prefix = _PACKAGED_RCCL_PREFIX if packaged else _SOURCE_RCCL_PREFIX
    if packaged:
        log_info(_OP, f"Installing RCCL on {distribution} from the ROCm dnf packages "
                      "(Azure Linux ships RCCL packaged)",
                 distro=distribution, source="dnf", rccl_prefix=rccl_prefix)
    else:
        log_info(_OP, f"Installing RCCL {config['version']} on {distribution} "
                      "from source (no usable RCCL package on this distro)",
                 distro=distribution, version=str(config["version"]),
                 source=str(config["url"]), branch=str(config.get("branch")),
                 commit=str(config.get("commit")), rccl_prefix=rccl_prefix)

    nproc = _nproc(env)
    if nproc is None:
        return 3

    try:
        # Start from an empty work directory (see the module docstring).
        shutil.rmtree(_WORK_DIR, ignore_errors=True)
        os.makedirs(_WORK_DIR, exist_ok=True)

        # ---- 1. RCCL ----------------------------------------------------------
        if packaged:
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
        if not _build_rccl_tests(env, rccl_prefix):
            return 3

        # ---- 4. perftest -------------------------------------------------------
        if not _build_perftest(env, is_ubuntu, nproc):
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
