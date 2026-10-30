"""install_nccl.py — Python port of components/install_nccl.sh.

What this step does
-------------------
NCCL (NVIDIA Collective Communications Library) is the library multi-GPU /
multi-node jobs use for all-reduce and friends. On NVIDIA images this step:

  1. Installs the build dependencies, once (see install_deps()).
  2. Installs NCCL and stops later package updates from replacing it
     (apt-mark hold on Ubuntu, dnf versionlock elsewhere):
       * Ubuntu 26.04: the libnccl2 / libnccl-dev packages from apt, and the
         installed version is read back with dpkg-query.
       * everything else: downloads and checksums the NCCL release tag
         tarball from GitHub, builds it (`make src.build`), packages it as
         .deb (Ubuntu) or .rpm (everything else) and installs those packages.
  3. On the standard InfiniBand network only (sku_network_mode() ==
     "standard_ib"), builds the NCCL RDMA SHARP plugin from the configured
     branch and commit against HPC-X into /usr/local/nccl-rdma-sharp-plugins,
     records the source it was built from, and adds its lib dir to the
     dynamic linker path (/etc/ld.so.conf.d/nccl-rdma-sharp-plugins.conf +
     ldconfig).
  4. Clones nccl-tests (the *_perf benchmarks), records the commit/branch it
     got, builds it with HPC-X MPI and moves the checkout to /opt/nccl-tests.
  5. Records NCCL (and, when built, NCCL-RDMA_SHARP_PLUGIN) in the component
     versions file. On Ubuntu 22.04 it then purges patchelf (build-time only;
     Jammy's patchelf has a CVE, USN-6036-1, whose fix needs Ubuntu Pro).

The original bash, for comparison (abridged; see components/install_nccl.sh):

    NCCL_VERSION=$(jq -r '.version' <<< $nccl_metadata)
    NCCL_SHA256=$(jq -r '.sha256 // empty' <<< $nccl_metadata)
    NCCL_RDMA_SHARP_REF=$(jq -r '.rdmasharpplugins.ref // .rdmasharpplugins.branch // empty')
    NCCL_RDMA_SHARP_COMMIT=$(jq -r '.rdmasharpplugins.commit' <<< $nccl_metadata)
    CUDA_DRIVER_VERSION=$(jq -r '.driver.version' <<< $cuda_metadata)
    MAKE_JOBS=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))
    TARBALL="v${NCCL_VERSION}.tar.gz"
    if ubuntu26.04:     apt install -y build-essential patchelf zlib1g-dev
    elif ubuntu:        apt install -y build-essential devscripts debhelper fakeroot patchelf
                        apt install -y zlib1g-dev # libibverbs-dev
    elif azurelinux3.0: dnf install -y rpm-build rpmdevtools autoconf automake git libtool patchelf
    else:               dnf install -y rpm-build rpmdevtools patchelf
    pushd /tmp
    if ubuntu26.04:
        apt-get install -y libnccl2 libnccl-dev
        apt-mark hold libnccl2; apt-mark hold libnccl-dev
        NCCL_VERSION=$(dpkg-query -W -f='${Version}' libnccl2 | sed 's/+cuda.*//')
    else:
        download_and_verify https://github.com/NVIDIA/nccl/archive/refs/tags/${TARBALL} ${NCCL_SHA256}
        tar -xvf ${TARBALL}
        pushd nccl-${NCCL_VERSION}
        make -j ${MAKE_JOBS} src.build
        if ubuntu:         make pkg.debian.build; cd build/pkg/deb/
                           apt install -y ./libnccl2_${V}+cuda${CUDA}_${ARCHITECTURE_DISTRO}.deb; apt-mark hold libnccl2
                           apt install -y ./libnccl-dev_${V}+cuda${CUDA}_${ARCHITECTURE_DISTRO}.deb; apt-mark hold libnccl-dev
        elif azurelinux3.0: make pkg.redhat.build
                           dnf install -y ./build/pkg/rpm/<arch>/libnccl{,-devel,-static}-${V}+cuda*.<arch>.rpm
                           dnf_pin_packages "libnccl*"
        else:              make pkg.redhat.build
                           rpm -i ./build/pkg/rpm/x86_64/libnccl{,-devel,-static}-${V}+cuda${CUDA}.x86_64.rpm
                           dnf_pin_packages "libnccl*"
        popd
    if [[ "$(sku_network_mode)" == "standard_ib" ]]; then
        module load mpi/hpcx; check HPCX_SHARP_DIR / HPCX_UCX_DIR (else exit 1)
        mkdir -p /usr/local/nccl-rdma-sharp-plugins
        git clone --branch ${REF} .../nccl-rdma-sharp-plugins.git; git checkout ${COMMIT}
        write_repo_source_info "NCCL-RDMA_SHARP_PLUGIN" . --commit "${COMMIT}" --ref "${REF}"
        [azurelinux3.0: libtoolize --verbose]; ./autogen.sh; ./configure ...; make; make install
        echo /usr/local/nccl-rdma-sharp-plugins/lib > /etc/ld.so.conf.d/nccl-rdma-sharp-plugins.conf
        ldconfig
        write_component_version "NCCL-RDMA_SHARP_PLUGIN" ${COMMIT}
        module unload mpi/hpcx
    fi
    module load mpi/hpcx
    git clone https://github.com/NVIDIA/nccl-tests.git
    pushd nccl-tests
    write_repo_source_info "NCCL_TESTS" .
    make -j ${MAKE_JOBS} MPI=1 MPI_HOME=${HPCX_MPI_DIR} CUDA_HOME=/usr/local/cuda
    popd; mv nccl-tests /opt/.; module unload mpi/hpcx
    popd
    write_component_version "NCCL" ${NCCL_VERSION}
    [ubuntu22.04: apt-get purge -y patchelf]
    rm -rf /tmp/${TARBALL} /tmp/nccl-${NCCL_VERSION}

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-nccl" Step (NVIDIA images only). Deleting `action=` from that Step
makes the build run the bash script again, which is why the .sh file is kept.

Single commands run with exec_program. The SHARP plugin and nccl-tests builds
need `module load mpi/hpcx` (and the HPCX_* variables it sets), and `module`
only affects the shell it runs in, so those parts run as `bash -c` scripts
with `set -e` (like install_rccl.py). Bash records the repo sources
(write_repo_source_info) in the middle of those parts, so the scripts are
split there and utils/repo_source.py records them from Python in the same
place: the SHARP part is two scripts (clone + checkout, then build), each
loading mpi/hpcx; nccl-tests is cloned with exec_program, recorded, then
built by one script. Versions are recorded from Python after the matching
script succeeded.

install() returns 0 for success or 3 for failure. It logs problems and returns
3; it never lets an exception escape to the build loop.

Dependencies: exact bash commands, installed once
-------------------------------------------------
install_deps() runs the exact bash commands with exec_program, and install()
calls it exactly once. We chose that over PackageInstaller because
PackageInstaller installs one package at a time and carries on after a
failure, whereas bash runs one `apt`/`dnf` command per line under `set -e`
and stops at the first failure. Same commands, same order, same fail-fast
behaviour.

Behaviour kept from bash
------------------------
  * Commands, arguments and order are the same, and the first failing command
    stops the step (bash's `set -e`), including inside the `bash -c`
    scripts (the HPC-X checks `exit 1` with bash's messages).
  * `make -j MAKE_JOBS`: nproc - 1, but at least 1 (upstream #653), for both
    the NCCL and the nccl-tests build.
  * The Azure Linux rpm paths are shell globs, expanded like bash does
    (sorted; a pattern that matches nothing is passed on literally, and dnf
    then fails on it). Ubuntu .deb and RHEL-family .rpm names are exact.
  * NCCL-RDMA_SHARP_PLUGIN records the configured commit; its _REPO_COMMIT /
    _REPO_REF entries the configured commit and ref. NCCL records the
    configured version (Ubuntu 26.04: the installed one, from dpkg-query).
  * /etc/ld.so.conf.d/nccl-rdma-sharp-plugins.conf is written by the script
    with the same heredoc, replacing any earlier copy.

Deliberate differences from bash
--------------------------------
  * Work directory: bash downloads and builds in /tmp itself. We use
    _WORK_DIR (/tmp/azhpc-nccl), emptied first so leftovers from an earlier,
    interrupted run can't break the download/tar/git clone, and removed at
    the end. That also removes the nccl-rdma-sharp-plugins clone, which bash
    leaves behind in /tmp.
  * Missing config fails up front (log + return 3) instead of building with
    "null" or an empty value. When NCCL is built from source (not Ubuntu
    26.04): nccl version and sha256; the CUDA driver version on Ubuntu and
    the RHEL family (it is part of the package file names; Azure Linux globs
    over it); ARCHITECTURE_DISTRO on Ubuntu (part of the .deb names). When
    the SHARP plugin is built: rdmasharpplugins.commit and the ref (.ref,
    else .branch).
  * nproc is read with `nproc` (run_capture) up front; a failing or
    non-numeric result fails the step (bash would compute a bogus job count).
  * A tarball that doesn't produce nccl-<version>/ fails with a clear message
    (bash's `pushd` would fail too, less clearly).
  * Ubuntu 26.04: an empty dpkg-query result fails the step (bash, without
    pipefail, would record NCCL as "").
  * The SHARP ref and commit are shell-quoted in the script (identical for
    normal refs).
  * The download uses utils/download.py (urllib, 3 tries) rather than wget.

Bash issues kept and flagged (comment + log_warn when they apply)
-----------------------------------------------------------------
  * The download and git clones write into bash's working dir (/tmp), and the
    SHARP clone is never removed, so a bash rerun trips over leftovers (git
    clone refuses an existing dir; wget saves a .1 copy). We warn when such
    bash leftovers are present in /tmp (we leave them alone).
  * The RHEL-family branch hard-codes x86_64 rpm paths, so it can't work on
    aarch64.
  * `mv nccl-tests /opt/.` fails if /opt/nccl-tests already exists and isn't
    empty (and silently replaces it if empty).
  * A rdmasharpplugins.commit that is a branch/tag rather than a hash leaves
    the plugin build unpinned (warned; versions.json uses hashes today).
"""

# Lets type hints such as `str | None` work on older Python versions.
from __future__ import annotations

import glob
import os
import re
import shlex
import shutil

from utils.component_config import config_for, write_component_version
from utils.download import download_and_verify
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.package_pinning import dnf_pin_packages
from utils.process import exec_program, run_capture
from utils.repo_source import write_repo_source_info
from utils.sku import NETWORK_MODE_STANDARD_IB, sku_network_mode

# The Step's `op` label in build_config.py; used on every log line.
_OP = "install-nccl"

_AZURE_LINUX = "azurelinux3.0"
# The one distro that installs NCCL from apt instead of building it.
_APT_NCCL_DISTRIBUTION = "ubuntu26.04"
# The one distro that purges patchelf afterwards.
_PURGE_PATCHELF_DISTRIBUTION = "ubuntu22.04"

# ---- Paths (module-level so tests can point them at a temporary directory) --
# Our scratch area for the tarball, the NCCL source and the git checkouts.
_WORK_DIR = "/tmp/azhpc-nccl"
# Where bash itself works (pushd /tmp); only checked for bash leftovers.
_BASH_TMP_DIR = "/tmp"
# nccl-tests is moved here (bash: mv nccl-tests /opt/.).
_OPT_DIR = "/opt"
_SHARP_PREFIX = "/usr/local/nccl-rdma-sharp-plugins"
_LD_SO_CONF = "/etc/ld.so.conf.d/nccl-rdma-sharp-plugins.conf"

_NCCL_TAG_URL = "https://github.com/NVIDIA/nccl/archive/refs/tags/"
_SHARP_GIT_URL = "https://github.com/Mellanox/nccl-rdma-sharp-plugins.git"
_NCCL_TESTS_GIT_URL = "https://github.com/NVIDIA/nccl-tests.git"

# The names write_repo_source_info records under (bash's first argument).
_SHARP_COMPONENT = "NCCL-RDMA_SHARP_PLUGIN"
_NCCL_TESTS_COMPONENT = "NCCL_TESTS"

# Build dependencies: the exact bash commands for each distro.
_UBUNTU2604_DEPS = [
    ["apt", "install", "-y", "build-essential", "patchelf", "zlib1g-dev"],
]
_UBUNTU_DEPS = [
    ["apt", "install", "-y", "build-essential", "devscripts", "debhelper",
     "fakeroot", "patchelf"],
    # libibverbs-dev is deliberately NOT installed (it's commented out in
    # bash): it conflicts on bare-metal 1P builds, and on VMs it's already
    # installed by install_utils.sh or install_doca.sh.
    ["apt", "install", "-y", "zlib1g-dev"],
]
_AZURE_LINUX_DEPS = [
    ["dnf", "install", "-y", "rpm-build", "rpmdevtools", "autoconf", "automake",
     "git", "libtool", "patchelf"],
]
_RHEL_DEPS = [   # RHEL family: AlmaLinux, Rocky Linux, RHEL, ...
    ["dnf", "install", "-y", "rpm-build", "rpmdevtools", "patchelf"],
]

# Ubuntu 26.04: install and hold the packaged NCCL.
_APT_NCCL_COMMANDS = [
    ["apt-get", "install", "-y", "libnccl2", "libnccl-dev"],
    ["apt-mark", "hold", "libnccl2"],
    ["apt-mark", "hold", "libnccl-dev"],
]
# bash: dpkg-query -W -f='${Version}' libnccl2 (the quotes stop bash expanding it).
_DPKG_QUERY_NCCL = ["dpkg-query", "-W", "-f=${Version}", "libnccl2"]

# A full or abbreviated git commit hash (as opposed to a branch or tag name).
_COMMIT_HASH_RE = re.compile(r"[0-9a-f]{7,40}")


# --------------------------------------------------------------------------- #
# Pure helpers: no side effects, so tests call them directly.
# --------------------------------------------------------------------------- #
def distro_family(distribution: str) -> str:
    """Which bash branch a distro takes: "ubuntu", "azurelinux" or "rhel".

    bash: [[ $DISTRIBUTION == *"ubuntu"* ]] / == "azurelinux3.0" / else.
    """
    if "ubuntu" in distribution:
        return "ubuntu"
    if distribution == _AZURE_LINUX:
        return "azurelinux"
    return "rhel"


def nccl_from_apt(distribution: str) -> bool:
    """True where bash installs NCCL from apt instead of building it (Ubuntu 26.04)."""
    return distribution == _APT_NCCL_DISTRIBUTION


def deps_commands(distribution: str) -> list[list[str]]:
    """The dependency install commands bash runs on `distribution`."""
    if nccl_from_apt(distribution):
        return _UBUNTU2604_DEPS
    return {"ubuntu": _UBUNTU_DEPS, "azurelinux": _AZURE_LINUX_DEPS,
            "rhel": _RHEL_DEPS}[distro_family(distribution)]


def tarball_name(version: str) -> str:
    """bash: TARBALL="v${NCCL_VERSION}.tar.gz"."""
    return f"v{version}.tar.gz"


def download_url(version: str) -> str:
    """bash: NCCL_DOWNLOAD_URL=https://github.com/NVIDIA/nccl/archive/refs/tags/${TARBALL}."""
    return _NCCL_TAG_URL + tarball_name(version)


def make_jobs(nproc: int) -> int:
    """bash: MAKE_JOBS=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 )).

    Leaves one core free for sshd/agent, but never -j 0 on a 1-CPU VM.
    """
    return nproc - 1 if nproc > 1 else 1


def apt_nccl_version(dpkg_query_output: str) -> str:
    """The NCCL version from `dpkg-query -W -f='${Version}' libnccl2`.

    Mirrors  $(... | sed 's/+cuda.*//') : on each line, drop everything from
    the first "+cuda"; $(...) then drops trailing newlines.

        "2.28.9-1+cuda13.0"  ->  "2.28.9-1"
    """
    lines = [line.split("+cuda", 1)[0] for line in dpkg_query_output.split("\n")]
    return "\n".join(lines).rstrip("\n")


def deb_packages(version: str, cuda: str, arch_distro: str) -> list[tuple[str, str]]:
    """(file, package) pairs bash apt-installs then apt-mark holds, in order."""
    suffix = f"_{version}+cuda{cuda}_{arch_distro}.deb"
    return [(f"libnccl2{suffix}", "libnccl2"), (f"libnccl-dev{suffix}", "libnccl-dev")]


def rpm_arch(architecture: str) -> str:
    """Azure Linux rpm dir/suffix: aarch64 if ARCHITECTURE is aarch64, else x86_64."""
    return "aarch64" if architecture == "aarch64" else "x86_64"


def azure_linux_rpm_patterns(version: str, architecture: str) -> list[str]:
    """The three globs bash passes to `dnf install -y` on Azure Linux."""
    arch = rpm_arch(architecture)
    return [f"./build/pkg/rpm/{arch}/{name}-{version}+cuda*.{arch}.rpm"
            for name in ("libnccl", "libnccl-devel", "libnccl-static")]


def rhel_rpm_paths(version: str, cuda: str) -> list[str]:
    """The three exact paths bash passes to `rpm -i` (always x86_64)."""
    return [f"./build/pkg/rpm/x86_64/{name}-{version}+cuda{cuda}.x86_64.rpm"
            for name in ("libnccl", "libnccl-devel", "libnccl-static")]


def expand_glob(pattern: str, cwd: str) -> list[str]:
    """Expand `pattern` relative to `cwd` the way bash would.

    Sorted matches; with no match, bash (without nullglob) passes the pattern
    on unchanged.
    """
    matches = sorted(glob.glob(pattern, root_dir=cwd))
    return matches or [pattern]


def is_commit_hash(ref: str) -> bool:
    """True if `ref` looks like a git commit hash rather than a branch/tag."""
    return _COMMIT_HASH_RE.fullmatch(ref) is not None


def bash_leftovers(tmp_dir: str, version: str) -> list[str]:
    """Paths an earlier bash run of install_nccl.sh may have left in `tmp_dir`.

    bash downloads and clones straight into /tmp and never removes the SHARP
    clone (nor anything at all if it fails half way).
    """
    names = (tarball_name(version), f"nccl-{version}", "nccl-rdma-sharp-plugins",
             "nccl-tests")
    return [os.path.join(tmp_dir, name) for name in names
            if os.path.lexists(os.path.join(tmp_dir, name))]


def _hpcx_preamble() -> str:
    """The start of every HPC-X script: `set -e`, then load the module."""
    return ("set -e\n"
            "source /etc/profile.d/modules.sh\n"
            "module load mpi/hpcx\n")


def sharp_clone_script(ref: str, commit: str) -> str:
    """Part 1 of the SHARP plugin: check HPC-X, then clone and check out.

    One shell, because the checks read HPCX_SHARP_DIR / HPCX_UCX_DIR, which
    `module load mpi/hpcx` sets only in the shell that runs it. `set -e`
    stops at the first failure, as in bash. The checks and messages are
    bash's own. Bash then runs write_repo_source_info, which install() does
    from Python between the two parts.
    """
    return (
        _hpcx_preamble()
        + "\n"
        'if [[ -z "${HPCX_SHARP_DIR:-}" || ! -d "${HPCX_SHARP_DIR}" ]]; then\n'
        '    echo "HPC-X module did not provide a valid HPCX_SHARP_DIR; '
        'cannot build NCCL RDMA SHARP plugin."\n'
        "    exit 1\n"
        "fi\n"
        'if [[ -z "${HPCX_UCX_DIR:-}" || ! -d "${HPCX_UCX_DIR}" ]]; then\n'
        '    echo "HPC-X module did not provide a valid HPCX_UCX_DIR; '
        'cannot build NCCL RDMA SHARP plugin."\n'
        "    exit 1\n"
        "fi\n"
        f"mkdir -p {_SHARP_PREFIX}\n"
        f"git clone --branch {shlex.quote(ref)} {_SHARP_GIT_URL}\n"
        "pushd nccl-rdma-sharp-plugins\n"
        f"git checkout {shlex.quote(commit)}\n"
    )


def sharp_build_script(distribution: str) -> str:
    """Part 2 of the SHARP plugin: build and install it with HPC-X loaded.

    A fresh shell, so it loads mpi/hpcx again (./configure finds SHARP/UCX
    through it) and enters the clone again.
    """
    libtoolize = "libtoolize --verbose\n" if distribution == _AZURE_LINUX else ""
    return (
        _hpcx_preamble()
        + "pushd nccl-rdma-sharp-plugins\n"
        f"{libtoolize}"
        "./autogen.sh\n"
        f"./configure --prefix={_SHARP_PREFIX} --with-cuda=/usr/local/cuda\n"
        "make\n"
        "make install\n"
        f"cat > {_LD_SO_CONF} <<EOF\n"
        f"{_SHARP_PREFIX}/lib\n"
        "EOF\n"
        "ldconfig\n"
        "popd\n"
        "module unload mpi/hpcx\n"
    )


def nccl_tests_script(jobs: int) -> str:
    """The bash script that builds nccl-tests with HPC-X MPI and moves it to /opt.

    The clone happens before (exec_program), and its source is recorded in
    between, as in bash. ${HPCX_MPI_DIR} is set by the module and expanded
    by that shell. (bash doesn't check it's set; an empty MPI_HOME makes the
    build fail on a missing mpi.h.)
    """
    return (
        _hpcx_preamble()
        + "pushd nccl-tests\n"
        f"make -j {jobs} MPI=1 MPI_HOME=${{HPCX_MPI_DIR}} CUDA_HOME=/usr/local/cuda\n"
        "popd\n"
        f"mv nccl-tests {_OPT_DIR}/.\n"
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
            log_error(_OP, f"NCCL install stopped: `{shown}` failed", exit_code=rc)
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


def _record_source(component: str, repo_dir: str, env: dict[str, str], *,
                   commit: str | None = None, ref: str | None = None) -> bool:
    """write_repo_source_info, with a log_error on failure (git failed)."""
    if not write_repo_source_info(component, repo_dir, _OP, commit=commit, ref=ref,
                                  env=env):
        log_error(_OP, f"NCCL install stopped: could not read the {component} "
                       f"source from {repo_dir}", component=component, path=repo_dir)
        return False
    return True


# --------------------------------------------------------------------------- #
# The phases. Each returns True on success, False after logging a failure.
# --------------------------------------------------------------------------- #
def install_deps(env: dict[str, str]) -> int:
    """Install NCCL's build dependencies with bash's exact commands.

    Returns 0 on success, 3 on failure (stopping at the first failing command).
    install() calls this exactly once.
    """
    distribution = env.get("DISTRIBUTION", "")
    commands = deps_commands(distribution)
    branch = distribution if nccl_from_apt(distribution) else distro_family(distribution)
    log_info(_OP, f"Installing NCCL build dependencies ({branch} branch of "
                  "install_nccl.sh)",
             distro=distribution, branch=branch,
             packages=[p for command in commands for p in command[3:]])
    return 0 if _run_all(commands, env) else 3


def _install_apt_nccl(env: dict[str, str]) -> str | None:
    """Ubuntu 26.04: install + hold the packaged NCCL; return its version."""
    log_info(_OP, f"{_APT_NCCL_DISTRIBUTION}: installing NCCL from apt (no source "
                  "build)", packages=["libnccl2", "libnccl-dev"], source="apt")
    if not _run_all(_APT_NCCL_COMMANDS, env):
        return None
    log_info(_OP, "Held libnccl2 and libnccl-dev with apt-mark",
             packages=["libnccl2", "libnccl-dev"], held=True)
    rc, output = run_capture(_DPKG_QUERY_NCCL, _OP, env=env)
    version = apt_nccl_version(output) if rc == 0 else ""
    if not version:
        log_error(_OP, "could not read the installed libnccl2 version with dpkg-query",
                  exit_code=rc, output=output.strip())
        return None
    log_info(_OP, f"Installed NCCL version is {version}", version=version,
             version_source="dpkg-query", dpkg_version=output.strip())
    return version


def _build_nccl(env: dict[str, str], version: str, sha256: str, source_dir: str,
                jobs: int) -> bool:
    """download_and_verify + tar + make src.build. Leaves the source tree in source_dir."""
    tarball = tarball_name(version)
    url = download_url(version)
    log_info(_OP, f"Downloading and extracting NCCL {version}", url=url, sha256=sha256,
             tarball=tarball, work_dir=_WORK_DIR, phase="download")
    try:
        download_and_verify(url, sha256, dest_dir=_WORK_DIR)
    except Exception as exc:          # network errors, checksum mismatch, disk
        log_error(_OP, f"NCCL install stopped: download/verify of {url} failed: {exc}",
                  url=url)
        return False
    if not _run_all([["tar", "-xvf", tarball]], env, cwd=_WORK_DIR):
        return False
    if not os.path.isdir(source_dir):
        log_error(_OP, f"NCCL source folder {source_dir} was not created by "
                       f"extracting {tarball}", tarball=tarball)
        return False

    log_info(_OP, f"Building NCCL {version} (make -j {jobs} src.build)",
             version=version, jobs=jobs, phase="src.build")
    return _run_all([["make", "-j", str(jobs), "src.build"]], env, cwd=source_dir)


def _package_ubuntu(env, source_dir, version, cuda, arch_distro) -> bool:
    log_info(_OP, "Packaging NCCL as .deb (make pkg.debian.build)", phase="pkg.debian.build")
    if not _run_all([["make", "pkg.debian.build"]], env, cwd=source_dir):
        return False
    deb_dir = os.path.join(source_dir, "build", "pkg", "deb")
    for deb, package in deb_packages(version, cuda, arch_distro):
        if not _run_all([["apt", "install", "-y", f"./{deb}"],
                         ["apt-mark", "hold", package]], env, cwd=deb_dir):
            return False
        log_info(_OP, f"Installed {deb} and held {package} with apt-mark",
                 package=package, file=deb, held=True)
    return True


def _package_azure_linux(env, source_dir, version, architecture) -> bool:
    log_info(_OP, "Packaging NCCL as .rpm (make pkg.redhat.build)", phase="pkg.redhat.build")
    if not _run_all([["make", "pkg.redhat.build"]], env, cwd=source_dir):
        return False
    for pattern in azure_linux_rpm_patterns(version, architecture):
        files = expand_glob(pattern, source_dir)
        if files == [pattern]:
            log_warn(_OP, f"no rpm matched {pattern}; passing it to dnf literally, "
                          "as bash does", pattern=pattern)
        if not _run_all([["dnf", "install", "-y", *files]], env, cwd=source_dir):
            return False
        log_info(_OP, f"Installed {' '.join(files)} with dnf", files=files)
    return _pin_libnccl(env)


def _package_rhel(env, source_dir, version, cuda, architecture) -> bool:
    if architecture != "x86_64":
        # bash issue: this branch hard-codes the x86_64 rpm paths.
        log_warn(_OP, f"install_nccl.sh hard-codes x86_64 rpm paths on {architecture}; "
                      "rpm -i will not find the packages, as in bash",
                 architecture=architecture)
    log_info(_OP, "Packaging NCCL as .rpm (make pkg.redhat.build)", phase="pkg.redhat.build")
    if not _run_all([["make", "pkg.redhat.build"]], env, cwd=source_dir):
        return False
    paths = rhel_rpm_paths(version, cuda)
    if not _run_all([["rpm", "-i", path] for path in paths], env, cwd=source_dir):
        return False
    log_info(_OP, f"Installed {len(paths)} NCCL rpms with rpm -i", files=paths)
    return _pin_libnccl(env)


def _pin_libnccl(env) -> bool:
    # dnf_pin_packages logs the pin itself (or that dnf is absent).
    if not dnf_pin_packages(["libnccl*"], _OP, env):
        log_error(_OP, "NCCL install stopped: could not pin libnccl* with dnf")
        return False
    return True


def _build_sharp_plugin(env, ref: str, commit: str, distribution: str) -> bool:
    if not is_commit_hash(commit):
        # bash issue: a branch/tag "commit" means what gets built moves over time.
        log_warn(_OP, f"rdmasharpplugins.commit {commit!r} is a branch or tag, not a "
                      "commit hash; the plugin build is not pinned (kept from bash)",
                 commit=commit)
    conf_existed = os.path.exists(_LD_SO_CONF)
    clone = sharp_clone_script(ref, commit)
    build = sharp_build_script(distribution)
    log_info(_OP, f"Building the NCCL RDMA SHARP plugin from {ref} at {commit} with "
                  f"HPC-X into {_SHARP_PREFIX}", ref=ref, commit=commit,
             prefix=_SHARP_PREFIX, url=_SHARP_GIT_URL,
             libtoolize=distribution == _AZURE_LINUX, phase="sharp-plugin")
    log_debug(_OP, f"SHARP plugin clone script:\n{clone}")
    if not _run_all([["bash", "-c", clone]], env, cwd=_WORK_DIR):
        return False
    if not _record_source(_SHARP_COMPONENT, os.path.join(_WORK_DIR, "nccl-rdma-sharp-plugins"),
                          env, commit=commit, ref=ref):
        return False
    log_debug(_OP, f"SHARP plugin build script:\n{build}")
    if not _run_all([["bash", "-c", build]], env, cwd=_WORK_DIR):
        return False
    log_info(_OP, f"{'Replaced' if conf_existed else 'Created'} {_LD_SO_CONF} "
                  f"({_SHARP_PREFIX}/lib) and ran ldconfig", path=_LD_SO_CONF,
             action="replaced" if conf_existed else "created")
    write_component_version(_SHARP_COMPONENT, commit)
    log_info(_OP, f"Recorded {_SHARP_COMPONENT}={commit}",
             component=_SHARP_COMPONENT, version=commit)
    return True


def _build_nccl_tests(env, jobs: int) -> bool:
    dest = os.path.join(_OPT_DIR, "nccl-tests")
    if os.path.lexists(dest):
        # bash issue: `mv nccl-tests /opt/.` fails if /opt/nccl-tests is a
        # non-empty dir (and replaces it if empty). Kept as is.
        log_warn(_OP, f"{dest} already exists; bash's `mv nccl-tests {_OPT_DIR}/.` "
                      "fails if it is not empty", path=dest)
    script = nccl_tests_script(jobs)
    log_info(_OP, f"Building nccl-tests with HPC-X MPI (make -j {jobs})",
             url=_NCCL_TESTS_GIT_URL, dest=dest, jobs=jobs, phase="nccl-tests")
    if not _run_all([["git", "clone", _NCCL_TESTS_GIT_URL]], env, cwd=_WORK_DIR):
        return False
    if not _record_source(_NCCL_TESTS_COMPONENT, os.path.join(_WORK_DIR, "nccl-tests"), env):
        return False
    log_debug(_OP, f"nccl-tests build script:\n{script}")
    if not _run_all([["bash", "-c", script]], env, cwd=_WORK_DIR):
        return False
    log_info(_OP, f"nccl-tests built and moved to {dest}", dest=dest)
    return True


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def _str_or_none(value) -> str | None:
    """A config value as text, or None when missing/empty (jq's "null")."""
    if value is None or str(value) in ("", "null"):
        return None
    return str(value)


def _jq_alternative(*values):
    """jq's `a // b`: the first value that is neither null (None) nor false."""
    for value in values:
        if value is not None and value is not False:
            return value
    return None


def _prepare_work_dir(version: str | None) -> None:
    leftovers = bash_leftovers(_BASH_TMP_DIR, version) if version else []
    if leftovers:
        # bash issue: install_nccl.sh downloads/clones into /tmp and never
        # removes the SHARP clone, so a bash rerun trips over these (git clone
        # refuses an existing dir; wget saves a .1 copy). We work elsewhere.
        log_warn(_OP, f"leftovers from an earlier bash install_nccl.sh run in "
                      f"{_BASH_TMP_DIR} would break a bash rerun; left alone "
                      f"(Python builds in {_WORK_DIR})", paths=leftovers)
    if os.path.lexists(_WORK_DIR):
        log_warn(_OP, f"removing leftovers of an interrupted run in {_WORK_DIR}",
                 path=_WORK_DIR)
    shutil.rmtree(_WORK_DIR, ignore_errors=True)
    os.makedirs(_WORK_DIR, exist_ok=True)


def install(env: dict[str, str]) -> int:
    """Build and install NCCL, the SHARP plugin (IB only) and nccl-tests. Returns 0 or 3.

    `env` is the build environment made by component_env() in build_config.py.
    """
    distribution = env.get("DISTRIBUTION", "")
    family = distro_family(distribution)
    from_apt = nccl_from_apt(distribution)
    architecture = env.get("ARCHITECTURE", "")
    arch_distro = env.get("ARCHITECTURE_DISTRO", "")
    sku = env.get("SKU", "")
    node_type = env.get("TARGET_NODE_TYPE", "")

    # ---- Inputs --------------------------------------------------------------
    nccl = config_for("nccl", env)
    cuda_cfg = config_for("cuda", env)
    nccl = nccl if isinstance(nccl, dict) else {}
    sharp = nccl.get("rdmasharpplugins")
    sharp = sharp if isinstance(sharp, dict) else {}
    version = _str_or_none(nccl.get("version"))
    sha256 = _str_or_none(nccl.get("sha256"))
    ref = _str_or_none(_jq_alternative(sharp.get("ref"), sharp.get("branch")))
    commit = _str_or_none(sharp.get("commit"))
    driver = cuda_cfg.get("driver") if isinstance(cuda_cfg, dict) else None
    cuda = _str_or_none(driver.get("version") if isinstance(driver, dict) else None)
    network_mode = sku_network_mode(sku, node_type)
    build_sharp = network_mode == NETWORK_MODE_STANDARD_IB

    problem = None
    if not from_apt and version is None:
        problem = "could not resolve nccl version from versions.json"
    elif not from_apt and sha256 is None:
        problem = "could not resolve nccl sha256 from versions.json (needed to verify the tarball)"
    elif not from_apt and cuda is None and family != "azurelinux":
        problem = "could not resolve cuda driver.version (part of the NCCL package names)"
    elif not from_apt and family == "ubuntu" and not arch_distro:
        problem = "ARCHITECTURE_DISTRO is not set (part of the NCCL .deb names)"
    elif build_sharp and commit is None:
        problem = "could not resolve nccl rdmasharpplugins.commit from versions.json"
    elif build_sharp and ref is None:
        problem = ("could not resolve nccl rdmasharpplugins.ref or .branch from "
                   "versions.json (git clone --branch needs it)")
    if problem:
        log_error(_OP, problem, distro=distribution, architecture=architecture)
        return 3

    source = "apt" if from_apt else "source"
    log_info(_OP, f"Installing NCCL {version if not from_apt else '(from apt)'} on "
                  f"{distribution} {architecture}", version=version, source=source,
             sha256=sha256, cuda=cuda, sharp_ref=ref, sharp_commit=commit,
             distro=distribution, architecture=architecture,
             architecture_distro=arch_distro, sku=sku, node_type=node_type,
             network_mode=network_mode,
             url=None if from_apt else download_url(version), work_dir=_WORK_DIR)

    # bash computes MAKE_JOBS first, for both the NCCL and the nccl-tests build.
    nproc = _nproc(env)
    if nproc is None:
        return 3
    jobs = make_jobs(nproc)
    log_info(_OP, f"Using make -j {jobs} ({nproc} CPUs, one left free)",
             nproc=nproc, jobs=jobs)

    # ---- 1. Build dependencies (exactly once) --------------------------------
    if install_deps(env) != 0:
        return 3

    source_dir = os.path.join(_WORK_DIR, f"nccl-{version}")
    try:
        _prepare_work_dir(version)

        # ---- 2. NCCL: apt (26.04) or build, package, install, hold/pin -------
        if from_apt:
            version = _install_apt_nccl(env)
            ok = version is not None
        else:
            ok = _build_nccl(env, version, sha256, source_dir, jobs)
            if ok and family == "ubuntu":
                ok = _package_ubuntu(env, source_dir, version, cuda, arch_distro)
            elif ok and family == "azurelinux":
                ok = _package_azure_linux(env, source_dir, version, architecture)
            elif ok:
                ok = _package_rhel(env, source_dir, version, cuda, architecture)
        if not ok:
            return 3

        # ---- 3. RDMA SHARP plugin (standard InfiniBand only) ------------------
        if build_sharp:
            log_info(_OP, f"Network mode {network_mode}: building the NCCL RDMA "
                          "SHARP plugin", network_mode=network_mode, sharp=True)
            if not _build_sharp_plugin(env, ref, commit, distribution):
                return 3
        else:
            log_info(_OP, f"Network mode {network_mode}: skipping the NCCL RDMA SHARP "
                          "plugin (no DOCA-OFED/SHARP/GPUDirect RDMA on this target)",
                     network_mode=network_mode, sharp=False)

        # ---- 4. nccl-tests ----------------------------------------------------
        if not _build_nccl_tests(env, jobs):
            return 3

        # ---- 5. Record, purge patchelf (22.04) and clean up ------------------
        write_component_version("NCCL", version)
        log_info(_OP, f"Recorded NCCL={version}", component="NCCL", version=version)

        if distribution == _PURGE_PATCHELF_DISTRIBUTION:
            # patchelf is only needed at build time (the SHARP plugin's make
            # install runs patchelf --set-soname). Jammy's patchelf has a
            # publishing-blocking CVE (USN-6036-1) whose fix needs Ubuntu Pro.
            log_info(_OP, f"{distribution}: purging patchelf (build-time only; "
                          "USN-6036-1 is unfixed without Ubuntu Pro)", package="patchelf")
            if not _run_all([["apt-get", "purge", "-y", "patchelf"]], env):
                return 3

        # bash: rm -rf /tmp/${TARBALL} /tmp/nccl-${NCCL_VERSION}
        if not from_apt:
            tarball_path = os.path.join(_WORK_DIR, tarball_name(version))
            if os.path.lexists(tarball_path):
                os.remove(tarball_path)
            shutil.rmtree(source_dir, ignore_errors=True)
            log_info(_OP, "Removed the NCCL tarball and source tree",
                     paths=[tarball_path, source_dir])
    except OSError as exc:
        # Work dir creation, tarball removal, and the versions file.
        log_error(_OP, f"file operation failed: {exc}",
                  path=getattr(exc, "filename", None))
        return 3
    finally:
        # Our own addition: never leave the scratch area (including the SHARP
        # plugin clone, which bash leaves in /tmp) behind.
        shutil.rmtree(_WORK_DIR, ignore_errors=True)

    log_info(_OP, f"NCCL {version} installed (packages "
                  f"{'held' if family == 'ubuntu' else 'pinned'}), SHARP plugin "
                  f"{'built' if build_sharp else 'skipped'}, nccl-tests in "
                  f"{os.path.join(_OPT_DIR, 'nccl-tests')}",
             version=version, source=source, cuda=cuda, distro=distribution,
             sharp_plugin=commit if build_sharp else None,
             nccl_tests=os.path.join(_OPT_DIR, "nccl-tests"))
    return 0
