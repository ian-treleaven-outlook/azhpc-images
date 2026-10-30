"""install_nccl.py — Python port of components/install_nccl.sh.

NCCL is the library multi-GPU / multi-node jobs use for collectives. On
NVIDIA images this step, in bash's order:

  1. Installs the build dependencies (install_deps()).
  2. Installs NCCL and stops package updates from replacing it:
       * Ubuntu 26.04: libnccl2 / libnccl-dev from apt, held with apt-mark;
         the installed version is read back with dpkg-query.
       * elsewhere: downloads the NCCL release tag tarball, builds it
         (`make src.build`), packages it as .deb (Ubuntu, then apt-mark hold)
         or .rpm (Azure Linux: dnf install; RHEL family: rpm -i; both then
         dnf versionlock) and installs those packages.
  3. On standard InfiniBand targets (sku_network_mode() == "standard_ib")
     builds the NCCL RDMA SHARP plugin against HPC-X into
     /usr/local/nccl-rdma-sharp-plugins and adds it to the linker path.
  4. Clones and builds nccl-tests with HPC-X MPI, then moves it to /opt.
  5. Records NCCL; on Ubuntu 22.04 purges patchelf (build-time only;
     Jammy's patchelf has a CVE, USN-6036-1, whose fix needs Ubuntu Pro).

Which of bash's if/elif branches a distro takes is decided once
(bash_branch()); the dependency lists and the NCCL packaging step are tables
keyed by it. The SHARP plugin and nccl-tests builds need `module load
mpi/hpcx`, so they run as run_script() shells. Bash records repo sources
(write_repo_source_info) in the middle of them, so the SHARP part is two
scripts (clone + checkout, then build) with the source recorded between, and
nccl-tests is cloned, recorded, then built.

utils/plans/steps.py makes install() the action of the "install-nccl" Step.

Behaviour kept from bash
------------------------
  * Commands, arguments and order are the same, and the first failure stops
    the step, including inside the scripts (the HPC-X checks `exit 1` with
    bash's messages).
  * `make -j MAKE_JOBS`: nproc - 1, but at least 1, for NCCL and nccl-tests.
  * The Azure Linux rpm paths are shell globs, expanded like bash (sorted; a
    pattern that matches nothing is passed on literally, and dnf fails).
  * Versions recorded: NCCL-RDMA_SHARP_PLUGIN = the configured commit (and
    its _REPO_COMMIT / _REPO_REF the configured commit and ref); NCCL = the
    configured version (Ubuntu 26.04: the installed one, from dpkg-query).
  * /etc/ld.so.conf.d/nccl-rdma-sharp-plugins.conf is written by the build
    script's heredoc, replacing any earlier copy.

Deliberate differences from bash
--------------------------------
  * Installs go through install_packages() (`apt-get install -y -o
    Dpkg::Use-Pty=0` / `dnf install -y`); `tar -xvf` is `tar -xf`.
  * The scripts get their `set -e` / `module load mpi/hpcx` header from
    run_script(); bash's `module unload mpi/hpcx` lines are gone (the shell
    exits).
  * Work directory: bash works in /tmp itself. We use _WORK_DIR
    (/tmp/azhpc-nccl), emptied first so leftovers of an interrupted run can't
    break the download/tar/git clone, and removed at the end (which covers
    bash's final `rm -rf` of the tarball and source tree, and also removes
    the SHARP plugin clone, which bash leaves in /tmp).
  * Missing config fails up front instead of building with "null": when
    NCCL is built (not Ubuntu 26.04), nccl version and sha256, the CUDA
    driver version (part of the package names; not on Azure Linux, which
    globs over it) and, on Ubuntu, ARCHITECTURE_DISTRO; when the SHARP
    plugin is built, rdmasharpplugins.commit and its ref (.ref, else .branch).
  * A tarball that doesn't produce nccl-<version>/ fails clearly; on Ubuntu
    26.04 an empty dpkg-query result fails (bash would record NCCL as "").
  * `nproc` is cpu_count() (the same CPU-affinity count, without a process);
    the dpkg-query read is installed_package_version()'s
    `dpkg-query -W -f ${Version} libnccl2` (bash: -f='${Version}').
  * The SHARP ref and commit are shell-quoted (identical for normal refs).
  * Azure Linux is any azurelinux* distro (bash: == "azurelinux3.0").

Bash issues kept and flagged (log_warn when they apply)
-------------------------------------------------------
  * Bash downloads and clones into /tmp and never removes the SHARP clone,
    so a bash rerun trips over leftovers; we warn if they are there.
  * The RHEL-family branch hard-codes x86_64 rpm paths (fails on aarch64).
  * `mv nccl-tests /opt/.` fails if /opt/nccl-tests exists and isn't empty.
  * A rdmasharpplugins.commit that is a branch/tag leaves the plugin unpinned.
"""

from __future__ import annotations

import glob
import os
import re
import shlex
from dataclasses import dataclass

from utils.component_config import config_for, record_version, require_config
from utils.distro import Distro
from utils.download import fetch_verified
from utils.files import extract_tarball, fresh_dir, remove_paths
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import (install_packages, installed_package_version,
                                     remove_packages)
from utils.package_pinning import dnf_pin_packages
from utils.process import cpu_count, run, run_all, run_script
from utils.repo_source import git_clone, write_repo_source_info
from utils.sku import NETWORK_MODE_STANDARD_IB, sku_network_mode

_OP = "install-nccl"

# The one distro that installs NCCL from apt instead of building it, and its
# bash_branch() name.
_APT_BRANCH = "ubuntu26.04"
# The one distro that purges patchelf afterwards.
_PURGE_PATCHELF_DISTRIBUTION = "ubuntu22.04"

# ---- Paths (module-level so tests can point them at a temporary directory) --
_WORK_DIR = "/tmp/azhpc-nccl"
# Where bash itself works (pushd /tmp); only checked for bash leftovers.
_BASH_TMP_DIR = "/tmp"
_OPT_DIR = "/opt"
_SHARP_PREFIX = "/usr/local/nccl-rdma-sharp-plugins"
_LD_SO_CONF = "/etc/ld.so.conf.d/nccl-rdma-sharp-plugins.conf"

_NCCL_TAG_URL = "https://github.com/NVIDIA/nccl/archive/refs/tags/"
_SHARP_GIT_URL = "https://github.com/Mellanox/nccl-rdma-sharp-plugins.git"
_NCCL_TESTS_GIT_URL = "https://github.com/NVIDIA/nccl-tests.git"
_HPCX = ["mpi/hpcx"]

_SHARP_COMPONENT = "NCCL-RDMA_SHARP_PLUGIN"
_NCCL_TESTS_COMPONENT = "NCCL_TESTS"

# Build dependencies per bash_branch(): one package list per bash install command.
_DEPS = {
    _APT_BRANCH: [["build-essential", "patchelf", "zlib1g-dev"]],
    "ubuntu": [
        ["build-essential", "devscripts", "debhelper", "fakeroot", "patchelf"],
        # libibverbs-dev is deliberately NOT installed (commented out in
        # bash): it conflicts on bare-metal 1P builds, and on VMs
        # install_utils.sh or install_doca.sh has installed it.
        ["zlib1g-dev"],
    ],
    "azurelinux": [["rpm-build", "rpmdevtools", "autoconf", "automake", "git",
                    "libtool", "patchelf"]],
    "rhel": [["rpm-build", "rpmdevtools", "patchelf"]],
}

_APT_NCCL_PACKAGES = ["libnccl2", "libnccl-dev"]
_APT_NCCL_HOLDS = [["apt-mark", "hold", "libnccl2"], ["apt-mark", "hold", "libnccl-dev"]]

_COMMIT_HASH_RE = re.compile(r"[0-9a-f]{7,40}")


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def bash_branch(distro: Distro) -> str:
    """Which branch of install_nccl.sh's if/elif `distro` takes.

    "ubuntu26.04" (NCCL from apt), "ubuntu", "azurelinux", or "rhel" for
    bash's `else` (the RHEL family, and anything unrecognised).
    """
    if distro.name == _APT_BRANCH:
        return _APT_BRANCH
    return distro.family if distro.family in ("ubuntu", "azurelinux") else "rhel"


def tarball_name(version: str) -> str:
    """bash: TARBALL="v${NCCL_VERSION}.tar.gz"."""
    return f"v{version}.tar.gz"


def download_url(version: str) -> str:
    """bash: NCCL_DOWNLOAD_URL=https://github.com/NVIDIA/nccl/archive/refs/tags/${TARBALL}."""
    return _NCCL_TAG_URL + tarball_name(version)


def make_jobs(nproc: int) -> int:
    """bash: MAKE_JOBS=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 )) (never -j 0)."""
    return nproc - 1 if nproc > 1 else 1


def apt_nccl_version(dpkg_query_output: str) -> str:
    """The NCCL version from dpkg-query, as bash's `| sed 's/+cuda.*//'`.

    sed drops everything from the first "+cuda" on each line; $(...) then
    drops trailing newlines: "2.28.9-1+cuda13.0" -> "2.28.9-1".
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
    """Expand `pattern` relative to `cwd` as bash would: sorted matches, or
    (without nullglob) the pattern itself when nothing matches."""
    return sorted(glob.glob(pattern, root_dir=cwd)) or [pattern]


def is_commit_hash(ref: str) -> bool:
    """True if `ref` looks like a git commit hash rather than a branch/tag."""
    return _COMMIT_HASH_RE.fullmatch(ref) is not None


def bash_leftovers(tmp_dir: str, version: str) -> list[str]:
    """Paths an earlier bash run of install_nccl.sh may have left in `tmp_dir`."""
    names = (tarball_name(version), f"nccl-{version}", "nccl-rdma-sharp-plugins",
             "nccl-tests")
    return [os.path.join(tmp_dir, name) for name in names
            if os.path.lexists(os.path.join(tmp_dir, name))]


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


def sharp_ref(sharp: dict) -> str | None:
    """bash: jq -r '.rdmasharpplugins.ref // .rdmasharpplugins.branch // empty'."""
    return _str_or_none(_jq_alternative(sharp.get("ref"), sharp.get("branch")))


def sharp_clone_lines(ref: str, commit: str) -> list[str]:
    """SHARP plugin part 1 (after `module load mpi/hpcx`): check HPC-X, clone, check out.

    The checks read HPCX_SHARP_DIR / HPCX_UCX_DIR, which the module sets only
    in its own shell; the checks and messages are bash's.
    """
    return [
        'if [[ -z "${HPCX_SHARP_DIR:-}" || ! -d "${HPCX_SHARP_DIR}" ]]; then',
        '    echo "HPC-X module did not provide a valid HPCX_SHARP_DIR; '
        'cannot build NCCL RDMA SHARP plugin."',
        "    exit 1",
        "fi",
        'if [[ -z "${HPCX_UCX_DIR:-}" || ! -d "${HPCX_UCX_DIR}" ]]; then',
        '    echo "HPC-X module did not provide a valid HPCX_UCX_DIR; '
        'cannot build NCCL RDMA SHARP plugin."',
        "    exit 1",
        "fi",
        f"mkdir -p {_SHARP_PREFIX}",
        f"git clone --branch {shlex.quote(ref)} {_SHARP_GIT_URL}",
        "pushd nccl-rdma-sharp-plugins",
        f"git checkout {shlex.quote(commit)}",
    ]


def sharp_build_lines(libtoolize: bool) -> list[str]:
    """SHARP plugin part 2: build, install, and add its lib dir to ld.so.conf.d.

    A fresh shell, so it enters the clone again; ./configure finds SHARP/UCX
    through the HPC-X module. bash runs libtoolize on Azure Linux only.
    """
    return [
        "pushd nccl-rdma-sharp-plugins",
        *(["libtoolize --verbose"] if libtoolize else []),
        "./autogen.sh",
        f"./configure --prefix={_SHARP_PREFIX} --with-cuda=/usr/local/cuda",
        "make",
        "make install",
        f"cat > {_LD_SO_CONF} <<EOF",
        f"{_SHARP_PREFIX}/lib",
        "EOF",
        "ldconfig",
        "popd",
    ]


def nccl_tests_lines(jobs: int) -> list[str]:
    """Build nccl-tests with HPC-X MPI and move it to /opt.

    ${HPCX_MPI_DIR} is set by the module and expanded by that shell (bash
    doesn't check it; an empty MPI_HOME fails the build on a missing mpi.h).
    """
    return [
        "pushd nccl-tests",
        f"make -j {jobs} MPI=1 MPI_HOME=${{HPCX_MPI_DIR}} CUDA_HOME=/usr/local/cuda",
        "popd",
        f"mv nccl-tests {_OPT_DIR}/.",
    ]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Inputs:
    """What install() resolves up front from env and versions.json."""

    distro: Distro
    branch: str               # bash_branch(distro)
    version: str | None       # configured NCCL version
    sha256: str | None
    cuda: str | None          # CUDA driver version, part of the package names
    architecture: str
    arch_distro: str
    sku: str
    node_type: str
    network_mode: str
    ref: str | None           # SHARP plugin branch/tag
    commit: str | None        # SHARP plugin commit

    @property
    def from_apt(self) -> bool:
        return self.branch == _APT_BRANCH

    @property
    def build_sharp(self) -> bool:
        return self.network_mode == NETWORK_MODE_STANDARD_IB


def _config(component: str, env: dict[str, str], keys: list[str]) -> dict | None:
    """require_config() when `keys` are needed; otherwise whatever config_for()
    finds ({} if nothing), since bash reads optional fields without failing."""
    if keys:
        return require_config(component, env, _OP, *keys)
    return config_for(component, env) or {}


def _resolve_inputs(env: dict[str, str]) -> _Inputs | None:
    """Read and check the config this distro/SKU needs; None (logged) if missing."""
    distro = Distro.from_env(env)
    branch = bash_branch(distro)
    builds = branch != _APT_BRANCH
    sku, node_type = env.get("SKU", ""), env.get("TARGET_NODE_TYPE", "")
    network_mode = sku_network_mode(sku, node_type)
    build_sharp = network_mode == NETWORK_MODE_STANDARD_IB

    nccl = _config("nccl", env, (["version", "sha256"] if builds else [])
                   + (["rdmasharpplugins.commit"] if build_sharp else []))
    cuda = _config("cuda", env,
                   ["driver.version"] if builds and branch != "azurelinux" else [])
    if nccl is None or cuda is None:
        return None
    sharp = nccl.get("rdmasharpplugins") or {}
    inputs = _Inputs(
        distro=distro, branch=branch, version=_str_or_none(nccl.get("version")),
        sha256=_str_or_none(nccl.get("sha256")),
        cuda=_str_or_none((cuda.get("driver") or {}).get("version")),
        architecture=env.get("ARCHITECTURE", ""),
        arch_distro=env.get("ARCHITECTURE_DISTRO", ""), sku=sku, node_type=node_type,
        network_mode=network_mode, ref=sharp_ref(sharp),
        commit=_str_or_none(sharp.get("commit")))

    if branch == "ubuntu" and not inputs.arch_distro:
        log_error(_OP, "ARCHITECTURE_DISTRO is not set (part of the NCCL .deb names)")
        return None
    if build_sharp and inputs.ref is None:
        log_error(_OP, "could not resolve nccl rdmasharpplugins.ref or .branch from "
                       "versions.json (git clone --branch needs it)")
        return None
    return inputs


def _log_start(inputs: _Inputs) -> None:
    source = "apt" if inputs.from_apt else "source"
    shown = "(from apt)" if inputs.from_apt else inputs.version
    log_info(_OP, f"Installing NCCL {shown} on {inputs.distro} {inputs.architecture}",
             version=inputs.version, source=source, sha256=inputs.sha256,
             cuda=inputs.cuda, sharp_ref=inputs.ref, sharp_commit=inputs.commit,
             distro=str(inputs.distro), architecture=inputs.architecture,
             architecture_distro=inputs.arch_distro, sku=inputs.sku,
             node_type=inputs.node_type, network_mode=inputs.network_mode,
             url=None if inputs.from_apt else download_url(inputs.version),
             work_dir=_WORK_DIR)


def _make_jobs() -> int:
    """bash's MAKE_JOBS, from `nproc` (cpu_count())."""
    nproc = cpu_count()
    jobs = make_jobs(nproc)
    log_info(_OP, f"Using make -j {jobs} ({nproc} CPUs, one left free)",
             nproc=nproc, jobs=jobs)
    return jobs


# --------------------------------------------------------------------------- #
# Steps: each returns True on success, False once the failure is logged.
# --------------------------------------------------------------------------- #
def install_deps(env: dict[str, str]) -> int:
    """Install NCCL's build dependencies, one install per bash command. 0 or 3."""
    distro = Distro.from_env(env)
    branch = bash_branch(distro)
    package_lists = _DEPS[branch]
    log_info(_OP, f"Installing NCCL build dependencies ({branch} branch of "
                  "install_nccl.sh)", distro=str(distro), branch=branch,
             packages=[p for packages in package_lists for p in packages])
    for packages in package_lists:
        if not install_packages(packages, env, _OP):
            return 3
    return 0


def _prepare_work_dir(version: str | None) -> bool:
    leftovers = bash_leftovers(_BASH_TMP_DIR, version) if version else []
    if leftovers:
        log_warn(_OP, f"leftovers from an earlier bash install_nccl.sh run in "
                      f"{_BASH_TMP_DIR} would break a bash rerun; left alone "
                      f"(Python builds in {_WORK_DIR})", paths=leftovers)
    if os.path.lexists(_WORK_DIR):
        log_warn(_OP, f"removing leftovers of an interrupted run in {_WORK_DIR}",
                 path=_WORK_DIR)
    return fresh_dir(_WORK_DIR, _OP)


def _install_apt_nccl(env: dict[str, str]) -> str | None:
    """Ubuntu 26.04: install + hold the packaged NCCL; return its version."""
    log_info(_OP, f"{_APT_BRANCH}: installing NCCL from apt (no source build)",
             packages=_APT_NCCL_PACKAGES, source="apt")
    if not install_packages(_APT_NCCL_PACKAGES, env, _OP):
        return None
    if not run_all(_APT_NCCL_HOLDS, _OP, env=env):
        return None
    log_info(_OP, "Held libnccl2 and libnccl-dev with apt-mark",
             packages=_APT_NCCL_PACKAGES, held=True)
    installed = installed_package_version("libnccl2", env, _OP)
    if installed is None:
        return None
    version = apt_nccl_version(installed)       # bash: | sed 's/+cuda.*//'
    if not version:
        log_error(_OP, f"no NCCL version left in libnccl2's {installed!r}",
                  dpkg_version=installed)
        return None
    log_info(_OP, f"Installed NCCL version is {version}", version=version,
             version_source="dpkg-query", dpkg_version=installed)
    return version


def _build_nccl(env: dict[str, str], inputs: _Inputs, source_dir: str, jobs: int) -> bool:
    """Download, verify and unpack the NCCL tag tarball, then `make src.build`."""
    version, tarball = inputs.version, tarball_name(inputs.version)
    url = download_url(version)
    log_info(_OP, f"Downloading and extracting NCCL {version}", url=url,
             sha256=inputs.sha256, tarball=tarball, work_dir=_WORK_DIR, phase="download")
    if fetch_verified(url, inputs.sha256, _OP, dest_dir=_WORK_DIR) is None:
        return False
    if not extract_tarball(tarball, _OP, cwd=_WORK_DIR, env=env):
        return False
    if not os.path.isdir(source_dir):
        log_error(_OP, f"NCCL source folder {source_dir} was not created by "
                       f"extracting {tarball}", tarball=tarball)
        return False
    log_info(_OP, f"Building NCCL {version} (make -j {jobs} src.build)",
             version=version, jobs=jobs, phase="src.build")
    return run(["make", "-j", str(jobs), "src.build"], _OP, cwd=source_dir, env=env)


def _package_debs(env: dict[str, str], inputs: _Inputs, source_dir: str) -> bool:
    """Ubuntu: make .debs, then install and apt-mark hold each in turn."""
    log_info(_OP, "Packaging NCCL as .deb (make pkg.debian.build)", phase="pkg.debian.build")
    if not run(["make", "pkg.debian.build"], _OP, cwd=source_dir, env=env):
        return False
    deb_dir = os.path.join(source_dir, "build", "pkg", "deb")
    for deb, package in deb_packages(inputs.version, inputs.cuda, inputs.arch_distro):
        if not install_packages([f"./{deb}"], env, _OP, cwd=deb_dir):
            return False
        if not run(["apt-mark", "hold", package], _OP, cwd=deb_dir, env=env):
            return False
        log_info(_OP, f"Installed {deb} and held {package} with apt-mark",
                 package=package, file=deb, held=True)
    return True


def _package_azure_linux_rpms(env: dict[str, str], inputs: _Inputs, source_dir: str) -> bool:
    """Azure Linux: make .rpms, dnf install each glob, pin libnccl*."""
    log_info(_OP, "Packaging NCCL as .rpm (make pkg.redhat.build)", phase="pkg.redhat.build")
    if not run(["make", "pkg.redhat.build"], _OP, cwd=source_dir, env=env):
        return False
    for pattern in azure_linux_rpm_patterns(inputs.version, inputs.architecture):
        files = expand_glob(pattern, source_dir)
        if files == [pattern]:
            log_warn(_OP, f"no rpm matched {pattern}; passing it to dnf literally, "
                          "as bash does", pattern=pattern)
        if not install_packages(files, env, _OP, cwd=source_dir):
            return False
    return dnf_pin_packages(["libnccl*"], _OP, env)


def _package_rhel_rpms(env: dict[str, str], inputs: _Inputs, source_dir: str) -> bool:
    """RHEL family: make .rpms, rpm -i the three exact x86_64 paths, pin libnccl*."""
    if inputs.architecture != "x86_64":
        log_warn(_OP, f"install_nccl.sh hard-codes x86_64 rpm paths on "
                      f"{inputs.architecture}; rpm -i will not find the packages, "
                      "as in bash", architecture=inputs.architecture)
    log_info(_OP, "Packaging NCCL as .rpm (make pkg.redhat.build)", phase="pkg.redhat.build")
    if not run(["make", "pkg.redhat.build"], _OP, cwd=source_dir, env=env):
        return False
    paths = rhel_rpm_paths(inputs.version, inputs.cuda)
    if not run_all([["rpm", "-i", path] for path in paths], _OP, cwd=source_dir, env=env):
        return False
    log_info(_OP, f"Installed {len(paths)} NCCL rpms with rpm -i", files=paths)
    return dnf_pin_packages(["libnccl*"], _OP, env)


# How each source-building branch packages and installs the built NCCL.
_PACKAGE_NCCL = {
    "ubuntu": _package_debs,
    "azurelinux": _package_azure_linux_rpms,
    "rhel": _package_rhel_rpms,
}


def _install_nccl(env: dict[str, str], inputs: _Inputs, source_dir: str,
                  jobs: int) -> str | None:
    """Install NCCL from apt or from source; the NCCL version to record, or None."""
    if inputs.from_apt:
        return _install_apt_nccl(env)
    if not _build_nccl(env, inputs, source_dir, jobs):
        return None
    if not _PACKAGE_NCCL[inputs.branch](env, inputs, source_dir):
        return None
    return inputs.version


def _build_sharp_plugin(env: dict[str, str], inputs: _Inputs) -> bool:
    """Clone, record and build the SHARP plugin, then record its commit."""
    ref, commit = inputs.ref, inputs.commit
    if not is_commit_hash(commit):
        # bash issue: a branch/tag "commit" means what gets built moves over time.
        log_warn(_OP, f"rdmasharpplugins.commit {commit!r} is a branch or tag, not a "
                      "commit hash; the plugin build is not pinned (kept from bash)",
                 commit=commit)
    # The build script's heredoc writes the conf, so note beforehand whether it
    # will replace one.
    conf_existed = os.path.exists(_LD_SO_CONF)
    log_info(_OP, f"Building the NCCL RDMA SHARP plugin from {ref} at {commit} with "
                  f"HPC-X into {_SHARP_PREFIX}", ref=ref, commit=commit,
             prefix=_SHARP_PREFIX, url=_SHARP_GIT_URL,
             libtoolize=inputs.branch == "azurelinux", phase="sharp-plugin")
    if not run_script(sharp_clone_lines(ref, commit), _OP, modules=_HPCX,
                      cwd=_WORK_DIR, env=env):
        return False
    if not write_repo_source_info(_SHARP_COMPONENT,
                                  os.path.join(_WORK_DIR, "nccl-rdma-sharp-plugins"),
                                  _OP, commit=commit, ref=ref, env=env):
        return False
    if not run_script(sharp_build_lines(inputs.branch == "azurelinux"), _OP,
                      modules=_HPCX, cwd=_WORK_DIR, env=env):
        return False
    action = "replaced" if conf_existed else "created"
    log_info(_OP, f"{action.capitalize()} {_LD_SO_CONF} ({_SHARP_PREFIX}/lib) and ran "
                  "ldconfig", path=_LD_SO_CONF, action=action)
    return record_version(_SHARP_COMPONENT, commit, _OP)


def _sharp_plugin(env: dict[str, str], inputs: _Inputs) -> bool:
    """Build the SHARP plugin on standard InfiniBand targets only."""
    mode = inputs.network_mode
    if not inputs.build_sharp:
        log_info(_OP, f"Network mode {mode}: skipping the NCCL RDMA SHARP plugin (no "
                      "DOCA-OFED/SHARP/GPUDirect RDMA on this target)",
                 network_mode=mode, sharp=False)
        return True
    log_info(_OP, f"Network mode {mode}: building the NCCL RDMA SHARP plugin",
             network_mode=mode, sharp=True)
    return _build_sharp_plugin(env, inputs)


def _build_nccl_tests(env: dict[str, str], jobs: int) -> bool:
    """Clone nccl-tests, record its source, build it and move it to /opt."""
    dest = os.path.join(_OPT_DIR, "nccl-tests")
    if os.path.lexists(dest):
        # bash issue: `mv nccl-tests /opt/.` fails if /opt/nccl-tests is a
        # non-empty dir (and replaces it if empty). Kept as is.
        log_warn(_OP, f"{dest} already exists; bash's `mv nccl-tests {_OPT_DIR}/.` "
                      "fails if it is not empty", path=dest)
    log_info(_OP, f"Building nccl-tests with HPC-X MPI (make -j {jobs})",
             url=_NCCL_TESTS_GIT_URL, dest=dest, jobs=jobs, phase="nccl-tests")
    checkout = git_clone(_NCCL_TESTS_GIT_URL, _OP, cwd=_WORK_DIR, env=env)
    if checkout is None or not write_repo_source_info(_NCCL_TESTS_COMPONENT, checkout,
                                                      _OP, env=env):
        return False
    if not run_script(nccl_tests_lines(jobs), _OP, modules=_HPCX, cwd=_WORK_DIR, env=env):
        return False
    log_info(_OP, f"nccl-tests built and moved to {dest}", dest=dest)
    return True


def _record_nccl(env: dict[str, str], distro: Distro, version: str) -> bool:
    """Record NCCL's version, then (Ubuntu 22.04) purge the build-time patchelf."""
    if not record_version("NCCL", version, _OP):
        return False
    if distro.name != _PURGE_PATCHELF_DISTRIBUTION:
        return True
    # patchelf is only needed at build time (the SHARP plugin's make install
    # runs patchelf --set-soname). Jammy's patchelf has a publishing-blocking
    # CVE (USN-6036-1) whose fix needs Ubuntu Pro.
    log_info(_OP, f"{distro}: purging patchelf (build-time only; USN-6036-1 is "
                  "unfixed without Ubuntu Pro)", package="patchelf")
    return remove_packages(["patchelf"], env, _OP, purge=True)


# --------------------------------------------------------------------------- #
# The step
# --------------------------------------------------------------------------- #
def install(env: dict[str, str]) -> int:
    """Install NCCL, the SHARP plugin (standard IB only) and nccl-tests. 0, or 3."""
    inputs = _resolve_inputs(env)
    if inputs is None:
        return 3
    _log_start(inputs)
    jobs = _make_jobs()
    if install_deps(env) != 0:
        return 3

    source_dir = os.path.join(_WORK_DIR, f"nccl-{inputs.version}")
    try:
        if not _prepare_work_dir(inputs.version):
            return 3
        version = _install_nccl(env, inputs, source_dir, jobs)
        if version is None:
            return 3
        if not _sharp_plugin(env, inputs) or not _build_nccl_tests(env, jobs):
            return 3
        if not _record_nccl(env, inputs.distro, version):
            return 3
    except OSError as exc:
        # write_repo_source_info raises if the versions file can't be written.
        log_error(_OP, f"file operation failed: {exc}",
                  path=getattr(exc, "filename", None))
        return 3
    finally:
        remove_paths(_WORK_DIR, op=_OP)

    nccl_tests = os.path.join(_OPT_DIR, "nccl-tests")
    log_info(_OP, f"NCCL {version} installed (packages "
                  f"{'held' if inputs.distro.is_ubuntu else 'pinned'}), SHARP plugin "
                  f"{'built' if inputs.build_sharp else 'skipped'}, nccl-tests in "
                  f"{nccl_tests}", version=version,
             source="apt" if inputs.from_apt else "source", cuda=inputs.cuda,
             distro=str(inputs.distro),
             sharp_plugin=inputs.commit if inputs.build_sharp else None,
             nccl_tests=nccl_tests)
    return 0