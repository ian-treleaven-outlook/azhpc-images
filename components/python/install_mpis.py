"""install_mpis.py — Python port of components/install_mpis.sh.

Installs the MPI libraries HPC jobs use to talk between processes and nodes,
each with an environment-module file (`module load mpi/<name>`), in bash's
order:

  1. HPC-X       NVIDIA's MPI bundle (Open MPI + UCX + HCOLL + SHARP),
                 unpacked into /opt. The variant depends on the GPU vendor and
                 network (hpcx_variant()). Usually its Open MPI (and on AMD /
                 Ubuntu 22.04 its UCX) is rebuilt with the bundled
                 hpcx_rebuild.sh. Its SHARP and UCX libraries are then
                 registered in /etc/ld.so.conf.d and its pkg-config files
                 linked into /usr/local/lib(64)/pkgconfig.
  2. MVAPICH     built from source (skipped on GB-family SKUs with Ubuntu 24.04
                 or Azure Linux 3.0, and on Ubuntu 26.04).
  3. Open MPI    built from source (skipped on Ubuntu 26.04).
  4. Intel MPI   Intel's offline installer (x86_64 only).

Each installed version is recorded (HPCX, MVAPICH, OMPI, IMPI, and PMIX when
Open MPI uses its internal PMIx). On RPM distros ucx*, openmpi and perftest
are pinned against dnf updates.

Two decisions run through the whole step, both made once in _plan():
  * Network: UCX SKUs (InfiniBand) use UCX + HCOLL; non-UCX SKUs (NCv6,
    "no_rdma") use libfabric's tcp provider and get modulefile settings that
    avoid hangs on MANA-only hardware.
  * PMIx: on Ubuntu 26.04, Azure Linux 3.0, EL10 and baremetal_3p nodes Open
    MPI builds its own PMIx/hwloc/libevent ("internal") and the PMIx version
    it ships is recorded; elsewhere it uses /opt/pmix/<version> from
    install_pmix.

utils/plans/steps.py sets install() as the `action=` of the "install-mpis"
Step; deleting `action=` runs the bash script again. Every phase returns
False (or None) at its first failure, which install() turns into 3, like
bash's `set -e`.

Kept from bash
--------------
  * Commands, arguments and order. The bash-rendered fixtures in
    tests/python/fixtures/install_mpis/ (UCX, NCv6, EL8 AMD, GB200 aarch64,
    Ubuntu 22.04 patched UCX, Azure Linux 3.0 vendor HPC-X, Ubuntu 26.04)
    pin the modulefiles and symlinks byte for byte.
  * Open MPI's UCX configure passes LD_LIBRARY_PATH=<caller's>:<hcoll>/lib as
    a configure ARGUMENT, an empty caller value giving ":<hcoll>/lib".
  * The /etc/ld.so.conf.d files are replaced (`cat >`) and the pkg-config
    links too (`ln -sf`).
  * BASH ISSUES (kept, warned at runtime):
      - `set CC=/usr/bin/gcc` doesn't set CC (it sets $1); the builds use the
        environment's compiler, in bash and here.
      - Modulefiles are appended to (`cat >>`) and the short names linked
        with plain `ln -s`, so a rerun duplicates the modulefile and then
        fails at the first existing link, where bash fails.
      - Open MPI's PMIX_FLAG has an unreachable azurelinux3.0/baremetal_3p
        branch (both use the internal PMIx); see ompi_pmix_flags().

Deliberate differences
----------------------
  * Every config (HPC-X variant, pmix unless internal, mvapich/ompi/impi
    unless skipped) is checked before anything is downloaded; bash resolves
    each just before use and carries on with blank values. A pmix/impi
    version too short for bash's ${VERSION:0:-2} fails up front too, as does
    an empty MODULE_FILES_DIRECTORY.
  * Downloads and builds happen in _WORK_DIR, removed at the end whatever
    happened; bash works in the distro build dir and ends with
    `rm -rf *.tbz *.tar.gz *offline.sh; rm -rf -- */`, which deletes EVERY
    subdirectory there (warned about, not ported).
  * The CPU count for `make -j` is read once, with cpu_count() (the affinity
    count `nproc` prints, without running it); a source folder tar didn't
    create fails clearly.
  * hpcx_rebuild.sh gets CUDA_HOME through its environment (bash:
    `env CUDA_HOME=... hpcx_rebuild.sh`); on failure the tails of its UCX and
    Open MPI logs are logged as error details (bash: `tail -n 40`).
  * The UCX netlink patch: `patch -i <file>` (exec_program has no stdin). A
    UCX source without src/ucs/sys/netlink.c fails (bash's grep fails, so it
    rebuilds unpatched); the scratch dir is removed even on failure.
  * An internal PMIx version pkg-config prints as empty fails.
  * Toolkit spellings: `tar -xf` for every unpack (bash `tar -xvf`, and
    `tar -xzf ... -C` for the UCX source), and write_file creates
    /etc/ld.so.conf.d if it is missing. `mkdir -p`, `mktemp -d`, `ln -s` /
    `ln -sf` and write_component_version go through make_dirs(),
    make_temp_dir(), symlink() and record_version(), which log their own
    failures. The `mv`s, `cp -r` and `sed -i` stay bash's commands.
"""

from __future__ import annotations

import glob
import os
import re
from collections import deque
from dataclasses import dataclass, field

from utils.component_config import DEFAULT_NODE_TYPE, record_version, require_config
from utils.distro import Distro
from utils.download import fetch_verified
from utils.files import (extract_tarball, fresh_dir, make_dirs, make_temp_dir, remove_paths,
                         symlink, write_file)
from utils.logger import log_debug, log_error, log_error_detail, log_info, log_warn
from utils.package_pinning import dnf_pin_packages
from utils.process import cpu_count, run, run_all, run_capture
from utils.sku import NETWORK_MODE_NO_RDMA, sku_network_mode, sku_uses_ucx

_OP = "install-mpis"

# bash matches these distro names exactly.
_AZURE_LINUX = "azurelinux3.0"
_UBUNTU_22 = "ubuntu22.04"
_UBUNTU_24 = "ubuntu24.04"
_UBUNTU_26 = "ubuntu26.04"
_EL8_DISTROS = ("almalinux8.10", "rocky8.10")      # plus rhel8*
_GB_FAMILY = "gb-family"
_BAREMETAL_3P = "baremetal_3p"

# ---- Paths (module level so tests can point them at a temporary directory) --
_INSTALL_PREFIX = "/opt"                      # bash: INSTALL_PREFIX=/opt
_WORK_DIR = "/tmp/azhpc-mpis"                 # our scratch area (see docstring)
_ROCM_DIR = "/opt/rocm"                       # only for the "is ROCm installed" check
_LD_SO_CONF_DIR = "/etc/ld.so.conf.d"
_PKGCONFIG_DIRS = ("/usr/local/lib/pkgconfig",)
_PKGCONFIG_DIRS_RPM = ("/usr/local/lib/pkgconfig", "/usr/local/lib64/pkgconfig")

# Literal in bash; only used in generated text, arguments and the environment.
_LIBFABRIC_PATH = "/opt/libfabric"
_ROCM_PATH = "/opt/rocm"
_CUDA_HOME = "/usr/local/cuda"
_INTERNAL_DEPS_ARGS = ("--with-pmix=internal", "--with-hwloc=internal",
                       "--with-libevent=internal")
# --ompi-extra-config replaces HPC-X's own Open MPI configure options; these
# are the compatible vendor options bash restores.
_OMPI_VENDOR_ARGS = ("--enable-mpi1-compatibility", "--without-xpmem", "--with-slurm",
                     "--enable-orterun-prefix-by-default")
_MELLANOX_PLATFORM = "--with-platform=contrib/platform/mellanox/optimized"

# ---- UCX netlink route-dump patch (upstream #644) ---------------------------
_UCX_NETLINK_PATCH = "patches/ucx-netlink-dump-done.patch"      # under COMPONENT_DIR
# The line only UCX versions with the hang have (bash greps for it with -F).
_UCX_NETLINK_BUGGY_LINE = ("msg_done = ucs_netlink_is_msg_done("
                           "(const struct nlmsghdr *)recv_msg);")
_UCX_NETLINK_SOURCE = "src/ucs/sys/netlink.c"
# How many lines of each hpcx_rebuild.sh log bash prints when the rebuild fails.
_REBUILD_LOG_TAIL = 40

# bash's sed expressions pointing hpcx_rebuild.sh's module/init files at
# hpcx-rebuild instead of hpcx-rebuild$mpi_version (a directory it never
# creates). Passed to sed exactly as bash's single-quoted strings.
_REBUILD_ENTRYPOINT_SED = (
    "-e", r"s|/hpcx-rebuild\$\{mpi_version\}|/hpcx-rebuild|g",
    "-e", r"s|/hpcx-rebuild\$mpi_version|/hpcx-rebuild|g",
    "-e", "s|/hpcx-rebuild[0-9]+|/hpcx-rebuild|g",
)


# --------------------------------------------------------------------------- #
# Pure helpers: bash's decisions and command lines, tested directly.
# --------------------------------------------------------------------------- #
def drop_last_two(value: str) -> str | None:
    """bash's ${value:0:-2}: everything but the last 2 characters.

    e.g. pmix "4.2.9-2" -> "4.2.9", impi "2021.16.1" -> "2021.16". None when
    bash would abort (fewer than 2 characters) or give an empty path (exactly 2).
    """
    if len(value) <= 2:
        return None
    return value[:-2]


def url_basename(url: str, suffix: str = "") -> str:
    """bash's $(basename URL [SUFFIX]): the last path part, minus SUFFIX.

    basename ignores trailing slashes and only removes SUFFIX when something
    is left.
    """
    name = url.rstrip("/").rsplit("/", 1)[-1]
    if suffix and name.endswith(suffix) and name != suffix:
        name = name[:-len(suffix)]
    return name


def hpcx_variant(gpu: str, network_mode: str) -> tuple[str, str]:
    """Which HPC-X config bash picks, and why: (component name, reason).

    Same choice as utils/version_validate.active_hpcx_component; a test keeps
    the two in step.
    """
    if gpu == "AMD":
        return "hpcx_amd", "AMD GPU: AMD has a regression on newer HPC-X versions"
    if network_mode == NETWORK_MODE_NO_RDMA:
        return ("hpcx_inbox", "network mode no_rdma: no DOCA-OFED on non-IB SKUs, "
                              "so use the inbox HPC-X (UCX linked against rdma-core)")
    return "hpcx", "default HPC-X (InfiniBand network)"


def internal_pmix_reason(distro: Distro, node_type: str) -> str | None:
    """Why Open MPI builds its own PMIx here (bash's USE_INTERNAL_PMIX), or None.

    bash: ubuntu26.04 | azurelinux3.0 | almalinux10* | rocky10* | rhel10*,
          elif ${TARGET_NODE_TYPE:-azure_vm_regular} == baremetal_3p
    """
    if distro.name == _AZURE_LINUX:
        return "azurelinux3.0 lacks the PMIx package published by the CycleCloud team"
    if distro.name == _UBUNTU_26 or (distro.is_rhel_family and distro.major == "10"):
        return f"{distro} has no external PMIx v5 to align to"
    if (node_type or DEFAULT_NODE_TYPE) == _BAREMETAL_3P:
        return "baremetal_3p nodes must avoid the external PMIx (package conflict)"
    return None


def uses_internal_pmix(distro: Distro, node_type: str) -> bool:
    """bash's USE_INTERNAL_PMIX (see internal_pmix_reason)."""
    return internal_pmix_reason(distro, node_type) is not None


def ompi_deps_args(internal_pmix: bool, pmix_path: str | None) -> list[str]:
    """bash's OMPI_DEPS_ARGS: Open MPI's PMIx/hwloc/libevent configure options.

    Internal: Open MPI builds them from its own sources into its prefix
    (pointing them at HPC-X's ompi5 tree would put the vendor libmpi ahead of
    the rebuild in every RUNPATH).
    """
    if internal_pmix:
        return list(_INTERNAL_DEPS_ARGS)
    return [f"--with-pmix={pmix_path}"]


def ompi_pmix_flags(internal_pmix: bool, distro: Distro, node_type: str,
                    pmix_path: str | None) -> list[str]:
    """Open MPI's ${PMIX_FLAG}, split into words as bash's unquoted expansion does."""
    if internal_pmix:
        return list(_INTERNAL_DEPS_ARGS)        # "${OMPI_DEPS_ARGS[*]}", unquoted
    # BASH ISSUE (kept): unreachable. azurelinux3.0 and baremetal_3p always
    # use the internal PMIx, which is checked first.
    if distro.name == _AZURE_LINUX or (node_type or DEFAULT_NODE_TYPE) == _BAREMETAL_3P:
        return ["--with-pmix"]
    return [f"--with-pmix={pmix_path}"]


def rebuild_ucx_reason(gpu: str, distro: Distro, uses_ucx: bool) -> str | None:
    """Why HPC-X's UCX is rebuilt (bash's REBUILD_UCX), or None.

    bash: sku_uses_ucx && ( "$GPU" == "AMD" || "$DISTRIBUTION" == "ubuntu22.04" )
    """
    if not uses_ucx:
        return None
    if gpu == "AMD":
        return "AMD GPU: rebuild UCX with ROCm support"
    if distro.name == _UBUNTU_22:
        return ("ubuntu22.04: linux-azure 5.15.0-1123+ triggers the UCX netlink "
                "route-dump hang, so rebuild UCX from the patched source")
    return None


def needs_hpcx_rebuild(*, rebuild_ucx: bool, internal_pmix: bool, gpu: str,
                       uses_ucx: bool) -> bool:
    """bash's REBUILD_HPCX: true unless nothing needs rebuilding.

    Only an NVIDIA UCX SKU with internal PMIx and no UCX rebuild keeps the
    vendor Open MPI. (hpcx_rebuild.sh rebuilds UCX, so a UCX rebuild always
    means an HPC-X rebuild.)
    """
    return not (not rebuild_ucx and internal_pmix and gpu == "NVIDIA" and uses_ucx)


def is_rpm_distro(distro: Distro) -> bool:
    """bash: almalinux* | rocky* | rhel* | "azurelinux3.0" (an exact match)."""
    return distro.is_rhel_family or distro.name == _AZURE_LINUX


def is_el8(distro: Distro) -> bool:
    """bash: "almalinux8.10" | "rocky8.10" | rhel8*."""
    return distro.name in _EL8_DISTROS or (distro.is_rhel and distro.major == "8")


def mvapich_skip_reason(distro: Distro, sku_family: str) -> str | None:
    """Why bash skips MVAPICH here, or None.

    bash: ((D == ubuntu24.04 || D == azurelinux3.0) && SKU_FAMILY == gb-family)
          || D == ubuntu26.04
    """
    if distro.name == _UBUNTU_26:
        return ("MVAPICH 4.1 does not build with ubuntu26.04's gcc 15 (bundled "
                "libfabric OPX/UCR providers, then the collectives source)")
    if distro.name in (_UBUNTU_24, _AZURE_LINUX) and sku_family == _GB_FAMILY:
        return (f"not supported on {distro} for {sku_family} SKUs "
                "(bash skips ubuntu24.04/azurelinux3.0 + gb-family)")
    return None


def ompi_skip_reason(distro: Distro) -> str | None:
    """Why bash skips the standalone Open MPI here (ubuntu26.04 only), or None."""
    if distro.name == _UBUNTU_26:
        return f"{distro} uses HPC-X's Open MPI only"
    return None


def impi_skip_reason(architecture: str) -> str | None:
    """Why bash skips Intel MPI here (aarch64), or None."""
    if architecture == "aarch64":
        return "Intel MPI is x86_64 only"
    return None


def impi_fi_provider(network_mode: str) -> str:
    """Intel MPI's libfabric provider: "tcp" on no_rdma, else "mlx"."""
    return "tcp" if network_mode == NETWORK_MODE_NO_RDMA else "mlx"


def hpcx_rebuild_command(hpcx_path: str, *, gpu: str, uses_ucx: bool, rebuild_ucx: bool,
                         deps_args: list[str]) -> tuple[list[str], dict[str, str]]:
    """hpcx_rebuild.sh's command line and the extra environment it runs with.

    Mirrors bash's HPCX_REBUILD_ARGS / HPCX_REBUILD_ENV, in the same order. The
    Open MPI options become one --ompi-extra-config value, joined with spaces
    as bash's "${HPCX_REBUILD_OMPI_ARGS[*]}".
    """
    args: list[str] = []
    extra_env: dict[str, str] = {}
    if rebuild_ucx:
        args.append("--rebuild-ucx")
    if gpu == "AMD":
        if rebuild_ucx:
            args += ["--ucx-extra-config", f"--with-rocm={_ROCM_PATH}"]
    elif gpu == "NVIDIA":
        args.append("--cuda")
        if rebuild_ucx:
            # Without CUDA_HOME, hpcx_rebuild.sh passes a bare --with-cuda and
            # UCX's configure misses /usr/local/cuda.
            extra_env["CUDA_HOME"] = _CUDA_HOME

    ompi_args = [*_OMPI_VENDOR_ARGS, *deps_args]
    if not uses_ucx:
        ompi_args += ["--without-ucx", f"--with-ofi={_LIBFABRIC_PATH}"]
    else:
        args.append("--with-hcoll")
        ompi_args.append(_MELLANOX_PLATFORM)
        if gpu == "AMD":
            ompi_args.append(f"--with-rocm={_ROCM_PATH}")
    args += ["--ompi-extra-config", " ".join(ompi_args)]
    return [f"{hpcx_path}/utils/hpcx_rebuild.sh", *args], extra_env


def ldd_libmpi(ldd_output: str) -> str:
    """bash's  $(ldd <ompi_info> | awk '$1 ~ /^libmpi\\.so/ {print $3}').

    awk splits each line on runs of blanks and prints the 3rd field ("" if
    none) of every line whose 1st field starts with "libmpi.so"; $(...) drops
    trailing newlines. e.g. "\\tlibmpi.so.40 => /x/lib/libmpi.so.40 (0x..)"
    -> "/x/lib/libmpi.so.40", and "libmpi.so.40 => not found" -> "not".
    """
    found = []
    for line in ldd_output.split("\n"):
        fields = re.split(r"[ \t]+", line.strip(" \t"))
        if fields[0].startswith("libmpi.so"):
            found.append(fields[2] if len(fields) > 2 else "")
    return "\n".join(found).rstrip("\n")


def mvapich_configure_command(distro: Distro, version: str, *, uses_ucx: bool,
                              ucx_path: str) -> tuple[list[str], str]:
    """MVAPICH's ./configure command and the transport lib dir it links against.

    The lib dir (UCX or libfabric) goes into the modulefile's LD_LIBRARY_PATH,
    so the runtime picks the UCX/libfabric MVAPICH was built with instead of
    the system DOCA-OFED copy.
    """
    command = ["./configure"]
    # Ubuntu's (22.04's) gfortran and Azure Linux's reject calls to one routine
    # with arguments of different types without this flag.
    if distro.is_ubuntu or distro.name == _AZURE_LINUX:
        command.append("FFLAGS=-fallow-argument-mismatch")
    command += [f"--prefix={_INSTALL_PREFIX}/mvapich-{version}",
                "--enable-g=none", "--enable-fast=yes"]
    if uses_ucx:
        command.append(f"--with-ucx={ucx_path}")
        return command, f"{ucx_path}/lib"
    # bash's unquoted "--with-device=ch4:ofi --with-libfabric=..." splits in two.
    command += ["--with-device=ch4:ofi", f"--with-libfabric={_LIBFABRIC_PATH}"]
    return command, f"{_LIBFABRIC_PATH}/lib"


def ompi_configure_command(version: str, *, uses_ucx: bool, pmix_flags: list[str],
                           ucx_path: str, hcoll_path: str,
                           ld_library_path: str) -> tuple[list[str], str]:
    """Open MPI's ./configure command and the transport lib dir (see MVAPICH)."""
    prefix = f"--prefix={_INSTALL_PREFIX}/openmpi-{version}"
    if uses_ucx:
        return ([
            "./configure",
            # A configure ARGUMENT (autoconf turns VAR=value into a variable),
            # built from the caller's LD_LIBRARY_PATH, exactly as bash does.
            f"LD_LIBRARY_PATH={ld_library_path}:{hcoll_path}/lib",
            prefix, f"--with-ucx={ucx_path}", f"--with-hcoll={hcoll_path}",
            *pmix_flags, "--enable-mpirun-prefix-by-default", _MELLANOX_PLATFORM,
        ], f"{ucx_path}/lib")
    # No UCX: drop --with-ucx, --with-hcoll (uses UCX) and the Mellanox platform file.
    return ([
        "./configure", prefix, "--without-ucx", f"--with-ofi={_LIBFABRIC_PATH}",
        *pmix_flags, "--enable-mpirun-prefix-by-default",
    ], f"{_LIBFABRIC_PATH}/lib")


# ---- Modulefile text ------------------------------------------------------
# bash builds the *_NON_UCX_EXTRAS blocks with `read -r -d '' VAR << 'EXTRAS'`,
# which drops the heredoc's final newline; the template then adds "\n" after
# ${VAR}. So non-UCX files end "<last setting>\n" and UCX files (empty VAR) end
# with a blank line. Checked against the bash-rendered fixtures.
_HPCX_NON_UCX_EXTRAS = (
    # PML cm instead of ob1 (BTL openib can't move data on MANA-only
    # hardware -> hangs); the libfabric tcp provider explicitly (auto-detection
    # trips on the docker bridge 172.17.0.1); no UCC (tl_ucp probes verbs on
    # MANA) and no hcoll (needs a Mellanox IB HCA).
    "setenv          OMPI_MCA_pml cm\n"
    "setenv          OMPI_MCA_mtl_ofi_provider_include tcp\n"
    "setenv          OMPI_MCA_coll_ucc_enable 0\n"
    "setenv          OMPI_MCA_coll_hcoll_enable 0")
_MVAPICH_NON_UCX_EXTRAS = (
    # tcp provider (MPICH4 asks for shared-AV, which tcp lacks, so
    # auto-detection picks the legacy sockets provider), and no CMA
    # (process_vm_readv fails with ptrace_scope=1 between siblings).
    "setenv          FI_PROVIDER tcp\n"
    "setenv          MPIR_CVAR_CH4_CMA_ENABLE 0")
_OMPI_NON_UCX_EXTRAS = (
    # Same PML auto-selection problem as HPC-X: pml=cm + tcp provider.
    "setenv          OMPI_MCA_pml cm\n"
    "setenv          OMPI_MCA_mtl_ofi_provider_include tcp")

# Appended to the Intel MPI modulefile on EL8 (bash's second `cat >>`).
_IMPI_EL8_APPEND = (
    "# see https://community.intel.com/t5/Intel-MPI-Library/Suspected-unfixed-"
    "Intel-MPI-race-condition-in-collectives/td-p/1693452 for Intel MPI bug\n"
    "setenv          I_MPI_STARTUP_MODE         pmi_shm\n")


def hpcx_module(hpcx_path: str, rebuilt: bool) -> str:
    """bash's HPCX_MODULE: the HPC-X module mpi/hpcx loads."""
    return f"{hpcx_path}/modulefiles/{'hpcx-rebuild' if rebuilt else 'hpcx'}"


def hpcx_modulefile(version: str, module: str, uses_ucx: bool) -> str:
    """Text of mpi/hpcx-<version> (mpi/hpcx-pmix-<version> is a symlink to it)."""
    extras = "" if uses_ucx else _HPCX_NON_UCX_EXTRAS
    return ("#%Module 1.0\n"
            "#\n"
            f"#  HPCx {version}\n"
            "#\n"
            "conflict        mpi\n"
            f"module load {module}\n"
            f"{extras}\n")


def _prefix_modulefile(title: str, home: str, ld_library_path: str, extras: str) -> str:
    """The MVAPICH/Open MPI modulefile layout (both are a /opt/<name> install)."""
    return ("#%Module 1.0\n"
            "#\n"
            f"#  {title}\n"
            "#\n"
            "conflict        mpi\n"
            f"prepend-path    PATH            {home}/bin\n"
            f"prepend-path    LD_LIBRARY_PATH {ld_library_path}\n"
            f"prepend-path    MANPATH         {home}/share/man\n"
            f"setenv          MPI_BIN         {home}/bin\n"
            f"setenv          MPI_INCLUDE     {home}/include\n"
            f"setenv          MPI_LIB         {home}/lib\n"
            f"setenv          MPI_MAN         {home}/share/man\n"
            f"setenv          MPI_HOME        {home}\n"
            f"{extras}\n")


def mvapich_modulefile(version: str, transport_lib: str, uses_ucx: bool) -> str:
    """Text of mpi/mvapich-<version>. Paths are literal /opt, as in bash."""
    home = f"/opt/mvapich-{version}"
    return _prefix_modulefile(f"MVAPICH {version}", home, f"{home}/lib:{transport_lib}",
                              "" if uses_ucx else _MVAPICH_NON_UCX_EXTRAS)


def openmpi_modulefile(version: str, hcoll_path: str, transport_lib: str,
                       uses_ucx: bool) -> str:
    """Text of mpi/openmpi-<version> (HCOLL's lib dir is listed even without UCX)."""
    home = f"/opt/openmpi-{version}"
    return _prefix_modulefile(f"OpenMPI {version}", home,
                              f"{home}/lib:{hcoll_path}/lib:{transport_lib}",
                              "" if uses_ucx else _OMPI_NON_UCX_EXTRAS)


def impi_modulefile(impi_version: str, fi_provider: str) -> str:
    """Text of mpi/impi_<2021 version> (without the EL8 append)."""
    home = f"/opt/intel/oneapi/mpi/{impi_version}"
    return ("#%Module 1.0\n"
            "#\n"
            f"#  Intel MPI {impi_version}\n"
            "#\n"
            "conflict        mpi\n"
            f"module load {home}/etc/modulefiles/impi/{impi_version}\n"
            f"setenv          MPI_BIN         {home}/bin\n"
            f"setenv          MPI_INCLUDE     {home}/include\n"
            f"setenv          MPI_LIB         {home}/lib\n"
            f"setenv          MPI_MAN         {home}/share/man\n"
            f"setenv          MPI_HOME        {home}\n"
            f"setenv          FI_PROVIDER     {fi_provider}\n")


# --------------------------------------------------------------------------- #
# The plan: every decision, made once before the system is touched.
# --------------------------------------------------------------------------- #
@dataclass
class _Plan:
    distro: Distro
    architecture: str
    gpu: str
    sku: str
    sku_family: str
    node_type: str
    network_mode: str
    uses_ucx: bool
    module_dir: str
    component_dir: str
    hpcx_component: str
    hpcx_reason: str
    internal_pmix_reason: str | None
    rebuild_ucx_reason: str | None
    mvapich_skip: str | None
    ompi_skip: str | None
    impi_skip: str | None
    # From versions.json (_resolve_configs); None for a skipped MPI.
    hpcx: dict | None = None
    pmix_version: str | None = None
    pmix_path: str | None = None
    mvapich: dict | None = None
    ompi: dict | None = None
    impi: dict | None = None
    impi_2021_version: str | None = None

    @property
    def internal_pmix(self) -> bool:
        return self.internal_pmix_reason is not None

    @property
    def rebuild_ucx(self) -> bool:
        return self.rebuild_ucx_reason is not None

    @property
    def rebuild_hpcx(self) -> bool:
        return needs_hpcx_rebuild(rebuild_ucx=self.rebuild_ucx,
                                  internal_pmix=self.internal_pmix, gpu=self.gpu,
                                  uses_ucx=self.uses_ucx)

    @property
    def rpm(self) -> bool:
        return is_rpm_distro(self.distro)

    @property
    def pkgconfig_dirs(self) -> tuple[str, ...]:
        return _PKGCONFIG_DIRS_RPM if self.rpm else _PKGCONFIG_DIRS

    @property
    def mpi_dir(self) -> str:
        """bash's MPI_MODULE_FILES_DIRECTORY."""
        return f"{self.module_dir}/mpi"


@dataclass(frozen=True)
class _Hpcx:
    """Where HPC-X's parts are (bash's HPCX_PATH and UCX_PATH)."""
    path: str
    ucx: str

    @classmethod
    def at(cls, path: str, *, rebuilt_ucx: bool) -> _Hpcx:
        # bash switches UCX_PATH to the rebuilt UCX before hpcx_rebuild.sh runs,
        # and everything after it (ld.so.conf, pkg-config, MPI builds) uses it.
        return cls(path, f"{path}/ucx/hpcx-rebuild" if rebuilt_ucx else f"{path}/ucx")

    @property
    def hcoll(self) -> str:
        return f"{self.path}/hcoll"

    @property
    def sharp(self) -> str:
        return f"{self.path}/sharp"


@dataclass
class _Report:
    """What the step changed, for the closing summary."""
    recorded: dict = field(default_factory=dict)
    symlinks: list = field(default_factory=list)
    pinned: list = field(default_factory=list)
    patched: list = field(default_factory=list)


def _decide(env: dict[str, str]) -> _Plan:
    """Every bash decision that depends on the machine, not on versions.json."""
    distro = Distro.from_env(env)
    gpu, sku = env.get("GPU", ""), env.get("SKU", "")
    architecture, sku_family = env.get("ARCHITECTURE", ""), env.get("SKU_FAMILY", "")
    node_type = env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE
    network_mode = sku_network_mode(sku, node_type)
    uses_ucx = sku_uses_ucx(sku)
    hpcx_component, hpcx_reason = hpcx_variant(gpu, network_mode)
    return _Plan(
        distro=distro, architecture=architecture, gpu=gpu, sku=sku, sku_family=sku_family,
        node_type=node_type, network_mode=network_mode, uses_ucx=uses_ucx,
        module_dir=env.get("MODULE_FILES_DIRECTORY", ""),
        component_dir=env.get("COMPONENT_DIR", ""),
        hpcx_component=hpcx_component, hpcx_reason=hpcx_reason,
        internal_pmix_reason=internal_pmix_reason(distro, node_type),
        rebuild_ucx_reason=rebuild_ucx_reason(gpu, distro, uses_ucx),
        mvapich_skip=mvapich_skip_reason(distro, sku_family),
        ompi_skip=ompi_skip_reason(distro),
        impi_skip=impi_skip_reason(architecture))


def _download_config(component: str, env: dict[str, str]) -> dict | None:
    """require_config() for version/url/sha256, as strings."""
    cfg = require_config(component, env, _OP, "version", "url", "sha256")
    if cfg is None:
        return None
    return {key: str(cfg[key]) for key in ("version", "url", "sha256")}


def _resolve_pmix_path(plan: _Plan, env: dict[str, str]) -> bool:
    """bash: PMIX_PATH=${INSTALL_PREFIX}/pmix/${PMIX_VERSION:0:-2}."""
    cfg = require_config("pmix", env, _OP, "version")
    if cfg is None:
        return False
    plan.pmix_version = str(cfg["version"])
    short = drop_last_two(plan.pmix_version)
    if short is None:
        log_error(_OP, f"pmix version {plan.pmix_version!r} is too short for "
                       "${PMIX_VERSION:0:-2}", component="pmix", version=plan.pmix_version)
        return False
    plan.pmix_path = f"{_INSTALL_PREFIX}/pmix/{short}"
    return True


def _resolve_impi(plan: _Plan, env: dict[str, str]) -> bool:
    """The impi config and bash's impi_2021_version=${IMPI_VERSION:0:-2}."""
    plan.impi = _download_config("impi", env)
    if plan.impi is None:
        return False
    plan.impi_2021_version = drop_last_two(plan.impi["version"])
    if plan.impi_2021_version is None:
        log_error(_OP, f"impi version {plan.impi['version']!r} is too short for "
                       "${IMPI_VERSION:0:-2}", component="impi", version=plan.impi["version"])
        return False
    return True


def _resolve_configs(plan: _Plan, env: dict[str, str]) -> bool:
    """Read the configs in bash's order, skipping those of skipped MPIs."""
    plan.hpcx = _download_config(plan.hpcx_component, env)
    if plan.hpcx is None:
        return False
    if not plan.internal_pmix and not _resolve_pmix_path(plan, env):
        return False
    if plan.mvapich_skip is None:
        plan.mvapich = _download_config("mvapich", env)
        if plan.mvapich is None:
            return False
    if plan.ompi_skip is None:
        plan.ompi = _download_config("ompi", env)
        if plan.ompi is None:
            return False
    return plan.impi_skip is not None or _resolve_impi(plan, env)


def _plan(env: dict[str, str]) -> _Plan | None:
    """The run's decisions and configs, or None (logged) if they are unusable."""
    if not env.get("MODULE_FILES_DIRECTORY"):
        log_error(_OP, "MODULE_FILES_DIRECTORY is not set")
        return None
    plan = _decide(env)
    return plan if _resolve_configs(plan, env) else None


def _log_plan(plan: _Plan) -> None:
    def version(cfg):
        return cfg["version"] if cfg else "skipped"

    log_info(_OP, f"Installing MPIs on {plan.distro} ({plan.architecture}) for "
                  f"{plan.gpu} {plan.sku}",
             distro=str(plan.distro), arch=plan.architecture, gpu=plan.gpu, sku=plan.sku,
             sku_family=plan.sku_family, node_type=plan.node_type,
             network_mode=plan.network_mode, uses_ucx=plan.uses_ucx,
             hpcx_component=plan.hpcx_component, hpcx_version=plan.hpcx["version"],
             internal_pmix=plan.internal_pmix, pmix_version=plan.pmix_version or "internal",
             pmix_path=plan.pmix_path or "internal", rebuild_ucx=plan.rebuild_ucx,
             rebuild_hpcx=plan.rebuild_hpcx, mvapich_version=version(plan.mvapich),
             ompi_version=version(plan.ompi), impi_version=version(plan.impi),
             module_dir=plan.module_dir)
    log_info(_OP, f"HPC-X variant {plan.hpcx_component}: {plan.hpcx_reason}",
             hpcx_component=plan.hpcx_component, gpu=plan.gpu, network_mode=plan.network_mode)
    if plan.internal_pmix:
        log_info(_OP, "Open MPI uses its internal PMIx, hwloc and libevent: "
                      f"{plan.internal_pmix_reason}", internal_pmix=True,
                 distro=str(plan.distro), node_type=plan.node_type)
    else:
        log_info(_OP, f"PMIx path for the MPI builds: {plan.pmix_path}", internal_pmix=False,
                 pmix_path=plan.pmix_path, pmix_version=plan.pmix_version)
    log_info(_OP, f"HPC-X rebuild: {'yes' if plan.rebuild_hpcx else 'no'}, UCX rebuild: "
                  f"{plan.rebuild_ucx_reason or 'no'}", rebuild_hpcx=plan.rebuild_hpcx,
             rebuild_ucx=plan.rebuild_ucx)
    # BASH ISSUE (kept): bash's `set` replaces the positional parameters ($1),
    # so the builds use the environment's default compiler; we don't set CC either.
    log_warn(_OP, "bash's `set CC=/usr/bin/gcc` / `set GCC=/usr/bin/gcc` do not set "
                  "CC or GCC (`set` replaces the positional parameters); the builds "
                  "use the default compiler, as in bash", bash_line="set CC=/usr/bin/gcc")


# --------------------------------------------------------------------------- #
# Small steps shared by the phases. Each logs its own failure.
# --------------------------------------------------------------------------- #
def _fetch(cfg: dict) -> bool:
    """bash's download_and_verify of cfg's url into the work dir."""
    return fetch_verified(cfg["url"], cfg["sha256"], _OP, _WORK_DIR) is not None


def _build_from_source(env: dict[str, str], cfg: dict, configure: list[str],
                       nproc: int) -> bool:
    """bash: download; tar -xvf <tarball>; cd <folder>; configure; make; make install."""
    url = cfg["url"]
    source = f"{_WORK_DIR}/{url_basename(url, '.tar.gz')}"
    if not (_fetch(cfg) and extract_tarball(url_basename(url), _OP, cwd=_WORK_DIR, env=env)):
        return False
    if not os.path.isdir(source):
        log_error(_OP, f"source folder {source} was not created by tar", path=source)
        return False
    return run_all([configure, ["make", f"-j{nproc}"], ["make", "install"]], _OP,
                   cwd=source, env=env)


def _record(component: str, version: str, report: _Report) -> bool:
    if not record_version(component, version, _OP):
        return False
    report.recorded[component] = version
    return True


def _pin(packages: list[str], env: dict[str, str], report: _Report) -> bool:
    """bash's dnf_pin_packages (a failure stops the script under set -e)."""
    if not dnf_pin_packages(packages, _OP, env):
        return False
    report.pinned.extend(packages)
    return True


def _skipped(mpi: str, reason: str, **fields) -> bool:
    log_info(_OP, f"Skipping {mpi}: {reason}", skipped=True, **fields)
    return True


def _append_modulefile(path: str, text: str, *, rerun_check: bool = True) -> bool:
    """bash `cat << EOF >> path`: append, creating the file if needed.

    rerun_check=False marks a deliberate second append in the same run (the
    EL8 Intel MPI lines), where the file is expected to exist.
    """
    if rerun_check and os.path.lexists(path):
        # BASH ISSUE (kept): `>>` appends, so a rerun adds a second copy of the
        # modulefile to the same file instead of replacing it.
        log_warn(_OP, f"{path} already exists; appending another copy, as bash's "
                      "`cat >>` does (not rerun-safe: the file gets duplicate settings)",
                 path=path)
    log_debug(_OP, f"{path} content appended:\n{text}")
    return write_file(path, text, _OP, append=True)


def _link_path(target: str, link: str) -> str:
    """Where GNU ln (without -n) puts the link: inside `link` if it's a directory."""
    if os.path.isdir(link):
        return os.path.join(link, os.path.basename(target))
    return link


def _ln_s(target: str, link: str, report: _Report, *, force: bool = False) -> bool:
    """bash `ln -s target link` (`ln -sf` with force=True)."""
    link = _link_path(target, link)
    if not force and os.path.lexists(link):
        # BASH ISSUE (kept): plain `ln -s` (no -f) fails when the link exists,
        # which stops the bash script on any rerun over an existing image.
        log_error(_OP, f"cannot create symlink {link}: it already exists. bash's "
                       "`ln -s` (without -f) fails here too, so install_mpis is not "
                       "rerun-safe; remove the old MPI modulefiles first",
                  path=link, target=target)
        return False
    if not symlink(target, link, _OP, force=force):
        return False
    report.symlinks.append(link)
    return True


def _tail_lines(path: str, count: int) -> list[str]:
    """`tail -n <count> path` as a list of lines (without their newlines)."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in deque(handle, maxlen=count)]


# --------------------------------------------------------------------------- #
# Phase 1: HPC-X.
# --------------------------------------------------------------------------- #
def _install_hpcx(env: dict[str, str], plan: _Plan, report: _Report) -> _Hpcx | None:
    """Unpack HPC-X, rebuild it if needed, register its libraries, write its modulefiles."""
    hpcx = _unpack_hpcx(env, plan, report)
    if hpcx is None:
        return None
    if plan.rebuild_hpcx:
        if not _rebuild_hpcx(env, plan, hpcx, report):
            return None
    else:
        log_info(_OP, "Not rebuilding HPC-X: NVIDIA UCX SKU with internal PMIx and no UCX "
                      "rebuild, so the vendor Open MPI is used as shipped",
                 phase="hpcx-rebuild", rebuild_hpcx=False)
    if plan.internal_pmix and not _record_internal_pmix(env, plan, hpcx, report):
        return None
    # hpcx_rebuild.sh installs fresh Open MPI (and, on AMD, UCX) metadata; fix it too.
    if not (_fix_ladir(env, hpcx) and _register_hpcx_libs(env, hpcx)
            and _link_pkgconfig(plan, hpcx, report)):
        return None
    if plan.rpm and not _pin(["ucx*"], env, report):        # exclude ucx from updates
        return None
    if not _write_hpcx_modulefiles(plan, hpcx, report):
        return None
    log_info(_OP, f"HPC-X {plan.hpcx['version']} installed at {hpcx.path}", phase="hpcx",
             version=plan.hpcx["version"], path=hpcx.path, ucx_path=hpcx.ucx,
             rebuilt=plan.rebuild_hpcx, module=hpcx_module(hpcx.path, plan.rebuild_hpcx))
    return hpcx


def _unpack_hpcx(env: dict[str, str], plan: _Plan, report: _Report) -> _Hpcx | None:
    """bash: download_and_verify; tar; mv <folder> /opt; hpcx_fix_ladir.sh; record HPCX."""
    cfg = plan.hpcx
    folder = url_basename(cfg["url"], ".tbz")
    hpcx = _Hpcx.at(f"{_INSTALL_PREFIX}/{folder}", rebuilt_ucx=plan.rebuild_ucx)
    log_info(_OP, f"HPC-X phase: installing {plan.hpcx_component} {cfg['version']} into "
                  f"{hpcx.path}", phase="hpcx", component=plan.hpcx_component,
             version=cfg["version"], url=cfg["url"], path=hpcx.path)
    # The relocated tree's .la/.pc metadata is fixed before the rebuild uses
    # the bundled UCX, HCOLL and SHARP.
    if not (_fetch(cfg)
            and extract_tarball(url_basename(cfg["url"]), _OP, cwd=_WORK_DIR, env=env)
            and run(["mv", folder, _INSTALL_PREFIX], _OP, cwd=_WORK_DIR, env=env)
            and _fix_ladir(env, hpcx)
            and _record("HPCX", cfg["version"], report)):
        return None
    return hpcx


def _fix_ladir(env: dict[str, str], hpcx: _Hpcx) -> bool:
    """bash: HPCX_DIR=${HPCX_PATH} ${HPCX_PATH}/utils/hpcx_fix_ladir.sh

    (HPCX_DIR is set for that one command only, hence the copied env.)
    """
    return run([f"{hpcx.path}/utils/hpcx_fix_ladir.sh"], _OP, cwd=_WORK_DIR,
               env={**env, "HPCX_DIR": hpcx.path})


def _rebuild_hpcx(env: dict[str, str], plan: _Plan, hpcx: _Hpcx, report: _Report) -> bool:
    """bash's `if [[ "$REBUILD_HPCX" == true ]]` block, then its checks and fixes."""
    if plan.rebuild_ucx:
        if not _patch_ucx_source_netlink_dump(env, plan, hpcx, report):
            return False
        log_info(_OP, f"Rebuilding UCX: {plan.rebuild_ucx_reason}; using the rebuilt UCX "
                      f"at {hpcx.ucx} from here on", ucx_path=hpcx.ucx)
    if plan.gpu == "AMD" and not os.path.isdir(_ROCM_DIR):
        log_error(_OP, "ROCm must be installed before rebuilding HPC-X UCX with "
                       "ROCm support.", path=_ROCM_DIR)
        return False
    return (_run_hpcx_rebuild(env, plan, hpcx)
            and (not plan.rebuild_ucx or _check_rebuilt_ucx(env, hpcx))
            and _check_rebuilt_libmpi(env, hpcx)
            and _fix_rebuild_entrypoints(env, hpcx)
            and run(["cp", "-r", f"{hpcx.path}/ompi/tests", f"{hpcx.path}/hpcx-rebuild"],
                    _OP, cwd=_WORK_DIR, env=env))


def _run_hpcx_rebuild(env: dict[str, str], plan: _Plan, hpcx: _Hpcx) -> bool:
    """Run hpcx_rebuild.sh; on failure show the tails of its logs."""
    command, extra_env = hpcx_rebuild_command(
        hpcx.path, gpu=plan.gpu, uses_ucx=plan.uses_ucx, rebuild_ucx=plan.rebuild_ucx,
        deps_args=ompi_deps_args(plan.internal_pmix, plan.pmix_path))
    log_info(_OP, "Rebuilding HPC-X Open MPI"
                  f"{' and UCX' if plan.rebuild_ucx else ''} with hpcx_rebuild.sh",
             phase="hpcx-rebuild", gpu=plan.gpu, uses_ucx=plan.uses_ucx,
             rebuild_ucx=plan.rebuild_ucx, internal_pmix=plan.internal_pmix,
             rebuild_args=" ".join(command[1:]),
             rebuild_env=" ".join(f"{k}={v}" for k, v in extra_env.items()))
    if run(command, _OP, cwd=_WORK_DIR, env={**env, **extra_env}):
        return True
    _log_rebuild_logs(hpcx.path)
    log_error(_OP, "hpcx_rebuild.sh failed; its UCX/Open MPI log tails are above",
              command=command[0])
    return False


def _log_rebuild_logs(hpcx_path: str) -> None:
    """Show the tail of hpcx_rebuild.sh's logs, which stay on the build VM."""
    patterns = (f"{hpcx_path}/sources/ucx-*/ucx-*.log",
                f"{hpcx_path}/sources/openmpi-gitclone/ompi-*.log")
    shown = 0
    for pattern in patterns:
        for log_file in sorted(glob.glob(pattern)):
            if not os.path.isfile(log_file):        # bash: [[ -f ]] || continue
                continue
            try:
                lines = _tail_lines(log_file, _REBUILD_LOG_TAIL)
            except OSError as exc:
                log_warn(_OP, f"could not read {log_file}: {exc}", path=log_file)
                continue
            log_error_detail(_OP, f"===== tail of {log_file}", "\n".join(lines) or "(empty)",
                             path=log_file, lines=len(lines))
            shown += 1
    if not shown:
        log_warn(_OP, "no hpcx_rebuild.sh UCX/Open MPI logs found to show",
                 patterns=list(patterns))


def _patch_ucx_source_netlink_dump(env: dict[str, str], plan: _Plan, hpcx: _Hpcx,
                                   report: _Report) -> bool:
    """bash's patch_hpcx_ucx_source_netlink_dump.

    Patches HPC-X's bundled UCX source tarball so `hpcx_rebuild.sh
    --rebuild-ucx` builds a UCX that doesn't hang in ucs_netlink_send_request()
    on kernels that append NLMSG_DONE to the last route-dump datagram
    (linux-azure 5.15.0-1123+; openucx/ucx#12070). Only UCX versions with the
    buggy line are patched; the tarball is repacked in place.
    """
    tarball = _find_ucx_tarball(hpcx)
    if tarball is None:
        return False
    work_dir = make_temp_dir(_OP)
    if work_dir is None:
        return False
    try:
        return _patch_ucx_tarball(env, plan, tarball, str(work_dir), report)
    finally:
        remove_paths(work_dir, op=_OP)


def _find_ucx_tarball(hpcx: _Hpcx) -> str | None:
    """bash: find ${HPCX_PATH}/sources -maxdepth 1 -name 'ucx-*.tar.gz' | head -n1"""
    # Sorted, so the choice is stable; HPC-X ships one UCX tarball.
    tarballs = sorted(glob.glob(f"{hpcx.path}/sources/ucx-*.tar.gz"))
    if not tarballs:
        log_error(_OP, f"HPC-X UCX source tarball not found under {hpcx.path}/sources",
                  path=f"{hpcx.path}/sources")
        return None
    if len(tarballs) > 1:
        log_warn(_OP, f"{len(tarballs)} UCX source tarballs in HPC-X; patching the "
                      f"first, {tarballs[0]}", tarballs=tarballs)
    return tarballs[0]


def _patch_ucx_tarball(env: dict[str, str], plan: _Plan, tarball: str, work_dir: str,
                       report: _Report) -> bool:
    """Unpack `tarball` into work_dir; if its UCX has the hang, patch and repack it."""
    ucx_name = url_basename(tarball, ".tar.gz")
    ucx_src = f"{work_dir}/{ucx_name}"
    if not extract_tarball(tarball, _OP, dest=work_dir, cwd=_WORK_DIR, env=env):
        return False
    buggy = _has_netlink_hang(ucx_src, tarball)
    if buggy is None:
        return False
    if not buggy:
        log_info(_OP, f"{ucx_name} does not have the netlink dump hang; skipping "
                      "UCX patch", ucx=ucx_name, tarball=tarball, patched=False)
        return True
    patch_file = os.path.join(plan.component_dir, _UCX_NETLINK_PATCH)
    if not os.path.isfile(patch_file):
        log_error(_OP, f"UCX patch {patch_file} not found (is COMPONENT_DIR set?)",
                  path=patch_file)
        return False
    if not run_all([["patch", "-d", ucx_src, "-p1", "-i", patch_file],
                    ["tar", "-czf", tarball, "-C", work_dir, ucx_name]],
                   _OP, cwd=_WORK_DIR, env=env):
        return False
    report.patched.append(tarball)
    log_info(_OP, f"Patched {ucx_name} for the netlink route-dump hang and "
                  f"repacked {tarball} (replaced)", ucx=ucx_name, tarball=tarball,
             patch=patch_file, patched=True, action="replaced")
    return True


def _has_netlink_hang(ucx_src: str, tarball: str) -> bool | None:
    """bash's grep -qF <buggy line> in netlink.c; None (logged) if it can't be read."""
    source = f"{ucx_src}/{_UCX_NETLINK_SOURCE}"
    try:
        with open(source, encoding="utf-8", errors="replace") as handle:
            return _UCX_NETLINK_BUGGY_LINE in handle.read()
    except OSError as exc:
        # bash's grep would fail and rebuild UCX unpatched; we can't tell
        # whether this UCX hangs, so we stop.
        log_error(_OP, f"cannot read {_UCX_NETLINK_SOURCE} in {tarball} ({exc}); "
                       "can't tell whether this UCX has the netlink route-dump "
                       "hang (bash would rebuild it unpatched)",
                  path=source, tarball=tarball)
        return None


def _check_rebuilt_ucx(env: dict[str, str], hpcx: _Hpcx) -> bool:
    """bash: LD_LIBRARY_PATH=${UCX_PATH}/lib ${UCX_PATH}/bin/ucx_info -v > /dev/null.

    hpcx_rebuild.sh reports, but doesn't fail on, UCX `make install` errors.
    """
    rc, output = run_capture([f"{hpcx.ucx}/bin/ucx_info", "-v"], _OP, cwd=_WORK_DIR,
                             env={**env, "LD_LIBRARY_PATH": f"{hpcx.ucx}/lib"})
    if rc != 0:
        log_error(_OP, f"Rebuilt HPC-X UCX at {hpcx.ucx} is not usable.",
                  ucx_path=hpcx.ucx, exit_code=rc)
        return False
    version = next((line.strip() for line in output.splitlines() if line.strip()), "")
    log_info(_OP, f"Rebuilt UCX at {hpcx.ucx} runs: {version}", ucx_path=hpcx.ucx,
             ucx_info=version)
    return True


def _check_rebuilt_libmpi(env: dict[str, str], hpcx: _Hpcx) -> bool:
    """The rebuilt ompi_info must load the rebuilt libmpi.

    A libmpi outside hpcx-rebuild means the vendor Open MPI shadows the
    rebuild, which otherwise fails silently at runtime. bash ignores ldd's exit
    code (no pipefail); only what awk prints decides.
    """
    ompi_info = f"{hpcx.path}/hpcx-rebuild/bin/ompi_info"
    expected = f"{hpcx.path}/hpcx-rebuild/lib/"
    rc, output = run_capture(["ldd", ompi_info], _OP, cwd=_WORK_DIR, env=env,
                             failure_is_error=False)
    libmpi = ldd_libmpi(output)
    if not libmpi.startswith(expected):
        log_error(_OP, f"Rebuilt HPC-X ompi_info loads {libmpi or '<unresolved>'} "
                       "instead of the rebuilt libmpi.",
                  ompi_info=ompi_info, libmpi=libmpi, expected_prefix=expected,
                  ldd_exit_code=rc)
        return False
    log_info(_OP, f"Rebuilt ompi_info loads {libmpi}", ompi_info=ompi_info, libmpi=libmpi)
    return True


def _fix_rebuild_entrypoints(env: dict[str, str], hpcx: _Hpcx) -> bool:
    """Point hpcx_rebuild.sh's module/init files at hpcx-rebuild.

    hpcx_rebuild.sh installs into hpcx-rebuild but writes files that compose
    the MPI prefix as hpcx-rebuild$mpi_version, a directory it never creates.
    Only that suffix is dropped (mpi_version is still used for ompi5/tests).
    """
    for entrypoint in (f"{hpcx.path}/modulefiles/hpcx-rebuild", f"{hpcx.path}/hpcx-rebuild.sh"):
        if not os.path.exists(entrypoint):      # bash: [[ -e ]] || continue
            log_info(_OP, f"{entrypoint} does not exist; nothing to fix", path=entrypoint)
            continue
        if not run(["sed", "-i", "--follow-symlinks", "-E", *_REBUILD_ENTRYPOINT_SED,
                    entrypoint], _OP, cwd=_WORK_DIR, env=env):
            return False
        log_info(_OP, f"Fixed the hpcx-rebuild prefix in {entrypoint} (edited in place)",
                 path=entrypoint, action="edited")
    return True


def _record_internal_pmix(env: dict[str, str], plan: _Plan, hpcx: _Hpcx,
                          report: _Report) -> bool:
    """Record the PMIx Open MPI ships when it uses its internal copy.

    The rebuild installs its own PMIx; otherwise the vendor ompi5 tree is used.
    """
    sub = "hpcx-rebuild" if plan.rebuild_hpcx else "ompi5"
    pkg_config_path = f"{hpcx.path}/{sub}/lib/pkgconfig"
    rc, output = run_capture(["pkg-config", "--modversion", "pmix"], _OP, cwd=_WORK_DIR,
                             env={**env, "PKG_CONFIG_PATH": pkg_config_path})
    version = output.rstrip("\n")       # bash's $(...) drops trailing newlines
    if rc != 0 or not version.strip():
        log_error(_OP, "could not read the internal PMIx version with pkg-config from "
                       f"{pkg_config_path}", pkg_config_path=pkg_config_path, exit_code=rc,
                  output=version)
        return False
    log_info(_OP, f"Internal PMIx {version} (from {pkg_config_path})",
             pkg_config_path=pkg_config_path, pmix_version=version)
    return _record("PMIX", version, report)


def _register_hpcx_libs(env: dict[str, str], hpcx: _Hpcx) -> bool:
    """bash: cat > /etc/ld.so.conf.d/hpcx-{sharp,ucx}.conf; ldconfig."""
    return (write_file(f"{_LD_SO_CONF_DIR}/hpcx-sharp.conf", f"{hpcx.sharp}/lib\n", _OP)
            and write_file(f"{_LD_SO_CONF_DIR}/hpcx-ucx.conf", f"{hpcx.ucx}/lib\n", _OP)
            and run(["ldconfig"], _OP, cwd=_WORK_DIR, env=env))


def _hpcx_pc_files(hpcx: _Hpcx) -> list[str]:
    """The HCOLL, SHARP and UCX .pc files (bash: for f in <tree>/lib/pkgconfig/*.pc;
    [[ -f ]] || continue). sorted() approximates bash's glob order; only the
    order the links are made in can differ, not the result."""
    return [pc_file for tree in (hpcx.hcoll, hpcx.sharp, hpcx.ucx)
            for pc_file in sorted(glob.glob(f"{tree}/lib/pkgconfig/*.pc"))
            if os.path.isfile(pc_file)]


def _link_pkgconfig(plan: _Plan, hpcx: _Hpcx, report: _Report) -> bool:
    """Symlink HCOLL/SHARP/UCX .pc files into the system pkg-config dirs.

    So module-free builds resolve the same HCOLL, SHARP and UCX that
    `module load mpi/hpcx` would put on PKG_CONFIG_PATH.
    """
    pc_files = _hpcx_pc_files(hpcx)
    for pkgconfig_dir in plan.pkgconfig_dirs:
        if not (make_dirs(pkgconfig_dir, op=_OP)
                and all(_ln_s(pc_file, f"{pkgconfig_dir}/{os.path.basename(pc_file)}",
                              report, force=True) for pc_file in pc_files)):
            return False
        log_info(_OP, f"Linked {len(pc_files)} HPC-X pkg-config file(s) into "
                      f"{pkgconfig_dir}", path=pkgconfig_dir, count=len(pc_files))
    return True


def _write_hpcx_modulefiles(plan: _Plan, hpcx: _Hpcx, report: _Report) -> bool:
    """mpi/hpcx-<v>, plus the hpcx-pmix-<v>, hpcx and hpcx-pmix symlinks."""
    mpi_dir, version = plan.mpi_dir, plan.hpcx["version"]
    module = hpcx_module(hpcx.path, plan.rebuild_hpcx)
    log_info(_OP, f"mpi/hpcx loads {module}", module=module, rebuilt=plan.rebuild_hpcx,
             uses_ucx=plan.uses_ucx)
    if not plan.uses_ucx:
        log_info(_OP, "Non-UCX SKU: adding PML cm / libfabric tcp settings and disabling "
                      "UCC and hcoll in the HPC-X modulefile")
    return (_append_modulefile(f"{mpi_dir}/hpcx-{version}",
                               hpcx_modulefile(version, module, plan.uses_ucx))
            and _ln_s(f"{mpi_dir}/hpcx-{version}", f"{mpi_dir}/hpcx-pmix-{version}", report)
            and _ln_s(f"{mpi_dir}/hpcx-{version}", f"{mpi_dir}/hpcx", report)
            and _ln_s(f"{mpi_dir}/hpcx-pmix-{version}", f"{mpi_dir}/hpcx-pmix", report))


# --------------------------------------------------------------------------- #
# Phases 2-4: MVAPICH, Open MPI, Intel MPI.
# --------------------------------------------------------------------------- #
def _install_mvapich(env: dict[str, str], plan: _Plan, hpcx: _Hpcx, nproc: int,
                     report: _Report) -> bool:
    """Build MVAPICH into /opt/mvapich-<version>, then its modulefile."""
    if plan.mvapich is None:
        return _skipped("MVAPICH", plan.mvapich_skip, phase="mvapich",
                        distro=str(plan.distro), sku_family=plan.sku_family)
    version = plan.mvapich["version"]
    configure, transport_lib = mvapich_configure_command(
        plan.distro, version, uses_ucx=plan.uses_ucx, ucx_path=hpcx.ucx)
    log_info(_OP, f"MVAPICH phase: building MVAPICH {version} "
                  f"({'UCX' if plan.uses_ucx else 'OFI/libfabric'} transport)",
             phase="mvapich", version=version, url=plan.mvapich["url"],
             transport_lib=transport_lib)
    if not _build_from_source(env, plan.mvapich, configure, nproc):
        return False
    return (_record("MVAPICH", version, report)
            and _write_mvapich_modulefile(plan, version, transport_lib, report))


def _write_mvapich_modulefile(plan: _Plan, version: str, transport_lib: str,
                              report: _Report) -> bool:
    if not plan.uses_ucx:
        log_info(_OP, "Non-UCX SKU: adding FI_PROVIDER=tcp and disabling CMA in the "
                      "MVAPICH modulefile")
    path = f"{plan.mpi_dir}/mvapich-{version}"
    return (_append_modulefile(path, mvapich_modulefile(version, transport_lib, plan.uses_ucx))
            and _ln_s(path, f"{plan.mpi_dir}/mvapich", report))


def _install_ompi(env: dict[str, str], plan: _Plan, hpcx: _Hpcx, nproc: int,
                  report: _Report) -> bool:
    """Build Open MPI into /opt/openmpi-<version>, then its modulefile."""
    if plan.ompi is None:
        return _skipped("the standalone Open MPI", plan.ompi_skip, phase="ompi",
                        distro=str(plan.distro))
    version = plan.ompi["version"]
    pmix_flags = ompi_pmix_flags(plan.internal_pmix, plan.distro, plan.node_type,
                                 plan.pmix_path)
    configure, transport_lib = ompi_configure_command(
        version, uses_ucx=plan.uses_ucx, pmix_flags=pmix_flags, ucx_path=hpcx.ucx,
        hcoll_path=hpcx.hcoll, ld_library_path=env.get("LD_LIBRARY_PATH", ""))
    log_info(_OP, f"Open MPI phase: building Open MPI {version} "
                  f"({'UCX + HCOLL' if plan.uses_ucx else 'OFI/libfabric'} transport)",
             phase="ompi", version=version, url=plan.ompi["url"],
             pmix_flag=" ".join(pmix_flags), transport_lib=transport_lib)
    if not _build_from_source(env, plan.ompi, configure, nproc):
        return False
    return (_record("OMPI", version, report)
            and _write_ompi_modulefile(plan, version, hpcx, transport_lib, report))


def _write_ompi_modulefile(plan: _Plan, version: str, hpcx: _Hpcx, transport_lib: str,
                           report: _Report) -> bool:
    if not plan.uses_ucx:
        log_info(_OP, "Non-UCX SKU: adding PML cm / libfabric tcp settings to the "
                      "Open MPI modulefile")
    path = f"{plan.mpi_dir}/openmpi-{version}"
    text = openmpi_modulefile(version, hpcx.hcoll, transport_lib, plan.uses_ucx)
    return _append_modulefile(path, text) and _ln_s(path, f"{plan.mpi_dir}/openmpi", report)


def _install_impi(env: dict[str, str], plan: _Plan, report: _Report) -> bool:
    """Install Intel MPI with its offline installer, then its modulefile."""
    if plan.impi is None:
        return _skipped("Intel MPI", plan.impi_skip, phase="impi", arch=plan.architecture)
    cfg = plan.impi
    modulefiles = (f"{_INSTALL_PREFIX}/intel/oneapi/mpi/{plan.impi_2021_version}"
                   "/etc/modulefiles")
    log_info(_OP, f"Intel MPI phase: installing Intel MPI {cfg['version']}", phase="impi",
             version=cfg["version"], url=cfg["url"],
             impi_2021_version=plan.impi_2021_version)
    if not (_fetch(cfg)
            and run(["bash", url_basename(cfg["url"]), "-s", "-a", "-s", "--eula", "accept"],
                    _OP, cwd=_WORK_DIR, env=env)
            and run(["mv", f"{modulefiles}/mpi", f"{modulefiles}/impi"], _OP,
                    cwd=_WORK_DIR, env=env)
            and _record("IMPI", cfg["version"], report)):
        return False
    return _write_impi_modulefile(plan, report)


def _write_impi_modulefile(plan: _Plan, report: _Report) -> bool:
    provider = impi_fi_provider(plan.network_mode)
    log_info(_OP, f"Intel MPI modulefile uses FI_PROVIDER={provider}",
             fi_provider=provider, network_mode=plan.network_mode)
    path = f"{plan.mpi_dir}/impi_{plan.impi_2021_version}"
    if not (_append_modulefile(path, impi_modulefile(plan.impi_2021_version, provider))
            and _ln_s(path, f"{plan.mpi_dir}/impi-2021", report)):
        return False
    if not is_el8(plan.distro):
        return True
    log_info(_OP, "EL8: adding I_MPI_STARTUP_MODE=pmi_shm to the Intel MPI modulefile "
                  "(Intel MPI collectives race condition)", distro=str(plan.distro))
    return _append_modulefile(path, _IMPI_EL8_APPEND, rerun_check=False)


# --------------------------------------------------------------------------- #
# The step.
# --------------------------------------------------------------------------- #
def _prepare_dirs(plan: _Plan) -> bool:
    """bash: mkdir -p ${MPI_MODULE_FILES_DIRECTORY}; plus our empty work dir."""
    if not make_dirs(plan.mpi_dir, op=_OP):
        return False
    log_info(_OP, f"MPI modulefiles go to {plan.mpi_dir}", path=plan.mpi_dir)
    return fresh_dir(_WORK_DIR, _OP)


def _install_all(env: dict[str, str], plan: _Plan, nproc: int,
                 report: _Report) -> _Hpcx | None:
    """bash's phases in order; None at the first failure."""
    hpcx = _install_hpcx(env, plan, report) if _prepare_dirs(plan) else None
    ok = (hpcx is not None
          and _install_mvapich(env, plan, hpcx, nproc, report)
          and _install_ompi(env, plan, hpcx, nproc, report)
          # exclude openmpi, perftest from updates
          and (not plan.rpm or _pin(["openmpi", "perftest"], env, report))
          and _install_impi(env, plan, report))
    return hpcx if ok else None


def _log_summary(plan: _Plan, hpcx: _Hpcx, report: _Report) -> None:
    # BASH ISSUE (not ported): on success bash finishes with `rm -rf -- */` in
    # the distro build dir, deleting every subdirectory there, not just its own.
    log_warn(_OP, "bash ends with `rm -rf -- */` in the distro build dir, deleting "
                  "every subdirectory there; the Python port only removed its own "
                  f"work dir {_WORK_DIR}", work_dir=_WORK_DIR)
    log_info(_OP, "MPIs installed: " + ", ".join(f"{k}={v}" for k, v in report.recorded.items()),
             recorded=report.recorded, hpcx_path=hpcx.path, ucx_path=hpcx.ucx,
             rebuilt_hpcx=plan.rebuild_hpcx, internal_pmix=plan.internal_pmix,
             symlinks=len(report.symlinks), pinned=report.pinned, patched=report.patched)


def install(env: dict[str, str]) -> int:
    """Install HPC-X, MVAPICH, Open MPI and Intel MPI plus modulefiles. Returns 0 or 3."""
    plan = _plan(env)
    if plan is None:
        return 3
    _log_plan(plan)
    report = _Report()
    try:
        # bash runs $(nproc) for each `make -j`; read once here.
        hpcx = _install_all(env, plan, cpu_count(), report)
    finally:
        remove_paths(_WORK_DIR, op=_OP)     # only the scratch area this step created
    if hpcx is None:
        return 3
    _log_summary(plan, hpcx, report)
    return 0
