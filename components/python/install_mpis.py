"""install_mpis.py — Python port of components/install_mpis.sh.

What this step does
-------------------
Installs the MPI (Message Passing Interface) libraries that HPC jobs use to
talk between processes and nodes, and the environment-module files that let a
user pick one with `module load mpi/<name>`:

  1. HPC-X       NVIDIA's MPI bundle (Open MPI + UCX + HCOLL + SHARP),
                 unpacked into /opt. Which HPC-X build is used depends on the
                 GPU vendor and network (see hpcx_variant()). Usually its Open
                 MPI (and on AMD / Ubuntu 22.04 its UCX) is then rebuilt with
                 the bundled hpcx_rebuild.sh (see "HPC-X rebuild" below).
                 Afterwards its SHARP and UCX libraries are registered with the
                 dynamic linker (/etc/ld.so.conf.d), its pkg-config files are
                 symlinked into /usr/local/lib(64)/pkgconfig, and its
                 modulefiles are written.
  2. MVAPICH     built from source into /opt/mvapich-<version> (skipped on
                 GB200/GB300 with Ubuntu 24.04 or Azure Linux 3.0, and on
                 Ubuntu 26.04).
  3. Open MPI    built from source into /opt/openmpi-<version> (skipped on
                 Ubuntu 26.04).
  4. Intel MPI   installed with Intel's offline installer (x86_64 only).

Each MPI's modulefile under $MODULE_FILES_DIRECTORY/mpi, and its short name
(mpi/hpcx, mpi/openmpi, ...) as a symlink, is written right after that MPI is
installed. mpi/hpcx-pmix-<version> is a symlink to mpi/hpcx-<version>.

Every installed version is recorded in the component-versions file (HPCX,
MVAPICH, OMPI, IMPI, and PMIX when Open MPI uses its internal PMIx). On RPM
distros (AlmaLinux, Rocky, RHEL, Azure Linux 3.0) ucx*, openmpi and perftest
are version-locked against later dnf updates.

Two network flavours run through the whole script:
  * UCX SKUs (every SKU with InfiniBand): MPIs use UCX (+ HCOLL).
  * Non-UCX SKUs (NCv6, network mode "no_rdma"): MPIs use libfabric
    (/opt/libfabric) with the tcp provider, and the modulefiles get extra
    settings that avoid hangs on MANA-only hardware.

PMIx: internal or external
--------------------------
On Ubuntu 26.04, Azure Linux 3.0, AlmaLinux/Rocky/RHEL 10 (no external PMIx v5
package to align to) and on baremetal_3p nodes (the external PMIx conflicts
with the node's packages), Open MPI builds its own PMIx, hwloc and libevent
("internal"), and the PMIx version it ships is read back with pkg-config and
recorded as PMIX. Everywhere else Open MPI is built against the PMIx that
install_pmix installed in /opt/pmix/<version without the last 2 chars>.

HPC-X rebuild
-------------
  * UCX is rebuilt (hpcx_rebuild.sh --rebuild-ucx) on UCX SKUs that are AMD
    (ROCm support) or Ubuntu 22.04. Before that, HPC-X's bundled UCX source
    tarball is patched with components/patches/ucx-netlink-dump-done.patch if
    it has the netlink route-dump hang (openucx/ucx#12070, hit on
    linux-azure 5.15.0-1123+).
  * HPC-X is rebuilt unless the SKU is an NVIDIA UCX SKU with internal PMIx and
    no UCX rebuild; that one case uses the vendor Open MPI as shipped
    (modulefiles/hpcx instead of modulefiles/hpcx-rebuild).
  * The rebuild passes the vendor's own Open MPI configure options back
    (--ompi-extra-config replaces them), --cuda on NVIDIA (the driver/CUDA
    step runs before this one; CUDA_HOME=/usr/local/cuda when UCX is rebuilt)
    and ROCm options on AMD (ROCm is installed before this step).
  * After the rebuild: the rebuilt UCX must run (`ucx_info -v`), the rebuilt
    ompi_info must load the rebuilt libmpi (`ldd`), and the generated
    module/init entrypoints are fixed with sed to point at hpcx-rebuild.

The original bash, for comparison (abridged; see components/install_mpis.sh):

    set CC=/usr/bin/gcc; set GCC=/usr/bin/gcc
    INSTALL_PREFIX=/opt
    mkdir -p ${MODULE_FILES_DIRECTORY}/mpi
    hpcx_metadata = hpcx_amd (AMD) | hpcx_inbox (no_rdma) | hpcx
    download_and_verify ${HPCX_DOWNLOAD_URL} ${HPCX_SHA256}; tar -xvf ${TARBALL}
    mv ${HPCX_FOLDER} ${INSTALL_PREFIX}
    HPCX_DIR=${HPCX_PATH} ${HPCX_PATH}/utils/hpcx_fix_ladir.sh
    write_component_version "HPCX" $HPCX_VERSION
    USE_INTERNAL_PMIX = ubuntu26.04|azurelinux3.0|almalinux10*|rocky10*|rhel10*|baremetal_3p
        else PMIX_PATH=${INSTALL_PREFIX}/pmix/${PMIX_VERSION:0:-2}
    OMPI_DEPS_ARGS = (--with-pmix=internal --with-hwloc=internal --with-libevent=internal)
                     | ("--with-pmix=${PMIX_PATH}")
    REBUILD_UCX = sku_uses_ucx && (AMD || ubuntu22.04)
    REBUILD_HPCX = !(!REBUILD_UCX && USE_INTERNAL_PMIX && NVIDIA && sku_uses_ucx)
    if REBUILD_HPCX:
        REBUILD_UCX: patch_hpcx_ucx_source_netlink_dump; --rebuild-ucx;
                     UCX_PATH=${HPCX_PATH}/ucx/hpcx-rebuild
        AMD: [[ -d /opt/rocm ]] || exit 1; (REBUILD_UCX) --ucx-extra-config "--with-rocm=/opt/rocm"
        NVIDIA: --cuda; (REBUILD_UCX) CUDA_HOME=/usr/local/cuda
        --ompi-extra-config "--enable-mpi1-compatibility --without-xpmem --with-slurm
            --enable-orterun-prefix-by-default ${OMPI_DEPS_ARGS[@]}
            + (non-UCX) --without-ucx --with-ofi=${LIBFABRIC_PATH}
            + (UCX) --with-platform=contrib/platform/mellanox/optimized [--with-rocm=/opt/rocm]"
            (UCX adds --with-hcoll to the rebuild args)
        env [CUDA_HOME=...] hpcx_rebuild.sh ... || { tail -n 40 of its logs; exit 1; }
        REBUILD_UCX: LD_LIBRARY_PATH=${UCX_PATH}/lib ${UCX_PATH}/bin/ucx_info -v || exit 1
        ldd hpcx-rebuild/bin/ompi_info: libmpi.so must be under hpcx-rebuild/lib/ || exit 1
        sed -i --follow-symlinks -E ... modulefiles/hpcx-rebuild hpcx-rebuild.sh
        cp -r ${HPCX_PATH}/ompi/tests ${HPCX_PATH}/hpcx-rebuild
    USE_INTERNAL_PMIX: write_component_version "PMIX" $(PKG_CONFIG_PATH=<hpcx-rebuild|ompi5>/lib/pkgconfig
                                                       pkg-config --modversion pmix)
    HPCX_DIR=${HPCX_PATH} ${HPCX_PATH}/utils/hpcx_fix_ladir.sh
    cat > /etc/ld.so.conf.d/hpcx-sharp.conf / hpcx-ucx.conf; ldconfig
    ln -sf <hcoll|sharp|ucx>/lib/pkgconfig/*.pc /usr/local/lib[64]/pkgconfig/
    RPM distros: dnf_pin_packages "ucx*"
    cat >> mpi/hpcx-${HPCX_VERSION} (module load <hpcx-rebuild|hpcx>)
    ln -s hpcx-${V} hpcx-pmix-${V}; ln -s hpcx-${V} hpcx; ln -s hpcx-pmix-${V} hpcx-pmix
    MVAPICH (unless GB-family on ubuntu24.04/azurelinux3.0, or ubuntu26.04):
        download, tar, ./configure [FFLAGS=...] --prefix=... <UCX|OFI args>,
        make -j$(nproc), make install, write_component_version "MVAPICH",
        cat >> mpi/mvapich-<v>; ln -s mvapich
    Open MPI (unless ubuntu26.04): download, tar,
        ./configure <UCX|OFI variant> <PMIX_FLAG>, make, make install,
        write_component_version "OMPI", cat >> mpi/openmpi-<v>; ln -s openmpi
    RPM distros: dnf_pin_packages openmpi perftest
    Intel MPI (not aarch64): download, bash <installer> -s -a -s --eula accept,
        mv .../etc/modulefiles/mpi .../etc/modulefiles/impi, write_component_version "IMPI",
        cat >> mpi/impi_<v>; ln -s impi-2021; EL8: cat >> I_MPI_STARTUP_MODE
    rm -rf *.tbz *.tar.gz *offline.sh; rm -rf -- */

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-mpis" Step. Deleting `action=` from that Step makes the build run the
bash script again, which is why the .sh file is kept.

install() receives the build environment (a dict) and returns 0 for success or
3 for failure. It logs problems and never lets an exception escape. Inside the
module, a failing command raises the private _StepFailed exception (after
logging), which install() turns into 3 — the same "stop at the first failure"
as bash's `set -e`, without an `if ...: return` after every line.

Behaviour kept from bash
------------------------
  * Commands, arguments and their order are bash's; external programs (tar,
    mv, cp, sed, patch, ldd, pkg-config, ucx_info, hpcx scripts, configure,
    make, ldconfig, the Intel installer) are still run, so they behave exactly
    as before. Each phase stops at the first failing command.
  * Generated files are byte-for-byte what bash writes. The modulefiles were
    checked against the real bash (tests/python/fixtures/install_mpis/,
    rendered by running the real install_mpis.sh in bash with stubbed
    commands for UCX, non-UCX, EL8 AMD, GB200/aarch64, Ubuntu 22.04
    (patched/rebuilt UCX), Azure Linux 3.0 (vendor HPC-X, internal PMIx) and
    Ubuntu 26.04 targets). That same run, with this module next to it, gave
    identical commands, arguments, recorded versions, pins and resulting
    files/symlinks for 14 targets (incl. RHEL 8/10, Rocky 9 NCv6,
    baremetal_3p and a failing hpcx_rebuild.sh).
  * Modulefiles are APPENDED to (bash `cat >> file`) and the short-name
    symlinks are made with plain `ln -s` (no -f), so a rerun on the same image
    duplicates the modulefile content and then fails at the first symlink —
    see "Bash issues" below.
  * The /etc/ld.so.conf.d files are replaced (`cat >`), and the pkg-config
    symlinks are replaced if present (`ln -sf`).
  * Open MPI's UCX configure passes LD_LIBRARY_PATH=<caller's
    LD_LIBRARY_PATH>:<hcoll>/lib as a configure ARGUMENT (not an environment
    variable), with an empty caller value giving ":<hcoll>/lib", as in bash.
  * The /opt/rocm check, the baremetal_3p internal-PMIx case and every
    distro/SKU gate (including ubuntu26.04 and EL10) are kept.

Deliberate differences from bash
--------------------------------
  * All configs (pmix unless Open MPI uses its internal PMIx, the HPC-X
    variant, mvapich and ompi unless skipped, impi on x86_64) are resolved and
    checked BEFORE anything is downloaded. A missing version/url/sha256 fails
    the step (3). Bash resolves each one just before using it and carries on
    with blank values, e.g. installing HPC-X and only then building MVAPICH
    from an empty URL.
  * A pmix or impi version shorter than 2 characters fails up front; bash's
    ${VERSION:0:-2} would abort mid-script ("substring expression < 0"), and an
    exactly-2-character one would give an empty path, which we also refuse.
  * An empty MODULE_FILES_DIRECTORY fails up front (bash would write to /mpi).
  * Work directory: bash downloads and builds in its current directory (the
    distro build dir) and ends with `rm -rf *.tbz *.tar.gz *offline.sh` and
    `rm -rf -- */`. We download/build in _WORK_DIR (emptied first) and remove
    only that directory at the end, whatever happened.
  * nproc is read once (run_capture) and reused for both `make -j`.
  * A source folder that tar didn't create fails with a clear message (bash's
    pushd/cd would fail less clearly).
  * Open MPI's LD_LIBRARY_PATH configure argument is passed as ONE argument;
    bash's unquoted expansion would also split it at whitespace (a path list
    with spaces is not something these images use).
  * hpcx_rebuild.sh gets CUDA_HOME through the environment we pass it, where
    bash runs `env CUDA_HOME=/usr/local/cuda hpcx_rebuild.sh ...`; the log
    line names the extra variable.
  * The UCX patch is applied with `patch -d <src> -p1 -i <file>` instead of
    `patch -d <src> -p1 < <file>` (exec_program has no stdin); same effect.
  * UCX netlink patch: if the extracted UCX source has no
    src/ucs/sys/netlink.c, bash's `grep -qF` fails, so bash prints "does not
    have the netlink dump hang" and rebuilds UCX unpatched. We fail instead:
    we can't tell whether the hang is there. The scratch directory the
    tarball is unpacked into is removed even when patching fails (bash leaves
    it behind on failure).
  * hpcx_rebuild.sh failure: bash echoes `tail -n 40` of each of its UCX and
    Open MPI logs; we log the same lines as error details.
  * An internal-PMIx version that pkg-config prints as empty fails instead of
    recording an empty PMIX version.

Bash issues kept (warned at runtime with log_warn, and commented below)
-----------------------------------------------------------------------
  * `set CC=/usr/bin/gcc` / `set GCC=/usr/bin/gcc` don't set CC or GCC: bash's
    `set` replaces the positional parameters ($1...). The builds use whatever
    compiler the environment already selects, in bash and here.
  * `rm -rf -- */` at the end of the bash deletes EVERY subdirectory of the
    distro build dir, not only what this step created. The port doesn't do
    that (see the work-directory difference above).
  * The modulefile appends and `ln -s` symlinks are not rerun-safe (see above).
    We warn when appending to a modulefile that already exists, and fail with
    an explanation when a symlink already exists, exactly where bash fails.
  * Open MPI's PMIX_FLAG has a branch for azurelinux3.0 / baremetal_3p
    (`--with-pmix`) that can't be reached any more: both now use the internal
    PMIx, which is checked first. It is mirrored in ompi_pmix_flags().
"""

# Lets type hints such as `str | None` work on older Python versions.
from __future__ import annotations

import glob
import os
import re
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass, field

from utils.component_config import DEFAULT_NODE_TYPE, config_for, write_component_version
from utils.download import download_and_verify
from utils.logger import log_debug, log_error, log_error_detail, log_info, log_warn
from utils.package_pinning import dnf_pin_packages
from utils.process import exec_program, run_capture
from utils.sku import NETWORK_MODE_NO_RDMA, sku_network_mode, sku_uses_ucx

# The Step's `op` label in build_config.py; used on every log line.
_OP = "install-mpis"

_AZURE_LINUX = "azurelinux3.0"
_UBUNTU_22 = "ubuntu22.04"
_UBUNTU_26 = "ubuntu26.04"
_GB_FAMILY = "gb-family"
_BAREMETAL_3P = "baremetal_3p"
# MVAPICH is skipped for gb-family SKUs on these two distros (exact matches).
_MVAPICH_SKIP_DISTROS = ("ubuntu24.04", _AZURE_LINUX)
# Intel MPI's EL8 race-condition workaround applies to these (plus rhel8*).
_EL8_DISTROS = ("almalinux8.10", "rocky8.10")
# Open MPI builds its own PMIx/hwloc/libevent on these distros (exact matches)
# and on any distro starting with one of the prefixes (bash: almalinux10* ...).
_INTERNAL_PMIX_DISTROS = (_UBUNTU_26, _AZURE_LINUX)
_INTERNAL_PMIX_PREFIXES = ("almalinux10", "rocky10", "rhel10")

# ---- Paths (module-level so tests can point them at a temporary directory) --
_INSTALL_PREFIX = "/opt"                      # bash: INSTALL_PREFIX=/opt
_WORK_DIR = "/tmp/azhpc-mpis"                 # our scratch area (see docstring)
_ROCM_DIR = "/opt/rocm"                       # only for the "is ROCm installed" check
_LD_SO_CONF_DIR = "/etc/ld.so.conf.d"
_PKGCONFIG_DIRS = ("/usr/local/lib/pkgconfig",)
_PKGCONFIG_DIRS_RPM = ("/usr/local/lib/pkgconfig", "/usr/local/lib64/pkgconfig")
# Parent of the scratch directory the UCX source is patched in. None means
# tempfile's default ($TMPDIR or /tmp), like bash's `mktemp -d`.
_PATCH_TMP_PARENT: str | None = None

# Used only inside generated text, configure arguments and the environment
# (literal in bash).
_LIBFABRIC_PATH = "/opt/libfabric"
_ROCM_PATH = "/opt/rocm"
_CUDA_HOME = "/usr/local/cuda"
_INTERNAL_DEPS_ARGS = ("--with-pmix=internal", "--with-hwloc=internal",
                       "--with-libevent=internal")
# Supplying --ompi-extra-config replaces HPC-X's original Open MPI configure
# options; these are the compatible vendor options bash restores.
_OMPI_VENDOR_ARGS = ("--enable-mpi1-compatibility", "--without-xpmem", "--with-slurm",
                     "--enable-orterun-prefix-by-default")
_MELLANOX_PLATFORM = "--with-platform=contrib/platform/mellanox/optimized"

# ---- UCX netlink route-dump patch (upstream #644) ---------------------------
# Relative to COMPONENT_DIR.
_UCX_NETLINK_PATCH = "patches/ucx-netlink-dump-done.patch"
# The line only UCX versions with the hang have (bash greps for it with -F).
_UCX_NETLINK_BUGGY_LINE = ("msg_done = ucs_netlink_is_msg_done("
                           "(const struct nlmsghdr *)recv_msg);")
_UCX_NETLINK_SOURCE = "src/ucs/sys/netlink.c"
# How many lines of each hpcx_rebuild.sh log bash prints when the rebuild fails.
_REBUILD_LOG_TAIL = 40

# sed expressions bash uses to point hpcx_rebuild.sh's module/init files at
# hpcx-rebuild instead of hpcx-rebuild$mpi_version (a directory it never
# creates). Passed to sed exactly as bash does (bash single-quotes them).
_REBUILD_ENTRYPOINT_SED = (
    "-e", r"s|/hpcx-rebuild\$\{mpi_version\}|/hpcx-rebuild|g",
    "-e", r"s|/hpcx-rebuild\$mpi_version|/hpcx-rebuild|g",
    "-e", "s|/hpcx-rebuild[0-9]+|/hpcx-rebuild|g",
)


class _StepFailed(Exception):
    """Raised (after logging why) to stop the step, like bash's `set -e`."""


# --------------------------------------------------------------------------- #
# Pure helpers: no side effects, so tests call them directly.
# --------------------------------------------------------------------------- #
def drop_last_two(value: str) -> str | None:
    """bash's ${value:0:-2}: everything but the last 2 characters.

    e.g. pmix "4.2.9-2" -> "4.2.9", impi "2021.16.1" -> "2021.16".
    Bash aborts ("substring expression < 0") when the value has fewer than 2
    characters; we return None for that, and also for an exactly-2-character
    value, whose empty result would make a broken path.
    """
    if len(value) <= 2:
        return None
    return value[:-2]


def url_basename(url: str, suffix: str = "") -> str:
    """bash's $(basename URL [SUFFIX]): the last path part, minus SUFFIX.

    basename ignores trailing slashes and only removes SUFFIX when something is
    left, e.g. url_basename(".../hpcx-v2.18-...-x86_64.tbz", ".tbz")
    -> "hpcx-v2.18-...-x86_64".
    """
    name = url.rstrip("/").rsplit("/", 1)[-1]
    if suffix and name.endswith(suffix) and name != suffix:
        name = name[:-len(suffix)]
    return name


def hpcx_variant(gpu: str, network_mode: str) -> tuple[str, str]:
    """Which HPC-X config bash picks, and why: (component name, reason).

    Same choice as utils/version_validate.active_hpcx_component (which keys on
    the SKU being NCv6, the only "no_rdma" SKU); a test keeps the two in step.
    """
    if gpu == "AMD":
        return "hpcx_amd", "AMD GPU: AMD has a regression on newer HPC-X versions"
    if network_mode == NETWORK_MODE_NO_RDMA:
        return ("hpcx_inbox", "network mode no_rdma: no DOCA-OFED on non-IB SKUs, "
                              "so use the inbox HPC-X (UCX linked against rdma-core)")
    return "hpcx", "default HPC-X (InfiniBand network)"


def internal_pmix_reason(distribution: str, node_type: str) -> str | None:
    """Why Open MPI builds its own PMIx here (bash's USE_INTERNAL_PMIX), or None.

    bash: [[ $D == "ubuntu26.04" ]] || [[ $D == "azurelinux3.0" ]] ||
          [[ $D == almalinux10* ]] || [[ $D == rocky10* ]] || [[ $D == rhel10* ]]
          elif [[ ${TARGET_NODE_TYPE:-azure_vm_regular} == "baremetal_3p" ]]
    """
    if distribution == _AZURE_LINUX:
        return "azurelinux3.0 lacks the PMIx package published by the CycleCloud team"
    if distribution in _INTERNAL_PMIX_DISTROS or distribution.startswith(_INTERNAL_PMIX_PREFIXES):
        return f"{distribution} has no external PMIx v5 to align to"
    if (node_type or DEFAULT_NODE_TYPE) == _BAREMETAL_3P:
        return "baremetal_3p nodes must avoid the external PMIx (package conflict)"
    return None


def uses_internal_pmix(distribution: str, node_type: str) -> bool:
    """bash's USE_INTERNAL_PMIX (see internal_pmix_reason)."""
    return internal_pmix_reason(distribution, node_type) is not None


def ompi_deps_args(internal_pmix: bool, pmix_path: str | None) -> list[str]:
    """bash's OMPI_DEPS_ARGS: Open MPI's PMIx/hwloc/libevent configure options.

    Internal: Open MPI builds PMIx, hwloc and libevent from its own sources
    into its prefix (pointing them at HPC-X's ompi5 tree would put the vendor
    libmpi ahead of the rebuild in every RUNPATH).
    """
    if internal_pmix:
        return list(_INTERNAL_DEPS_ARGS)
    return [f"--with-pmix={pmix_path}"]


def ompi_pmix_flags(internal_pmix: bool, distribution: str, node_type: str,
                    pmix_path: str | None) -> list[str]:
    """Open MPI's ${PMIX_FLAG}, split into words as bash's unquoted expansion does."""
    if internal_pmix:
        # PMIX_FLAG="${OMPI_DEPS_ARGS[*]}", expanded unquoted: three words.
        return list(_INTERNAL_DEPS_ARGS)
    # BASH ISSUE (kept): unreachable. azurelinux3.0 and baremetal_3p always
    # use the internal PMIx, which is checked first.
    if distribution == _AZURE_LINUX or (node_type or DEFAULT_NODE_TYPE) == _BAREMETAL_3P:
        return ["--with-pmix"]
    return [f"--with-pmix={pmix_path}"]


def rebuild_ucx_reason(gpu: str, distribution: str, uses_ucx: bool) -> str | None:
    """Why HPC-X's UCX is rebuilt (bash's REBUILD_UCX), or None.

    bash: sku_uses_ucx && ( "$GPU" == "AMD" || "$DISTRIBUTION" == "ubuntu22.04" )
    """
    if not uses_ucx:
        return None
    if gpu == "AMD":
        return "AMD GPU: rebuild UCX with ROCm support"
    if distribution == _UBUNTU_22:
        return ("ubuntu22.04: linux-azure 5.15.0-1123+ triggers the UCX netlink "
                "route-dump hang, so rebuild UCX from the patched source")
    return None


def needs_hpcx_rebuild(*, rebuild_ucx: bool, internal_pmix: bool, gpu: str,
                       uses_ucx: bool) -> bool:
    """bash's REBUILD_HPCX: true unless nothing needs rebuilding.

    Only an NVIDIA UCX SKU with internal PMIx and no UCX rebuild keeps the
    vendor Open MPI. (A UCX rebuild is done by hpcx_rebuild.sh, so it always
    means an HPC-X rebuild.)
    """
    return not (not rebuild_ucx and internal_pmix and gpu == "NVIDIA" and uses_ucx)


def hpcx_rebuild_command(hpcx_path: str, *, gpu: str, uses_ucx: bool, rebuild_ucx: bool,
                         deps_args: list[str]) -> tuple[list[str], dict[str, str]]:
    """hpcx_rebuild.sh's command line and the extra environment it runs with.

    Mirrors how bash builds HPCX_REBUILD_ARGS / HPCX_REBUILD_ENV, in the same
    order. The Open MPI options are joined with spaces into one
    --ompi-extra-config value, as bash's "${HPCX_REBUILD_OMPI_ARGS[*]}".
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
    """bash's  ldd <ompi_info> | awk '$1 ~ /^libmpi\\.so/ {print $3}'  inside $(...).

    awk splits each line on runs of blanks (ignoring leading ones) and prints
    the 3rd field ("" if there isn't one) of every line whose 1st field starts
    with "libmpi.so"; $(...) then drops the trailing newlines. e.g.
    "\\tlibmpi.so.40 => /opt/h/hpcx-rebuild/lib/libmpi.so.40 (0x...)" ->
    "/opt/h/hpcx-rebuild/lib/libmpi.so.40", and "libmpi.so.40 => not found"
    -> "not".
    """
    found = []
    for line in ldd_output.split("\n"):
        fields = re.split(r"[ \t]+", line.strip(" \t"))
        if fields[0].startswith("libmpi.so"):
            found.append(fields[2] if len(fields) > 2 else "")
    return "\n".join(found).rstrip("\n")


def is_rpm_distro(distribution: str) -> bool:
    """bash: [[ $D == almalinux* ]] || [[ $D == rocky* ]] || [[ $D == rhel* ]] || [[ $D == "azurelinux3.0" ]]."""
    return distribution.startswith(("almalinux", "rocky", "rhel")) or distribution == _AZURE_LINUX


def skips_mvapich(distribution: str, sku_family: str) -> bool:
    """bash: ((D == ubuntu24.04 || D == azurelinux3.0) && SKU_FAMILY == gb-family) || D == ubuntu26.04."""
    return ((distribution in _MVAPICH_SKIP_DISTROS and sku_family == _GB_FAMILY)
            or distribution == _UBUNTU_26)


def skips_ompi(distribution: str) -> bool:
    """bash: [[ "$DISTRIBUTION" != "ubuntu26.04" ]] around the Open MPI build, negated."""
    return distribution == _UBUNTU_26


def is_el8(distribution: str) -> bool:
    """bash: [[ $D == "almalinux8.10" ]] || [[ $D == "rocky8.10" ]] || [[ $D == rhel8* ]]."""
    return distribution in _EL8_DISTROS or distribution.startswith("rhel8")


def impi_fi_provider(network_mode: str) -> str:
    """Intel MPI's libfabric provider: "tcp" on no_rdma, else "mlx"."""
    return "tcp" if network_mode == NETWORK_MODE_NO_RDMA else "mlx"


def mvapich_configure_command(distribution: str, version: str, *, uses_ucx: bool,
                              ucx_path: str) -> tuple[list[str], str]:
    """MVAPICH's ./configure command and the transport lib dir it links against.

    The lib dir (UCX or libfabric) goes into the modulefile's LD_LIBRARY_PATH,
    so the runtime picks the same UCX/libfabric MVAPICH was built with instead
    of the system DOCA-OFED copy.
    """
    command = ["./configure"]
    # Ubuntu (22.04's gfortran) / Azure Linux: "The Fortran compiler gfortran
    # will not compile files that call the same routine with arguments of
    # different types" without this flag.
    if "ubuntu" in distribution or distribution == _AZURE_LINUX:
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
# with a blank line "\n". Checked against bash-rendered fixtures.
_HPCX_NON_UCX_EXTRAS = (
    # Force PML cm (MTL-based) instead of ob1 (BTL openib can't move data on
    # MANA-only hardware -> hangs); the libfabric tcp provider explicitly
    # (auto-detection trips on the docker bridge 172.17.0.1); and disable UCC
    # (tl_ucp probes verbs on MANA) and hcoll (needs a Mellanox IB HCA).
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
# Resolved inputs and what the step did.
# --------------------------------------------------------------------------- #
@dataclass
class _Inputs:
    """Everything install() decides up front, so each phase just reads it."""
    distribution: str
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
    hpcx: dict
    internal_pmix: bool
    internal_pmix_reason: str | None
    pmix_version: str | None        # None: internal PMIx (no pmix config needed)
    pmix_path: str | None
    rebuild_ucx: bool
    rebuild_ucx_reason: str | None
    rebuild_hpcx: bool
    mvapich: dict | None            # None: skipped (GB-family, ubuntu26.04)
    ompi: dict | None               # None: skipped (ubuntu26.04)
    impi: dict | None               # None: skipped (aarch64)
    impi_2021_version: str | None

    @property
    def mpi_dir(self) -> str:
        """bash's MPI_MODULE_FILES_DIRECTORY."""
        return f"{self.module_dir}/mpi"


@dataclass
class _Hpcx:
    """HPC-X locations later phases and the modulefiles use."""
    path: str
    hcoll: str
    sharp: str
    ucx: str


@dataclass
class _Report:
    """What the step changed, for the closing summary."""
    recorded: dict = field(default_factory=dict)
    files: list = field(default_factory=list)
    symlinks: list = field(default_factory=list)
    pinned: list = field(default_factory=list)
    patched: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Small helpers with side effects. They raise _StepFailed after logging.
# --------------------------------------------------------------------------- #
def _run(command: list[str], what: str, env: dict[str, str], cwd: str) -> None:
    """Run one command; stop the step if it fails (bash `set -e`).

    exec_program already logs the command, its exit code and, on failure, its
    last output lines; we add what the failure means.
    """
    rc = exec_program(command, _OP, cwd=cwd, env=env)
    if rc != 0:
        log_error(_OP, f"{what} failed; stopping, as bash's set -e would",
                  command=command[0], exit_code=rc)
        raise _StepFailed


def _download(url: str, sha256: str, what: str) -> None:
    """download_and_verify into the work directory, or stop the step."""
    try:
        download_and_verify(url, sha256, dest_dir=_WORK_DIR)
    except Exception as exc:  # network errors, checksum mismatch, ...
        log_error(_OP, f"download/verify of {what} failed: {exc}", url=url)
        raise _StepFailed from exc


def _require_dir(path: str, what: str) -> None:
    if not os.path.isdir(path):
        log_error(_OP, f"{what} folder {path} was not created by tar", path=path)
        raise _StepFailed


def _record(component: str, version: str, report: _Report) -> None:
    write_component_version(component, version)
    report.recorded[component] = version
    log_info(_OP, f"Recorded {component}={version}", component=component, version=version)


def _pin(packages: list[str], env: dict[str, str], report: _Report) -> None:
    """bash's dnf_pin_packages (a failure stops the script under set -e)."""
    if not dnf_pin_packages(packages, _OP, env):
        raise _StepFailed
    report.pinned.extend(packages)


def _write_file(path: str, text: str, report: _Report) -> None:
    """bash `cat > path`: create or replace the file."""
    existed = os.path.lexists(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    action = "replaced" if existed else "created"
    report.files.append(path)
    log_info(_OP, f"Wrote {path} ({action})", path=path, action=action,
             content=text.strip())


def _append_file(path: str, text: str, report: _Report, *, new_file: bool = True) -> None:
    """bash `cat << EOF >> path`: append, creating the file if needed.

    new_file=False marks a deliberate second append in the same run (the EL8
    Intel MPI lines), where an existing file is expected, not a rerun.
    """
    existed = os.path.lexists(path)
    if existed and new_file:
        # BASH ISSUE (kept): `>>` appends, so a rerun adds a second copy of the
        # modulefile to the same file instead of replacing it.
        log_warn(_OP, f"{path} already exists; appending another copy, as bash's "
                      "`cat >>` does (not rerun-safe: the file gets duplicate settings)",
                 path=path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)
    action = "appended" if existed else "created"
    report.files.append(path)
    log_info(_OP, f"Wrote modulefile {path} ({action})", path=path, action=action)
    log_debug(_OP, f"{path} content written:\n{text}")


def _link_path(target: str, link: str) -> str:
    """Where GNU ln (without -n) puts the link: inside `link` if it's a directory."""
    if os.path.isdir(link):
        return os.path.join(link, os.path.basename(target))
    return link


def _ln_s(target: str, link: str, report: _Report) -> None:
    """bash `ln -s target link`: fails if `link` already exists."""
    link = _link_path(target, link)
    if os.path.lexists(link):
        # BASH ISSUE (kept): plain `ln -s` (no -f) fails when the link exists,
        # which stops the bash script on any rerun over an existing image.
        log_error(_OP, f"cannot create symlink {link}: it already exists. bash's "
                       "`ln -s` (without -f) fails here too, so install_mpis is not "
                       "rerun-safe; remove the old MPI modulefiles first",
                  path=link, target=target)
        raise _StepFailed
    os.symlink(target, link)
    report.symlinks.append(link)
    log_info(_OP, f"Created symlink {link} -> {target}", path=link, target=target)


def _ln_sf(target: str, link: str, report: _Report) -> None:
    """bash `ln -sf target link`: replace an existing file/link."""
    link = _link_path(target, link)
    if os.path.lexists(link):
        os.remove(link)     # a directory here raises, as `ln -sf` fails on one
    os.symlink(target, link)
    report.symlinks.append(link)
    log_info(_OP, f"Created symlink {link} -> {target}", path=link, target=target)


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


def _tail_lines(path: str, count: int) -> list[str]:
    """`tail -n <count> path` as a list of lines (without their newlines)."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in deque(handle, maxlen=count)]


# --------------------------------------------------------------------------- #
# HPC-X rebuild pieces, in bash's order.
# --------------------------------------------------------------------------- #
def _patch_ucx_source_netlink_dump(env: dict[str, str], inp: _Inputs, hpcx: _Hpcx,
                                   report: _Report) -> None:
    """bash's patch_hpcx_ucx_source_netlink_dump.

    Patch HPC-X's bundled UCX source tarball so `hpcx_rebuild.sh --rebuild-ucx`
    builds a UCX that doesn't hang in ucs_netlink_send_request() on kernels
    that append NLMSG_DONE to the last route-dump datagram (Ubuntu 5.15.0-198,
    hence linux-azure 5.15.0-1123+; openucx/ucx#12070). Only UCX versions that
    have the buggy line are patched; the tarball is repacked in place.
    """
    patch_file = os.path.join(inp.component_dir, _UCX_NETLINK_PATCH)
    # bash: find "${HPCX_PATH}/sources" -maxdepth 1 -name 'ucx-*.tar.gz' | head -n1
    # (sorted here so the choice is stable; HPC-X ships one UCX tarball).
    tarballs = sorted(glob.glob(f"{hpcx.path}/sources/ucx-*.tar.gz"))
    if not tarballs:
        log_error(_OP, f"HPC-X UCX source tarball not found under {hpcx.path}/sources",
                  path=f"{hpcx.path}/sources")
        raise _StepFailed
    if len(tarballs) > 1:
        log_warn(_OP, f"{len(tarballs)} UCX source tarballs in HPC-X; patching the "
                      f"first, {tarballs[0]}", tarballs=tarballs)
    tarball = tarballs[0]
    ucx_name = url_basename(tarball, ".tar.gz")

    work_dir = tempfile.mkdtemp(dir=_PATCH_TMP_PARENT)
    try:
        _run(["tar", "-xzf", tarball, "-C", work_dir], "extracting the HPC-X UCX source",
             env, _WORK_DIR)
        ucx_src = f"{work_dir}/{ucx_name}"
        source = f"{ucx_src}/{_UCX_NETLINK_SOURCE}"
        try:
            with open(source, encoding="utf-8", errors="replace") as handle:
                buggy = _UCX_NETLINK_BUGGY_LINE in handle.read()
        except OSError as exc:
            # bash's grep -qF would fail and skip the patch, rebuilding UCX
            # unpatched; we stop because we can't tell whether it hangs.
            log_error(_OP, f"cannot read {_UCX_NETLINK_SOURCE} in {tarball} ({exc}); "
                           "can't tell whether this UCX has the netlink route-dump "
                           "hang (bash would rebuild it unpatched)",
                      path=source, tarball=tarball)
            raise _StepFailed from exc

        if not buggy:
            log_info(_OP, f"{ucx_name} does not have the netlink dump hang; skipping "
                          "UCX patch", ucx=ucx_name, tarball=tarball, patched=False)
            return
        if not os.path.isfile(patch_file):
            log_error(_OP, f"UCX patch {patch_file} not found (is COMPONENT_DIR set?)",
                      path=patch_file)
            raise _StepFailed
        _run(["patch", "-d", ucx_src, "-p1", "-i", patch_file],
             f"applying {_UCX_NETLINK_PATCH} to {ucx_name}", env, _WORK_DIR)
        _run(["tar", "-czf", tarball, "-C", work_dir, ucx_name],
             f"repacking the patched {ucx_name} source", env, _WORK_DIR)
        report.patched.append(tarball)
        log_info(_OP, f"Patched {ucx_name} for the netlink route-dump hang and "
                      f"repacked {tarball} (replaced)", ucx=ucx_name, tarball=tarball,
                 patch=patch_file, patched=True, action="replaced")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _log_rebuild_logs(hpcx_path: str) -> None:
    """On a failed rebuild, show the tail of hpcx_rebuild.sh's logs.

    hpcx_rebuild.sh only names its logs, which stay on the build VM.
    """
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


def _check_rebuilt_ucx(env: dict[str, str], hpcx: _Hpcx) -> None:
    """bash: LD_LIBRARY_PATH=${UCX_PATH}/lib ${UCX_PATH}/bin/ucx_info -v > /dev/null.

    hpcx_rebuild.sh reports, but doesn't fail on, UCX `make install` errors.
    """
    rc, output = run_capture([f"{hpcx.ucx}/bin/ucx_info", "-v"], _OP, cwd=_WORK_DIR,
                             env={**env, "LD_LIBRARY_PATH": f"{hpcx.ucx}/lib"})
    if rc != 0:
        log_error(_OP, f"Rebuilt HPC-X UCX at {hpcx.ucx} is not usable.",
                  ucx_path=hpcx.ucx, exit_code=rc)
        raise _StepFailed
    version = next((line.strip() for line in output.splitlines() if line.strip()), "")
    log_info(_OP, f"Rebuilt UCX at {hpcx.ucx} runs: {version}", ucx_path=hpcx.ucx,
             ucx_info=version)


def _check_rebuilt_libmpi(env: dict[str, str], hpcx: _Hpcx) -> None:
    """The rebuilt ompi_info must load the rebuilt libmpi.

    A libmpi resolved outside hpcx-rebuild means the vendor Open MPI shadows
    the rebuild, which otherwise fails silently at runtime. bash ignores ldd's
    exit code (no pipefail); only what awk prints decides.
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
        raise _StepFailed
    log_info(_OP, f"Rebuilt ompi_info loads {libmpi}", ompi_info=ompi_info, libmpi=libmpi)


def _fix_rebuild_entrypoints(env: dict[str, str], hpcx: _Hpcx, report: _Report) -> None:
    """Point hpcx_rebuild.sh's module/init files at hpcx-rebuild.

    hpcx_rebuild.sh installs into hpcx-rebuild but writes files that compose
    the MPI prefix as hpcx-rebuild$mpi_version, a directory it never creates.
    Only that suffix is dropped (mpi_version is still used for ompi5/tests).
    """
    for entrypoint in (f"{hpcx.path}/modulefiles/hpcx-rebuild", f"{hpcx.path}/hpcx-rebuild.sh"):
        if not os.path.exists(entrypoint):      # bash: [[ -e ]] || continue
            log_info(_OP, f"{entrypoint} does not exist; nothing to fix", path=entrypoint)
            continue
        _run(["sed", "-i", "--follow-symlinks", "-E", *_REBUILD_ENTRYPOINT_SED, entrypoint],
             f"fixing the hpcx-rebuild prefix in {entrypoint}", env, _WORK_DIR)
        report.files.append(entrypoint)
        log_info(_OP, f"Fixed the hpcx-rebuild prefix in {entrypoint} (edited in place)",
                 path=entrypoint, action="edited")


def _rebuild_hpcx(env: dict[str, str], inp: _Inputs, hpcx: _Hpcx, report: _Report) -> None:
    """bash's `if [[ "$REBUILD_HPCX" == true ]]` block."""
    if inp.rebuild_ucx:
        _patch_ucx_source_netlink_dump(env, inp, hpcx, report)
        hpcx.ucx = f"{hpcx.path}/ucx/hpcx-rebuild"
        log_info(_OP, f"Rebuilding UCX: {inp.rebuild_ucx_reason}; using the rebuilt UCX "
                      f"at {hpcx.ucx} from here on", ucx_path=hpcx.ucx)
    if inp.gpu == "AMD" and not os.path.isdir(_ROCM_DIR):
        log_error(_OP, "ROCm must be installed before rebuilding HPC-X UCX with "
                       "ROCm support.", path=_ROCM_DIR)
        raise _StepFailed

    command, extra_env = hpcx_rebuild_command(
        hpcx.path, gpu=inp.gpu, uses_ucx=inp.uses_ucx, rebuild_ucx=inp.rebuild_ucx,
        deps_args=ompi_deps_args(inp.internal_pmix, inp.pmix_path))
    log_info(_OP, "Rebuilding HPC-X Open MPI"
                  f"{' and UCX' if inp.rebuild_ucx else ''} with hpcx_rebuild.sh",
             phase="hpcx-rebuild", gpu=inp.gpu, uses_ucx=inp.uses_ucx,
             rebuild_ucx=inp.rebuild_ucx, internal_pmix=inp.internal_pmix,
             rebuild_args=" ".join(command[1:]),
             rebuild_env=" ".join(f"{k}={v}" for k, v in extra_env.items()))
    rc = exec_program(command, _OP, cwd=_WORK_DIR, env={**env, **extra_env})
    if rc != 0:
        _log_rebuild_logs(hpcx.path)
        log_error(_OP, "hpcx_rebuild.sh failed; its UCX/Open MPI log tails are above",
                  command=command[0], exit_code=rc)
        raise _StepFailed

    if inp.rebuild_ucx:
        _check_rebuilt_ucx(env, hpcx)
    _check_rebuilt_libmpi(env, hpcx)
    _fix_rebuild_entrypoints(env, hpcx, report)
    _run(["cp", "-r", f"{hpcx.path}/ompi/tests", f"{hpcx.path}/hpcx-rebuild"],
         "copying the HPC-X ompi tests", env, _WORK_DIR)


def _record_internal_pmix(env: dict[str, str], inp: _Inputs, hpcx: _Hpcx,
                          report: _Report) -> None:
    """Record the PMIx Open MPI actually ships when it uses its internal copy.

    The rebuild installs its own PMIx; otherwise the vendor ompi5 tree is used.
    """
    sub = "hpcx-rebuild" if inp.rebuild_hpcx else "ompi5"
    pkg_config_path = f"{hpcx.path}/{sub}/lib/pkgconfig"
    rc, output = run_capture(["pkg-config", "--modversion", "pmix"], _OP, cwd=_WORK_DIR,
                             env={**env, "PKG_CONFIG_PATH": pkg_config_path})
    version = output.rstrip("\n")       # bash's $(...) drops trailing newlines
    if rc != 0 or not version.strip():
        log_error(_OP, f"could not read the internal PMIx version with pkg-config from "
                       f"{pkg_config_path}", pkg_config_path=pkg_config_path, exit_code=rc,
                  output=version)
        raise _StepFailed
    log_info(_OP, f"Internal PMIx {version} (from {pkg_config_path})",
             pkg_config_path=pkg_config_path, pmix_version=version)
    _record("PMIX", version, report)


# --------------------------------------------------------------------------- #
# Phases, in bash's order.
# --------------------------------------------------------------------------- #
def _install_hpcx(env: dict[str, str], inp: _Inputs, report: _Report) -> _Hpcx:
    cfg = inp.hpcx
    version, url = cfg["version"], cfg["url"]
    tarball = url_basename(url)
    folder = url_basename(url, ".tbz")
    hpcx_path = f"{_INSTALL_PREFIX}/{folder}"
    log_info(_OP, f"HPC-X phase: installing {inp.hpcx_component} {version} into {hpcx_path}",
             phase="hpcx", component=inp.hpcx_component, version=version, url=url,
             path=hpcx_path)

    _download(url, cfg["sha256"], "HPC-X")
    _run(["tar", "-xvf", tarball], "extracting HPC-X", env, _WORK_DIR)
    _run(["mv", folder, _INSTALL_PREFIX], f"moving {folder} to {_INSTALL_PREFIX}",
         env, _WORK_DIR)

    # Fix the relocated .la/.pc metadata before the rebuild uses the bundled
    # UCX, HCOLL and SHARP trees. `HPCX_DIR=... script` sets HPCX_DIR for that
    # one command only, hence the copied env.
    fix_ladir = [f"{hpcx_path}/utils/hpcx_fix_ladir.sh"]
    fix_env = {**env, "HPCX_DIR": hpcx_path}
    _run(fix_ladir, "hpcx_fix_ladir.sh", fix_env, _WORK_DIR)
    hpcx = _Hpcx(path=hpcx_path, hcoll=f"{hpcx_path}/hcoll",
                 sharp=f"{hpcx_path}/sharp", ucx=f"{hpcx_path}/ucx")
    _record("HPCX", version, report)

    if inp.rebuild_hpcx:
        _rebuild_hpcx(env, inp, hpcx, report)
    else:
        log_info(_OP, "Not rebuilding HPC-X: NVIDIA UCX SKU with internal PMIx and no UCX "
                      "rebuild, so the vendor Open MPI is used as shipped",
                 phase="hpcx-rebuild", rebuild_hpcx=False, internal_pmix=inp.internal_pmix,
                 rebuild_ucx=inp.rebuild_ucx)

    if inp.internal_pmix:
        _record_internal_pmix(env, inp, hpcx, report)

    # hpcx_rebuild.sh installs fresh Open MPI (and, on AMD, UCX) metadata; fix it too.
    _run(fix_ladir, "hpcx_fix_ladir.sh (after rebuild)", fix_env, _WORK_DIR)

    _write_file(f"{_LD_SO_CONF_DIR}/hpcx-sharp.conf", f"{hpcx.sharp}/lib\n", report)
    _write_file(f"{_LD_SO_CONF_DIR}/hpcx-ucx.conf", f"{hpcx.ucx}/lib\n", report)
    _run(["ldconfig"], "ldconfig", env, _WORK_DIR)

    _link_pkgconfig(inp, hpcx, report)

    if is_rpm_distro(inp.distribution):
        _pin(["ucx*"], env, report)          # exclude ucx from updates

    _write_hpcx_modulefiles(inp, hpcx, report)
    log_info(_OP, f"HPC-X {version} installed at {hpcx_path}", phase="hpcx",
             version=version, path=hpcx_path, ucx_path=hpcx.ucx,
             rebuilt=inp.rebuild_hpcx, module=hpcx_module(hpcx_path, inp.rebuild_hpcx))
    return hpcx


def _link_pkgconfig(inp: _Inputs, hpcx: _Hpcx, report: _Report) -> None:
    """Symlink HCOLL/SHARP/UCX .pc files into the system pkg-config dirs.

    So module-free builds resolve the same HCOLL, SHARP and UCX that
    `module load mpi/hpcx` would put on PKG_CONFIG_PATH.
    """
    dirs = _PKGCONFIG_DIRS_RPM if is_rpm_distro(inp.distribution) else _PKGCONFIG_DIRS
    for pkgconfig_dir in dirs:
        os.makedirs(pkgconfig_dir, exist_ok=True)
        linked = 0
        for tree in (hpcx.hcoll, hpcx.sharp, hpcx.ucx):
            # sorted() approximates the order of bash's glob; only the order
            # the links are made in can differ, not the result.
            for pc_file in sorted(glob.glob(f"{tree}/lib/pkgconfig/*.pc")):
                if not os.path.isfile(pc_file):     # bash: [[ -f ]] || continue
                    continue
                _ln_sf(pc_file, f"{pkgconfig_dir}/{os.path.basename(pc_file)}", report)
                linked += 1
        log_info(_OP, f"Linked {linked} HPC-X pkg-config file(s) into {pkgconfig_dir}",
                 path=pkgconfig_dir, count=linked)


def _write_hpcx_modulefiles(inp: _Inputs, hpcx: _Hpcx, report: _Report) -> None:
    """mpi/hpcx-<v>, plus the hpcx-pmix-<v>, hpcx and hpcx-pmix symlinks."""
    mpi_dir = inp.mpi_dir
    version = inp.hpcx["version"]
    module = hpcx_module(hpcx.path, inp.rebuild_hpcx)
    log_info(_OP, f"mpi/hpcx loads {module}", module=module, rebuilt=inp.rebuild_hpcx,
             uses_ucx=inp.uses_ucx)
    if not inp.uses_ucx:
        log_info(_OP, "Non-UCX SKU: adding PML cm / libfabric tcp settings and disabling "
                      "UCC and hcoll in the HPC-X modulefile")
    _append_file(f"{mpi_dir}/hpcx-{version}", hpcx_modulefile(version, module, inp.uses_ucx),
                 report)
    _ln_s(f"{mpi_dir}/hpcx-{version}", f"{mpi_dir}/hpcx-pmix-{version}", report)
    _ln_s(f"{mpi_dir}/hpcx-{version}", f"{mpi_dir}/hpcx", report)
    _ln_s(f"{mpi_dir}/hpcx-pmix-{version}", f"{mpi_dir}/hpcx-pmix", report)


def _install_mvapich(env: dict[str, str], inp: _Inputs, hpcx: _Hpcx, nproc: int,
                     report: _Report) -> None:
    if inp.mvapich is None:
        if inp.distribution == _UBUNTU_26:
            why = ("MVAPICH 4.1 does not build with ubuntu26.04's gcc 15 (bundled "
                   "libfabric OPX/UCR providers, then the collectives source)")
        else:
            why = (f"not supported on {inp.distribution} for {inp.sku_family} SKUs "
                   "(bash skips ubuntu24.04/azurelinux3.0 + gb-family)")
        log_info(_OP, f"Skipping MVAPICH: {why}", phase="mvapich", skipped=True,
                 distro=inp.distribution, sku_family=inp.sku_family)
        return
    cfg = inp.mvapich
    version, url = cfg["version"], cfg["url"]
    folder = url_basename(url, ".tar.gz")
    configure, transport_lib = mvapich_configure_command(
        inp.distribution, version, uses_ucx=inp.uses_ucx, ucx_path=hpcx.ucx)
    log_info(_OP, f"MVAPICH phase: building MVAPICH {version} "
                  f"({'UCX' if inp.uses_ucx else 'OFI/libfabric'} transport)",
             phase="mvapich", version=version, url=url, transport_lib=transport_lib)

    _download(url, cfg["sha256"], "MVAPICH")
    _run(["tar", "-xvf", url_basename(url)], "extracting MVAPICH", env, _WORK_DIR)
    source = f"{_WORK_DIR}/{folder}"
    _require_dir(source, "MVAPICH source")
    _run(configure, "MVAPICH configure", env, source)
    _run(["make", f"-j{nproc}"], "MVAPICH make", env, source)
    _run(["make", "install"], "MVAPICH make install", env, source)
    _record("MVAPICH", version, report)

    if not inp.uses_ucx:
        log_info(_OP, "Non-UCX SKU: adding FI_PROVIDER=tcp and disabling CMA in the "
                      "MVAPICH modulefile")
    _write_mvapich_modulefile(inp, version, transport_lib, report)
    log_info(_OP, f"MVAPICH {version} installed to {_INSTALL_PREFIX}/mvapich-{version}",
             phase="mvapich", version=version)


def _write_mvapich_modulefile(inp: _Inputs, version: str, transport_lib: str,
                              report: _Report) -> None:
    mpi_dir = inp.mpi_dir
    _append_file(f"{mpi_dir}/mvapich-{version}",
                 mvapich_modulefile(version, transport_lib, inp.uses_ucx), report)
    _ln_s(f"{mpi_dir}/mvapich-{version}", f"{mpi_dir}/mvapich", report)


def _install_ompi(env: dict[str, str], inp: _Inputs, hpcx: _Hpcx, nproc: int,
                  report: _Report) -> None:
    if inp.ompi is None:
        log_info(_OP, f"Skipping the standalone Open MPI on {inp.distribution} (HPC-X's "
                      "Open MPI only)", phase="ompi", skipped=True, distro=inp.distribution)
        return
    cfg = inp.ompi
    version, url = cfg["version"], cfg["url"]
    folder = url_basename(url, ".tar.gz")
    pmix_flags = ompi_pmix_flags(inp.internal_pmix, inp.distribution, inp.node_type,
                                 inp.pmix_path)
    configure, transport_lib = ompi_configure_command(
        version, uses_ucx=inp.uses_ucx, pmix_flags=pmix_flags, ucx_path=hpcx.ucx,
        hcoll_path=hpcx.hcoll, ld_library_path=env.get("LD_LIBRARY_PATH", ""))
    log_info(_OP, f"Open MPI phase: building Open MPI {version} "
                  f"({'UCX + HCOLL' if inp.uses_ucx else 'OFI/libfabric'} transport)",
             phase="ompi", version=version, url=url, pmix_flag=" ".join(pmix_flags),
             transport_lib=transport_lib)

    _download(url, cfg["sha256"], "Open MPI")
    _run(["tar", "-xvf", url_basename(url)], "extracting Open MPI", env, _WORK_DIR)
    source = f"{_WORK_DIR}/{folder}"
    _require_dir(source, "Open MPI source")
    _run(configure, "Open MPI configure", env, source)
    _run(["make", f"-j{nproc}"], "Open MPI make", env, source)
    _run(["make", "install"], "Open MPI make install", env, source)
    _record("OMPI", version, report)

    if not inp.uses_ucx:
        log_info(_OP, "Non-UCX SKU: adding PML cm / libfabric tcp settings to the "
                      "Open MPI modulefile")
    _write_ompi_modulefile(inp, version, hpcx, transport_lib, report)
    log_info(_OP, f"Open MPI {version} installed to {_INSTALL_PREFIX}/openmpi-{version}",
             phase="ompi", version=version)


def _write_ompi_modulefile(inp: _Inputs, version: str, hpcx: _Hpcx, transport_lib: str,
                           report: _Report) -> None:
    mpi_dir = inp.mpi_dir
    _append_file(f"{mpi_dir}/openmpi-{version}",
                 openmpi_modulefile(version, hpcx.hcoll, transport_lib, inp.uses_ucx), report)
    _ln_s(f"{mpi_dir}/openmpi-{version}", f"{mpi_dir}/openmpi", report)


def _install_impi(env: dict[str, str], inp: _Inputs, report: _Report) -> None:
    if inp.impi is None:
        log_info(_OP, "Skipping Intel MPI on aarch64 (x86_64 only)", phase="impi",
                 skipped=True, arch=inp.architecture)
        return
    cfg = inp.impi
    version, url = cfg["version"], cfg["url"]
    impi_version = inp.impi_2021_version
    log_info(_OP, f"Intel MPI phase: installing Intel MPI {version}", phase="impi",
             version=version, url=url, impi_2021_version=impi_version)

    _download(url, cfg["sha256"], "the Intel MPI installer")
    _run(["bash", url_basename(url), "-s", "-a", "-s", "--eula", "accept"],
         "Intel MPI installer", env, _WORK_DIR)
    modulefiles = f"{_INSTALL_PREFIX}/intel/oneapi/mpi/{impi_version}/etc/modulefiles"
    _run(["mv", f"{modulefiles}/mpi", f"{modulefiles}/impi"],
         "renaming Intel MPI's modulefiles/mpi to modulefiles/impi", env, _WORK_DIR)
    log_info(_OP, f"Renamed {modulefiles}/mpi to {modulefiles}/impi",
             path=f"{modulefiles}/impi")
    _record("IMPI", version, report)
    _write_impi_modulefile(inp, report)


def _write_impi_modulefile(inp: _Inputs, report: _Report) -> None:
    mpi_dir = inp.mpi_dir
    impi_version = inp.impi_2021_version
    provider = impi_fi_provider(inp.network_mode)
    log_info(_OP, f"Intel MPI modulefile uses FI_PROVIDER={provider}",
             fi_provider=provider, network_mode=inp.network_mode)
    impi_file = f"{mpi_dir}/impi_{impi_version}"
    _append_file(impi_file, impi_modulefile(impi_version, provider), report)
    _ln_s(impi_file, f"{mpi_dir}/impi-2021", report)
    if is_el8(inp.distribution):
        log_info(_OP, "EL8: adding I_MPI_STARTUP_MODE=pmi_shm to the Intel MPI "
                      "modulefile (Intel MPI collectives race condition)",
                 distro=inp.distribution)
        _append_file(impi_file, _IMPI_EL8_APPEND, report, new_file=False)


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def _download_config(component: str, env: dict[str, str]) -> dict | None:
    """config_for() + check version/url/sha256; values as strings. None if unusable."""
    cfg = config_for(component, env)
    if not isinstance(cfg, dict):
        log_error(_OP, f"could not resolve {component} from versions.json",
                  component=component)
        return None
    missing = [key for key in ("version", "url", "sha256") if not cfg.get(key)]
    if missing:
        log_error(_OP, f"{component} config in versions.json is missing "
                       f"{', '.join(missing)}", component=component, missing=missing)
        return None
    return {key: str(cfg[key]) for key in ("version", "url", "sha256")}


def _resolve_inputs(env: dict[str, str]) -> _Inputs | None:
    """Resolve every config this run needs, before touching the system."""
    distribution = env.get("DISTRIBUTION", "")
    architecture = env.get("ARCHITECTURE", "")
    gpu, sku = env.get("GPU", ""), env.get("SKU", "")
    sku_family = env.get("SKU_FAMILY", "")
    node_type = env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE
    network_mode = sku_network_mode(sku, node_type)
    uses_ucx = sku_uses_ucx(sku)

    module_dir = env.get("MODULE_FILES_DIRECTORY", "")
    if not module_dir:
        log_error(_OP, "MODULE_FILES_DIRECTORY is not set")
        return None

    hpcx_component, hpcx_reason = hpcx_variant(gpu, network_mode)
    hpcx = _download_config(hpcx_component, env)
    if hpcx is None:
        return None

    pmix_reason = internal_pmix_reason(distribution, node_type)
    internal_pmix = pmix_reason is not None
    pmix_version, pmix_path = None, None
    if not internal_pmix:
        pmix = config_for("pmix", env)
        pmix_version = str(pmix.get("version") or "") if isinstance(pmix, dict) else ""
        pmix_short = drop_last_two(pmix_version)
        if pmix_short is None:
            log_error(_OP, f"could not resolve a usable pmix version from versions.json "
                           f"(got {pmix_version!r}; PMIX_PATH needs ${{PMIX_VERSION:0:-2}})",
                      component="pmix", version=pmix_version)
            return None
        pmix_path = f"{_INSTALL_PREFIX}/pmix/{pmix_short}"

    ucx_reason = rebuild_ucx_reason(gpu, distribution, uses_ucx)
    rebuild_ucx = ucx_reason is not None
    rebuild_hpcx = needs_hpcx_rebuild(rebuild_ucx=rebuild_ucx, internal_pmix=internal_pmix,
                                      gpu=gpu, uses_ucx=uses_ucx)

    mvapich = None
    if not skips_mvapich(distribution, sku_family):
        mvapich = _download_config("mvapich", env)
        if mvapich is None:
            return None

    ompi = None
    if not skips_ompi(distribution):
        ompi = _download_config("ompi", env)
        if ompi is None:
            return None

    impi, impi_2021_version = None, None
    if architecture != "aarch64":
        impi = _download_config("impi", env)
        if impi is None:
            return None
        impi_2021_version = drop_last_two(impi["version"])
        if impi_2021_version is None:
            log_error(_OP, f"impi version {impi['version']!r} is too short for "
                           "${IMPI_VERSION:0:-2}", component="impi", version=impi["version"])
            return None

    return _Inputs(
        distribution=distribution, architecture=architecture, gpu=gpu, sku=sku,
        sku_family=sku_family, node_type=node_type, network_mode=network_mode,
        uses_ucx=uses_ucx, module_dir=module_dir,
        component_dir=env.get("COMPONENT_DIR", ""),
        hpcx_component=hpcx_component, hpcx_reason=hpcx_reason, hpcx=hpcx,
        internal_pmix=internal_pmix, internal_pmix_reason=pmix_reason,
        pmix_version=pmix_version, pmix_path=pmix_path,
        rebuild_ucx=rebuild_ucx, rebuild_ucx_reason=ucx_reason, rebuild_hpcx=rebuild_hpcx,
        mvapich=mvapich, ompi=ompi, impi=impi, impi_2021_version=impi_2021_version)


def install(env: dict[str, str]) -> int:
    """Install HPC-X, MVAPICH, Open MPI and Intel MPI plus modulefiles. Returns 0 or 3.

    `env` is the build environment made by component_env() in build_config.py.
    """
    inp = _resolve_inputs(env)
    if inp is None:
        return 3

    log_info(_OP, f"Installing MPIs on {inp.distribution} ({inp.architecture}) for "
                  f"{inp.gpu} {inp.sku}",
             distro=inp.distribution, arch=inp.architecture, gpu=inp.gpu, sku=inp.sku,
             sku_family=inp.sku_family, node_type=inp.node_type,
             network_mode=inp.network_mode, uses_ucx=inp.uses_ucx,
             hpcx_component=inp.hpcx_component, hpcx_version=inp.hpcx["version"],
             internal_pmix=inp.internal_pmix,
             pmix_version=inp.pmix_version or "internal",
             pmix_path=inp.pmix_path or "internal",
             rebuild_ucx=inp.rebuild_ucx, rebuild_hpcx=inp.rebuild_hpcx,
             mvapich_version=inp.mvapich["version"] if inp.mvapich else "skipped",
             ompi_version=inp.ompi["version"] if inp.ompi else "skipped",
             impi_version=inp.impi["version"] if inp.impi else "skipped",
             module_dir=inp.module_dir)
    log_info(_OP, f"HPC-X variant {inp.hpcx_component}: {inp.hpcx_reason}",
             hpcx_component=inp.hpcx_component, gpu=inp.gpu, network_mode=inp.network_mode)
    if inp.internal_pmix:
        log_info(_OP, f"Open MPI uses its internal PMIx, hwloc and libevent: "
                      f"{inp.internal_pmix_reason}", internal_pmix=True,
                 distro=inp.distribution, node_type=inp.node_type)
    else:
        log_info(_OP, f"PMIx path for the MPI builds: {inp.pmix_path}",
                 internal_pmix=False, pmix_path=inp.pmix_path, pmix_version=inp.pmix_version)
    log_info(_OP, f"HPC-X rebuild: {'yes' if inp.rebuild_hpcx else 'no'}, UCX rebuild: "
                  f"{inp.rebuild_ucx_reason or 'no'}", rebuild_hpcx=inp.rebuild_hpcx,
             rebuild_ucx=inp.rebuild_ucx)
    # BASH ISSUE (kept): `set CC=/usr/bin/gcc` and `set GCC=/usr/bin/gcc` don't
    # set CC/GCC; bash's `set` replaces the positional parameters ($1). So the
    # builds use the environment's default compiler; we don't set CC either.
    log_warn(_OP, "bash's `set CC=/usr/bin/gcc` / `set GCC=/usr/bin/gcc` do not set "
                  "CC or GCC (`set` replaces the positional parameters); the builds "
                  "use the default compiler, as in bash", bash_line="set CC=/usr/bin/gcc")

    nproc = _nproc(env)
    if nproc is None:
        return 3

    report = _Report()
    try:
        # bash: mkdir -p ${MPI_MODULE_FILES_DIRECTORY}, before anything else.
        os.makedirs(inp.mpi_dir, exist_ok=True)
        log_info(_OP, f"MPI modulefiles go to {inp.mpi_dir}", path=inp.mpi_dir)
        # Start from an empty work directory (see the module docstring).
        shutil.rmtree(_WORK_DIR, ignore_errors=True)
        os.makedirs(_WORK_DIR, exist_ok=True)

        hpcx = _install_hpcx(env, inp, report)
        _install_mvapich(env, inp, hpcx, nproc, report)
        _install_ompi(env, inp, hpcx, nproc, report)
        if is_rpm_distro(inp.distribution):
            _pin(["openmpi", "perftest"], env, report)   # exclude from updates
        _install_impi(env, inp, report)
    except _StepFailed:
        return 3
    except OSError as exc:
        # Work dir, ld.so.conf.d, pkg-config links, modulefiles, versions file.
        log_error(_OP, f"file operation failed: {exc}",
                  path=getattr(exc, "filename", None))
        return 3
    finally:
        # Our own cleanup: only the scratch area this step created.
        shutil.rmtree(_WORK_DIR, ignore_errors=True)

    # BASH ISSUE (not ported): on success bash finishes with `rm -rf -- */` in
    # the distro build dir, deleting every subdirectory there, not just its own.
    log_warn(_OP, "bash ends with `rm -rf -- */` in the distro build dir, deleting "
                  "every subdirectory there; the Python port only removed its own "
                  f"work dir {_WORK_DIR}", work_dir=_WORK_DIR)
    log_info(_OP, "MPIs installed: " + ", ".join(f"{k}={v}" for k, v in report.recorded.items()),
             recorded=report.recorded, hpcx_path=hpcx.path, ucx_path=hpcx.ucx,
             rebuilt_hpcx=inp.rebuild_hpcx, internal_pmix=inp.internal_pmix,
             files=report.files, symlinks=len(report.symlinks), pinned=report.pinned,
             patched=report.patched)
    return 0
