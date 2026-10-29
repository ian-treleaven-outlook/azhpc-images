"""install_mpis.py — Python port of components/install_mpis.sh.

What this step does
-------------------
Installs the MPI (Message Passing Interface) libraries that HPC jobs use to
talk between processes and nodes, and the environment-module files that let a
user pick one with `module load mpi/<name>`:

  1. HPC-X       NVIDIA's MPI bundle (Open MPI + UCX + HCOLL + SHARP),
                 unpacked into /opt and rebuilt against PMIx with the bundled
                 hpcx_rebuild.sh. Which HPC-X build is used depends on the GPU
                 vendor and network (see hpcx_variant()). Afterwards its SHARP
                 and UCX libraries are registered with the dynamic linker
                 (/etc/ld.so.conf.d) and its pkg-config files are symlinked
                 into /usr/local/lib(64)/pkgconfig.
  2. MVAPICH     built from source into /opt/mvapich-<version> (skipped on
                 GB200/GB300 with Ubuntu 24.04 or Azure Linux 3.0).
  3. Open MPI    built from source into /opt/openmpi-<version>.
  4. Intel MPI   installed with Intel's offline installer (x86_64 only).
  5. Modulefiles under $MODULE_FILES_DIRECTORY/mpi for each of the above, plus
                 the short names (mpi/hpcx, mpi/openmpi, ...) as symlinks.

Every installed version is recorded in the component-versions file (HPCX,
MVAPICH, OMPI, IMPI). On RPM distros (AlmaLinux, Rocky, RHEL, Azure Linux 3.0)
ucx*, openmpi and perftest are version-locked against later dnf updates.

Two network flavours run through the whole script:
  * UCX SKUs (every SKU with InfiniBand): MPIs use UCX (+ HCOLL).
  * Non-UCX SKUs (NCv6, network mode "no_rdma"): MPIs use libfabric
    (/opt/libfabric) with the tcp provider, and the modulefiles get extra
    settings that avoid hangs on MANA-only hardware.

The original bash, for comparison (abridged; see components/install_mpis.sh):

    set CC=/usr/bin/gcc; set GCC=/usr/bin/gcc
    INSTALL_PREFIX=/opt
    PMIX_PATH=${INSTALL_PREFIX}/pmix/${PMIX_VERSION:0:-2}
    hpcx_metadata = hpcx_amd (AMD) | hpcx_inbox (no_rdma) | hpcx
    download_and_verify ${HPCX_DOWNLOAD_URL} ${HPCX_SHA256}; tar -xvf ${TARBALL}
    mv ${HPCX_FOLDER} ${INSTALL_PREFIX}
    HPCX_DIR=${HPCX_PATH} ${HPCX_PATH}/utils/hpcx_fix_ladir.sh
    write_component_version "HPCX" $HPCX_VERSION
    AMD + UCX: [[ -d /opt/rocm ]] || exit 1
               HPCX_REBUILD_UCX_ARGS=(--rebuild-ucx --ucx-extra-config "--with-rocm=/opt/rocm")
    ${HPCX_PATH}/utils/hpcx_rebuild.sh ...   (3 variants, see hpcx_rebuild_command())
    cp -r ${HPCX_PATH}/ompi/tests ${HPCX_PATH}/hpcx-rebuild
    UCX_PATH=${HPCX_PATH}/ucx/hpcx-rebuild   (only when UCX was rebuilt)
    HPCX_DIR=${HPCX_PATH} ${HPCX_PATH}/utils/hpcx_fix_ladir.sh
    cat > /etc/ld.so.conf.d/hpcx-sharp.conf / hpcx-ucx.conf; ldconfig
    ln -sf <hcoll|sharp|ucx>/lib/pkgconfig/*.pc /usr/local/lib[64]/pkgconfig/
    RPM distros: dnf_pin_packages "ucx*"
    MVAPICH (unless GB-family on ubuntu24.04/azurelinux3.0):
        download, tar, ./configure [FFLAGS=...] --prefix=... <UCX|OFI args>,
        make -j$(nproc), make install, write_component_version "MVAPICH"
    Open MPI: download, tar, ./configure <UCX|OFI variant>, make, make install,
        write_component_version "OMPI"; RPM distros: dnf_pin_packages openmpi perftest
    Intel MPI (not aarch64): download, bash <installer> -s -a -s --eula accept,
        mv .../etc/modulefiles/mpi .../etc/modulefiles/impi, write_component_version "IMPI"
    cat << EOF >> modulefiles; ln -s short names
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
    mv, cp, hpcx scripts, configure, make, ldconfig, the Intel installer) are
    still run, so they behave exactly as before. Each phase stops at the first
    failing command.
  * Generated files are byte-for-byte what bash writes. The modulefiles were
    checked against the real bash heredocs (tests/python/fixtures/install_mpis/,
    rendered with bash for UCX, non-UCX, EL8 and GB200/aarch64 targets).
    When the port was written, the real bash script and this module were
    also run side by side against the same stubbed commands (tar, mv, cp,
    make, ldconfig, hpcx scripts, configure, installer) for 11 targets plus
    a rerun: the commands, arguments, recorded versions, pins and resulting
    files/symlinks were identical.
  * Modulefiles are APPENDED to (bash `cat >> file`) and the short-name
    symlinks are made with plain `ln -s` (no -f), so a rerun on the same image
    duplicates the modulefile content and then fails at the first symlink —
    see "Bash issues" below.
  * The /etc/ld.so.conf.d files are replaced (`cat >`), and the pkg-config
    symlinks are replaced if present (`ln -sf`).
  * Open MPI's UCX configure passes LD_LIBRARY_PATH=<caller's
    LD_LIBRARY_PATH>:<hcoll>/lib as a configure ARGUMENT (not an environment
    variable), with an empty caller value giving ":<hcoll>/lib", as in bash.
  * The /opt/rocm check, the baremetal_3p "bundled PMIx" case and every
    distro/SKU gate are kept.

Deliberate differences from bash
--------------------------------
  * All configs (pmix, the HPC-X variant, mvapich unless skipped, ompi, impi
    on x86_64) are resolved and checked BEFORE anything is downloaded. A
    missing version/url/sha256 fails the step (3). Bash resolves each one just
    before using it and carries on with blank values, e.g. installing HPC-X
    and only then building MVAPICH from an empty URL.
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
"""

# Lets type hints such as `str | None` work on older Python versions.
from __future__ import annotations

import glob
import os
import shutil
from dataclasses import dataclass, field

from utils.component_config import DEFAULT_NODE_TYPE, config_for, write_component_version
from utils.download import download_and_verify
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.package_pinning import dnf_pin_packages
from utils.process import exec_program, run_capture
from utils.sku import NETWORK_MODE_NO_RDMA, sku_network_mode, sku_uses_ucx

# The Step's `op` label in build_config.py; used on every log line.
_OP = "install-mpis"

_AZURE_LINUX = "azurelinux3.0"
_GB_FAMILY = "gb-family"
_BAREMETAL_3P = "baremetal_3p"
# MVAPICH is skipped for gb-family SKUs on these two distros (exact matches).
_MVAPICH_SKIP_DISTROS = ("ubuntu24.04", _AZURE_LINUX)
# Intel MPI's EL8 race-condition workaround applies to these (plus rhel8*).
_EL8_DISTROS = ("almalinux8.10", "rocky8.10")

# ---- Paths (module-level so tests can point them at a temporary directory) --
_INSTALL_PREFIX = "/opt"                      # bash: INSTALL_PREFIX=/opt
_WORK_DIR = "/tmp/azhpc-mpis"                 # our scratch area (see docstring)
_ROCM_DIR = "/opt/rocm"                       # only for the "is ROCm installed" check
_LD_SO_CONF_DIR = "/etc/ld.so.conf.d"
_PKGCONFIG_DIRS = ("/usr/local/lib/pkgconfig",)
_PKGCONFIG_DIRS_RPM = ("/usr/local/lib/pkgconfig", "/usr/local/lib64/pkgconfig")

# Used only inside generated text and configure arguments (literal in bash).
_LIBFABRIC_PATH = "/opt/libfabric"
_ROCM_UCX_ARGS = ("--rebuild-ucx", "--ucx-extra-config", "--with-rocm=/opt/rocm")


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


def uses_bundled_pmix(distribution: str, node_type: str) -> bool:
    """bash: [[ $DISTRIBUTION == "azurelinux3.0" || ${TARGET_NODE_TYPE:-...} == "baremetal_3p" ]].

    These targets build against the PMIx bundled in HPC-X (`--with-pmix` with
    no path): on baremetal nodes the standalone PMIx conflicts with the
    Mellanox Open MPI package.
    """
    return distribution == _AZURE_LINUX or (node_type or DEFAULT_NODE_TYPE) == _BAREMETAL_3P


def is_rpm_distro(distribution: str) -> bool:
    """bash: [[ $D == almalinux* ]] || [[ $D == rocky* ]] || [[ $D == rhel* ]] || [[ $D == "azurelinux3.0" ]]."""
    return distribution.startswith(("almalinux", "rocky", "rhel")) or distribution == _AZURE_LINUX


def skips_mvapich(distribution: str, sku_family: str) -> bool:
    """bash: ("$DISTRIBUTION" == ubuntu24.04 || == azurelinux3.0) && "$SKU_FAMILY" == gb-family."""
    return distribution in _MVAPICH_SKIP_DISTROS and sku_family == _GB_FAMILY


def is_el8(distribution: str) -> bool:
    """bash: [[ $D == "almalinux8.10" ]] || [[ $D == "rocky8.10" ]] || [[ $D == rhel8* ]]."""
    return distribution in _EL8_DISTROS or distribution.startswith("rhel8")


def impi_fi_provider(network_mode: str) -> str:
    """Intel MPI's libfabric provider: "tcp" on no_rdma, else "mlx"."""
    return "tcp" if network_mode == NETWORK_MODE_NO_RDMA else "mlx"


def hpcx_rebuild_command(hpcx_path: str, *, bundled_pmix: bool, uses_ucx: bool,
                         ucx_args: tuple[str, ...], pmix_path: str) -> list[str]:
    """The hpcx_rebuild.sh command line, one of bash's three variants.

    ucx_args is HPCX_REBUILD_UCX_ARGS (empty unless AMD + UCX).
    """
    script = f"{hpcx_path}/utils/hpcx_rebuild.sh"
    if bundled_pmix:
        return [script, "--with-hcoll", *ucx_args, "--ompi-extra-config",
                "--with-pmix --enable-orterun-prefix-by-default"]
    if not uses_ucx:
        return [script, "--ompi-extra-config",
                f"--with-pmix={pmix_path} --enable-orterun-prefix-by-default "
                f"--without-ucx --with-ofi={_LIBFABRIC_PATH}"]
    return [script, "--with-hcoll", *ucx_args, "--ompi-extra-config",
            f"--with-pmix={pmix_path} --enable-orterun-prefix-by-default"]


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


def ompi_configure_command(version: str, *, uses_ucx: bool, pmix_flag: str,
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
            pmix_flag, "--enable-mpirun-prefix-by-default",
            "--with-platform=contrib/platform/mellanox/optimized",
        ], f"{ucx_path}/lib")
    # No UCX: drop --with-ucx, --with-hcoll (uses UCX) and the Mellanox platform file.
    return ([
        "./configure", prefix, "--without-ucx", f"--with-ofi={_LIBFABRIC_PATH}",
        pmix_flag, "--enable-mpirun-prefix-by-default",
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


def hpcx_modulefile(version: str, hpcx_path: str, uses_ucx: bool) -> str:
    """Text of mpi/hpcx-<version> and mpi/hpcx-pmix-<version> (bash writes the same)."""
    extras = "" if uses_ucx else _HPCX_NON_UCX_EXTRAS
    return ("#%Module 1.0\n"
            "#\n"
            f"#  HPCx {version}\n"
            "#\n"
            "conflict        mpi\n"
            f"module load {hpcx_path}/modulefiles/hpcx-rebuild\n"
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
    hpcx_component: str
    hpcx_reason: str
    hpcx: dict
    pmix_version: str
    pmix_path: str
    mvapich: dict | None            # None: skipped (GB-family)
    ompi: dict
    impi: dict | None               # None: skipped (aarch64)
    impi_2021_version: str | None


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

    ucx_args: tuple[str, ...] = ()
    if inp.gpu == "AMD" and inp.uses_ucx:
        if not os.path.isdir(_ROCM_DIR):
            log_error(_OP, "ROCm must be installed before rebuilding HPC-X UCX with "
                           "ROCm support.", path=_ROCM_DIR)
            raise _StepFailed
        ucx_args = _ROCM_UCX_ARGS
        log_info(_OP, "AMD GPU with UCX: rebuilding HPC-X's UCX with ROCm support",
                 rebuild_ucx_args=" ".join(ucx_args))

    bundled = uses_bundled_pmix(inp.distribution, inp.node_type)
    if bundled:
        why = (f"{inp.distribution}/{inp.node_type}: PMIx bundled in HPC-X "
               "(standalone PMIx conflicts with the Mellanox Open MPI package on baremetal)")
    elif not inp.uses_ucx:
        why = f"non-UCX SKU: standalone PMIx {inp.pmix_path}, OFI/libfabric instead of UCX"
    else:
        why = f"UCX SKU: standalone PMIx {inp.pmix_path} with HCOLL"
    rebuild = hpcx_rebuild_command(hpcx_path, bundled_pmix=bundled, uses_ucx=inp.uses_ucx,
                                   ucx_args=ucx_args, pmix_path=inp.pmix_path)
    log_info(_OP, f"Rebuilding HPC-X Open MPI: {why}", phase="hpcx-rebuild",
             bundled_pmix=bundled, uses_ucx=inp.uses_ucx, rebuild_args=" ".join(rebuild[1:]))
    _run(rebuild, "hpcx_rebuild.sh", env, _WORK_DIR)
    _run(["cp", "-r", f"{hpcx_path}/ompi/tests", f"{hpcx_path}/hpcx-rebuild"],
         "copying the HPC-X ompi tests", env, _WORK_DIR)

    if ucx_args:
        hpcx.ucx = f"{hpcx_path}/ucx/hpcx-rebuild"
        log_info(_OP, f"Using the rebuilt UCX at {hpcx.ucx}", ucx_path=hpcx.ucx)
    # hpcx_rebuild.sh installs fresh Open MPI (and, on AMD, UCX) metadata; fix it too.
    _run(fix_ladir, "hpcx_fix_ladir.sh (after rebuild)", fix_env, _WORK_DIR)

    _write_file(f"{_LD_SO_CONF_DIR}/hpcx-sharp.conf", f"{hpcx.sharp}/lib\n", report)
    _write_file(f"{_LD_SO_CONF_DIR}/hpcx-ucx.conf", f"{hpcx.ucx}/lib\n", report)
    _run(["ldconfig"], "ldconfig", env, _WORK_DIR)

    _link_pkgconfig(inp, hpcx, report)

    if is_rpm_distro(inp.distribution):
        _pin(["ucx*"], env, report)          # exclude ucx from updates
    log_info(_OP, f"HPC-X {version} installed at {hpcx_path}", phase="hpcx",
             version=version, path=hpcx_path, ucx_path=hpcx.ucx)
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


def _install_mvapich(env: dict[str, str], inp: _Inputs, hpcx: _Hpcx, nproc: int,
                     report: _Report) -> str | None:
    """Build MVAPICH; returns its transport lib dir, or None when skipped."""
    if inp.mvapich is None:
        log_info(_OP, f"Skipping MVAPICH: not supported on {inp.distribution} for "
                      f"{inp.sku_family} SKUs (bash skips ubuntu24.04/azurelinux3.0 "
                      "+ gb-family)", phase="mvapich", skipped=True,
                 distro=inp.distribution, sku_family=inp.sku_family)
        return None
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
    log_info(_OP, f"MVAPICH {version} installed to {_INSTALL_PREFIX}/mvapich-{version}",
             phase="mvapich", version=version)
    return transport_lib


def _install_ompi(env: dict[str, str], inp: _Inputs, hpcx: _Hpcx, nproc: int,
                  report: _Report) -> str:
    """Build Open MPI; returns its transport lib dir."""
    cfg = inp.ompi
    version, url = cfg["version"], cfg["url"]
    folder = url_basename(url, ".tar.gz")
    if uses_bundled_pmix(inp.distribution, inp.node_type):
        pmix_flag = "--with-pmix"
    else:
        pmix_flag = f"--with-pmix={inp.pmix_path}"
    configure, transport_lib = ompi_configure_command(
        version, uses_ucx=inp.uses_ucx, pmix_flag=pmix_flag, ucx_path=hpcx.ucx,
        hcoll_path=hpcx.hcoll, ld_library_path=env.get("LD_LIBRARY_PATH", ""))
    log_info(_OP, f"Open MPI phase: building Open MPI {version} "
                  f"({'UCX + HCOLL' if inp.uses_ucx else 'OFI/libfabric'} transport)",
             phase="ompi", version=version, url=url, pmix_flag=pmix_flag,
             transport_lib=transport_lib)

    _download(url, cfg["sha256"], "Open MPI")
    _run(["tar", "-xvf", url_basename(url)], "extracting Open MPI", env, _WORK_DIR)
    source = f"{_WORK_DIR}/{folder}"
    _require_dir(source, "Open MPI source")
    _run(configure, "Open MPI configure", env, source)
    _run(["make", f"-j{nproc}"], "Open MPI make", env, source)
    _run(["make", "install"], "Open MPI make install", env, source)
    _record("OMPI", version, report)

    if is_rpm_distro(inp.distribution):
        _pin(["openmpi", "perftest"], env, report)   # exclude from updates
    log_info(_OP, f"Open MPI {version} installed to {_INSTALL_PREFIX}/openmpi-{version}",
             phase="ompi", version=version)
    return transport_lib


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


def _write_modulefiles(inp: _Inputs, hpcx: _Hpcx, mvapich_lib: str | None,
                       ompi_lib: str, report: _Report) -> None:
    """Write the MPI modulefiles and symlinks, in bash's order."""
    mpi_dir = f"{inp.module_dir}/mpi"
    os.makedirs(mpi_dir, exist_ok=True)
    log_info(_OP, f"Writing MPI modulefiles to {mpi_dir}", phase="modulefiles",
             path=mpi_dir, uses_ucx=inp.uses_ucx)
    if not inp.uses_ucx:
        log_info(_OP, "Non-UCX SKU: adding libfabric tcp / PML cm settings to the "
                      "HPC-X, MVAPICH and Open MPI modulefiles")

    hpcx_version = inp.hpcx["version"]
    hpcx_text = hpcx_modulefile(hpcx_version, hpcx.path, inp.uses_ucx)
    # mpi/hpcx is the public HPC-X entrypoint; both files load the rebuilt HPC-X.
    _append_file(f"{mpi_dir}/hpcx-{hpcx_version}", hpcx_text, report)
    _append_file(f"{mpi_dir}/hpcx-pmix-{hpcx_version}", hpcx_text, report)

    if inp.mvapich is not None:
        mvapich_version = inp.mvapich["version"]
        _append_file(f"{mpi_dir}/mvapich-{mvapich_version}",
                     mvapich_modulefile(mvapich_version, mvapich_lib, inp.uses_ucx), report)
        _ln_s(f"{mpi_dir}/mvapich-{mvapich_version}", f"{mpi_dir}/mvapich", report)

    ompi_version = inp.ompi["version"]
    _append_file(f"{mpi_dir}/openmpi-{ompi_version}",
                 openmpi_modulefile(ompi_version, hpcx.hcoll, ompi_lib, inp.uses_ucx), report)

    if inp.impi is not None:
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

    _ln_s(f"{mpi_dir}/hpcx-{hpcx_version}", f"{mpi_dir}/hpcx", report)
    _ln_s(f"{mpi_dir}/hpcx-pmix-{hpcx_version}", f"{mpi_dir}/hpcx-pmix", report)
    _ln_s(f"{mpi_dir}/openmpi-{ompi_version}", f"{mpi_dir}/openmpi", report)


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

    module_dir = env.get("MODULE_FILES_DIRECTORY", "")
    if not module_dir:
        log_error(_OP, "MODULE_FILES_DIRECTORY is not set")
        return None

    pmix = config_for("pmix", env)
    pmix_version = str(pmix.get("version") or "") if isinstance(pmix, dict) else ""
    pmix_short = drop_last_two(pmix_version)
    if pmix_short is None:
        log_error(_OP, f"could not resolve a usable pmix version from versions.json "
                       f"(got {pmix_version!r}; PMIX_PATH needs ${{PMIX_VERSION:0:-2}})",
                  component="pmix", version=pmix_version)
        return None

    hpcx_component, hpcx_reason = hpcx_variant(gpu, network_mode)
    hpcx = _download_config(hpcx_component, env)
    if hpcx is None:
        return None

    mvapich = None
    if not skips_mvapich(distribution, sku_family):
        mvapich = _download_config("mvapich", env)
        if mvapich is None:
            return None

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
        uses_ucx=sku_uses_ucx(sku), module_dir=module_dir,
        hpcx_component=hpcx_component, hpcx_reason=hpcx_reason, hpcx=hpcx,
        pmix_version=pmix_version, pmix_path=f"{_INSTALL_PREFIX}/pmix/{pmix_short}",
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
             pmix_version=inp.pmix_version, pmix_path=inp.pmix_path,
             mvapich_version=inp.mvapich["version"] if inp.mvapich else "skipped",
             ompi_version=inp.ompi["version"],
             impi_version=inp.impi["version"] if inp.impi else "skipped",
             module_dir=inp.module_dir)
    log_info(_OP, f"HPC-X variant {inp.hpcx_component}: {inp.hpcx_reason}",
             hpcx_component=inp.hpcx_component, gpu=inp.gpu, network_mode=inp.network_mode)
    log_info(_OP, f"PMIx path for the MPI builds: {inp.pmix_path}",
             pmix_path=inp.pmix_path, pmix_version=inp.pmix_version)
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
        # Start from an empty work directory (see the module docstring).
        shutil.rmtree(_WORK_DIR, ignore_errors=True)
        os.makedirs(_WORK_DIR, exist_ok=True)

        hpcx = _install_hpcx(env, inp, report)
        mvapich_lib = _install_mvapich(env, inp, hpcx, nproc, report)
        ompi_lib = _install_ompi(env, inp, hpcx, nproc, report)
        _install_impi(env, inp, report)
        _write_modulefiles(inp, hpcx, mvapich_lib, ompi_lib, report)
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
             files=report.files, symlinks=len(report.symlinks), pinned=report.pinned)
    return 0
