"""install_doca.py — Python port of components/install_doca.sh.

What this step does
-------------------
DOCA-OFED is NVIDIA's packaging of the InfiniBand / RDMA driver stack (the
successor to MLNX_OFED). On InfiniBand SKUs this step:

  1. Gets the DOCA host repo package: downloads and checksums it, or, for a
     "private" entry in versions.json, checksums the copy in internal_bits/.
  2. Installs the repo package and, per distro family, a marker package that
     tells apt/dnf "HPC-X provides Open MPI/UCX/SHARP", so DOCA does not
     install its own copies (HPC-X is installed later by install_mpis.sh).
  3. Adds DKMS patch overlays under /etc/dkms that DKMS applies whenever it
     (re)builds the DOCA kernel modules, then installs doca-ofed and checks
     DKMS really built the modules for the running kernel.
  4. Records DOCA and OFED in the component versions file, and sets up the
     openibd boot service (and turns IPoIB off where the target doesn't use it).

Per distro family (bash: *"ubuntu"* / == "azurelinux3.0" / else):

  * Ubuntu        `apt install` the repo .deb; pin DOCA's old dkms out (the
                  CUDA repo's newer dkms is installed earlier by
                  install_cuda_repo_and_dkms.sh, from the bash bootstrap
                  install_utils.sh); equivs marker package; DKMS patches
                  for mlnx-ofed-kernel; doca-ofed; DKMS status check.
  * Azure Linux   `rpm -i` the repo rpm; rpmbuild marker package; doca-ofed.
                  No DKMS patches or check (as in bash).
  * RHEL family   (AlmaLinux, Rocky, RHEL) like Azure Linux, plus: dnf.conf's
                  exclude= lines are lifted during the install, DKMS patches
                  for mlnx-ofa_kernel (incl. the DPLL patch when the kernel
                  needs it), DKMS status check, and a baseos excludepkgs pin
                  for packages DOCA replaces.

DKMS patch overlays (upstream #642, #633):

  * peer-mem-gpl.patch, every SKU except V100: exports the IB peer-memory API
    as GPL-only symbols, so DKMS can rebuild DOCA and the GPU driver in either
    order. V100's proprietary nvidia-peermem can't use GPL-only exports.
    Written to /etc/dkms/<module>/patches/ and listed (overwriting) in
    /etc/dkms/<module>.conf.
  * dpll-ffo-param.patch, RHEL family only, when the running kernel's
    linux/dpll.h has Red Hat's newer DPLL callback: appended to that .conf.
  * xdp-fragments.patch, Ubuntu 22.04 with DOCA 3.2.3 only: listed in the
    version-specific /etc/dkms/mlnx-ofed-kernel-25.10.OFED.25.10.3.1.8.1-1.conf.

The original bash, abridged (heredoc texts are in the constants below, and
tests compare them with install_doca.sh):

    if [[ "$DOCA_SOURCE" == "private" ]]; then
        DOCA_FILE="$TOP_DIR/internal_bits/$(jq -r '.repo_file')"
        verify_checksum "$DOCA_FILE" "$DOCA_SHA256"
    else
        download_and_verify $DOCA_URL $DOCA_SHA256      # into the working dir
        DOCA_FILE=$(basename ${DOCA_URL})
    fi
    if ubuntu:
        apt install -y "$(realpath "${DOCA_FILE}")"
        cat > /etc/apt/preferences.d/doca-dkms-pin      # dkms from DOCA-HOST: -1
        apt-get update
        install_hpcx_doca_ofed_deps_apt_marker          # equivs; sharp only < 3.3.0
        configure_doca_dkms_patches mlnx-ofed-kernel
        configure_mlnx_ofed_kernel_dkms_xdp_patch       # ubuntu22.04 + DOCA 3.2.3
        apt-get -y install doca-ofed
        check_dkms_status mlnx-ofed-kernel iser isert srp
    elif azurelinux3.0:
        download_and_verify $DOCA_URL $DOCA_SHA256      # again (see bash bugs)
        rpm -i $DOCA_FILE; dnf clean all
        install_hpcx_doca_ofed_deps_rpm_marker
        dnf -y install doca-ofed
    else:
        download_and_verify $DOCA_URL $DOCA_SHA256      # again
        rpm -i $DOCA_FILE; dnf clean all
        cp /etc/dnf/dnf.conf /etc/dnf/dnf.conf.bak; sed -i '/^exclude=/d' /etc/dnf/dnf.conf
        install_hpcx_doca_ofed_deps_rpm_marker
        configure_doca_dkms_patches mlnx-ofa_kernel
        dnf -y install doca-ofed
        check_dkms_status mlnx-ofa_kernel iser isert srp
        mv /etc/dnf/dnf.conf.bak /etc/dnf/dnf.conf
        doca_pkgs = installed packages from doca* repos (error if none)
        baseos_repo = baseos, or rhel-<major>-for-<arch>-baseos-rhui-rpms on rhel*
        conflicts = doca_pkgs that baseos also offers
        [conflicts]: dnf config-manager --save --setopt="<repo>.excludepkgs=<conflicts>"
    write_component_version "DOCA" $DOCA_VERSION
    write_component_version "OFED" $(ofed_info | sed -n 1p | awk -F- '{print $3"-"$4}' | tr -d ':')
    cat > /etc/systemd/system/openibd.service.d/override.conf
    sku_uses_ipoib || echo -e "\\n# Load IPoIB\\nIPOIB_LOAD=no" | tee -a /etc/infiniband/openib.conf
    systemctl daemon-reload; systemctl enable openibd

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets this module's install() as the `action=` of the
"install-doca" Step (InfiniBand SKUs only). Deleting `action=` makes the
build run the bash script again, which is why the .sh file is kept.
install() returns 0 for success or 3 for failure; it logs problems and never
lets an exception escape.

Behaviour kept from bash
------------------------
  * Same commands, arguments and order; the first failing command stops the
    step (bash's `set -e`). File contents are byte-for-byte bash's heredocs,
    with the same overwrite (`cat >`) or append (`cat >>`, `tee -a`).
  * The public package is downloaded into bash's working directory, the
    distro build dir (distros/<distro>), where install.sh's cleanup removes
    *.deb files, and `rpm -i` gets bash's relative file name there.
  * `dpkg --compare-versions` decides "DOCA < 3.3.0" (exit 0 = yes).
  * The marker packages' temp files are removed only after success, as bash.

Deliberate differences from bash
--------------------------------
  * Installs go through install_packages(), which spells them
    `apt-get install -y -o Dpkg::Use-Pty=0` / `dnf install -y`.
  * Missing config fails up front: version always; url + sha256 for a public
    package; repo_file + sha256 for a private one.
  * OFED: a failing `ofed_info` or a first line with fewer than four
    '-'-separated fields fails the step. Bash (no pipefail) would record ""
    or a fragment such as "-".
  * RHEL family: dnf.conf is restored from the .bak even if the install
    fails (bash's `set -e` exits first and leaves the edited copy).
  * Package lists bash sorts with `sort -u` (locale order) are sorted in
    plain character order. Only the order of names in the marker spec's
    Provides lines and in excludepkgs can differ; rpm/dnf ignore it.
  * The repeated download on Azure Linux / RHEL (see below) replaces the
    file; bash's wget keeps the first copy and saves the second as "<name>.1".

Bash bugs kept (commented, log_warn when they apply)
----------------------------------------------------
  * Azure Linux and RHEL download the package a second time, although it was
    already downloaded and verified at the top. With a "private" source,
    DOCA_URL is unset there, so bash's download_and_verify fails; this port
    fails at the same point with a clear error.
  * `tee -a` appends the IPoIB lines again on every rerun.
  * `dpkg --compare-versions` errors (exit 2, e.g. an invalid version) count
    as "not older than 3.3.0", so the sharp provide is left out.
"""

# Lets type hints such as `str | None` work on older Python versions.
from __future__ import annotations

import glob
import os
import re
import shutil
from pathlib import Path

from utils.component_config import config_for, write_component_version
from utils.dkms import check_dkms_status, running_kernel
from utils.download import download_and_verify, verify_checksum
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import exec_program, run_capture
from utils.sku import sku_uses_ipoib

# The Step's `op` label in plans/steps.py; used on every log line.
_OP = "install-doca"

_AZURE_LINUX = "azurelinux3.0"

# ---- Paths (module-level so tests can point them at a temporary directory) --
_DKMS_PIN = "/etc/apt/preferences.d/doca-dkms-pin"
# DKMS reads /etc/dkms/<module>.conf and applies patches from
# /etc/dkms/<module>/patches/ (install_cuda_repo_and_dkms.sh installs a dkms
# new enough for these overlays).
_DKMS_ETC_DIR = "/etc/dkms"
_KERNEL_SRC_DIR = "/usr/src/kernels"
# bash builds the marker packages in /tmp.
_MARKER_WORK_DIR = "/tmp"
_DNF_CONF = "/etc/dnf/dnf.conf"
_OPENIBD_DROPIN_DIR = "/etc/systemd/system/openibd.service.d"
_OPENIB_CONF = "/etc/infiniband/openib.conf"

# bash: HPCX_DOCA_OFED_DEPS_MARKER
_MARKER = "hpcx-provides-doca-ofed-deps"

# The DKMS patch overlays: (file in components/patches/, name under /etc/dkms).
_PEER_MEM_PATCH = ("mlnx-ofa-kernel-peer-mem-gpl.patch", "peer-mem-gpl.patch")
_DPLL_PATCH = ("mlnx-ofa-kernel-dpll-ffo-param.patch", "dpll-ffo-param.patch")
_XDP_PATCH = ("mlnx-ofed-kernel-xdp-fragments.patch", "xdp-fragments.patch")
# What grep -q looks for in linux/dpll.h ('struct dpll_ffo_param \*ffo,').
_DPLL_FFO_SIGNATURE = "struct dpll_ffo_param *ffo,"
# The only build that gets the XDP patch, and the module version DKMS knows it by.
_XDP_DISTRIBUTION = "ubuntu22.04"
_XDP_DOCA_VERSION = "3.2.3"
_XDP_MODULE_VERSION = "25.10.OFED.25.10.3.1.8.1-1"

# V100's proprietary nvidia-peermem cannot use GPL-only exports.
_NO_PEER_MEM_PATCH_SKU = "V100"

# DOCA from this version on no longer needs the marker to provide sharp.
_SHARP_PROVIDE_BEFORE = "3.3.0"

_DOCA_OFED = "doca-ofed"
# What builds the marker packages.
_APT_MARKER_TOOLS = ["equivs"]
_RPM_MARKER_TOOLS = ["rpm-build"]

_UBUNTU_DKMS_MODULES = ["mlnx-ofed-kernel", "iser", "isert", "srp"]
_RHEL_DKMS_MODULES = ["mlnx-ofa_kernel", "iser", "isert", "srp"]

# The DOCA-OFED dependencies the RPM marker provides (the awk regex in bash).
_RPM_MARKER_NAMES = re.compile(
    r"(openmpi|sharp|hcoll|mpitests_openmpi|ucx|ucx-cma|ucx-devel|ucx-ib|"
    r"ucx-ib-mlx5|ucx-knem|ucx-rdmacm|ucx-xpmem)")

# ---- File contents (bash heredocs, byte for byte) ---------------------------
# Prefer the CUDA repo's newer dkms and ignore the older copy from DOCA.
_DKMS_PIN_CONTENT = """Package: dkms
Pin: release l=DOCA-HOST*
Pin-Priority: -1
"""

# openibd should start only after udev settles and restart if it fails.
_OPENIBD_OVERRIDE = """[Unit]
After=systemd-udev-settle.service
Wants=systemd-udev-settle.service

[Service]
Restart=on-failure
RestartSec=5
"""

# bash: echo -e "\n# Load IPoIB\nIPOIB_LOAD=no" | tee -a ...  (echo adds the final \n)
_IPOIB_OFF = "\n# Load IPoIB\nIPOIB_LOAD=no\n"

# The quoted (<<'EOF') second half of the RPM marker spec: nothing expanded.
_RPM_MARKER_SPEC_BODY = """
%description
HPC-X, installed later by install_mpis.sh into /opt, provides the Open MPI,
UCX, HCOLL, and SHARP runtime stack used by this image. This marker package
satisfies DOCA OFED RPM dependencies that would otherwise install the
DOCA-bundled MPI-adjacent packages.

%prep

%build

%install
mkdir -p %{buildroot}/usr/share/doc/%{name}
cat > %{buildroot}/usr/share/doc/%{name}/README <<'README'
HPC-X provides the MPI-adjacent DOCA OFED dependencies for this image.
README

%files
/usr/share/doc/%{name}/README
"""


# --------------------------------------------------------------------------- #
# Pure helpers: no side effects, so tests call them directly.
# --------------------------------------------------------------------------- #
def parse_apt_cache_version(apt_cache_output: str) -> str | None:
    """The version from `apt-cache show <pkg>` output, or None.

    Mirrors: awk '/^Version:/ {print $2; exit}'. The first line starting with
    "Version:" decides: its second whitespace-separated field, or None when it
    has none (awk prints "", which bash then rejects as empty).
    """
    for line in apt_cache_output.split("\n"):
        if line.startswith("Version:"):
            fields = line.split()
            return fields[1] if len(fields) >= 2 else None
    return None


def parse_ofed_version(ofed_info_output: str) -> str | None:
    """Extract the OFED version from `ofed_info` output.

    Mirrors: ofed_info | sed -n '1,1p' | awk -F'-' 'OFS="-" {print $3,$4}' | tr -d ':'
    i.e. take the first line, split on '-', join the 3rd and 4th fields with '-',
    and drop any ':'. Returns None if the first line has too few fields.
    """
    lines = ofed_info_output.splitlines()
    if not lines:
        return None
    fields = lines[0].split("-")
    if len(fields) < 4:
        return None
    result = f"{fields[2]}-{fields[3]}".replace(":", "")
    return result or None


def strip_dnf_excludes(conf_text: str) -> str:
    """Drop every line starting with 'exclude=' (mirrors sed -i '/^exclude=/d').

    Lines keep their own line endings, so a missing final newline stays
    missing, as with sed.
    """
    lines = re.findall(r"[^\n]*\n|[^\n]+$", conf_text)
    return "".join(line for line in lines if not line.startswith("exclude="))


def apt_marker_control(openmpi_version: str, ucx_version: str, doca_version: str,
                       sharp_version: str | None) -> str:
    """The equivs control file bash writes to /tmp/hpcx-provides-doca-ofed-deps.

    sharp_version is None from DOCA 3.3.0 on (no sharp provide).
    """
    sharp_provides = f", sharp (= {sharp_version})" if sharp_version else ""
    return (
        "Section: misc\n"
        "Priority: optional\n"
        "Homepage: https://github.com/Azure/azhpc-images\n"
        "Standards-Version: 3.9.2\n"
        "\n"
        f"Package: {_MARKER}\n"
        f"Provides: openmpi (= {openmpi_version}), openmpi-bin, libopenmpi-dev, "
        f"openmpi-common, ucx (= {ucx_version}), libucx0{sharp_provides}\n"
        "Conflicts: openmpi-bin, libopenmpi-dev, openmpi-common\n"
        f"Version: {doca_version}\n"
        "Maintainer: Azure HPC Platform team <hpcplat@microsoft.com>\n"
        "Description: marker package to indicate that HPC-X provides DOCA OFED dependencies\n"
        " HPC-X (installed by install_mpis.sh into /opt) provides Open MPI, UCX, and\n"
        " SHARP at runtime, so the DOCA-bundled openmpi and ucx packages are redundant.\n"
        " Before DOCA 3.3.0, the bundled sharp package is redundant too. The DOCA\n"
        " openmpi collides with /etc/pmix-mca-params.conf from the separately-installed\n"
        " pmix package, Canonical's Open MPI can pull unsuitable PMIx dependencies, and\n"
        " older DOCA SHARP packages can inject stale libtool dependencies on DOCA UCX\n"
        " paths under /usr/lib. The libucx0 virtual provide also satisfies later\n"
        " Ubuntu/ROCm dependency chains that require libucx0.\n"
    )


def rpm_marker_provides(repoquery_output: str) -> list[str]:
    """The Provides values for the RPM marker, from `dnf repoquery --requires doca-ofed`.

    Mirrors the awk filter + `sort -u`:
        $1 ~ /^(openmpi|sharp|...)$/ {
            if ($2 ~ /^[<>=]+$/ && $3 != "") print $1 " = " $3; else print $1 }

        "openmpi = 4.1.7rc1-1"  -> "openmpi = 4.1.7rc1-1"
        "ucx >= 1.19"           -> "ucx = 1.19"    (any operator becomes "=")
        "sharp"                 -> "sharp"
    """
    provides = set()
    for line in repoquery_output.split("\n"):
        fields = line.split()
        if not fields or not _RPM_MARKER_NAMES.fullmatch(fields[0]):
            continue
        if len(fields) >= 3 and re.fullmatch(r"[<>=]+", fields[1]):
            provides.add(f"{fields[0]} = {fields[2]}")
        else:
            provides.add(fields[0])
    return sorted(provides)


def rpm_marker_spec(doca_version: str, provides: list[str]) -> str:
    """The marker .spec: bash's <<EOF header, one `Provides:` line per value
    (printf 'Provides: %s\\n'), then the quoted <<'EOF' body."""
    header = (
        f"Name: {_MARKER}\n"
        f"Version: {doca_version}\n"
        "Release: 1\n"
        "Summary: Marker package to indicate that HPC-X provides DOCA OFED dependencies\n"
        "License: MIT\n"
        "BuildArch: noarch\n"
    )
    return header + "".join(f"Provides: {p}\n" for p in provides) + _RPM_MARKER_SPEC_BODY


def installed_doca_packages(repoquery_output: str) -> list[str]:
    """Installed package names whose repo starts with "doca".

    Mirrors: dnf repoquery --installed --qf '%{name} %{from_repo}\\n'
             | awk '$2 ~ /^doca/ {print $1}' | sort -u
    """
    names = set()
    for line in repoquery_output.split("\n"):
        fields = line.split()
        if len(fields) >= 2 and fields[1].startswith("doca"):
            names.add(fields[0])
    return sorted(names)


def baseos_repo_name(distribution: str, architecture: str) -> str:
    """The baseos repo id bash pins: "baseos", or RHEL's RHUI repo on rhel*.

        rhel9.8 + x86_64 -> rhel-9-for-x86_64-baseos-rhui-rpms
    """
    if distribution.startswith("rhel"):
        major = distribution[len("rhel"):].split(".")[0]    # ${rhel_version%%.*}
        return f"rhel-{major}-for-{architecture}-baseos-rhui-rpms"
    return "baseos"


def expand_glob(pattern: str) -> list[str]:
    """Expand `pattern` like bash: sorted matches, or the pattern itself if none."""
    return sorted(glob.glob(pattern)) or [pattern]


# --------------------------------------------------------------------------- #
# Small helpers with side effects.
# --------------------------------------------------------------------------- #
def _run(command: list[str], env: dict[str, str], cwd: str | None = None) -> bool:
    """Run one command; True on success. exec_program logs the command, its
    exit code and, on failure, its last output lines."""
    return exec_program(command, _OP, cwd=cwd, env=env) == 0


def _run_all(commands: list[list[str]], env: dict[str, str],
             cwd: str | None = None) -> bool:
    """Run commands in order, stopping at the first failure (like `set -e`)."""
    return all(_run(command, env, cwd) for command in commands)


def _command_exists(name: str, env: dict[str, str]) -> bool:
    """bash `command -v <name>`, on the build env's PATH."""
    return shutil.which(name, path=env.get("PATH")) is not None


def _write_file(path: str, text: str, *, append: bool = False) -> None:
    """Write (bash `cat >`) or append (`cat >>` / `tee -a`) text and log it."""
    existed = os.path.exists(path)
    with open(path, "a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)
    action = ("appended" if append else "replaced") if existed else "created"
    log_info(_OP, f"{action.capitalize()} {path}", path=path, action=action)
    log_debug(_OP, f"{path} {'appended' if append else 'content'}:\n{text}")


def _copy_patch(env: dict[str, str], patch: tuple[str, str], patch_dir: str) -> None:
    """cp "${COMPONENT_DIR}/patches/<source>" "<patch_dir>/<name>"."""
    source = os.path.join(env.get("COMPONENT_DIR", ""), "patches", patch[0])
    dest = os.path.join(patch_dir, patch[1])
    shutil.copy(source, dest)
    log_info(_OP, f"Installed DKMS patch {dest}", source=source, path=dest,
             patch=patch[1])


def _bash_work_dir(env: dict[str, str]) -> str:
    """Where bash runs install_doca.sh (and so downloads to): the distro build dir."""
    return os.path.join(env.get("TOP_DIR", "."), "distros", env.get("DISTRIBUTION", ""))


# --------------------------------------------------------------------------- #
# The phases. Each returns True on success, False after logging a failure.
# File errors raise OSError, which install() turns into a logged rc=3.
# --------------------------------------------------------------------------- #
def _get_package(env: dict[str, str], cfg: dict, source: str,
                 work_dir: str) -> str | None:
    """Verify or download the DOCA repo package; return bash's DOCA_FILE."""
    sha256 = cfg.get("sha256")
    if source == "private":
        repo_file = cfg.get("repo_file")
        if not repo_file or not sha256:
            log_error(_OP, "private DOCA entry needs repo_file and sha256 in versions.json",
                      repo_file=repo_file, source=source)
            return None
        path = os.path.join(env.get("TOP_DIR", ""), "internal_bits", repo_file)
        log_info(_OP, f"Using the private DOCA package {path}", path=path,
                 source=source, sha256=sha256)
        if not verify_checksum(path, sha256):
            log_error(_OP, f"checksum of the private DOCA package {path} is wrong",
                      path=path)
            return None
        return path

    url = cfg.get("url")
    if not url or not sha256:
        log_error(_OP, "could not resolve doca url and sha256 from versions.json",
                  url=url, source=source or "public")
        return None
    if not _download(url, sha256, work_dir):
        return None
    return os.path.basename(url)


def _download(url: str, sha256: str, work_dir: str) -> bool:
    log_info(_OP, f"Downloading the DOCA host package into {work_dir}", url=url,
             sha256=sha256, dest_dir=work_dir)
    try:
        download_and_verify(url, sha256, dest_dir=work_dir)
    except Exception as exc:          # network errors, checksum mismatch, disk
        log_error(_OP, f"download/verify of {url} failed: {exc}", url=url)
        return False
    return True


def _download_again(cfg: dict, source: str, work_dir: str) -> bool:
    """bash runs download_and_verify $DOCA_URL $DOCA_SHA256 again on Azure
    Linux and the RHEL family."""
    if source == "private":
        # BASH BUG: DOCA_URL is only set for public packages, so bash's
        # download_and_verify fails here (no URL). Fail the same way.
        log_error(_OP, "install_doca.sh downloads $DOCA_URL again on this distro, but a "
                       "private package has no url, so bash fails here too",
                  source=source)
        return False
    # BASH BUG (kept): the package was already downloaded and verified above.
    log_warn(_OP, "downloading the DOCA package a second time, as install_doca.sh "
                  "does on Azure Linux and the RHEL family", url=cfg.get("url"))
    return _download(cfg["url"], cfg["sha256"], work_dir)


def _apt_cache_version(package: str, env: dict[str, str]) -> str | None:
    """apt-cache show <package> 2>/dev/null | awk '/^Version:/ {print $2; exit}'."""
    _rc, output = run_capture(["apt-cache", "show", package], _OP, env=env,
                              failure_is_error=False)
    version = parse_apt_cache_version(output)
    if not version:
        log_error(_OP, f"could not read {package} version from DOCA repo", package=package)
    return version


def _install_apt_marker(env: dict[str, str], doca_version: str) -> bool:
    """install_hpcx_doca_ofed_deps_apt_marker: an empty equivs package that
    Provides Open MPI / UCX (and, before DOCA 3.3.0, SHARP), so doca-ofed's
    strict dependencies don't pull DOCA's own copies, and Conflicts with
    Canonical's Open MPI."""
    if not install_packages(_APT_MARKER_TOOLS, env, _OP):
        return False
    openmpi = _apt_cache_version("openmpi", env)
    if not openmpi:
        return False
    ucx = _apt_cache_version("ucx", env)
    if not ucx:
        return False

    sharp = None
    rc = exec_program(["dpkg", "--compare-versions", doca_version, "lt",
                       _SHARP_PROVIDE_BEFORE], _OP, env=env, failure_is_error=False)
    if rc not in (0, 1):
        # BASH BUG (kept): `if dpkg --compare-versions ...` treats an error the
        # same as "not older", so the marker won't provide sharp.
        log_warn(_OP, f"dpkg could not compare DOCA version {doca_version!r} with "
                      f"{_SHARP_PROVIDE_BEFORE}; treating it as not older, as bash does",
                 exit_code=rc, doca_version=doca_version)
    if rc == 0:
        sharp = _apt_cache_version("sharp", env)
        if not sharp:
            return False
    log_info(_OP, "Building the equivs marker package "
                  f"{_MARKER} (HPC-X provides DOCA's MPI-adjacent packages)",
             openmpi_version=openmpi, ucx_version=ucx, sharp_version=sharp,
             sharp_provided=sharp is not None, doca_version=doca_version)

    control = os.path.join(_MARKER_WORK_DIR, _MARKER)
    _write_file(control, apt_marker_control(openmpi, ucx, doca_version, sharp))
    # bash: ( cd /tmp; equivs-build ...; apt install -y /tmp/<marker>_*_all.deb )
    if not _run(["equivs-build", control], env, cwd=_MARKER_WORK_DIR):
        return False
    debs = expand_glob(os.path.join(_MARKER_WORK_DIR, f"{_MARKER}_*_all.deb"))
    if not install_packages(debs, env, _OP, cwd=_MARKER_WORK_DIR):
        return False
    # bash: rm -f <debs> <control>
    for path in [*debs, control]:
        if os.path.lexists(path):
            os.remove(path)
    log_info(_OP, f"Installed the {_MARKER} marker package", package=_MARKER,
             files=debs)
    return True


def _install_rpm_marker(env: dict[str, str], doca_version: str) -> bool:
    """install_hpcx_doca_ofed_deps_rpm_marker: an empty rpm that Provides the
    Open MPI / UCX / HCOLL / SHARP packages doca-ofed requires."""
    if not _command_exists("rpmbuild", env):
        log_info(_OP, "rpmbuild not found: installing rpm-build")
        if not install_packages(_RPM_MARKER_TOOLS, env, _OP):
            return False

    topdir = os.path.join(_MARKER_WORK_DIR, f"{_MARKER}-rpmbuild")
    spec_file = os.path.join(topdir, "SPECS", f"{_MARKER}.spec")
    shutil.rmtree(topdir, ignore_errors=True)
    for sub in ("BUILD", "BUILDROOT", "RPMS", "SOURCES", "SPECS", "SRPMS"):
        os.makedirs(os.path.join(topdir, sub), exist_ok=True)

    # bash ignores this pipeline's exit code; an empty result is the error.
    _rc, output = run_capture(["dnf", "repoquery", "--quiet", "--requires", "doca-ofed"],
                              _OP, env=env, failure_is_error=False)
    provides = rpm_marker_provides(output)
    if not provides:
        log_error(_OP, "could not derive DOCA OFED Open MPI/UCX/SHARP RPM dependencies",
                  package="doca-ofed")
        return False
    log_info(_OP, f"Building the marker rpm {_MARKER} providing {len(provides)} "
                  "DOCA OFED dependencies", provides=provides, doca_version=doca_version)

    _write_file(spec_file, rpm_marker_spec(doca_version, provides))
    if not _run(["rpmbuild", "--define", f"_topdir {topdir}", "-bb", spec_file], env):
        return False
    # bash: find "${rpm_topdir}/RPMS" -type f -name "<marker>-*.rpm" | head -n1
    built = sorted(glob.glob(os.path.join(topdir, "RPMS", "**", f"{_MARKER}-*.rpm"),
                             recursive=True))
    built = [path for path in built if os.path.isfile(path)]
    if not built:
        log_error(_OP, f"failed to build {_MARKER} marker RPM", rpms_dir=f"{topdir}/RPMS")
        return False
    if not _run(["rpm", "-Uvh", "--replacepkgs", built[0]], env):
        return False
    shutil.rmtree(topdir)
    log_info(_OP, f"Installed the {_MARKER} marker rpm", package=_MARKER, rpm=built[0])
    return True


def _configure_doca_dkms_patches(env: dict[str, str], module: str) -> None:
    """configure_doca_dkms_patches <module>: the PeerDirect GPL patch (not on
    V100) and, for mlnx-ofa_kernel, the DPLL patch when the kernel needs it."""
    patch_dir = os.path.join(_DKMS_ETC_DIR, module, "patches")
    dkms_conf = os.path.join(_DKMS_ETC_DIR, f"{module}.conf")
    os.makedirs(patch_dir, exist_ok=True)

    sku = env.get("SKU", "")
    if sku != _NO_PEER_MEM_PATCH_SKU:
        _copy_patch(env, _PEER_MEM_PATCH, patch_dir)
        _write_file(dkms_conf, f'PATCH+=("{_PEER_MEM_PATCH[1]}")\n')
    else:
        log_info(_OP, "V100: not adding the PeerDirect GPL-export DKMS patch (its "
                      "proprietary nvidia-peermem can't use GPL-only exports)",
                 sku=sku, module=module)

    # Alma/Rocky/RHEL 9.8 kernels use Red Hat's newer DPLL ffo_get callback
    # signature, while DOCA 3.2.x / MLNX OFED 25.10 still ships the older one.
    if module != "mlnx-ofa_kernel":
        return
    header = os.path.join(_KERNEL_SRC_DIR, running_kernel(), "include", "linux", "dpll.h")
    try:
        with open(header, "r", encoding="utf-8", errors="replace") as handle:
            needs_patch = _DPLL_FFO_SIGNATURE in handle.read()
    except OSError:
        # bash: [[ -f "${kernel_header}" ]] || return 0 (and grep errors return 0)
        log_info(_OP, f"{header} not found: no DPLL DKMS patch", path=header)
        return
    if not needs_patch:
        log_info(_OP, "Kernel uses the older DPLL ffo_get signature: no DPLL DKMS patch",
                 path=header)
        return
    log_info(_OP, "Kernel uses Red Hat's newer DPLL ffo_get signature: adding the DPLL "
                  "DKMS patch", path=header)
    _copy_patch(env, _DPLL_PATCH, patch_dir)
    _write_file(dkms_conf, f'PATCH+=("{_DPLL_PATCH[1]}")\n', append=True)


def _configure_xdp_patch(env: dict[str, str], doca_version: str) -> None:
    """configure_mlnx_ofed_kernel_dkms_xdp_patch: Ubuntu 22.04 + DOCA 3.2.3 only."""
    distribution = env.get("DISTRIBUTION", "")
    if distribution != _XDP_DISTRIBUTION or doca_version != _XDP_DOCA_VERSION:
        log_info(_OP, "No XDP fragments DKMS patch (only for DOCA "
                      f"{_XDP_DOCA_VERSION} on {_XDP_DISTRIBUTION})",
                 distro=distribution, doca_version=doca_version)
        return
    # Installed even on older kernels so later DKMS rebuilds get the fix.
    # TODO (upstream): remove when DOCA fixes detection of partially
    # backported XDP APIs.
    patch_dir = os.path.join(_DKMS_ETC_DIR, "mlnx-ofed-kernel", "patches")
    dkms_conf = os.path.join(_DKMS_ETC_DIR, f"mlnx-ofed-kernel-{_XDP_MODULE_VERSION}.conf")
    os.makedirs(patch_dir, exist_ok=True)
    _copy_patch(env, _XDP_PATCH, patch_dir)
    _write_file(dkms_conf, f'PATCH+=("{_XDP_PATCH[1]}")\n')


def _check_dkms(env: dict[str, str], modules: list[str]) -> bool:
    if not check_dkms_status(modules, _OP, env):
        log_error(_OP, "DOCA-OFED kernel modules were not built by DKMS", modules=modules)
        return False
    return True


# --------------------------------------------------------------------------- #
# Per-distro handlers.
# --------------------------------------------------------------------------- #
def _install_ubuntu(env, doca_version: str, doca_file: str, work_dir: str) -> bool:
    deb = os.path.realpath(os.path.join(work_dir, doca_file))   # "$(realpath ...)"
    if not install_packages([deb], env, _OP, cwd=work_dir):
        return False
    _write_file(_DKMS_PIN, _DKMS_PIN_CONTENT)
    if not _run(["apt-get", "update"], env):
        return False
    if not _install_apt_marker(env, doca_version):
        return False
    _configure_doca_dkms_patches(env, "mlnx-ofed-kernel")
    _configure_xdp_patch(env, doca_version)
    if not install_packages([_DOCA_OFED], env, _OP):
        return False
    return _check_dkms(env, _UBUNTU_DKMS_MODULES)


def _install_azurelinux(env, doca_version: str, doca_file: str, work_dir: str,
                        cfg: dict, source: str) -> bool:
    if not _download_again(cfg, source, work_dir):
        return False
    if not _run_all([["rpm", "-i", doca_file], ["dnf", "clean", "all"]], env, cwd=work_dir):
        return False
    if not _install_rpm_marker(env, doca_version):
        return False
    return install_packages([_DOCA_OFED], env, _OP)


def _install_rhel(env, doca_version: str, doca_file: str, work_dir: str,
                  cfg: dict, source: str) -> bool:
    if not _download_again(cfg, source, work_dir):
        return False
    if not _run_all([["rpm", "-i", doca_file], ["dnf", "clean", "all"]], env, cwd=work_dir):
        return False

    # Lift dnf.conf's exclude= pins while doca-ofed installs, then restore it.
    backup = f"{_DNF_CONF}.bak"
    shutil.copy(_DNF_CONF, backup)
    original = Path(_DNF_CONF).read_text(encoding="utf-8")
    Path(_DNF_CONF).write_text(strip_dnf_excludes(original), encoding="utf-8")
    log_info(_OP, f"Backed up {_DNF_CONF} to {backup} and removed its exclude= lines",
             path=_DNF_CONF, backup=backup, action="edited")
    try:
        ok = (_install_rpm_marker(env, doca_version)
              and _configure_rhel_dkms_and_install(env))
    finally:
        # bash: mv /etc/dnf/dnf.conf.bak /etc/dnf/dnf.conf (only on success;
        # we restore on failure too).
        os.replace(backup, _DNF_CONF)
        log_info(_OP, f"Restored {_DNF_CONF} from {backup}", path=_DNF_CONF,
                 action="restored")
    return ok and _pin_baseos_conflicts(env)


def _configure_rhel_dkms_and_install(env) -> bool:
    _configure_doca_dkms_patches(env, "mlnx-ofa_kernel")
    if not install_packages([_DOCA_OFED], env, _OP):
        return False
    return _check_dkms(env, _RHEL_DKMS_MODULES)


def _pin_baseos_conflicts(env) -> bool:
    """Exclude, in the baseos repo only, the packages DOCA's repos replaced.

    EL9.x baseos ships its own rdma-core stack (ABI-bumped libibverbs etc.)
    that conflicts with DOCA's; without this, later `dnf update`s abort. A
    repo-local exclude keeps `dnf check-update` meaningful.
    """
    rc, output = run_capture(["dnf", "repoquery", "--installed", "--quiet", "--qf",
                              "%{name} %{from_repo}\\n"], _OP, env=env,
                             failure_is_error=False)
    if rc != 0:
        log_warn(_OP, "dnf repoquery --installed failed; using its output anyway, "
                      "as bash does", exit_code=rc)
    doca_pkgs = installed_doca_packages(output)
    if not doca_pkgs:
        log_error(_OP, "no packages found from doca* repos after doca-ofed install")
        return False

    distribution = env.get("DISTRIBUTION", "")
    repo = baseos_repo_name(distribution, env.get("ARCHITECTURE", ""))
    rc, output = run_capture(["dnf", "repoquery", "--quiet", f"--repo={repo}", "--qf",
                              "%{name}\\n", "*"], _OP, env=env, failure_is_error=False)
    if rc != 0:
        log_warn(_OP, f"dnf repoquery of {repo} failed; bash carries on, so nothing "
                      "may be pinned", repo=repo, exit_code=rc)
    baseos_pkgs = {line for line in output.split("\n") if line}
    conflicts = sorted(set(doca_pkgs) & baseos_pkgs)
    log_info(_OP, f"{len(doca_pkgs)} installed packages come from doca* repos; "
                  f"{len(conflicts)} of them are also in {repo}",
             repo=repo, doca_packages=len(doca_pkgs), conflicts=conflicts)
    if not conflicts:
        return True
    log_info(_OP, f"Pinning {len(conflicts)} baseos package(s) shadowed by DOCA: "
                  f"{' '.join(conflicts)}", repo=repo, excludepkgs=conflicts)
    return _run(["dnf", "config-manager", "--save",
                 f"--setopt={repo}.excludepkgs={' '.join(conflicts)}"], env)


# --------------------------------------------------------------------------- #
# Common tail.
# --------------------------------------------------------------------------- #
def _record_versions(env, doca_version: str) -> bool:
    write_component_version("DOCA", doca_version)
    log_info(_OP, f"Recorded DOCA={doca_version}", component="DOCA", version=doca_version)
    rc, output = run_capture(["ofed_info"], _OP, env=env)
    ofed = parse_ofed_version(output) if rc == 0 else None
    if not ofed:
        first = output.splitlines()[0] if output.splitlines() else ""
        log_error(_OP, "could not read the OFED version from ofed_info",
                  exit_code=rc, first_line=first)
        return False
    write_component_version("OFED", ofed)
    log_info(_OP, f"Recorded OFED={ofed}", component="OFED", version=ofed)
    return True


def _configure_openibd(env) -> bool:
    os.makedirs(_OPENIBD_DROPIN_DIR, exist_ok=True)
    _write_file(os.path.join(_OPENIBD_DROPIN_DIR, "override.conf"), _OPENIBD_OVERRIDE)

    sku, node_type = env.get("SKU", ""), env.get("TARGET_NODE_TYPE", "")
    if sku_uses_ipoib(sku, node_type):
        log_info(_OP, "This target uses IPoIB: leaving openib.conf alone",
                 sku=sku, node_type=node_type, ipoib=True)
    else:
        log_info(_OP, f"This target doesn't use IPoIB: disabling it in {_OPENIB_CONF}",
                 sku=sku, node_type=node_type, ipoib=False)
        try:
            already = _IPOIB_OFF in Path(_OPENIB_CONF).read_text(encoding="utf-8")
        except OSError:
            already = False
        if already:
            # BASH BUG (kept): `tee -a` appends the lines again on every rerun.
            log_warn(_OP, f"{_OPENIB_CONF} already disables IPoIB; appending again, "
                          "as bash does", path=_OPENIB_CONF)
        _write_file(_OPENIB_CONF, _IPOIB_OFF, append=True)

    # Enable only; do NOT restart here: restarting probes the build VM's IB
    # hardware (which may be absent on general-purpose build SKUs).
    if not _run_all([["systemctl", "daemon-reload"], ["systemctl", "enable", "openibd"]],
                    env):
        log_error(_OP, "could not enable the openibd service", service="openibd")
        return False
    log_info(_OP, "Enabled the openibd service (not started at build time)",
             service="openibd")
    return True


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def install(env: dict[str, str]) -> int:
    """Install DOCA-OFED, record DOCA/OFED and set up openibd. Returns 0 or 3."""
    cfg = config_for("doca", env)
    if not isinstance(cfg, dict) or not cfg.get("version"):
        log_error(_OP, "could not resolve doca version from versions.json",
                  distro=env.get("DISTRIBUTION", ""))
        return 3

    version = str(cfg["version"])
    source = str(cfg.get("source") or "")
    distribution = env.get("DISTRIBUTION", "")
    if "ubuntu" in distribution:
        family = "ubuntu"
    elif distribution == _AZURE_LINUX:
        family = "azurelinux"
    else:  # RHEL family: AlmaLinux, Rocky Linux, RHEL, etc.
        family = "rhel"
    work_dir = _bash_work_dir(env)
    log_info(_OP, f"Installing DOCA-OFED {version} on {distribution} ({family} branch)",
             version=version, source=source or "public", url=cfg.get("url"),
             repo_file=cfg.get("repo_file"), distro=distribution, branch=family,
             architecture=env.get("ARCHITECTURE", ""), sku=env.get("SKU", ""),
             node_type=env.get("TARGET_NODE_TYPE", ""), work_dir=work_dir)

    try:
        doca_file = _get_package(env, cfg, source, work_dir)
        if doca_file is None:
            return 3
        if family == "ubuntu":
            ok = _install_ubuntu(env, version, doca_file, work_dir)
        elif family == "azurelinux":
            ok = _install_azurelinux(env, version, doca_file, work_dir, cfg, source)
        else:
            ok = _install_rhel(env, version, doca_file, work_dir, cfg, source)
        if not ok:
            log_error(_OP, "DOCA-OFED installation failed", distro=distribution,
                      branch=family)
            return 3
        if not _record_versions(env, version) or not _configure_openibd(env):
            return 3
    except OSError as exc:
        # File writes/copies (/etc/dkms, dnf.conf, marker files, versions file).
        log_error(_OP, f"file operation failed: {exc}",
                  path=getattr(exc, "filename", None))
        return 3

    log_info(_OP, f"DOCA-OFED {version} installed", version=version, distro=distribution,
             branch=family)
    return 0
