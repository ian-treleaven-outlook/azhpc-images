"""install_doca.py — Python port of components/install_doca.sh.

DOCA-OFED is NVIDIA's packaging of the InfiniBand / RDMA driver stack (the
successor to MLNX_OFED). On InfiniBand SKUs this step:

  1. Gets the DOCA host repo package: downloads and checksums it, or, for a
     "private" entry in versions.json, checksums the copy in internal_bits/.
  2. Installs the repo package and a marker package that tells apt/dnf
     "HPC-X provides Open MPI/UCX/SHARP", so DOCA does not install its own
     copies (HPC-X is installed later by install_mpis.sh).
  3. Adds DKMS patch overlays under /etc/dkms that DKMS applies whenever it
     (re)builds the DOCA kernel modules, installs doca-ofed and checks DKMS
     built the modules for the running kernel.
  4. Records DOCA and OFED, and sets up the openibd boot service (turning
     IPoIB off where the target doesn't use it).

The distro decides the branch of bash's if/elif (bash_branch()), looked up
once in _INSTALLERS:

  * Ubuntu        `apt install` the repo .deb; pin DOCA's old dkms out (the
                  CUDA repo's newer dkms comes from install_cuda_repo_and_dkms.sh);
                  equivs marker; DKMS patches for mlnx-ofed-kernel; doca-ofed;
                  DKMS status check.
  * Azure Linux   `rpm -i` the repo rpm; rpmbuild marker; doca-ofed. No DKMS
                  patches or check (as in bash).
  * RHEL family   like Azure Linux, plus: dnf.conf's exclude= lines are lifted
                  during the install, DKMS patches for mlnx-ofa_kernel, DKMS
                  status check, and a baseos excludepkgs pin for the packages
                  DOCA replaces.

DKMS patch overlays (upstream #642, #633), each copied from
components/patches and listed in /etc/dkms/<module>.conf:

  * peer-mem-gpl.patch, every SKU except V100: exports the IB peer-memory API
    as GPL-only symbols, so DKMS can rebuild DOCA and the GPU driver in either
    order (the .conf is overwritten).
  * dpll-ffo-param.patch, RHEL family, when the running kernel's linux/dpll.h
    has Red Hat's newer DPLL callback (appended to the .conf).
  * xdp-fragments.patch, Ubuntu 22.04 with DOCA 3.2.3 only, in the
    version-specific mlnx-ofed-kernel-25.10.OFED.25.10.3.1.8.1-1.conf.

File texts are bash's heredocs byte for byte (tests compare them with
install_doca.sh). utils/plans/steps.py makes install() the action of the
"install-doca" Step.

Behaviour kept from bash
------------------------
  * Same commands, arguments and order; the first failure stops the step.
    Files are overwritten (`cat >`) or appended to (`cat >>`, `tee -a`) as
    in bash.
  * The public package is downloaded into bash's working directory, the
    distro build dir (distros/<distro>), where install.sh's cleanup removes
    *.deb files, and `rpm -i` gets bash's relative file name there.
  * `dpkg --compare-versions` decides "DOCA < 3.3.0" (exit 0 = yes).
  * The marker packages' temp files are removed only after success.

Deliberate differences from bash
--------------------------------
  * Installs go through install_packages() (`apt-get install -y -o
    Dpkg::Use-Pty=0` / `dnf install -y`); systemctl through utils.services.
  * Missing config fails up front: version always; url + sha256 for a public
    package; repo_file + sha256 for a private one.
  * OFED: a failing `ofed_info` or a first line with fewer than four
    '-'-separated fields fails the step (bash would record "" or "-").
  * RHEL family: dnf.conf is restored from the .bak even if the install
    fails (bash's `set -e` exits first and leaves the edited copy).
  * Package lists bash sorts with `sort -u` (locale order) are sorted in
    plain character order; only the order of names in the marker spec's
    Provides lines and in excludepkgs can differ, which rpm/dnf ignore.
  * The repeated download on Azure Linux / RHEL replaces the file; bash's
    wget keeps the first copy and saves the second as "<name>.1".
  * Removing the marker temp files (remove_paths, like `rm -f` / `rm -rf`)
    never fails the step.

Bash bugs kept (commented, log_warn when they apply)
----------------------------------------------------
  * Azure Linux and RHEL download the package a second time. With a
    "private" source DOCA_URL is unset there, so bash's download_and_verify
    fails; this port fails at the same point with a clear error.
  * `tee -a` appends the IPoIB lines again on every rerun.
  * `dpkg --compare-versions` errors (exit 2) count as "not older than
    3.3.0", so the sharp provide is left out.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from utils.component_config import config_for, require_config, write_component_version
from utils.distro import Distro
from utils.dkms import check_dkms_status, running_kernel
from utils.download import fetch_verified, verify_checksum
from utils.files import remove_paths, write_file
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import command_exists, exec_program, run, run_all, run_capture
from utils.services import systemctl
from utils.sku import sku_uses_ipoib

_OP = "install-doca"

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
_RPMBUILD_DIRS = ("BUILD", "BUILDROOT", "RPMS", "SOURCES", "SPECS", "SRPMS")

# The DOCA kernel module's DKMS name per family (Azure Linux has no DKMS
# patches or check), and the modules check_dkms_status checks after the install.
_UBUNTU_KERNEL_MODULE = "mlnx-ofed-kernel"
_RHEL_KERNEL_MODULE = "mlnx-ofa_kernel"
_UBUNTU_DKMS_MODULES = [_UBUNTU_KERNEL_MODULE, "iser", "isert", "srp"]
_RHEL_DKMS_MODULES = [_RHEL_KERNEL_MODULE, "iser", "isert", "srp"]

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


def baseos_repo_name(distro: Distro, architecture: str) -> str:
    """The baseos repo id bash pins: "baseos", or RHEL's RHUI repo on rhel*.

        rhel9.8 + x86_64 -> rhel-9-for-x86_64-baseos-rhui-rpms
    """
    if distro.is_rhel:
        return f"rhel-{distro.major}-for-{architecture}-baseos-rhui-rpms"
    return "baseos"


def bash_branch(distro: Distro) -> str:
    """Which branch of install_doca.sh's if/elif `distro` takes: "ubuntu",
    "azurelinux", or "rhel" for bash's `else` (the RHEL family)."""
    return distro.family if distro.family in ("ubuntu", "azurelinux") else "rhel"


def expand_glob(pattern: str) -> list[str]:
    """Expand `pattern` like bash: sorted matches, or the pattern itself if none."""
    return sorted(glob.glob(pattern)) or [pattern]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Doca:
    """The DOCA host package versions.json describes."""

    version: str
    private: bool             # bash: [[ "$DOCA_SOURCE" == "private" ]]
    sha256: str
    url: str | None           # public packages
    repo_file: str | None     # private packages, in internal_bits/


def _doca_config(env: dict[str, str]) -> _Doca | None:
    """Resolve the doca entry, requiring the keys its source needs; None (logged)."""
    private = (config_for("doca", env) or {}).get("source") == "private"
    keys = ("repo_file", "sha256") if private else ("url", "sha256")
    cfg = require_config("doca", env, _OP, "version", *keys)
    if cfg is None:
        return None
    return _Doca(version=str(cfg["version"]), private=private, sha256=cfg["sha256"],
                 url=cfg.get("url"), repo_file=cfg.get("repo_file"))


def _bash_work_dir(env: dict[str, str]) -> str:
    """Where bash runs install_doca.sh (and so downloads to): the distro build dir."""
    return os.path.join(env.get("TOP_DIR", "."), "distros", env.get("DISTRIBUTION", ""))


# --------------------------------------------------------------------------- #
# The DOCA host package
# --------------------------------------------------------------------------- #
def _get_package(env: dict[str, str], doca: _Doca, work_dir: str) -> str | None:
    """Verify the private package or download the public one; bash's DOCA_FILE."""
    if doca.private:
        path = os.path.join(env.get("TOP_DIR", ""), "internal_bits", doca.repo_file)
        log_info(_OP, f"Using the private DOCA package {path}", path=path,
                 source="private", sha256=doca.sha256)
        return path if verify_checksum(path, doca.sha256) else None
    if not _download(doca, work_dir):
        return None
    return os.path.basename(doca.url)


def _download(doca: _Doca, work_dir: str) -> bool:
    log_info(_OP, f"Downloading the DOCA host package into {work_dir}", url=doca.url,
             sha256=doca.sha256, dest_dir=work_dir)
    return fetch_verified(doca.url, doca.sha256, _OP, dest_dir=work_dir) is not None


def _add_doca_repo_rpm(env: dict[str, str], doca: _Doca, doca_file: str,
                       work_dir: str) -> bool:
    """Azure Linux / RHEL: download again (as bash does), rpm -i, dnf clean all."""
    if doca.private:
        # BASH BUG: DOCA_URL is only set for public packages, so bash's
        # download_and_verify fails here (no URL). Fail the same way.
        log_error(_OP, "install_doca.sh downloads $DOCA_URL again on this distro, but a "
                       "private package has no url, so bash fails here too",
                  source="private")
        return False
    # BASH BUG (kept): the package was already downloaded and verified above.
    log_warn(_OP, "downloading the DOCA package a second time, as install_doca.sh "
                  "does on Azure Linux and the RHEL family", url=doca.url)
    if not _download(doca, work_dir):
        return False
    return run_all([["rpm", "-i", doca_file], ["dnf", "clean", "all"]], _OP,
                   cwd=work_dir, env=env)


# --------------------------------------------------------------------------- #
# The marker packages
# --------------------------------------------------------------------------- #
def _apt_cache_version(package: str, env: dict[str, str]) -> str | None:
    """apt-cache show <package> 2>/dev/null | awk '/^Version:/ {print $2; exit}'."""
    _rc, output = run_capture(["apt-cache", "show", package], _OP, env=env,
                              failure_is_error=False)
    version = parse_apt_cache_version(output)
    if not version:
        log_error(_OP, f"could not read {package} version from DOCA repo", package=package)
    return version


def _older_than_sharp_cutoff(env: dict[str, str], doca_version: str) -> bool:
    """bash: `if dpkg --compare-versions "${DOCA_VERSION}" lt 3.3.0`."""
    rc = exec_program(["dpkg", "--compare-versions", doca_version, "lt",
                       _SHARP_PROVIDE_BEFORE], _OP, env=env, failure_is_error=False)
    if rc not in (0, 1):
        # BASH BUG (kept): `if` treats an error the same as "not older", so
        # the marker won't provide sharp.
        log_warn(_OP, f"dpkg could not compare DOCA version {doca_version!r} with "
                      f"{_SHARP_PROVIDE_BEFORE}; treating it as not older, as bash does",
                 exit_code=rc, doca_version=doca_version)
    return rc == 0


def _apt_marker_control(env: dict[str, str], doca_version: str) -> str | None:
    """Read the DOCA repo's openmpi/ucx (and, before 3.3.0, sharp) versions
    and render the equivs control file; None (logged) if one is missing."""
    openmpi = _apt_cache_version("openmpi", env)
    if not openmpi:
        return None
    ucx = _apt_cache_version("ucx", env)
    if not ucx:
        return None
    sharp = None
    if _older_than_sharp_cutoff(env, doca_version):
        sharp = _apt_cache_version("sharp", env)
        if not sharp:
            return None
    log_info(_OP, "Building the equivs marker package "
                  f"{_MARKER} (HPC-X provides DOCA's MPI-adjacent packages)",
             openmpi_version=openmpi, ucx_version=ucx, sharp_version=sharp,
             sharp_provided=sharp is not None, doca_version=doca_version)
    return apt_marker_control(openmpi, ucx, doca_version, sharp)


def _install_apt_marker(env: dict[str, str], doca_version: str) -> bool:
    """install_hpcx_doca_ofed_deps_apt_marker: an empty equivs package that
    Provides Open MPI / UCX (and, before DOCA 3.3.0, SHARP), so doca-ofed's
    strict dependencies don't pull DOCA's own copies, and Conflicts with
    Canonical's Open MPI."""
    if not install_packages(_APT_MARKER_TOOLS, env, _OP):
        return False
    control_text = _apt_marker_control(env, doca_version)
    control = os.path.join(_MARKER_WORK_DIR, _MARKER)
    if control_text is None or not write_file(control, control_text, _OP):
        return False
    # bash: ( cd /tmp; equivs-build ...; apt install -y /tmp/<marker>_*_all.deb )
    if not run(["equivs-build", control], _OP, cwd=_MARKER_WORK_DIR, env=env):
        return False
    debs = expand_glob(os.path.join(_MARKER_WORK_DIR, f"{_MARKER}_*_all.deb"))
    if not install_packages(debs, env, _OP, cwd=_MARKER_WORK_DIR):
        return False
    remove_paths(*debs, control, op=_OP)
    log_info(_OP, f"Installed the {_MARKER} marker package", package=_MARKER, files=debs)
    return True


def _rpm_marker_provides(env: dict[str, str]) -> list[str] | None:
    """What doca-ofed requires that HPC-X provides; None (logged) if nothing."""
    # bash ignores this pipeline's exit code; an empty result is the error.
    _rc, output = run_capture(["dnf", "repoquery", "--quiet", "--requires", "doca-ofed"],
                              _OP, env=env, failure_is_error=False)
    provides = rpm_marker_provides(output)
    if not provides:
        log_error(_OP, "could not derive DOCA OFED Open MPI/UCX/SHARP RPM dependencies",
                  package="doca-ofed")
        return None
    return provides


def _find_marker_rpm(topdir: str) -> str | None:
    """bash: find "${rpm_topdir}/RPMS" -type f -name "<marker>-*.rpm" | head -n1."""
    built = sorted(glob.glob(os.path.join(topdir, "RPMS", "**", f"{_MARKER}-*.rpm"),
                             recursive=True))
    built = [path for path in built if os.path.isfile(path)]
    if not built:
        log_error(_OP, f"failed to build {_MARKER} marker RPM", rpms_dir=f"{topdir}/RPMS")
        return None
    return built[0]


def _install_rpm_marker(env: dict[str, str], doca_version: str) -> bool:
    """install_hpcx_doca_ofed_deps_rpm_marker: an empty rpm that Provides the
    Open MPI / UCX / HCOLL / SHARP packages doca-ofed requires."""
    if not command_exists("rpmbuild", env):
        log_info(_OP, "rpmbuild not found: installing rpm-build")
        if not install_packages(_RPM_MARKER_TOOLS, env, _OP):
            return False
    topdir = os.path.join(_MARKER_WORK_DIR, f"{_MARKER}-rpmbuild")
    spec_file = os.path.join(topdir, "SPECS", f"{_MARKER}.spec")
    remove_paths(topdir, op=_OP)
    for sub in _RPMBUILD_DIRS:
        os.makedirs(os.path.join(topdir, sub), exist_ok=True)

    provides = _rpm_marker_provides(env)
    if provides is None:
        return False
    log_info(_OP, f"Building the marker rpm {_MARKER} providing {len(provides)} "
                  "DOCA OFED dependencies", provides=provides, doca_version=doca_version)
    if not write_file(spec_file, rpm_marker_spec(doca_version, provides), _OP):
        return False
    if not run(["rpmbuild", "--define", f"_topdir {topdir}", "-bb", spec_file], _OP,
               env=env):
        return False
    marker_rpm = _find_marker_rpm(topdir)
    if marker_rpm is None or not run(["rpm", "-Uvh", "--replacepkgs", marker_rpm], _OP,
                                     env=env):
        return False
    remove_paths(topdir, op=_OP)
    log_info(_OP, f"Installed the {_MARKER} marker rpm", package=_MARKER, rpm=marker_rpm)
    return True


# --------------------------------------------------------------------------- #
# DKMS patch overlays
# --------------------------------------------------------------------------- #
def _add_patch(env: dict[str, str], patch: tuple[str, str], module: str,
               dkms_conf: str, *, append: bool = False) -> bool:
    """cp the patch into /etc/dkms/<module>/patches and list it in `dkms_conf`
    (`cat >`, or `cat >>` with append=True)."""
    patch_dir = os.path.join(_DKMS_ETC_DIR, module, "patches")
    os.makedirs(patch_dir, exist_ok=True)
    source = os.path.join(env.get("COMPONENT_DIR", ""), "patches", patch[0])
    dest = os.path.join(patch_dir, patch[1])
    # TODO(ADO 1001): toolkit candidate: copy_file(source, dest, op) -> bool (bash `cp`)
    shutil.copy(source, dest)
    log_info(_OP, f"Installed DKMS patch {dest}", source=source, path=dest,
             patch=patch[1])
    return write_file(dkms_conf, f'PATCH+=("{patch[1]}")\n', _OP, append=append)


def _kernel_needs_dpll_patch() -> bool:
    """True if the running kernel's linux/dpll.h has Red Hat's newer DPLL
    ffo_get callback, which DOCA 3.2.x / MLNX OFED 25.10 doesn't know yet."""
    header = os.path.join(_KERNEL_SRC_DIR, running_kernel(), "include", "linux", "dpll.h")
    try:
        with open(header, "r", encoding="utf-8", errors="replace") as handle:
            needs_patch = _DPLL_FFO_SIGNATURE in handle.read()
    except OSError:
        # bash: [[ -f "${kernel_header}" ]] || return 0 (and grep errors return 0)
        log_info(_OP, f"{header} not found: no DPLL DKMS patch", path=header)
        return False
    if needs_patch:
        log_info(_OP, "Kernel uses Red Hat's newer DPLL ffo_get signature: adding the "
                      "DPLL DKMS patch", path=header)
    else:
        log_info(_OP, "Kernel uses the older DPLL ffo_get signature: no DPLL DKMS patch",
                 path=header)
    return needs_patch


def _configure_doca_dkms_patches(env: dict[str, str], module: str) -> bool:
    """configure_doca_dkms_patches <module>: the PeerDirect GPL patch (not on
    V100) and, for mlnx-ofa_kernel, the DPLL patch when the kernel needs it."""
    dkms_conf = os.path.join(_DKMS_ETC_DIR, f"{module}.conf")
    # bash: mkdir -p "${patch_dir}", even when no patch goes in it.
    os.makedirs(os.path.join(_DKMS_ETC_DIR, module, "patches"), exist_ok=True)
    sku = env.get("SKU", "")
    if sku == _NO_PEER_MEM_PATCH_SKU:
        log_info(_OP, "V100: not adding the PeerDirect GPL-export DKMS patch (its "
                      "proprietary nvidia-peermem can't use GPL-only exports)",
                 sku=sku, module=module)
    elif not _add_patch(env, _PEER_MEM_PATCH, module, dkms_conf):
        return False
    if module == _RHEL_KERNEL_MODULE and _kernel_needs_dpll_patch():
        return _add_patch(env, _DPLL_PATCH, module, dkms_conf, append=True)
    return True


def _configure_xdp_patch(env: dict[str, str], doca_version: str) -> bool:
    """configure_mlnx_ofed_kernel_dkms_xdp_patch: Ubuntu 22.04 + DOCA 3.2.3 only."""
    distribution = env.get("DISTRIBUTION", "")
    if distribution != _XDP_DISTRIBUTION or doca_version != _XDP_DOCA_VERSION:
        log_info(_OP, "No XDP fragments DKMS patch (only for DOCA "
                      f"{_XDP_DOCA_VERSION} on {_XDP_DISTRIBUTION})",
                 distro=distribution, doca_version=doca_version)
        return True
    # Installed even on older kernels so later DKMS rebuilds get the fix.
    # TODO (upstream): remove when DOCA fixes detection of partially
    # backported XDP APIs.
    dkms_conf = os.path.join(_DKMS_ETC_DIR,
                             f"{_UBUNTU_KERNEL_MODULE}-{_XDP_MODULE_VERSION}.conf")
    return _add_patch(env, _XDP_PATCH, _UBUNTU_KERNEL_MODULE, dkms_conf)


# --------------------------------------------------------------------------- #
# Per-distro installers: each returns True on success, False once logged.
# --------------------------------------------------------------------------- #
def _install_ubuntu(env: dict[str, str], doca: _Doca, doca_file: str,
                    work_dir: str) -> bool:
    deb = os.path.realpath(os.path.join(work_dir, doca_file))   # "$(realpath ...)"
    if not install_packages([deb], env, _OP, cwd=work_dir):
        return False
    if not write_file(_DKMS_PIN, _DKMS_PIN_CONTENT, _OP):
        return False
    # TODO(ADO 1001): toolkit candidate: refresh_package_index(env, op) (apt-get update)
    if not run(["apt-get", "update"], _OP, env=env):
        return False
    if not _install_apt_marker(env, doca.version):
        return False
    if not _configure_doca_dkms_patches(env, _UBUNTU_KERNEL_MODULE):
        return False
    if not _configure_xdp_patch(env, doca.version):
        return False
    if not install_packages([_DOCA_OFED], env, _OP):
        return False
    return check_dkms_status(_UBUNTU_DKMS_MODULES, _OP, env)


def _install_azurelinux(env: dict[str, str], doca: _Doca, doca_file: str,
                        work_dir: str) -> bool:
    return (_add_doca_repo_rpm(env, doca, doca_file, work_dir)
            and _install_rpm_marker(env, doca.version)
            and install_packages([_DOCA_OFED], env, _OP))


@contextmanager
def _dnf_excludes_lifted():
    """bash: cp dnf.conf dnf.conf.bak; sed -i '/^exclude=/d' dnf.conf; ...;
    mv dnf.conf.bak dnf.conf. We restore on failure too."""
    backup = f"{_DNF_CONF}.bak"
    shutil.copy(_DNF_CONF, backup)
    original = Path(_DNF_CONF).read_text(encoding="utf-8")
    Path(_DNF_CONF).write_text(strip_dnf_excludes(original), encoding="utf-8")
    log_info(_OP, f"Backed up {_DNF_CONF} to {backup} and removed its exclude= lines",
             path=_DNF_CONF, backup=backup, action="edited")
    try:
        yield
    finally:
        os.replace(backup, _DNF_CONF)
        log_info(_OP, f"Restored {_DNF_CONF} from {backup}", path=_DNF_CONF,
                 action="restored")


def _install_rhel(env: dict[str, str], doca: _Doca, doca_file: str,
                  work_dir: str) -> bool:
    if not _add_doca_repo_rpm(env, doca, doca_file, work_dir):
        return False
    with _dnf_excludes_lifted():
        ok = (_install_rpm_marker(env, doca.version)
              and _configure_doca_dkms_patches(env, _RHEL_KERNEL_MODULE)
              and install_packages([_DOCA_OFED], env, _OP)
              and check_dkms_status(_RHEL_DKMS_MODULES, _OP, env))
    return ok and _pin_baseos_conflicts(env)


_INSTALLERS = {
    "ubuntu": _install_ubuntu,
    "azurelinux": _install_azurelinux,
    "rhel": _install_rhel,
}


def _repoquery_names(command: list[str], env: dict[str, str], what: str) -> str:
    """Run a `dnf repoquery` whose exit code bash ignores (it reads the output)."""
    rc, output = run_capture(command, _OP, env=env, failure_is_error=False)
    if rc != 0:
        log_warn(_OP, f"dnf repoquery of {what} failed; using its output anyway, "
                      "as bash does", exit_code=rc)
    return output


def _pin_baseos_conflicts(env: dict[str, str]) -> bool:
    """Exclude, in the baseos repo only, the packages DOCA's repos replaced.

    EL9.x baseos ships its own rdma-core stack (ABI-bumped libibverbs etc.)
    that conflicts with DOCA's; without this, later `dnf update`s abort. A
    repo-local exclude keeps `dnf check-update` meaningful.
    """
    doca_pkgs = installed_doca_packages(_repoquery_names(
        ["dnf", "repoquery", "--installed", "--quiet", "--qf", "%{name} %{from_repo}\\n"],
        env, "installed packages"))
    if not doca_pkgs:
        log_error(_OP, "no packages found from doca* repos after doca-ofed install")
        return False
    repo = baseos_repo_name(Distro.from_env(env), env.get("ARCHITECTURE", ""))
    output = _repoquery_names(["dnf", "repoquery", "--quiet", f"--repo={repo}", "--qf",
                               "%{name}\\n", "*"], env, repo)
    conflicts = sorted(set(doca_pkgs) & {line for line in output.split("\n") if line})
    log_info(_OP, f"{len(doca_pkgs)} installed packages come from doca* repos; "
                  f"{len(conflicts)} of them are also in {repo}",
             repo=repo, doca_packages=len(doca_pkgs), conflicts=conflicts)
    if not conflicts:
        return True
    log_info(_OP, f"Pinning {len(conflicts)} baseos package(s) shadowed by DOCA: "
                  f"{' '.join(conflicts)}", repo=repo, excludepkgs=conflicts)
    return run(["dnf", "config-manager", "--save",
                f"--setopt={repo}.excludepkgs={' '.join(conflicts)}"], _OP, env=env)


# --------------------------------------------------------------------------- #
# Common tail
# --------------------------------------------------------------------------- #
def _record_versions(env: dict[str, str], doca_version: str) -> bool:
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


def _disable_ipoib_if_unused(env: dict[str, str]) -> bool:
    """bash: sku_uses_ipoib || echo -e "\\n# Load IPoIB\\nIPOIB_LOAD=no" | tee -a openib.conf."""
    sku, node_type = env.get("SKU", ""), env.get("TARGET_NODE_TYPE", "")
    if sku_uses_ipoib(sku, node_type):
        log_info(_OP, "This target uses IPoIB: leaving openib.conf alone",
                 sku=sku, node_type=node_type, ipoib=True)
        return True
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
    return write_file(_OPENIB_CONF, _IPOIB_OFF, _OP, append=True)


def _configure_openibd(env: dict[str, str]) -> bool:
    """The openibd drop-in, IPoIB off where unused, and enable (not start) openibd."""
    if not write_file(os.path.join(_OPENIBD_DROPIN_DIR, "override.conf"),
                      _OPENIBD_OVERRIDE, _OP):
        return False
    if not _disable_ipoib_if_unused(env):
        return False
    # Enable only; do NOT restart here: restarting probes the build VM's IB
    # hardware (which may be absent on general-purpose build SKUs).
    if not (systemctl("daemon-reload", op=_OP, env=env)
            and systemctl("enable", "openibd", op=_OP, env=env)):
        return False
    log_info(_OP, "Enabled the openibd service (not started at build time)",
             service="openibd")
    return True


# --------------------------------------------------------------------------- #
# The step
# --------------------------------------------------------------------------- #
def install(env: dict[str, str]) -> int:
    """Install DOCA-OFED, record DOCA/OFED and set up openibd. Returns 0 or 3."""
    doca = _doca_config(env)
    if doca is None:
        return 3
    distro = Distro.from_env(env)
    branch = bash_branch(distro)
    work_dir = _bash_work_dir(env)
    log_info(_OP, f"Installing DOCA-OFED {doca.version} on {distro} ({branch} branch)",
             version=doca.version, source="private" if doca.private else "public",
             url=doca.url, repo_file=doca.repo_file, distro=str(distro), branch=branch,
             architecture=env.get("ARCHITECTURE", ""), sku=env.get("SKU", ""),
             node_type=env.get("TARGET_NODE_TYPE", ""), work_dir=work_dir)
    try:
        doca_file = _get_package(env, doca, work_dir)
        if doca_file is None:
            return 3
        if not _INSTALLERS[branch](env, doca, doca_file, work_dir):
            log_error(_OP, "DOCA-OFED installation failed", distro=str(distro),
                      branch=branch)
            return 3
        if not _record_versions(env, doca.version) or not _configure_openibd(env):
            return 3
    except OSError as exc:
        # Patch copies, dnf.conf, the marker build dirs and the versions file.
        log_error(_OP, f"file operation failed: {exc}",
                  path=getattr(exc, "filename", None))
        return 3
    log_info(_OP, f"DOCA-OFED {doca.version} installed", version=doca.version,
             distro=str(distro), branch=branch)
    return 0