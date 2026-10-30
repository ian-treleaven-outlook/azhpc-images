"""install_dcgm.py — Python port of components/install_dcgm.sh.

What this step does
-------------------
Installs NVIDIA DCGM (Data Center GPU Manager), the GPU health-monitoring and
diagnostics service, with its multinode diagnostic packages (upstream #600);
records its version as DCGM; and runs the nvidia-dcgm service's nv-hostengine
inside a screen session with HPC-X's Open MPI loaded, so `dcgmi mndiag` finds
an mpirun (upstream #649). The NVIDIA package repos were added by the
driver/CUDA steps.

DCGM packages are built per CUDA major version (e.g. ...-cuda13), so the step
first asks the installed driver which CUDA version it supports. Like bash, it
compiles and runs a tiny C program (_CUDA_PROBE_SOURCE, the bash heredoc byte
for byte) that calls cuDriverGetVersion(): 1000 * major + 10 * minor, e.g.
13000 for CUDA 13.0. That needs the driver's user-space library, not a GPU,
so it works on a build VM (nvidia-smi would not).

Per distro family (_INSTALLERS):

  * Ubuntu 22.04/24.04  the six datacenter-gpu-manager-4 packages pinned to
                  versions.json's "dcgm" version, plus every sub-package they
                  depend on with "(= <that version>)", found by walking
                  `apt-cache show` (bash's resolve_exact_version_deps,
                  upstream #639: otherwise apt picks the sub-packages' newest
                  revision and fails to resolve).
  * Ubuntu 26.04  the same six packages unpinned; the version is read back
                  with dpkg-query.
  * Azure Linux   an "hpcx-provides-openmpi" marker rpm first (unless
                  installed: it Provides openmpi, so the multinode package
                  doesn't pull Azure Linux's openmpi, whose PMIx files clash
                  with the pmix package), then cuda13, multinode and
                  multinode-cuda13 at the "dcgm" version.
  * RHEL family   (AlmaLinux, Rocky, RHEL; bash's final `else`) cuda<N> and
                  multinode-cuda<N> unpinned; the version is read back from
                  `dcgmi --version`.

When the SKU's CUDA version in versions.json (cuda.driver.version, e.g. 12.9
for a V100) has a lower major than the driver's, Ubuntu and the RHEL family
also install that major's cuda (Ubuntu: and proprietary-cuda) and
multinode-cuda packages: DCGM needs both (NVIDIA/DCGM issue #254).

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets install() as the "install-dcgm" Step's `action=`;
deleting `action=` runs install_dcgm.sh again, which is why it is kept.

Kept bash quirk
---------------
The Azure Linux branch picks cuda12 only for `[ "$1" = "V100" ]`, but
install.sh runs install_dcgm.sh with no arguments, so every SKU (V100
included) gets the cuda13 packages. This port does the same, and warns on a
V100 so its build log shows the packages it really got.

Deliberate differences from bash
--------------------------------
  * Installs go through install_packages(), which spells them
    `apt-get install -y -o Dpkg::Use-Pty=0` / `dnf install -y` (RHEL family:
    `dnf install -y --setopt=install_weak_deps=True` for bash's `--assumeyes`).
  * A missing "dcgm" version (Ubuntu 22.04/24.04, Azure Linux) or a
    missing/non-numeric SKU CUDA version (Ubuntu, RHEL family, where it is
    compared) fails before installing anything. Bash would look for
    "...=null", or compare "null" (bash arithmetic: 0) and install
    "...-cudanull". Azure Linux never uses the SKU CUDA version, so, as in
    bash, it may be missing.
  * Probe output that isn't a positive integer fails the step.
  * RHEL family: a failing `dcgmi --version` or an empty parsed version fails
    the step; bash (no pipefail) would record "". Likewise an empty
    dpkg-query result on Ubuntu 26.04.
  * The probe's and the marker's temporary directories come from
    tempfile.mkdtemp() (like `mktemp -d`) and are removed in all cases; bash
    leaves the marker's behind when rpmbuild fails.
  * resolve_exact_version_deps sorts each package's dependencies in plain
    character order (bash: `sort -u`, locale order). That only changes the
    order of the packages in the apt install.
  * If `screen` is still not on PATH after installing it, the step fails;
    bash would write a drop-in whose ExecStart has no program.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import tempfile

from utils.component_config import config_for, require_config, write_component_version
from utils.distro import Distro
from utils.files import remove_paths, write_file
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import command_exists, exec_program, run, run_capture
from utils.services import service_is_active, systemctl

_OP = "install-dcgm"

# Parent of the probe's and the marker's temporary directories. None is
# Python's default temp location, like `mktemp -d`; tests use their own.
_TEMP_PARENT_DIR: str | None = None
_PROBE_SOURCE_NAME = "cuda_driver_api_version.c"
_PROBE_BINARY_NAME = "cuda_driver_api_version"

# The C program from install_dcgm.sh's quoted heredoc (<<'EOF'), byte for
# byte: a quoted heredoc expands nothing. The raw string keeps C's "\n"
# escapes as written. A test checks it against the .sh file.
_CUDA_PROBE_SOURCE = r"""#include <dlfcn.h>
#include <stdio.h>

typedef int (*cuDriverGetVersionFn)(int *);

int main(void) {
    int exit_code = 1;
    int cuda_driver_api_version = 0;
    void *cuda_library = NULL;

    cuda_library = dlopen("libcuda.so.1", RTLD_NOW);
    if (cuda_library == NULL) {
        fprintf(stderr, "Unable to load libcuda.so.1: %s\n", dlerror());
        goto cleanup;
    }

    cuDriverGetVersionFn cuDriverGetVersion = (cuDriverGetVersionFn)dlsym(cuda_library, "cuDriverGetVersion");
    if (cuDriverGetVersion == NULL) {
        fprintf(stderr, "Unable to find cuDriverGetVersion in libcuda.so.1: %s\n", dlerror());
        goto cleanup;
    }

    int result = cuDriverGetVersion(&cuda_driver_api_version);
    if (result != 0 || cuda_driver_api_version <= 0) {
        fprintf(stderr, "Unable to query CUDA driver API version from libcuda.so.1\n");
        goto cleanup;
    }

    printf("%d\n", cuda_driver_api_version);
    exit_code = 0;

cleanup:
    if (cuda_library != NULL) {
        dlclose(cuda_library);
    }

    return exit_code;
}
"""

_SERVICE = "nvidia-dcgm"
# See "Kept bash quirk" above.
_AZURE_LINUX_CUDA_MAJOR = 13
# The one Ubuntu that installs DCGM unpinned and reads the version back.
_UNPINNED_UBUNTU = "ubuntu26.04"
# The extra dnf option bash passes when installing cuda<N> on the RHEL family.
_RHEL_CUDA_OPTIONS = ["--setopt=install_weak_deps=True"]
# bash: DCGM_VERSION=$(dpkg-query -W -f='${Version}' datacenter-gpu-manager-4-core)
_DPKG_QUERY_DCGM = ["dpkg-query", "-W", "-f=${Version}", "datacenter-gpu-manager-4-core"]

_AZURE_LINUX_MARKER = "hpcx-provides-openmpi"
_RPMBUILD_DIRS = ("BUILD", "BUILDROOT", "RPMS", "SOURCES", "SPECS", "SRPMS")
# The spec from bash's unquoted heredoc (only the marker name is expanded).
# {{ and }} are literal braces in an f-string: %{buildroot} stays as is.
_AZURE_LINUX_MARKER_SPEC = f"""Name: {_AZURE_LINUX_MARKER}
Version: 1.0
Release: 1
Summary: Marker package to indicate that HPC-X provides Open MPI
License: MIT
BuildArch: noarch
Provides: openmpi

%description
HPC-X, installed earlier by install_mpis.sh into /opt, provides Open MPI at
runtime for this image. This marker package satisfies the "openmpi"
dependency of datacenter-gpu-manager-4-multinode-cuda<N> so dnf does not
install the Azure Linux openmpi RPM, whose newest build bundles PMIx files
that conflict with the separately-installed pmix package.

%install
mkdir -p %{{buildroot}}/usr/share/doc/%{{name}}
cat > %{{buildroot}}/usr/share/doc/%{{name}}/README <<'README'
HPC-X provides Open MPI for this image; see install_mpis.sh.
README

%files
/usr/share/doc/%{{name}}/README
"""

# A module constant so tests can point it at a temporary directory.
_DROPIN_DIR = "/etc/systemd/system/nvidia-dcgm.service.d"
_DROPIN_NAME = "99-mndiag-hpcx.conf"

# What bash word splitting and awk's default field splitting treat as
# separators: spaces, tabs and newlines.
_BLANKS = re.compile(r"[ \t\n]+")
# An apt dependency pinned to an exact version: "name (= version)".
_EXACT_DEP = re.compile(r"([^ ]+) \(= ([^)]+)\)")


# --------------------------------------------------------------------------- #
# Pure helpers: they only transform values, so tests call them directly.
# --------------------------------------------------------------------------- #
def parse_cuda_driver_version(output: str) -> int | None:
    """The positive integer the CUDA probe printed ("13000\\n" -> 13000), or None."""
    text = output.strip()
    if not re.fullmatch(r"[0-9]+", text) or int(text) <= 0:
        return None
    return int(text)


def cuda_major(driver_api_version: int) -> int:
    """bash $((CUDA_DRIVER_SUPPORTED_VERSION / 1000)): 13000 -> 13, 12090 -> 12."""
    return driver_api_version // 1000


def sku_cuda_major(cuda_config) -> int | None:
    """The major of cuda.driver.version from versions.json ("12.9" -> 12).

    bash: jq -r '.driver.version' | cut -d'.' -f1. None when the value is
    missing or its major isn't a number, where bash would carry on with "null".
    """
    driver = cuda_config.get("driver") if isinstance(cuda_config, dict) else None
    version = driver.get("version") if isinstance(driver, dict) else None
    if version is None:
        return None
    major = str(version).split(".")[0]
    return int(major) if re.fullmatch(r"[0-9]+", major) else None


def dcgmi_version(output: str) -> str | None:
    """The DCGM version from `dcgmi --version` output, or None.

    bash: DCGM_VERSION=$(dcgmi --version | awk '{print $3}'), then the unquoted
    ${DCGM_VERSION}: awk prints field 3 of every line ("" if a line is
    shorter) and write_component_version records only the first word.

        "dcgmi  version: 4.4.1"  ->  "4.4.1"
    """
    values = []
    for line in output.split("\n"):
        fields = [field for field in _BLANKS.split(line) if field]
        values.append(fields[2] if len(fields) >= 3 else "")
    words = [word for word in _BLANKS.split("\n".join(values)) if word]
    return words[0] if words else None


def ubuntu_packages(cuda: int) -> list[str]:
    """The six Ubuntu packages bash installs for CUDA major `cuda`."""
    return [
        f"datacenter-gpu-manager-4-cuda{cuda}",
        "datacenter-gpu-manager-4-core",
        "datacenter-gpu-manager-4-proprietary",
        f"datacenter-gpu-manager-4-proprietary-cuda{cuda}",
        "datacenter-gpu-manager-4-multinode",
        f"datacenter-gpu-manager-4-multinode-cuda{cuda}",
    ]


def ubuntu_extra_packages(sku_cuda: int) -> list[str]:
    """The three extra Ubuntu packages for a lower SKU CUDA major."""
    return [
        f"datacenter-gpu-manager-4-cuda{sku_cuda}",
        f"datacenter-gpu-manager-4-proprietary-cuda{sku_cuda}",
        f"datacenter-gpu-manager-4-multinode-cuda{sku_cuda}",
    ]


def azure_linux_packages(cuda: int, dcgm_version: str) -> list[str]:
    """The three Azure Linux packages, `<name>-<version>` as dnf takes them."""
    return [
        f"datacenter-gpu-manager-4-cuda{cuda}-{dcgm_version}",
        f"datacenter-gpu-manager-4-multinode-{dcgm_version}",
        f"datacenter-gpu-manager-4-multinode-cuda{cuda}-{dcgm_version}",
    ]


def rhel_installs(cuda: int) -> list[tuple[list[str], list[str]]]:
    """The cuda<N> and multinode-cuda<N> installs bash runs on the RHEL family,
    as (packages, dnf options), one per bash command."""
    return [([f"datacenter-gpu-manager-4-cuda{cuda}"], _RHEL_CUDA_OPTIONS),
            ([f"datacenter-gpu-manager-4-multinode-cuda{cuda}"], [])]


def exact_version_deps(apt_cache_output: str, version: str) -> list[str]:
    """Dependencies declared as "(= <version>)" in `apt-cache show` output.

    Mirrors the awk filter + `sort -u` in bash's resolve_exact_version_deps:
    on each (Pre-)Depends line, split on "," and "|", keep "name (= ver)"
    entries whose ver is `version`, and drop any ":arch" from the name.

        "Depends: libdcgm4 (= 1:4.7.0-2), dcgmi (>= 4)"  ->  ["libdcgm4"]
    """
    names = set()
    for line in apt_cache_output.split("\n"):
        if not re.match(r"(Pre-)?Depends:", line):
            continue
        rest = re.sub(r"^[^:]*:[ \t]*", "", line, count=1)
        for dep in re.split(r"[,|]", rest):
            match = _EXACT_DEP.fullmatch(dep.strip(" \t"))
            if match and match.group(2) == version:
                names.add(match.group(1).split(":", 1)[0])
    return sorted(names)


def unit_exec_start(unit_text: str) -> str:
    """The last ExecStart= value in a systemd unit, or "".

    bash: sed -n 's/^ExecStart=//p' "${DCGM_UNIT_FILE}" | tail -n1
    """
    values = [line[len("ExecStart="):] for line in unit_text.split("\n")
              if line.startswith("ExecStart=")]
    return values[-1] if values else ""


def hostengine_dropin(screen: str, hostengine_cmd: str) -> str:
    """The 99-mndiag-hpcx.conf drop-in from bash's unquoted heredoc.

    `screen` is $(command -v screen) and `hostengine_cmd` the unit's original
    ExecStart. bash's `\\$\\$` becomes `$$`, which systemd turns into one `$`,
    so `$(command -v mpirun)` runs inside screen after mpi/hpcx is loaded.
    """
    return (
        "[Service]\n"
        "LimitMEMLOCK=infinity\n"
        "UnsetEnvironment=DCGM_MNDIAG_MPIRUN_PATH LD_LIBRARY_PATH\n"
        "RuntimeDirectory=nvidia-dcgm-screen\n"
        "RuntimeDirectoryMode=0700\n"
        'Environment="SCREENDIR=/run/nvidia-dcgm-screen"\n'
        'Environment="OMPI_MCA_mca_base_env_list=LD_LIBRARY_PATH;PATH"\n'
        "Restart=always\n"
        "ExecStart=\n"
        f"ExecStart={screen} -c /dev/null -DmS nvidia-dcgm /bin/bash -c "
        "'source /etc/profile.d/modules.sh && module load mpi/hpcx && "
        'export DCGM_MNDIAG_MPIRUN_PATH="$$(command -v mpirun)" && '
        f"exec {hostengine_cmd}'\n"
    )


# --------------------------------------------------------------------------- #
# Small jobs the toolkit doesn't cover yet.
# --------------------------------------------------------------------------- #
# TODO(ADO 1001): toolkit candidate: make_temp_dir(op, *, prefix="tmp",
# subdirs=(), parent=None) -> Path | None (bash `mktemp -d`, then `mkdir -p`
# inside it; logs failures).
def _make_temp_dir(prefix: str, subdirs=()) -> str | None:
    """bash `mktemp -d` (plus `mkdir -p` of `subdirs` inside it), or None (logged)."""
    path = None
    try:
        path = tempfile.mkdtemp(prefix=prefix, dir=_TEMP_PARENT_DIR)
        for sub in subdirs:
            os.makedirs(os.path.join(path, sub), exist_ok=True)
        return path
    except OSError as exc:
        log_error(_OP, f"could not create a temporary directory: {exc}",
                  path=getattr(exc, "filename", None))
        if path:
            remove_paths(path, op=_OP)
        return None


# TODO(ADO 1001): toolkit candidate: command_path(name, env) -> str | None
# (bash `$(command -v name)`; command_exists() only answers yes/no).
def _command_path(name: str, env: dict[str, str]) -> str | None:
    """bash `command -v <name>`: its path on the build env's PATH, or None."""
    return shutil.which(name, path=env.get("PATH"))


# --------------------------------------------------------------------------- #
# 1. Which CUDA major the installed driver supports.
# --------------------------------------------------------------------------- #
def _driver_cuda_major(env) -> int | None:
    """Compile and run the C probe in a temporary directory; its CUDA major,
    or None (logged)."""
    probe_dir = _make_temp_dir("cuda_probe.")
    if probe_dir is None:
        return None
    try:
        driver_api = _run_cuda_probe(probe_dir, env)
    finally:
        remove_paths(probe_dir, op=_OP)     # bash: trap 'rm -rf ...' EXIT
    if driver_api is None:
        return None
    cuda = cuda_major(driver_api)
    log_info(_OP, f"Probed CUDA driver API version {driver_api} (CUDA {cuda})",
             cuda_driver_api_version=driver_api, cuda_major=cuda)
    return cuda


def _run_cuda_probe(probe_dir: str, env) -> int | None:
    """cuDriverGetVersion() from the probe built in `probe_dir`, or None (logged)."""
    source = os.path.join(probe_dir, _PROBE_SOURCE_NAME)
    binary = os.path.join(probe_dir, _PROBE_BINARY_NAME)
    if not (write_file(source, _CUDA_PROBE_SOURCE, _OP)
            and run(["gcc", source, "-ldl", "-o", binary], _OP, env=env)):
        return None
    # The probe prints the version on stdout; its errors go to stderr.
    rc, output = run_capture([binary], _OP, env=env)
    if rc != 0:
        log_error(_OP, "the CUDA driver probe could not query libcuda.so.1 "
                       "(is the NVIDIA driver installed?)", exit_code=rc)
        return None
    version = parse_cuda_driver_version(output)
    if version is None:
        log_error(_OP, f"unexpected CUDA driver probe output: {output.strip()!r}")
    return version


# --------------------------------------------------------------------------- #
# 2. The packages, per distro. Each installer returns the DCGM version to
#    record, or None on failure (logged).
# --------------------------------------------------------------------------- #
def _config_dcgm_version(env) -> str | None:
    """versions.json's "dcgm" version, or None (logged by require_config)."""
    cfg = require_config("dcgm", env, _OP, "version")
    return None if cfg is None else str(cfg["version"])


def _have_sku_cuda(sku_cuda: int | None) -> bool:
    """Whether the SKU CUDA major the Ubuntu/RHEL comparison needs is known."""
    if sku_cuda is None:
        log_error(_OP, "could not resolve a numeric cuda driver version for this "
                       "SKU from versions.json")
    return sku_cuda is not None


def _needs_sku_packages(cuda: int, sku_cuda: int) -> bool:
    """Whether a lower SKU CUDA major needs its own DCGM packages too (logged)."""
    needed = sku_cuda < cuda
    decision = (f"also installing the cuda{sku_cuda} DCGM packages" if needed
                else "no extra DCGM packages")
    log_info(_OP, f"SKU CUDA major {sku_cuda}, driver CUDA major {cuda}: {decision}",
             cuda_major=cuda, sku_cuda_major=sku_cuda)
    return needed


def _resolve_exact_version_deps(version: str, packages: list[str],
                                env) -> list[str] | None:
    """bash resolve_exact_version_deps: "<pkg>=<version>" for `packages` and,
    recursively, every dependency they declare as "(= <version>)".

    Breadth-first, each package once, in bash's order. None (logged) if a
    package isn't available at that version.
    """
    queue = list(packages)
    seen: set[str] = set()
    resolved: list[str] = []
    while queue:
        package = queue.pop(0)
        if package in seen:
            continue
        seen.add(package)
        # bash: $(apt-cache show "${pkg}=${version}" 2>/dev/null || true)
        _rc, output = run_capture(["apt-cache", "show", f"{package}={version}"], _OP,
                                  env=env, failure_is_error=False)
        if not output.rstrip("\n"):
            log_error(_OP, f"{package}={version} is not available from the configured "
                           "apt repositories", package=package, version=version)
            return None
        resolved.append(f"{package}={version}")
        deps = exact_version_deps(output, version)
        log_debug(_OP, f"{package}={version} depends on exact-version {deps}",
                  package=package, deps=deps)
        queue.extend(deps)
    log_info(_OP, f"Resolved {len(resolved)} DCGM packages pinned to {version}",
             version=version, requested=packages, packages=resolved)
    return resolved


def _apt_install_dcgm(env, packages: list[str], dcgm: str | None) -> bool:
    """One apt install: pinned with the exact-version dependencies when `dcgm`
    is given, else unpinned (Ubuntu 26.04)."""
    if dcgm is not None:
        packages = _resolve_exact_version_deps(dcgm, packages, env)
        if packages is None:
            return False
    return install_packages(packages, env, _OP)


# TODO(ADO 1001): toolkit candidate: installed_package_version(package, env,
# op) -> str | None (dpkg-query -W -f='${Version}' / rpm -q --qf '%{VERSION}').
def _dpkg_dcgm_version(env) -> str | None:
    """The installed datacenter-gpu-manager-4-core version, or None (logged)."""
    rc, output = run_capture(_DPKG_QUERY_DCGM, _OP, env=env)
    dcgm = output.rstrip("\n") if rc == 0 else ""
    if not dcgm:
        log_error(_OP, "could not read the installed datacenter-gpu-manager-4-core "
                       "version with dpkg-query", exit_code=rc)
        return None
    log_info(_OP, f"Installed DCGM version is {dcgm}", dcgm_version=dcgm,
             version_source="dpkg-query")
    return dcgm


def _dcgmi_reported_version(env) -> str | None:
    """The version `dcgmi --version` reports, or None (logged)."""
    rc, output = run_capture(["dcgmi", "--version"], _OP, env=env)
    dcgm = dcgmi_version(output) if rc == 0 else None
    if dcgm is None:
        log_error(_OP, f"could not read the DCGM version from `dcgmi --version`: "
                       f"{output.strip()!r}", exit_code=rc)
        return None
    log_info(_OP, f"Installed DCGM version is {dcgm}", dcgm_version=dcgm,
             version_source="dcgmi")
    return dcgm


def _install_ubuntu(env, distro: Distro, cuda: int, sku_cuda: int | None) -> str | None:
    """apt-install DCGM, pinned to the "dcgm" version except on Ubuntu 26.04,
    which installs the latest and reads its version back."""
    pinned = distro.name != _UNPINNED_UBUNTU
    dcgm = _config_dcgm_version(env) if pinned else None
    if (pinned and dcgm is None) or not _have_sku_cuda(sku_cuda):
        return None
    log_info(_OP, f"Ubuntu: installing DCGM {dcgm or '(latest)'} for CUDA {cuda} "
                  "from apt", distro=distro.name, dcgm_version=dcgm, pinned=pinned,
             cuda_major=cuda)

    if not _apt_install_dcgm(env, ubuntu_packages(cuda), dcgm):
        return None
    if (_needs_sku_packages(cuda, sku_cuda)
            and not _apt_install_dcgm(env, ubuntu_extra_packages(sku_cuda), dcgm)):
        return None
    return dcgm if pinned else _dpkg_dcgm_version(env)


def _install_azurelinux(env, distro: Distro, cuda: int,
                        sku_cuda: int | None) -> str | None:
    """The marker rpm, then the cuda13 DCGM packages at the "dcgm" version."""
    dcgm = _config_dcgm_version(env)
    if dcgm is None or not _install_azure_linux_marker(env):
        return None
    packages = azure_linux_packages(_AZURE_LINUX_CUDA_MAJOR, dcgm)
    log_info(_OP, f"Azure Linux: installing DCGM {dcgm} (always the "
                  f"cuda{_AZURE_LINUX_CUDA_MAJOR} packages, like bash)",
             dcgm_version=dcgm, cuda_major=cuda, sku_cuda_major=sku_cuda,
             packages=packages)
    # The kept bash quirk only matters on a V100, which bash meant to get cuda12.
    if env.get("SKU", "") == "V100":
        log_warn(_OP, "Azure Linux V100: installing the cuda13 DCGM packages "
                      "because install.sh passes no SKU argument to "
                      "install_dcgm.sh (kept for bash parity)",
                 sku="V100", packages=packages)
    return dcgm if install_packages(packages, env, _OP) else None


def _install_azure_linux_marker(env) -> bool:
    """Build and install the hpcx-provides-openmpi marker rpm unless installed."""
    if exec_program(["rpm", "-q", _AZURE_LINUX_MARKER], _OP, env=env,
                    failure_is_error=False) == 0:
        log_info(_OP, f"Marker rpm {_AZURE_LINUX_MARKER} is already installed",
                 package=_AZURE_LINUX_MARKER)
        return True
    topdir = _make_temp_dir("tmp", subdirs=_RPMBUILD_DIRS)
    if topdir is None:
        return False
    try:
        marker_rpm = _build_marker_rpm(topdir, env)
        return marker_rpm is not None and run(
            ["rpm", "-Uvh", "--replacepkgs", marker_rpm], _OP, env=env)
    finally:
        remove_paths(topdir, op=_OP)


def _build_marker_rpm(topdir: str, env) -> str | None:
    """rpmbuild the marker spec under `topdir`; the rpm's path, or None (logged)."""
    spec_file = os.path.join(topdir, f"{_AZURE_LINUX_MARKER}.spec")
    if not write_file(spec_file, _AZURE_LINUX_MARKER_SPEC, _OP):
        return None
    log_info(_OP, f"Building the marker rpm {_AZURE_LINUX_MARKER}",
             package=_AZURE_LINUX_MARKER, spec=spec_file)
    if not run(["rpmbuild", "--define", f"_topdir {topdir}", "-bb", spec_file],
               _OP, env=env):
        return None
    # bash: find "${rpm_topdir}/RPMS" -type f -name "<marker>-*.rpm" | head -n1
    built = sorted(path for path in glob.glob(
        os.path.join(topdir, "RPMS", "**", f"{_AZURE_LINUX_MARKER}-*.rpm"),
        recursive=True) if os.path.isfile(path))
    if not built:
        log_error(_OP, f"failed to build {_AZURE_LINUX_MARKER} marker RPM",
                  rpms_dir=os.path.join(topdir, "RPMS"))
        return None
    return built[0]


def _install_rhel(env, distro: Distro, cuda: int, sku_cuda: int | None) -> str | None:
    """dnf-install the latest DCGM for the driver's (and a lower SKU's) CUDA
    major, then read the version from dcgmi."""
    if not _have_sku_cuda(sku_cuda):
        return None
    log_info(_OP, f"RHEL family: installing the latest DCGM for CUDA {cuda} from dnf",
             distro=distro.name, cuda_major=cuda)
    installs = rhel_installs(cuda)
    if _needs_sku_packages(cuda, sku_cuda):
        installs += rhel_installs(sku_cuda)
    if not run(["dnf", "clean", "expire-cache"], _OP, env=env):
        return None
    for packages, options in installs:
        if not install_packages(packages, env, _OP, options=options):
            return None
    return _dcgmi_reported_version(env)


# Keyed by Distro.family; anything else takes bash's final `else` (RHEL family).
_INSTALLERS = {"ubuntu": _install_ubuntu, "azurelinux": _install_azurelinux}


def _record_version(dcgm: str) -> bool:
    """write_component_version("DCGM", ...), logging a file error."""
    try:
        write_component_version("DCGM", dcgm)
    except OSError as exc:
        log_error(_OP, f"could not record the DCGM version: {exc}",
                  path=getattr(exc, "filename", None))
        return False
    log_info(_OP, f"Recorded DCGM version {dcgm}", component="DCGM", version=dcgm)
    return True


# --------------------------------------------------------------------------- #
# 3. nv-hostengine under screen with HPC-X, then the service.
# --------------------------------------------------------------------------- #
def _configure_hostengine(env) -> bool:
    """Write the drop-in that runs nv-hostengine in screen with mpi/hpcx loaded.

    dcgmi mndiag runs the mpirun in nv-hostengine's DCGM_MNDIAG_MPIRUN_PATH;
    the multinode package points it at a distro Open MPI this image doesn't
    have. The module is loaded inside screen (which drops LD_* variables on
    some distros); screen exits cleanly when nv-hostengine dies, hence
    Restart=always; UCX needs unlimited memlock. Attach with
    `sudo SCREENDIR=/run/nvidia-dcgm-screen screen -r nvidia-dcgm`.
    """
    screen = _ensure_screen(env)
    if screen is None:
        return False
    hostengine_cmd = _hostengine_command(env)
    if hostengine_cmd is None:
        return False
    log_info(_OP, f"nv-hostengine command: {hostengine_cmd}",
             exec_start=hostengine_cmd, screen=screen)
    dropin = os.path.join(_DROPIN_DIR, _DROPIN_NAME)
    return (write_file(dropin, hostengine_dropin(screen, hostengine_cmd), _OP)
            and systemctl("daemon-reload", op=_OP, env=env))


def _ensure_screen(env) -> str | None:
    """bash `command -v screen || <install screen>`; screen's path, or None (logged)."""
    if not command_exists("screen", env):
        log_info(_OP, "screen not found: installing it", package="screen")
        if not install_packages(["screen"], env, _OP):
            return None
    screen = _command_path("screen", env)
    if screen is None:
        log_error(_OP, "screen is still not on PATH after installing it; the drop-in "
                       "needs its path", path=env.get("PATH"))
    return screen


def _hostengine_command(env) -> str | None:
    """The nvidia-dcgm unit's (last) nv-hostengine ExecStart, or None (logged)."""
    rc, output = run_capture(["systemctl", "show", "-p", "FragmentPath", "--value",
                              f"{_SERVICE}.service"], _OP, env=env)
    if rc != 0:
        log_error(_OP, f"could not find the {_SERVICE} unit file", exit_code=rc)
        return None
    unit_file = output.rstrip("\n")
    try:
        with open(unit_file, "r", encoding="utf-8", errors="replace") as handle:
            hostengine_cmd = unit_exec_start(handle.read())
    except OSError as exc:
        log_warn(_OP, f"could not read {unit_file!r}: {exc}", path=unit_file)
        hostengine_cmd = ""
    if "/nv-hostengine" not in hostengine_cmd:
        log_error(_OP, f"could not find the nv-hostengine ExecStart in {unit_file}",
                  path=unit_file, exec_start=hostengine_cmd)
        return None
    return hostengine_cmd


def _restart_service(env) -> bool:
    """Enable and restart nvidia-dcgm (so a running instance picks up the
    drop-in), then check it is active."""
    if not (systemctl("--now", "enable", _SERVICE, op=_OP, env=env)    # bash's order
            and systemctl("restart", _SERVICE, op=_OP, env=env)):
        return False
    if not service_is_active(_SERVICE, op=_OP, env=env):
        log_error(_OP, "DCGM is inactive!", service=_SERVICE)      # bash's message
        return False
    log_info(_OP, f"{_SERVICE} service is active", service=_SERVICE)
    return True


def install(env: dict[str, str]) -> int:
    """Install DCGM, record its version and (re)start nvidia-dcgm. Returns 0 or 3."""
    distro = Distro.from_env(env)
    log_info(_OP, "Installing DCGM", distro=distro.name, sku=env.get("SKU", ""),
             node_type=env.get("TARGET_NODE_TYPE", ""))

    cuda = _driver_cuda_major(env)
    if cuda is None:
        return 3
    sku_cuda = sku_cuda_major(config_for("cuda", env))
    log_info(_OP, f"SKU CUDA major from versions.json: {sku_cuda}",
             sku_cuda_major=sku_cuda, cuda_major=cuda)

    installer = _INSTALLERS.get(distro.family, _install_rhel)
    dcgm = installer(env, distro, cuda, sku_cuda)
    if dcgm is None or not _record_version(dcgm):
        return 3
    if not _configure_hostengine(env) or not _restart_service(env):
        return 3

    log_info(_OP, "DCGM installed", distro=distro.name, dcgm_version=dcgm,
             cuda_major=cuda, sku_cuda_major=sku_cuda)
    return 0
