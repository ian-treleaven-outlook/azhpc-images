"""install_dcgm.py — Python port of components/install_dcgm.sh.

What this step does
-------------------
Installs NVIDIA DCGM (Data Center GPU Manager), the GPU health-monitoring and
diagnostics service, including its multinode diagnostic packages (upstream
#600), records its version as DCGM in the component-versions file, and runs
the nvidia-dcgm service's nv-hostengine inside a screen session with HPC-X's
Open MPI loaded, so `dcgmi mndiag` finds an mpirun (upstream #649). The
NVIDIA package repos were already added by the driver/CUDA steps.

DCGM packages are built per CUDA major version (e.g. ...-cuda13), so the step
first asks the installed NVIDIA driver which CUDA version it supports. Like
bash, it compiles and runs a tiny C program that loads libcuda.so.1 and calls
cuDriverGetVersion(), which returns 1000 * major + 10 * minor (13000 for CUDA
13.0). This needs the driver's user-space library, not GPU hardware, so it
works on a build VM without a GPU (nvidia-smi would not).

Per distro family:

  * Ubuntu 22.04/24.04  install the six datacenter-gpu-manager-4 packages
                  (cuda<N>, core, proprietary, proprietary-cuda<N>, multinode,
                  multinode-cuda<N>) pinned to the "dcgm" version in
                  versions.json, plus every sub-package they depend on with
                  "(= <that version>)", found by walking `apt-cache show`
                  (upstream #639: DCGM 4.7+ is split into many exactly-versioned
                  sub-packages, and apt would otherwise pick their newest
                  revision and fail to resolve).
  * Ubuntu 26.04  install the same six packages unpinned, then read the
                  installed version from dpkg-query.
  * Azure Linux   build and install an "hpcx-provides-openmpi" marker rpm
                  (unless installed), so the multinode package's openmpi
                  dependency doesn't pull Azure Linux's openmpi (its PMIx files
                  clash with the pmix package); then install cuda13,
                  multinode and multinode-cuda13 at the "dcgm" version (see
                  "Bash bug, mirrored" below).
  * RHEL family   (AlmaLinux, Rocky, RHEL) install cuda<N> and
                  multinode-cuda<N> (unpinned), then read the installed
                  version from `dcgmi`.

When the SKU's CUDA version in versions.json (e.g. 12.9 for a V100) has a
lower major than the driver supports, Ubuntu and RHEL also install the cuda,
(Ubuntu: proprietary-cuda,) and multinode-cuda packages for that lower major
(DCGM needs both; NVIDIA/DCGM issue #254).

The original bash, for comparison (comments trimmed; the C program is in
_CUDA_PROBE_SOURCE below, byte for byte, and the heredocs in
_AZURE_LINUX_MARKER_SPEC and hostengine_dropin()):

    CUDA_DRIVER_SUPPORTED_VERSION=$( <compile and run the C probe in mktemp -d> )
    CUDA_VERSION=$((CUDA_DRIVER_SUPPORTED_VERSION / 1000))
    SKU_CUDA_VERSION=$(jq -r '.driver.version' <<< $cuda_metadata | cut -d'.' -f1)

    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        if ubuntu26.04:
            apt-get install -y <the six packages for CUDA_VERSION>
        else:
            DCGM_VERSION=$(jq -r '.version' <<< $dcgm_metadata)
            dcgm_packages=$(resolve_exact_version_deps "${DCGM_VERSION}" <the six packages>)
            apt-get install -y ${dcgm_packages}
        if [[ "${SKU_CUDA_VERSION}" -lt "${CUDA_VERSION}" ]]; then
            same, for cuda${SKU}, proprietary-cuda${SKU}, multinode-cuda${SKU}
        fi
        [ubuntu26.04: DCGM_VERSION=$(dpkg-query -W -f='${Version}' datacenter-gpu-manager-4-core)]
    elif [[ $DISTRIBUTION == *"azurelinux"* ]]; then
        DCGM_VERSION=$(jq -r '.version' <<< $dcgm_metadata)
        if ! rpm -q hpcx-provides-openmpi; then
            rpm_topdir=$(mktemp -d); <write spec>; rpmbuild --define "_topdir ..." -bb <spec>
            rpm -Uvh --replacepkgs $(find .../RPMS -name 'hpcx-provides-openmpi-*.rpm' | head -n1)
            rm -rf "${rpm_topdir}"
        fi
        if [ "$1" = "V100" ]; then
            dnf install -y ...-cuda12-${V} ...-multinode-${V} ...-multinode-cuda12-${V}
        else
            dnf install -y ...-cuda13-${V} ...-multinode-${V} ...-multinode-cuda13-${V}
        fi
    else
        dnf clean expire-cache
        dnf install --assumeyes --setopt=install_weak_deps=True datacenter-gpu-manager-4-cuda${CUDA_VERSION}
        dnf install --assumeyes datacenter-gpu-manager-4-multinode-cuda${CUDA_VERSION}
        [SKU lower: the same two for ${SKU_CUDA_VERSION}]
        DCGM_VERSION=$(dcgmi --version | awk '{print $3}')
    fi

    write_component_version "DCGM" ${DCGM_VERSION}

    command -v screen || apt-get install -y screen  (else: dnf install -y screen)
    DCGM_UNIT_FILE=$(systemctl show -p FragmentPath --value nvidia-dcgm.service)
    DCGM_HOSTENGINE_CMD=$(sed -n 's/^ExecStart=//p' "${DCGM_UNIT_FILE}" | tail -n1)
    [[ "${DCGM_HOSTENGINE_CMD}" != */nv-hostengine* ]] && exit 1
    cat > /etc/systemd/system/nvidia-dcgm.service.d/99-mndiag-hpcx.conf <<EOF ... EOF
    systemctl daemon-reload
    systemctl --now enable nvidia-dcgm
    systemctl restart nvidia-dcgm
    systemctl is-active --quiet nvidia-dcgm     # set -e: a failure stops the script

Bash bug, mirrored: the Azure Linux branch checks `[ "$1" = "V100" ]`, but
install.sh runs install_dcgm.sh with no arguments, so $1 is always empty and
every SKU (V100 included) gets the cuda13 packages. This port always installs
cuda13 on Azure Linux too, does not try to "fix" it from SKU, and warns on V100.

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-dcgm" Step. Deleting `action=` from that Step makes the build run the
bash script again, which is why the .sh file is kept.

install() receives the build environment (a dict) and returns 0 for success or
3 for failure. It logs problems and never lets an exception escape.

Deliberate differences from bash
--------------------------------
  * A missing "dcgm" version (Ubuntu 22.04/24.04, Azure Linux) or a
    missing/non-numeric CUDA driver version for the SKU (Ubuntu, RHEL family,
    where it is compared) fails before installing anything. Bash would look
    for "...=null", or compare "null" (which bash arithmetic treats as 0)
    and install "...-cudanull". Azure Linux never uses the SKU CUDA version,
    so, as in bash, it may be missing.
  * Output from the CUDA probe that isn't a positive integer fails the step.
  * On the RHEL family, a failing `dcgmi --version` or an empty parsed
    version fails the step; bash (no pipefail) would record "". Likewise an
    empty dpkg-query result on Ubuntu 26.04.
  * The probe's temporary directory is created by Python's tempfile.mkdtemp()
    (like `mktemp -d`, under $TMPDIR or /tmp) and removed afterwards in all
    cases, like bash's EXIT trap. The Azure Linux marker's rpmbuild
    directory is removed in all cases too (bash leaves it behind on failure).
  * resolve_exact_version_deps sorts each package's dependencies in plain
    character order (bash: `sort -u`, locale order). That only changes the
    order of the arguments to `apt-get install`.
  * If `screen` is still not on PATH after installing it, the step fails;
    bash would write a drop-in whose ExecStart has no program.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import tempfile

from utils.component_config import config_for, write_component_version
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.process import exec_program, run_capture

# The Step's `op` label in build_config.py; used on every log line.
_OP = "install-dcgm"

# Where the probe's (and the Azure Linux marker's) temporary directory is
# created. None means Python's default temp location, like `mktemp -d`. The
# tests point it at their own directory.
_PROBE_PARENT_DIR: str | None = None
_PROBE_SOURCE_NAME = "cuda_driver_api_version.c"
_PROBE_BINARY_NAME = "cuda_driver_api_version"

# The C program from install_dcgm.sh's quoted heredoc (<<'EOF'), byte for byte:
# a quoted heredoc expands nothing, and its text ends with a newline after "}".
# A test checks this against the .sh file so the two can't drift apart.
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
# (The r"""...""" "raw string" keeps the C "\n" escapes as the two characters
# backslash and n, exactly as they appear in the bash heredoc.)

# The DCGM service, and the Azure Linux package's fixed CUDA major (see above).
_SERVICE = "nvidia-dcgm"
_AZURE_LINUX_CUDA_MAJOR = 13
# The one Ubuntu that installs DCGM unpinned and reads the version back.
_UNPINNED_UBUNTU = "ubuntu26.04"

# The dnf options bash passes on the RHEL family.
_RHEL_DNF_INSTALL = ["dnf", "install", "--assumeyes",
                     "--setopt=install_weak_deps=True"]
_RHEL_DNF_INSTALL_MULTINODE = ["dnf", "install", "--assumeyes"]

# bash: DCGM_VERSION=$(dpkg-query -W -f='${Version}' datacenter-gpu-manager-4-core)
_DPKG_QUERY_DCGM = ["dpkg-query", "-W", "-f=${Version}", "datacenter-gpu-manager-4-core"]

# ---- Azure Linux marker rpm ---------------------------------------------------
_AZURE_LINUX_MARKER = "hpcx-provides-openmpi"
# The spec from bash's unquoted heredoc (only the marker name is expanded).
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
# (In an f-string, {{ and }} are literal braces, so the spec's %{buildroot}
# and %{name} come out unchanged.)

# ---- nv-hostengine under screen (upstream #649) -------------------------------
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
    """The integer the CUDA probe printed (e.g. "13000\\n" -> 13000), or None.

    Bash's $(...) drops the trailing newline and $(( ... )) does the maths. We
    only accept a plain positive integer; anything else is None.
    """
    text = output.strip()
    # fullmatch() requires the WHOLE string to match: digits only.
    if not re.fullmatch(r"[0-9]+", text) or int(text) <= 0:
        return None
    return int(text)


def cuda_major(driver_api_version: int) -> int:
    """$((CUDA_DRIVER_SUPPORTED_VERSION / 1000)): 13000 -> 13, 12090 -> 12.

    `//` is integer division (it drops the remainder), like bash's `/`.
    """
    return driver_api_version // 1000


def sku_cuda_major(cuda_config) -> int | None:
    """The major of cuda.driver.version from versions.json ("12.9" -> 12).

    Mirrors  jq -r '.driver.version' | cut -d'.' -f1 . Returns None when the
    value is missing or its major isn't a number, where bash would carry on
    with "null" (see the module docstring).
    """
    driver = cuda_config.get("driver") if isinstance(cuda_config, dict) else None
    version = driver.get("version") if isinstance(driver, dict) else None
    if version is None:
        return None
    # str() handles a version written as a JSON number; split(".")[0] is cut -f1.
    major = str(version).split(".")[0]
    return int(major) if re.fullmatch(r"[0-9]+", major) else None


def dcgmi_version(output: str) -> str | None:
    """The DCGM version from `dcgmi --version` output, or None.

    Mirrors  DCGM_VERSION=$(dcgmi --version | awk '{print $3}')  followed by the
    unquoted ${DCGM_VERSION}: awk prints field 3 of every line ("" if a line is
    shorter), and write_component_version records only the first word.

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


def exact_version_deps(apt_cache_output: str, version: str) -> list[str]:
    """Dependencies declared as "(= <version>)" in `apt-cache show` output.

    Mirrors resolve_exact_version_deps' awk filter + `sort -u`:

        /^(Pre-)?Depends:/ {
            sub(/^[^:]*:[ \\t]*/, "")                    drop "Depends: "
            n = split($0, deps, /[,|]/)                 alternatives too
            for each d: trim blanks; if d ~ /^[^ ]+ \\(= [^)]+\\)$/:
                name = d before " (= ", minus any ":arch"; ver = inside
                if (ver == version) print name
        }

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

    Mirrors  sed -n 's/^ExecStart=//p' "${DCGM_UNIT_FILE}" | tail -n1 .
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
# Helpers with side effects.
# --------------------------------------------------------------------------- #
def _probe_cuda_driver_version(env: dict[str, str]) -> int | None:
    """Compile and run the C probe; return cuDriverGetVersion(), or None (logged)."""
    # mkdtemp() creates a new, uniquely named directory and returns its path.
    probe_dir = tempfile.mkdtemp(prefix="cuda_probe.", dir=_PROBE_PARENT_DIR)
    source = os.path.join(probe_dir, _PROBE_SOURCE_NAME)
    binary = os.path.join(probe_dir, _PROBE_BINARY_NAME)
    try:
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(_CUDA_PROBE_SOURCE)
        log_debug(_OP, f"Wrote CUDA probe source {source}", path=source)

        if exec_program(["gcc", source, "-ldl", "-o", binary], _OP, env=env) != 0:
            log_error(_OP, "could not compile the CUDA driver probe", path=source)
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
    finally:
        # Like bash's `trap 'rm -rf ...' EXIT`: always remove the directory.
        shutil.rmtree(probe_dir, ignore_errors=True)


def _run_all(commands: list[list[str]], env: dict[str, str]) -> bool:
    """Run commands in order, stopping at the first failure (like `set -e`)."""
    for command in commands:
        if exec_program(command, _OP, env=env) != 0:
            return False
    return True


def _command_path(name: str, env: dict[str, str]) -> str | None:
    """bash `command -v <name>`: its path on the build env's PATH, or None."""
    return shutil.which(name, path=env.get("PATH"))


def _config_dcgm_version(env: dict[str, str]) -> str | None:
    """The "dcgm" version from versions.json, or None (logged)."""
    cfg = config_for("dcgm", env)
    if not isinstance(cfg, dict) or not cfg.get("version"):
        log_error(_OP, "could not resolve dcgm version from versions.json",
                  distro=env.get("DISTRIBUTION", ""))
        return None
    return str(cfg["version"])


def _needs_sku_packages(cuda: int, sku_cuda: int) -> bool:
    """Log and return whether extra packages for the SKU's CUDA major are needed."""
    needed = sku_cuda < cuda
    if needed:
        log_info(_OP, f"SKU CUDA major {sku_cuda} is lower than the driver's "
                      f"{cuda}: also installing the cuda{sku_cuda} DCGM packages",
                 cuda_major=cuda, sku_cuda_major=sku_cuda)
    else:
        log_info(_OP, f"SKU CUDA major {sku_cuda} is not lower than the driver's "
                      f"{cuda}: no extra DCGM packages", cuda_major=cuda,
                 sku_cuda_major=sku_cuda)
    return needed


def _resolve_exact_version_deps(version: str, packages: list[str],
                                env: dict[str, str]) -> list[str] | None:
    """Port of resolve_exact_version_deps: "<pkg>=<version>" for `packages` and,
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
    """apt-get install -y the packages: pinned with their exact-version
    dependencies (dcgm version given), or unpinned (Ubuntu 26.04, dcgm None)."""
    if dcgm is not None:
        resolved = _resolve_exact_version_deps(dcgm, packages, env)
        if resolved is None:
            return False
        packages = resolved
    return _run_all([["apt-get", "install", "-y", *packages]], env)


def _install_azure_linux_marker(env) -> bool:
    """Build and install the hpcx-provides-openmpi marker rpm unless installed."""
    rc = exec_program(["rpm", "-q", _AZURE_LINUX_MARKER], _OP, env=env,
                      failure_is_error=False)
    if rc == 0:
        log_info(_OP, f"Marker rpm {_AZURE_LINUX_MARKER} is already installed",
                 package=_AZURE_LINUX_MARKER)
        return True

    # bash: rpm_topdir=$(mktemp -d)
    topdir = tempfile.mkdtemp(dir=_PROBE_PARENT_DIR)
    try:
        spec_file = os.path.join(topdir, f"{_AZURE_LINUX_MARKER}.spec")
        for sub in ("BUILD", "BUILDROOT", "RPMS", "SOURCES", "SPECS", "SRPMS"):
            os.makedirs(os.path.join(topdir, sub), exist_ok=True)
        with open(spec_file, "w", encoding="utf-8") as handle:
            handle.write(_AZURE_LINUX_MARKER_SPEC)
        log_info(_OP, f"Building the marker rpm {_AZURE_LINUX_MARKER} (Provides: openmpi, "
                      "so the multinode package doesn't pull Azure Linux's openmpi)",
                 package=_AZURE_LINUX_MARKER, spec=spec_file)
        if not _run_all([["rpmbuild", "--define", f"_topdir {topdir}", "-bb", spec_file]],
                        env):
            return False
        # bash: find "${rpm_topdir}/RPMS" -type f -name "<marker>-*.rpm" | head -n1
        built = sorted(path for path in glob.glob(
            os.path.join(topdir, "RPMS", "**", f"{_AZURE_LINUX_MARKER}-*.rpm"),
            recursive=True) if os.path.isfile(path))
        if not built:
            log_error(_OP, f"failed to build {_AZURE_LINUX_MARKER} marker RPM",
                      rpms_dir=os.path.join(topdir, "RPMS"))
            return False
        if not _run_all([["rpm", "-Uvh", "--replacepkgs", built[0]]], env):
            return False
        log_info(_OP, f"Installed the marker rpm {_AZURE_LINUX_MARKER}",
                 package=_AZURE_LINUX_MARKER, rpm=os.path.basename(built[0]))
        return True
    finally:
        shutil.rmtree(topdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Per-distro handlers. Each returns the DCGM version to record, or None on
# failure (already logged).
# --------------------------------------------------------------------------- #
def _install_ubuntu(env, cuda: int, sku_cuda: int | None) -> str | None:
    distribution = env.get("DISTRIBUTION", "")
    unpinned = distribution == _UNPINNED_UBUNTU
    dcgm = None
    if not unpinned:
        dcgm = _config_dcgm_version(env)
        if dcgm is None:
            return None
    if sku_cuda is None:
        log_error(_OP, "could not resolve a numeric cuda driver version for this "
                       "SKU from versions.json")
        return None
    if unpinned:
        log_info(_OP, f"{distribution}: installing the latest DCGM for CUDA {cuda} "
                      "from apt (unpinned)", distro=distribution, cuda_major=cuda,
                 pinned=False)
    else:
        log_info(_OP, f"Ubuntu: installing DCGM {dcgm} for CUDA {cuda} from apt, with "
                      "its exact-version dependencies", dcgm_version=dcgm,
                 version_source="versions.json", cuda_major=cuda, distro=distribution,
                 pinned=True)

    if not _apt_install_dcgm(env, ubuntu_packages(cuda), dcgm):
        return None
    if _needs_sku_packages(cuda, sku_cuda):
        if not _apt_install_dcgm(env, ubuntu_extra_packages(sku_cuda), dcgm):
            return None
    if not unpinned:
        return dcgm

    rc, output = run_capture(_DPKG_QUERY_DCGM, _OP, env=env)
    dcgm = output.rstrip("\n") if rc == 0 else ""
    if not dcgm:
        log_error(_OP, "could not read the installed datacenter-gpu-manager-4-core "
                       "version with dpkg-query", exit_code=rc)
        return None
    log_info(_OP, f"Installed DCGM version is {dcgm}", dcgm_version=dcgm,
             version_source="dpkg-query")
    return dcgm


def _install_azurelinux(env, cuda: int, sku_cuda: int | None) -> str | None:
    dcgm = _config_dcgm_version(env)
    if dcgm is None:
        return None
    if not _install_azure_linux_marker(env):
        return None
    # BASH BUG (mirrored on purpose): install_dcgm.sh picks cuda12 only when
    # its first argument is "V100", but install.sh never passes one, so it
    # always installs the cuda13 packages, whatever the SKU or probed CUDA major.
    packages = azure_linux_packages(_AZURE_LINUX_CUDA_MAJOR, dcgm)
    log_info(_OP, f"Azure Linux: installing DCGM {dcgm} (always the "
                  f"cuda{_AZURE_LINUX_CUDA_MAJOR} packages, like bash)",
             dcgm_version=dcgm, version_source="versions.json", cuda_major=cuda,
             sku_cuda_major=sku_cuda, packages=packages)
    # The bug only changes the result for a V100: bash meant it to get cuda12.
    # Warn so a V100 image build's log shows the packages it really received.
    sku = env.get("SKU", "")
    if sku == "V100":
        log_warn(_OP, "Azure Linux V100: installing the cuda13 DCGM packages "
                      "because install.sh passes no SKU argument to "
                      "install_dcgm.sh (kept for bash parity)",
                 sku=sku, packages=packages)
    return dcgm if _run_all([["dnf", "install", "-y", *packages]], env) else None


def _rhel_commands(cuda: int) -> list[list[str]]:
    """The cuda<N> and multinode-cuda<N> installs bash runs on the RHEL family."""
    return [[*_RHEL_DNF_INSTALL, f"datacenter-gpu-manager-4-cuda{cuda}"],
            [*_RHEL_DNF_INSTALL_MULTINODE, f"datacenter-gpu-manager-4-multinode-cuda{cuda}"]]


def _install_rhel(env, cuda: int, sku_cuda: int | None) -> str | None:
    if sku_cuda is None:
        log_error(_OP, "could not resolve a numeric cuda driver version for this "
                       "SKU from versions.json")
        return None
    log_info(_OP, f"RHEL family: installing the latest DCGM for CUDA {cuda} from dnf",
             distro=env.get("DISTRIBUTION", ""), cuda_major=cuda)

    commands = [["dnf", "clean", "expire-cache"], *_rhel_commands(cuda)]
    if _needs_sku_packages(cuda, sku_cuda):
        commands.extend(_rhel_commands(sku_cuda))
    if not _run_all(commands, env):
        return None

    # The RHEL family installs unpinned, so record what dcgmi reports.
    rc, output = run_capture(["dcgmi", "--version"], _OP, env=env)
    dcgm = dcgmi_version(output) if rc == 0 else None
    if dcgm is None:
        log_error(_OP, f"could not read the DCGM version from `dcgmi --version`: "
                       f"{output.strip()!r}", exit_code=rc)
        return None
    log_info(_OP, f"Installed DCGM version is {dcgm}", dcgm_version=dcgm,
             version_source="dcgmi")
    return dcgm


# --------------------------------------------------------------------------- #
# nv-hostengine under screen with HPC-X, then the service.
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
    distribution = env.get("DISTRIBUTION", "")
    if _command_path("screen", env) is None:
        command = (["apt-get", "install", "-y", "screen"] if "ubuntu" in distribution
                   else ["dnf", "install", "-y", "screen"])
        log_info(_OP, "screen not found: installing it", package="screen")
        if not _run_all([command], env):
            return False
    screen = _command_path("screen", env)
    if screen is None:
        log_error(_OP, "screen is still not on PATH after installing it; the drop-in "
                       "needs its path", path=env.get("PATH"))
        return False

    rc, output = run_capture(["systemctl", "show", "-p", "FragmentPath", "--value",
                              f"{_SERVICE}.service"], _OP, env=env)
    if rc != 0:
        log_error(_OP, f"could not find the {_SERVICE} unit file", exit_code=rc)
        return False
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
        return False
    log_info(_OP, f"nv-hostengine command from {unit_file}: {hostengine_cmd}",
             path=unit_file, exec_start=hostengine_cmd, screen=screen)

    os.makedirs(_DROPIN_DIR, exist_ok=True)
    dropin = os.path.join(_DROPIN_DIR, _DROPIN_NAME)
    existed = os.path.exists(dropin)
    text = hostengine_dropin(screen, hostengine_cmd)
    with open(dropin, "w", encoding="utf-8") as handle:
        handle.write(text)
    action = "replaced" if existed else "created"
    log_info(_OP, f"{action.capitalize()} {dropin} (nv-hostengine runs under screen "
                  "with mpi/hpcx loaded)", path=dropin, action=action)
    log_debug(_OP, f"{dropin}:\n{text}")
    return _run_all([["systemctl", "daemon-reload"]], env)


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def install(env: dict[str, str]) -> int:
    """Install DCGM, record its version and (re)start nvidia-dcgm. Returns 0 or 3."""
    distribution = env.get("DISTRIBUTION", "")
    log_info(_OP, "Installing DCGM", distro=distribution, sku=env.get("SKU", ""),
             node_type=env.get("TARGET_NODE_TYPE", ""))

    # ---- 1. CUDA major supported by the installed driver -------------------
    try:
        driver_api = _probe_cuda_driver_version(env)
    except OSError as exc:
        log_error(_OP, f"could not set up the CUDA driver probe: {exc}")
        return 3
    if driver_api is None:
        return 3
    cuda = cuda_major(driver_api)
    log_info(_OP, f"Probed CUDA driver API version {driver_api} (CUDA {cuda})",
             cuda_driver_api_version=driver_api, cuda_major=cuda)

    # ---- 2. CUDA major of this SKU's driver in versions.json ---------------
    sku_cuda = sku_cuda_major(config_for("cuda", env))
    log_info(_OP, f"SKU CUDA major from versions.json: {sku_cuda}",
             sku_cuda_major=sku_cuda, cuda_major=cuda)

    try:
        # ---- 3. packages, per distro ----------------------------------------
        if "ubuntu" in distribution:
            handler = _install_ubuntu
        elif "azurelinux" in distribution:
            handler = _install_azurelinux
        else:  # RHEL family: AlmaLinux, Rocky Linux, RHEL, etc.
            handler = _install_rhel
        dcgm = handler(env, cuda, sku_cuda)
        if dcgm is None:
            log_error(_OP, "DCGM installation failed", distro=distribution)
            return 3

        # ---- 4. record the version ------------------------------------------
        write_component_version("DCGM", dcgm)
        log_info(_OP, f"Recorded DCGM version {dcgm}", component="DCGM", version=dcgm)

        # ---- 5. nv-hostengine under screen ----------------------------------
        if not _configure_hostengine(env):
            log_error(_OP, "could not set up nv-hostengine under screen with HPC-X",
                      service=_SERVICE)
            return 3
    except OSError as exc:
        # The marker rpm files, the drop-in and the versions file.
        log_error(_OP, f"file operation failed: {exc}",
                  path=getattr(exc, "filename", None))
        return 3

    # ---- 6. service: restart so a running instance picks up the drop-in ------
    log_info(_OP, f"Enabling and restarting the {_SERVICE} service", service=_SERVICE)
    if not _run_all([["systemctl", "--now", "enable", _SERVICE],
                     ["systemctl", "restart", _SERVICE]], env):
        log_error(_OP, f"could not enable/restart the {_SERVICE} service",
                  service=_SERVICE)
        return 3
    if exec_program(["systemctl", "is-active", "--quiet", _SERVICE], _OP, env=env) != 0:
        log_error(_OP, "DCGM is inactive!", service=_SERVICE)
        return 3
    log_info(_OP, f"{_SERVICE} service is active", service=_SERVICE)

    log_info(_OP, "DCGM installed", distro=distribution, dcgm_version=dcgm,
             cuda_major=cuda, sku_cuda_major=sku_cuda)
    return 0
