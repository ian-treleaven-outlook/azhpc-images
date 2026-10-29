"""install_dynolog_drl.py — Python port of components/install_dynolog_drl.sh.

What this step does
-------------------
dynolog is a telemetry daemon that collects CPU and (through NVIDIA DCGM) GPU
performance metrics. dyno-relay-logger forwards what dynolog collects. Both are
built from source, at pinned git commits from versions.json ("dynolog" and
"dyno_relay_logger"), on NVIDIA images only:

  1. Install the build dependencies for the distro (cmake, Rust, ninja, ...).
  2. Build dynolog and move its `dynolog` and `dyno` binaries to
     /opt/dynolog/bin.
  3. Find the DCGM library with `ldconfig -p` and write
     /etc/systemd/system/dynolog.service, which points dynolog at it.
  4. Build dyno-relay-logger, move `dynorelaylogger` and
     `dynorelayloggerinfo` to /opt/dynolog/bin, and write
     /etc/systemd/system/dyno-relay-logger.service.
  5. Record both versions.

Neither service is enabled: dynolog holds an exclusive GPU profiling context
(the DCGM_FI_PROF_* fields), which clashes with dcgm-exporter, since only one
profiling client per GPU can run at a time. The services are installed so they
can be switched on later.

The original bash, summarised (see components/install_dynolog_drl.sh):

    if [[ "$GPU" == "NVIDIA" ]]; then
        ORIGINAL_PATH="$PATH"                 # (never used)
        mkdir -p /opt/dynolog/bin
        <read version/commit/url for dynolog and dyno_relay_logger>

        if [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
            dnf install -y cmake rust cargo ninja-build build-essential
        elif [[ $DISTRIBUTION == *"ubuntu"* ]]; then
            apt-get install -y cmake rustc-1.82 cargo-1.82 ninja-build build-essential
            apt-get install -y g++ pkg-config uuid-dev libssl-dev
            export PATH="/usr/lib/rust-1.82/bin:$PATH"
        elif [[ $DISTRIBUTION == almalinux* ]] || [[ $DISTRIBUTION == rocky* ]]; then
            dnf install -y cmake rust cargo ninja-build libuuid-devel gcc-toolset-12
            dnf install -y openssl3-devel   (almalinux8.10 / rocky8.10)
                       or openssl-devel     (otherwise)
            source /opt/rh/gcc-toolset-12/enable
        fi

        git clone --no-checkout ${DYNOLOG_URL} /tmp/dynolog
        pushd /tmp/dynolog
        git checkout ${DYNOLOG_COMMIT}; git submodule update --init --recursive
        ./scripts/build.sh -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        mv build/dynolog/src/dynolog build/release/dyno -> /opt/dynolog/bin
        popd; rm -rf /tmp/dynolog

        ldconfig
        DCGM_LIB=$(ldconfig -p | awk '/libdcgm\\.so(\\.[0-9]+)* / {print $NF; exit}')
        [[ -z "$DCGM_LIB" ]] && { echo "FATAL: libdcgm.so not found"; exit 1; }
        cat <<-EOF > /etc/systemd/system/dynolog.service ... EOF
        systemctl daemon-reload
        # systemctl enable dynolog.service

        git clone --no-checkout ${DRL_URL} /tmp/dyno-relay-logger
        pushd ...; git checkout ${DRL_COMMIT}; git submodule update --init --recursive
        mkdir build && cd build
        <cmake .. per distro: almalinux8.10/rocky8.10 with OPENSSL_* exports and
         -D options; Ubuntu with OPENSSL_LIB_DIR=/usr/lib/<multiarch>; else plain>
        cmake --build . -j$(nproc)
        mv dynorelaylogger dynorelayloggerinfo -> /opt/dynolog/bin
        popd; rm -rf /tmp/dyno-relay-logger
        cat <<-EOF > /etc/systemd/system/dyno-relay-logger.service ... EOF
        systemctl daemon-reload
        # systemctl enable dyno-relay-logger.service

        write_component_version "dynolog" ${DYNOLOG_VERSION}
        write_component_version "dyno_relay_logger" ${DRL_VERSION}
    fi

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-dynolog-drl" Step. The build then calls install(env) instead of
running the bash script. Deleting `action=` runs the bash script again, which
is why the .sh file is kept.

In bash, `export PATH=...` (Ubuntu) and `source /opt/rh/gcc-toolset-12/enable`
(Alma/Rocky) change the environment of every later command, so the builds use
Rust 1.82 or GCC 12. Python can't keep a `source` between separate commands,
so each build (dynolog's build.sh; dyno-relay-logger's cmake configure + build)
runs as one `bash -c` script with `set -e` that first applies the same
environment change (see build_environment()). The git commands, file moves,
ldconfig and systemctl don't depend on that environment and run on their own
with exec_program. The service files are written with plain Python.

install() returns 0 for success or 3 for failure. It logs problems and returns
3; it does not let exceptions escape to the build loop.

Deliberate differences from bash
--------------------------------
  * Both configs are checked (version, commit and url) before anything is
    changed. Bash would create /opt/dynolog/bin first and then try to clone
    "null".
  * `ldconfig -p` failing fails the step (bash only looks at awk's result).
  * `mkdir build && cd build`: if build/ already existed, bash's `set -e`
    would not stop on the failed mkdir (it is not the last command of the &&
    list) and cmake would run in the wrong directory. We fail instead.
  * A failed `rm -rf` of the clone directories doesn't fail the step.

Kept as in bash: other distros get no build dependencies but are still built
(logged as a warning), and a leftover /tmp/dynolog or /tmp/dyno-relay-logger
from an earlier failed run makes `git clone` fail (also logged as a warning
before the clone).
"""

# Lets type hints such as `str | None` work on Python versions older than 3.10.
from __future__ import annotations

import os
import re
import shutil

from utils.component_config import config_for, write_component_version
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.process import exec_program, run_capture

# The Step's `op` label in build_config.py. Every log line from this module uses it.
_OP = "install-dynolog-drl"

_INSTALL_DIR = "/opt/dynolog/bin"
_DYNOLOG_SRC = "/tmp/dynolog"
_DRL_SRC = "/tmp/dyno-relay-logger"
_SYSTEMD_DIR = "/etc/systemd/system"

_AZURE_LINUX = "azurelinux3.0"
# The two distros that need OpenSSL 3 from the separate openssl3-devel package.
_EL8_OPENSSL3 = ("almalinux8.10", "rocky8.10")

# Built files moved to _INSTALL_DIR, as paths relative to each source dir.
_DYNOLOG_BINARIES = ("build/dynolog/src/dynolog", "build/release/dyno")
_DRL_BINARIES = ("build/dynorelaylogger", "build/dynorelayloggerinfo")

# Why the services are left disabled (bash's comment), for the log.
_DISABLED_REASON = ("dynolog holds an exclusive GPU profiling context "
                    "(DCGM_FI_PROF_* fields), which clashes with dcgm-exporter")

# The CMake options every cmake call uses.
_CMAKE_BASE = "cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release"

# awk's /libdcgm\.so(\.[0-9]+)* / as a Python regex (same syntax here). Like
# awk's, it matches anywhere in the line; the trailing space stops it matching
# e.g. "libdcgm.so.4x".
_DCGM_PATTERN = re.compile(r"libdcgm\.so(\.[0-9]+)* ")

# The unit file bash writes with `cat <<-EOF` (no leading tabs in the script,
# so nothing is stripped). Only ${DCGM_LIB} is expanded; {dcgm_lib} below
# takes its place. Compared byte for byte with bash's output.
_DYNOLOG_SERVICE = """\
[Unit]
Description=dynolog
After=nvidia-dcgm.service

[Service]
Environment="GLOG_logtostderr=1" "GLOG_minloglevel=2"
ExecStart=/opt/dynolog/bin/dynolog -enable_ipc_monitor=true -enable_gpu_monitor=true \
-kernel_monitor_reporting_interval_s=10 -dcgm_lib_path={dcgm_lib} -dcgm_reporting_interval_s=10 \
-use_udsrelay=true -dcgm_fields="100,155,204,1001,1005,1008,1009,1010,1011,1012,1013,1014"
Restart=always
RestartSec=60s
User=root
Group=root

[Install]
WantedBy=multi-user.target
"""

_DRL_SERVICE = """\
[Unit]
Description=dyno-relay-logger
After=nvidia-dcgm.service

[Service]
Environment="GLOG_logtostderr=1" "GLOG_minloglevel=2"
ExecStart=/opt/dynolog/bin/dynorelaylogger --forward=aehubs
Restart=always
RestartSec=60s
User=root
Group=root

[Install]
WantedBy=multi-user.target
"""


# --------------------------------------------------------------------------- #
# Pure helpers: no side effects, so tests call them directly.
# --------------------------------------------------------------------------- #
def dependency_branch(distribution: str) -> str | None:
    """Which of bash's dependency branches a distro takes, or None for none."""
    if distribution == _AZURE_LINUX:
        return "azurelinux"
    if "ubuntu" in distribution:
        return "ubuntu"
    if distribution.startswith(("almalinux", "rocky")):
        return "almalinux-rocky"
    return None


def dependency_commands(distribution: str) -> list[list[str]]:
    """The dependency install commands bash runs for a distro (maybe none)."""
    branch = dependency_branch(distribution)
    if branch == "azurelinux":
        return [["dnf", "install", "-y", "cmake", "rust", "cargo", "ninja-build",
                 "build-essential"]]
    if branch == "ubuntu":
        return [["apt-get", "install", "-y", "cmake", "rustc-1.82", "cargo-1.82",
                 "ninja-build", "build-essential"],
                ["apt-get", "install", "-y", "g++", "pkg-config", "uuid-dev", "libssl-dev"]]
    if branch == "almalinux-rocky":
        openssl = "openssl3-devel" if distribution in _EL8_OPENSSL3 else "openssl-devel"
        return [["dnf", "install", "-y", "cmake", "rust", "cargo", "ninja-build",
                 "libuuid-devel", "gcc-toolset-12"],
                ["dnf", "install", "-y", openssl]]
    return []


def build_environment(distribution: str) -> str:
    """The shell lines bash runs after the dependencies, which change the
    environment of every later command. Put at the top of each build script.
    """
    # Bash also saves ORIGINAL_PATH="$PATH" at the start, apparently to restore
    # PATH afterwards, but never uses it again. Nothing needs restoring here:
    # these lines only affect the one `bash -c` build script they start, and
    # the build's own PATH is never changed. So that variable is not ported.
    branch = dependency_branch(distribution)
    if branch == "ubuntu":
        # Use Rust 1.82 from the rustc-1.82 package.
        return 'export PATH="/usr/lib/rust-1.82/bin:$PATH"\n'
    if branch == "almalinux-rocky":
        # Use GCC 12 from gcc-toolset-12.
        return "source /opt/rh/gcc-toolset-12/enable\n"
    return ""


def dynolog_build_script(distribution: str) -> str:
    """The bash script that builds dynolog (run in /tmp/dynolog)."""
    return ("set -e\n"
            + build_environment(distribution)
            + "./scripts/build.sh -DCMAKE_POLICY_VERSION_MINIMUM=3.5\n")


def drl_build_script(distribution: str) -> str:
    """The bash script that configures and builds dyno-relay-logger.

    Run in /tmp/dyno-relay-logger/build. The exports only affect this script,
    as in bash they only matter for the cmake calls that follow them.
    """
    script = "set -e\n" + build_environment(distribution)
    if distribution in _EL8_OPENSSL3:
        # Bash's workaround: cmake can't find OpenSSL 3 on Alma/Rocky 8.10
        # without these.
        script += ("export OPENSSL_DIR=/usr\n"
                   "export OPENSSL_INCLUDE_DIR=/usr/include/openssl3\n"
                   "export OPENSSL_LIB_DIR=/usr/lib64/openssl3\n"
                   f"{_CMAKE_BASE}"
                   " -DOPENSSL_ROOT_DIR=/usr/include/openssl3"
                   " -DOPENSSL_INCLUDE_DIR=/usr/include/openssl3"
                   " -DOPENSSL_CRYPTO_LIBRARY=/usr/lib64/openssl3/libcrypto.so"
                   " -DOPENSSL_SSL_LIBRARY=/usr/lib64/openssl3/libssl.so\n")
    elif "ubuntu" in distribution:
        # On Ubuntu the OpenSSL libraries are in the multiarch path
        # (e.g. /usr/lib/x86_64-linux-gnu).
        script += ("export OPENSSL_LIB_DIR=/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)\n"
                   f"{_CMAKE_BASE}\n")
    else:
        script += f"{_CMAKE_BASE}\n"
    return script + "cmake --build . -j$(nproc)\n"


def find_dcgm_lib(ldconfig_output: str) -> str | None:
    """The DCGM library path from `ldconfig -p` output, or None.

    Mirrors awk '/libdcgm\\.so(\\.[0-9]+)* / {print $NF; exit}': the last
    field of the first matching line. A typical line:

        libdcgm.so.4 (libc6,x86-64) => /lib/x86_64-linux-gnu/libdcgm.so.4
    """
    for line in ldconfig_output.split("\n"):
        if _DCGM_PATTERN.search(line):
            # split() splits on runs of blanks like awk; [-1] is awk's $NF.
            return line.split()[-1]
    return None


def dynolog_service(dcgm_lib: str) -> str:
    """The text of dynolog.service for this DCGM library."""
    return _DYNOLOG_SERVICE.format(dcgm_lib=dcgm_lib)


# --------------------------------------------------------------------------- #
# Steps. Each returns True on success, or False after logging why not.
# --------------------------------------------------------------------------- #
def _run_all(commands, env: dict[str, str], cwd: str | None = None,
             *, what: str) -> bool:
    """Run commands in order, stopping at the first failure (like `set -e`).

    exec_program already logs a failed command with its output; we add the
    consequence (`what` names the phase that failed).
    """
    for command in commands:
        rc = exec_program(command, _OP, cwd=cwd, env=env)
        if rc != 0:
            log_error(_OP, f"{what} failed: `{' '.join(command)}` exited with {rc}",
                      command=command[0], exit_code=rc, phase=what)
            return False
    return True


def _resolve(component: str, env: dict[str, str]) -> dict | None:
    config = config_for(component, env)
    # isinstance() guards against a missing (None) or oddly shaped config.
    if (not isinstance(config, dict)
            or not all(config.get(field) for field in ("version", "commit", "url"))):
        log_error(_OP, f"could not resolve {component} version, commit and url "
                       "from versions.json", component=component)
        return None
    return config


def _checkout(name: str, config: dict, source_dir: str, env: dict[str, str]) -> bool:
    """git clone --no-checkout, checkout the pinned commit, update submodules."""
    if os.path.exists(source_dir):
        log_warn(_OP, f"{source_dir} is left over from an earlier run; git clone "
                      "will fail, as it does in bash", path=source_dir)
    log_info(_OP, f"Fetching {name} source at commit {config['commit']}",
             component=name, url=config["url"], commit=config["commit"], path=source_dir)
    return (_run_all([["git", "clone", "--no-checkout", config["url"], source_dir]],
                     env, what=f"{name} clone")
            and _run_all([["git", "checkout", config["commit"]],
                          ["git", "submodule", "update", "--init", "--recursive"]],
                         env, cwd=source_dir, what=f"{name} checkout"))


def _install_binaries(source_dir: str, binaries) -> bool:
    """Move built files into _INSTALL_DIR, as `mv <file> /opt/dynolog/bin` does."""
    for relative in binaries:
        built = os.path.join(source_dir, relative)
        target = os.path.join(_INSTALL_DIR, os.path.basename(relative))
        try:
            # shutil.move to the full target path replaces an existing file, as mv does.
            shutil.move(built, target)
        except OSError as exc:
            log_error(_OP, f"could not move {built} to {target}: {exc}",
                      source=built, path=target)
            return False
        log_info(_OP, f"Installed {target}", path=target, action="moved")
    return True


def _write_service(name: str, text: str, env: dict[str, str]) -> bool:
    """Write a unit file (bash's `cat <<-EOF > file`), then daemon-reload."""
    path = os.path.join(_SYSTEMD_DIR, name)
    action = "replaced" if os.path.exists(path) else "created"
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    except OSError as exc:
        log_error(_OP, f"could not write {path}: {exc}", path=path)
        return False
    log_info(_OP, f"Wrote {path} ({action})", path=path, action=action)
    log_debug(_OP, f"{path} contents:\n{text}")

    if not _run_all([["systemctl", "daemon-reload"]], env, what="systemd reload"):
        return False
    # Bash keeps `systemctl enable` commented out; see the module docstring.
    log_info(_OP, f"{name} is installed but left disabled: {_DISABLED_REASON}",
             service=name, enabled=False)
    return True


def _build_dynolog(config: dict, distribution: str, env: dict[str, str]) -> bool:
    if not _checkout("dynolog", config, _DYNOLOG_SRC, env):
        return False
    log_info(_OP, "Building dynolog (scripts/build.sh)", component="dynolog",
             build_env=build_environment(distribution).strip())
    if not _run_all([["bash", "-c", dynolog_build_script(distribution)]],
                    env, cwd=_DYNOLOG_SRC, what="dynolog build"):
        return False
    if not _install_binaries(_DYNOLOG_SRC, _DYNOLOG_BINARIES):
        return False
    # Bash's `rm -rf /tmp/dynolog`; a leftover only matters for a rerun.
    shutil.rmtree(_DYNOLOG_SRC, ignore_errors=True)
    return True


def _find_dcgm(env: dict[str, str]) -> str | None:
    """Refresh the linker cache and find libdcgm.so (bash's DCGM_LIB)."""
    if not _run_all([["ldconfig"]], env, what="linker cache refresh"):
        return None
    rc, output = run_capture(["ldconfig", "-p"], _OP, env=env)
    if rc != 0:
        log_error(_OP, "could not list the linker cache to find libdcgm.so",
                  exit_code=rc)
        return None
    dcgm_lib = find_dcgm_lib(output)
    if dcgm_lib is None:
        # Bash: "FATAL: libdcgm.so not found".
        log_error(_OP, "libdcgm.so not found in the linker cache (is DCGM installed?)")
        return None
    log_info(_OP, f"Found the DCGM library: {dcgm_lib}", dcgm_lib=dcgm_lib)
    return dcgm_lib


def _build_drl(config: dict, distribution: str, env: dict[str, str]) -> bool:
    if not _checkout("dyno-relay-logger", config, _DRL_SRC, env):
        return False
    build_dir = os.path.join(_DRL_SRC, "build")
    try:
        # os.mkdir (not makedirs) fails if it exists, like `mkdir` without -p.
        os.mkdir(build_dir)
    except OSError as exc:
        log_error(_OP, f"could not create {build_dir}: {exc}", path=build_dir)
        return False

    if distribution in _EL8_OPENSSL3:
        variant = "openssl3 workaround"
    elif "ubuntu" in distribution:
        variant = "ubuntu multiarch OpenSSL"
    else:
        variant = "default"
    log_info(_OP, f"Building dyno-relay-logger (cmake, {variant})",
             component="dyno-relay-logger", cmake_variant=variant,
             build_env=build_environment(distribution).strip())
    if not _run_all([["bash", "-c", drl_build_script(distribution)]],
                    env, cwd=build_dir, what="dyno-relay-logger build"):
        return False
    if not _install_binaries(_DRL_SRC, _DRL_BINARIES):
        return False
    shutil.rmtree(_DRL_SRC, ignore_errors=True)
    return True


def _record(component: str, version: str) -> bool:
    try:
        write_component_version(component, version)
    except OSError as exc:
        log_error(_OP, f"could not record {component}={version}: {exc}",
                  component=component, version=version)
        return False
    log_info(_OP, f"Recorded {component}={version}", component=component, version=version)
    return True


def install(env: dict[str, str]) -> int:
    """Build and install dynolog and dyno-relay-logger (NVIDIA only).

    `env` is the build environment made by component_env() in build_config.py.
    Uses GPU, DISTRIBUTION and the values config_for() reads.

    Returns 0 on success (including the non-NVIDIA skip) and 3 on failure.
    """
    gpu = env.get("GPU", "")
    if gpu != "NVIDIA":
        log_info(_OP, "Skipping dynolog and dyno-relay-logger: they are only built "
                      "for NVIDIA GPUs (same as bash)", gpu=gpu, skipped=True)
        return 0

    # ---- 1. Checks before any changes are made ----------------------------
    dynolog = _resolve("dynolog", env)
    drl = _resolve("dyno_relay_logger", env)
    if dynolog is None or drl is None:
        return 3

    distribution = env.get("DISTRIBUTION", "")
    log_info(_OP, "Installing dynolog and dyno-relay-logger", distro=distribution,
             dynolog_version=dynolog["version"], dynolog_commit=dynolog["commit"],
             dynolog_url=dynolog["url"], drl_version=drl["version"],
             drl_commit=drl["commit"], drl_url=drl["url"], install_dir=_INSTALL_DIR)

    try:
        os.makedirs(_INSTALL_DIR, exist_ok=True)
    except OSError as exc:
        log_error(_OP, f"could not create {_INSTALL_DIR}: {exc}", path=_INSTALL_DIR)
        return 3

    # ---- 2. Build dependencies --------------------------------------------
    branch = dependency_branch(distribution)
    if branch is None:
        log_warn(_OP, f"No build dependencies are defined for {distribution!r}; "
                      "building anyway (same as bash)", distro=distribution)
    else:
        log_info(_OP, f"Installing build dependencies ({branch})", distro=distribution,
                 branch=branch, build_env=build_environment(distribution).strip())
        if not _run_all(dependency_commands(distribution), env,
                        what="build dependency install"):
            return 3

    # ---- 3. dynolog ----------------------------------------------------------
    if not _build_dynolog(dynolog, distribution, env):
        return 3
    dcgm_lib = _find_dcgm(env)
    if dcgm_lib is None:
        return 3
    if not _write_service("dynolog.service", dynolog_service(dcgm_lib), env):
        return 3

    # ---- 4. dyno-relay-logger -----------------------------------------------
    if not _build_drl(drl, distribution, env):
        return 3
    if not _write_service("dyno-relay-logger.service", _DRL_SERVICE, env):
        return 3

    # ---- 5. Record versions ----------------------------------------------------
    if not _record("dynolog", dynolog["version"]):
        return 3
    if not _record("dyno_relay_logger", drl["version"]):
        return 3

    log_info(_OP, f"dynolog {dynolog['version']} and dyno-relay-logger "
                  f"{drl['version']} installed to {_INSTALL_DIR} (services disabled)",
             dynolog_version=dynolog["version"], drl_version=drl["version"],
             dcgm_lib=dcgm_lib, install_dir=_INSTALL_DIR, distro=distribution)
    return 0
