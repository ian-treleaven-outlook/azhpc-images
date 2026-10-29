"""install_docker.py — Python port of components/install_docker.sh.

What this step does
-------------------
Installs Docker, using Microsoft's build of it ("moby"), and starts the docker
service. On NVIDIA images it also installs the NVIDIA Container Toolkit, so
containers can use the GPUs, and checks that containerd came back up. Finally
it records two versions in the component-versions file:

  * DOCKER       from `docker --version`        e.g. "29.1.4"
  * MOBY_ENGINE  the installed moby-engine package version
                 e.g. "29.1.4-ubuntu24.04u1" (Ubuntu)

The packages differ per distro family:

  * Ubuntu        apt-get install moby-engine, moby-cli, moby-buildx.
                  On baremetal_3p nodes (TARGET_NODE_TYPE), instead install
                  moby-engine=<V> and moby-cli=<V>, pinned to the "moby"
                  version in versions.json, and no buildx.
  * Azure Linux   dnf install moby-engine, moby-cli, docker-buildx.
  * RHEL family   (AlmaLinux, Rocky, RHEL) dnf install moby-engine, moby-cli,
                  moby-buildx.

The original bash, for comparison (comments trimmed):

    INSTALL_NVIDIA_CONTAINER_TOOLKIT=false
    if [[ "${GPU:-NVIDIA}" == "NVIDIA" ]]; then
        INSTALL_NVIDIA_CONTAINER_TOOLKIT=true
    fi

    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        if [[ "${TARGET_NODE_TYPE:-azure_vm_regular}" == "baremetal_3p" ]]; then
            moby_metadata=$(get_component_config "moby")
            MOBY_VERSION=$(jq -r '.version' <<< $moby_metadata)
            apt-get install -y moby-engine=${MOBY_VERSION}
            apt-get install -y moby-cli=${MOBY_VERSION}
        else
            apt-get install -y moby-engine
            apt-get install -y moby-cli
            apt-get install -y moby-buildx
        fi
    elif [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
        dnf install -y moby-engine
        dnf install -y moby-cli
        dnf install -y docker-buildx
    else
        dnf install -y moby-engine
        dnf install -y moby-cli
        dnf install -y moby-buildx
    fi

    if [[ "$INSTALL_NVIDIA_CONTAINER_TOOLKIT" == true ]]; then
        $COMPONENT_DIR/install_nvidia_container_toolkit.sh
    fi

    systemctl enable docker
    systemctl restart docker

    if [[ "$INSTALL_NVIDIA_CONTAINER_TOOLKIT" == true ]]; then
        systemctl restart containerd
        for i in $(seq 1 30); do
            if [ -S /run/containerd/containerd.sock ]; then
                break
            fi
            echo "Waiting for containerd socket... ($i/30)"
            sleep 1
        done
        ctr plugin ls
    fi

    docker_version=$(docker --version | awk -F' ' '{print $3}')
    write_component_version "DOCKER" ${docker_version::-1}

    if [[ $DISTRIBUTION == ubuntu* ]]; then
        moby_version=$(apt list --installed | grep moby-engine | awk -F' ' '{print $2}')
    elif [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
        moby_version=$(rpm -qa | grep moby | cut -d'-' -f3,4)
    else
        moby_version=$(dnf list installed | grep moby-engine | awk -F' ' '{print $2}')
    fi
    write_component_version "MOBY_ENGINE" ${moby_version}

Note the two different Ubuntu tests, mirrored here: the install uses
*"ubuntu"* ("contains"), the version lookup uses ubuntu* ("starts with").

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-docker" Step. Deleting `action=` from that Step makes the build run the
bash script again, which is why the .sh file is kept.

install() receives the build environment (a dict) and returns 0 for success or
3 for failure. It logs problems and never lets an exception escape.

The NVIDIA Container Toolkit is a separate component with its own bash script
(components/install_nvidia_container_toolkit.sh). It is still run as bash, with
the same environment, exactly as install_docker.sh runs it.

Like install.sh, build_plan runs this step for every vendor. On AMD images it
installs Docker without the NVIDIA container toolkit, exactly as bash does.

Deliberate differences from bash
--------------------------------
  * An empty or unparseable DOCKER or MOBY_ENGINE version fails (returns 3)
    instead of being recorded. Bash records "" for an empty moby version (and
    stops with a "substring expression" error for an empty docker version).
  * A failing `docker --version`, `apt list`, `rpm -qa` or `dnf list` fails the
    step. Bash, without `set -o pipefail`, ignores the exit code of the first
    command in a pipeline and parses whatever it printed.
  * A missing "moby" version in versions.json on baremetal_3p fails up front.
    Bash would run `apt-get install -y moby-engine=null`, which fails anyway.
  * The "Waiting for containerd socket" lines are logged at DEBUG, and a socket
    that never appears is logged as a warning. Like bash, it does not fail the
    step by itself (the following `ctr plugin ls` usually will).

Bash's unquoted ${moby_version} is split into words, and write_component_version
records only its first argument. When several lines match (e.g. on Azure Linux
`rpm -qa | grep moby` lists moby-engine, moby-cli, moby-containerd, ...), only
the first word is recorded. The parse helpers below reproduce that; they were
compared against real grep/awk/cut in bash. Bash would also glob-expand such a
word; versions never contain *, ? or [, so that is not reproduced.
"""

from __future__ import annotations

import os
import re
import stat
import time

from utils.component_config import DEFAULT_NODE_TYPE, config_for, write_component_version
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.process import exec_program, run_capture

# The Step's `op` label in build_config.py; used on every log line.
_OP = "install-docker"

# The only distro handled by the Azure Linux branch (an exact match, as in bash).
_AZURE_LINUX = "azurelinux3.0"

# The node type whose Ubuntu moby packages are pinned to a versions.json version.
_BAREMETAL_3P = "baremetal_3p"

# The socket containerd creates once it is ready, and how long bash waits for
# it: up to 30 checks, sleeping 1 second after each failed check.
# Module-level so the tests can point it at a temporary directory.
_CONTAINERD_SOCKET = "/run/containerd/containerd.sock"
_SOCKET_WAIT_TRIES = 30
_SOCKET_WAIT_SECONDS = 1

# The nested component script that installs the NVIDIA Container Toolkit.
_TOOLKIT_SCRIPT = "install_nvidia_container_toolkit.sh"

# Package install commands per distro family, in bash's order. Each package is
# installed by its own command, as in bash (so one failure names one package).
_UBUNTU_PACKAGES = ["moby-engine", "moby-cli", "moby-buildx"]
_UBUNTU_PINNED_PACKAGES = ["moby-engine", "moby-cli"]          # baremetal_3p
_AZURE_LINUX_PACKAGES = ["moby-engine", "moby-cli", "docker-buildx"]
_RHEL_PACKAGES = ["moby-engine", "moby-cli", "moby-buildx"]

# What bash's default word splitting (and awk's default field splitting) treats
# as separators: spaces, tabs and newlines. Python's str.split() would also
# split on a few other characters (\r, \f, \v), so we use an explicit pattern.
_BLANKS = re.compile(r"[ \t\n]+")


# --------------------------------------------------------------------------- #
# Pure helpers: they only transform strings, so tests call them directly.
# --------------------------------------------------------------------------- #
def _lines(output: str) -> list[str]:
    """Split command output into lines the way grep/awk/cut see them.

    split("\\n") (not splitlines()) because these tools only break on "\\n".
    A final newline ends the last line; it does not start an empty extra one.
    """
    if not output:
        return []
    lines = output.split("\n")
    if output.endswith("\n"):
        lines.pop()          # drop the "" after the last newline
    return lines


def _awk_field(line: str, number: int) -> str:
    """awk's $<number> with the default field separator ("" if missing).

    awk -F' ' is the same as awk's default: split on runs of blanks, ignoring
    leading and trailing ones.
    """
    # A list comprehension filters out the "" pieces that leading/trailing
    # blanks would leave behind.
    fields = [field for field in _BLANKS.split(line) if field]
    return fields[number - 1] if len(fields) >= number else ""


def first_word(text: str) -> str | None:
    """The first word of an unquoted bash expansion, or None if there is none.

    `write_component_version "X" ${value}` splits ${value} into words, and the
    function records only the first one ($2). Empty words disappear entirely.
    """
    words = [word for word in _BLANKS.split(text) if word]
    return words[0] if words else None


def docker_version(output: str) -> str | None:
    """The DOCKER version from `docker --version` output.

    Mirrors:
        docker_version=$(docker --version | awk -F' ' '{print $3}')
        write_component_version "DOCKER" ${docker_version::-1}

    e.g. "Docker version 29.1.4, build 0e5a6ab"  ->  $3 "29.1.4,"  ->  "29.1.4"

    ${v::-1} drops the last character of the whole captured text (normally the
    comma after the version). Returns None if nothing is left, where bash would
    fail ("substring expression < 0") or record an empty value.
    """
    # awk prints one line per input line; $(...) then removes trailing newlines.
    captured = "\n".join(_awk_field(line, 3) for line in _lines(output)).rstrip("\n")
    if not captured:
        return None
    # A slice like captured[:-1] means "all but the last character", like ${v::-1}.
    return first_word(captured[:-1])


def grep_awk_version(output: str, pattern: str) -> str | None:
    """Mirrors `... | grep <pattern> | awk -F' ' '{print $2}'` + first_word().

    Used for the moby-engine version on Ubuntu (`apt list --installed`) and the
    RHEL family (`dnf list installed`), e.g.
        "moby-engine/now 29.1.4-ubuntu24.04u1 amd64 [installed,local]"
            ->  "29.1.4-ubuntu24.04u1"
        "moby-engine.x86_64   28.5.1-1.el9   @packages-microsoft-com-prod"
            ->  "28.5.1-1.el9"

    `pattern` is a fixed string here ("moby-engine" has no regex characters
    that matter), so a plain substring test matches grep.
    """
    values = [_awk_field(line, 2) for line in _lines(output) if pattern in line]
    return first_word("\n".join(values))


def rpm_moby_version(output: str) -> str | None:
    """Mirrors `rpm -qa | grep moby | cut -d'-' -f3,4` + first_word().

    e.g. "moby-engine-25.0.3-11.azl3.x86_64"  ->  "25.0.3-11.azl3.x86_64"

    cut keeps fields 3 and 4 (joined by "-"), prints an empty line when a line
    has fewer fields, and prints a line with no "-" at all unchanged.

    Bash quirk, mirrored: every installed package whose name contains "moby"
    matches (moby-engine, moby-cli, moby-containerd, moby-runc, ...), and only
    the first in rpm's listing order is recorded, which need not be moby-engine.
    """
    values = []
    for line in _lines(output):
        if "moby" not in line:
            continue
        if "-" not in line:
            values.append(line)
            continue
        fields = line.split("-")
        # fields[2:4] is the 3rd and 4th fields (Python counts from 0); a
        # slice past the end just returns fewer items, never an error.
        values.append("-".join(fields[2:4]))
    return first_word("\n".join(values))


def moby_version_command(distribution: str) -> list[str]:
    """The command whose output holds the installed moby-engine version."""
    # startswith() mirrors bash's ubuntu* pattern (the install uses "contains").
    if distribution.startswith("ubuntu"):
        return ["apt", "list", "--installed"]
    if distribution == _AZURE_LINUX:
        return ["rpm", "-qa"]
    return ["dnf", "list", "installed"]


def parse_moby_version(distribution: str, output: str) -> str | None:
    """Parse the output of moby_version_command(distribution)."""
    if distribution == _AZURE_LINUX:
        return rpm_moby_version(output)
    # Ubuntu and the RHEL family use the same grep | awk '{print $2}' pipeline.
    return grep_awk_version(output, "moby-engine")


def wants_container_toolkit(env: dict[str, str]) -> bool:
    """True when the NVIDIA Container Toolkit is installed.

    Mirrors [[ "${GPU:-NVIDIA}" == "NVIDIA" ]]: an unset or empty GPU counts as
    NVIDIA. `or` returns its right side when the left side is "" or None.
    """
    return (env.get("GPU") or "NVIDIA") == "NVIDIA"


# --------------------------------------------------------------------------- #
# Small helpers with side effects.
# --------------------------------------------------------------------------- #
def _run_all(commands: list[list[str]], env: dict[str, str]) -> bool:
    """Run commands in order, stopping at the first failure (like `set -e`)."""
    for command in commands:
        if exec_program(command, _OP, env=env) != 0:
            return False
    return True


def _is_socket(path: str) -> bool:
    """bash's `[ -S path ]`: True if `path` exists and is a Unix socket."""
    try:
        # os.stat() follows symlinks, like test -S.
        return stat.S_ISSOCK(os.stat(path).st_mode)
    except OSError:          # it doesn't exist (or can't be inspected)
        return False


def _wait_for_containerd_socket() -> int | None:
    """Wait like bash for containerd's socket. Returns the seconds waited, or
    None if it never appeared (bash then carries on regardless)."""
    for attempt in range(1, _SOCKET_WAIT_TRIES + 1):
        if _is_socket(_CONTAINERD_SOCKET):
            return attempt - 1
        log_debug(_OP, f"Waiting for containerd socket... ({attempt}/{_SOCKET_WAIT_TRIES})")
        time.sleep(_SOCKET_WAIT_SECONDS)
    return None


def _moby_commands(env: dict[str, str]) -> list[list[str]] | None:
    """The package install commands for this distro/node type, or None (logged)
    if the baremetal_3p moby version is missing from versions.json."""
    distribution = env.get("DISTRIBUTION", "")
    # Mirrors ${TARGET_NODE_TYPE:-azure_vm_regular}: unset or empty -> default.
    node_type = env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE

    if "ubuntu" in distribution:
        if node_type == _BAREMETAL_3P:
            cfg = config_for("moby", env)
            # isinstance() guards against a config that isn't a JSON object.
            if not isinstance(cfg, dict) or not cfg.get("version"):
                log_error(_OP, "could not resolve moby version from versions.json "
                               "(required on baremetal_3p)",
                          distro=distribution, node_type=node_type)
                return None
            moby = str(cfg["version"])
            log_info(_OP, "Ubuntu baremetal_3p: pinning moby-engine and moby-cli "
                          "to the versions.json version (no moby-buildx)",
                     distro=distribution, node_type=node_type, moby_version=moby)
            return [["apt-get", "install", "-y", f"{package}={moby}"]
                    for package in _UBUNTU_PINNED_PACKAGES]
        log_info(_OP, "Ubuntu: installing the latest moby-engine, moby-cli and "
                      "moby-buildx from apt", distro=distribution, node_type=node_type)
        return [["apt-get", "install", "-y", package] for package in _UBUNTU_PACKAGES]

    if distribution == _AZURE_LINUX:
        packages = _AZURE_LINUX_PACKAGES
        branch = "Azure Linux"
    else:  # RHEL family: AlmaLinux, Rocky Linux, RHEL, etc.
        packages = _RHEL_PACKAGES
        branch = "RHEL family"
    log_info(_OP, f"{branch}: installing {', '.join(packages)} from dnf",
             distro=distribution)
    return [["dnf", "install", "-y", package] for package in packages]


def _record(component: str, version: str) -> bool:
    """Write one entry to the component-versions file; False (logged) on error."""
    try:
        write_component_version(component, version)
    except OSError as exc:
        log_error(_OP, f"could not record {component}={version}: {exc}",
                  component=component, version=version)
        return False
    log_info(_OP, f"Recorded {component} version {version}",
             component=component, version=version)
    return True


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def install(env: dict[str, str]) -> int:
    """Install moby/Docker (+ the NVIDIA Container Toolkit on NVIDIA), start
    the services, and record the DOCKER and MOBY_ENGINE versions. Returns 0 or 3.
    """
    distribution = env.get("DISTRIBUTION", "")
    toolkit = wants_container_toolkit(env)
    gpu = env.get("GPU") or "NVIDIA"
    log_info(_OP, "Installing Docker (moby)", distro=distribution, gpu=gpu,
             node_type=env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE,
             container_toolkit=toolkit)

    # ---- 1. moby packages ----------------------------------------------------
    commands = _moby_commands(env)
    if commands is None:
        return 3                 # already logged by _moby_commands
    if not _run_all(commands, env):
        log_error(_OP, "moby package installation failed", distro=distribution)
        return 3

    # ---- 2. NVIDIA Container Toolkit (nested bash component) -----------------
    if toolkit:
        log_info(_OP, "Installing the NVIDIA Container Toolkit (GPU is NVIDIA)",
                 gpu=gpu, script=_TOOLKIT_SCRIPT)
        script = os.path.join(env.get("COMPONENT_DIR", ""), _TOOLKIT_SCRIPT)
        if exec_program([script], _OP, env=env) != 0:
            log_error(_OP, "NVIDIA Container Toolkit installation failed", script=script)
            return 3
    else:
        log_info(_OP, f"Skipping the NVIDIA Container Toolkit (GPU is {gpu}, "
                      "not NVIDIA)", gpu=gpu)

    # ---- 3. services ---------------------------------------------------------
    log_info(_OP, "Enabling and restarting the docker service", service="docker")
    if not _run_all([["systemctl", "enable", "docker"],
                     ["systemctl", "restart", "docker"]], env):
        log_error(_OP, "could not enable/restart the docker service", service="docker")
        return 3

    if toolkit:
        log_info(_OP, "Restarting containerd to load the NVIDIA runtime config",
                 service="containerd")
        if exec_program(["systemctl", "restart", "containerd"], _OP, env=env) != 0:
            log_error(_OP, "could not restart the containerd service",
                      service="containerd")
            return 3
        waited = _wait_for_containerd_socket()
        if waited is None:
            log_warn(_OP, f"containerd socket did not appear after "
                          f"{_SOCKET_WAIT_TRIES} checks; continuing like bash",
                     socket=_CONTAINERD_SOCKET, waited_s=_SOCKET_WAIT_TRIES)
        else:
            log_info(_OP, f"containerd socket ready after {waited}s",
                     socket=_CONTAINERD_SOCKET, waited_s=waited)
        # Lists containerd's plugins (including snapshotters) into the log.
        if exec_program(["ctr", "plugin", "ls"], _OP, env=env) != 0:
            log_error(_OP, "`ctr plugin ls` failed; containerd is not usable",
                      socket=_CONTAINERD_SOCKET)
            return 3

    # ---- 4. versions ---------------------------------------------------------
    rc, output = run_capture(["docker", "--version"], _OP, env=env)
    docker = docker_version(output) if rc == 0 else None
    if docker is None:
        log_error(_OP, f"could not determine the docker version from "
                       f"`docker --version`: {output.strip()!r}", exit_code=rc)
        return 3
    if not _record("DOCKER", docker):
        return 3

    command = moby_version_command(distribution)
    rc, output = run_capture(command, _OP, env=env)
    moby = parse_moby_version(distribution, output) if rc == 0 else None
    log_debug(_OP, f"`{' '.join(command)}` output:\n{output}")
    if moby is None:
        log_error(_OP, f"could not determine the installed moby-engine version "
                       f"from `{' '.join(command)}`", distro=distribution, exit_code=rc)
        return 3
    if not _record("MOBY_ENGINE", moby):
        return 3

    log_info(_OP, "Docker installed", distro=distribution, docker_version=docker,
             moby_engine_version=moby, container_toolkit=toolkit)
    return 0
