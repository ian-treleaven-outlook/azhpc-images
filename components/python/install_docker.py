"""install_docker.py — Python port of components/install_docker.sh.

Installs Microsoft's Docker build ("moby"), starts the docker service and, on
NVIDIA images, installs the NVIDIA Container Toolkit and waits for containerd
to come back. Records DOCKER (`docker --version`, e.g. "29.1.4") and
MOBY_ENGINE (the installed moby-engine package, e.g. "29.1.4-ubuntu24.04u1").

Packages, one install command each, as in bash (_MOBY_PACKAGES):

  * Ubuntu: moby-engine, moby-cli, moby-buildx. On baremetal_3p nodes,
    moby-engine=<V> and moby-cli=<V> pinned to versions.json "moby", no
    buildx. Ubuntu 26.04 (any node type) has no moby packages: one
    `apt-get install docker.io docker-buildx`, and MOBY_ENGINE records the
    docker.io version (bash `dpkg-query -W -f='${Version}' docker.io`, here
    installed_package_version()).
  * Azure Linux: moby-engine, moby-cli, docker-buildx.
  * RHEL family (and, as bash's else branch, anything else): moby-engine,
    moby-cli, moby-buildx.

The toolkit is still the bash component install_nvidia_container_toolkit.sh,
run with the build env as install_docker.sh runs it. GPU unset or empty
counts as NVIDIA (bash's ${GPU:-NVIDIA}); AMD images get Docker only.

utils/plans/steps.py makes install() the action of the "install-docker" Step;
deleting `action=` runs the bash again.

Deliberate differences from bash
--------------------------------
  * An empty or unparseable DOCKER or MOBY_ENGINE version fails the step
    (bash records "" or stops on `${v::-1}` of an empty string), and so does
    a failing `docker --version`, `apt list`, `rpm -qa` or `dnf list` (bash
    has no pipefail, so it parses whatever they printed).
  * A missing "moby" version on Ubuntu baremetal_3p fails before installing
    (bash runs `apt-get install moby-engine=null`, which fails anyway).
  * The containerd socket wait logs at DEBUG and warns if the socket never
    appears; like bash, that alone doesn't fail the step.
  * Packages install through install_packages (`apt-get install -y -o
    Dpkg::Use-Pty=0` / `dnf install -y`). On an unknown distro bash's else
    branch would try dnf; install_packages fails instead.
  * Distro tests use Distro: bash's `*"ubuntu"*` (install) and `ubuntu*`
    (version) tests and `== azurelinux3.0` become is_ubuntu / family (the
    same for every distro built).

Kept bash quirks
----------------
  * `write_component_version "X" ${value}` is unquoted, so only the first
    word is recorded (first_word()). On Azure Linux `rpm -qa | grep moby`
    matches every moby-* package and the first listed wins, which may not be
    moby-engine (warned). The parsers were compared with real grep/awk/cut.
  * `${docker_version::-1}` trims the last character of the whole capture,
    not of each line.
"""

from __future__ import annotations

import os
import re
import stat
import time
from functools import partial

from utils.component_config import DEFAULT_NODE_TYPE, record_version, require_config
from utils.distro import Distro
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.package_installer import install_packages, installed_package_version
from utils.process import run, run_capture
from utils.services import systemctl

_OP = "install-docker"

# Canonical's docker.io instead of moby, checked before baremetal_3p as in bash.
_UBUNTU_DOCKER_IO = "ubuntu26.04"
_UBUNTU_DOCKER_IO_PACKAGES = ["docker.io", "docker-buildx"]     # one command

# Ubuntu nodes whose moby is pinned to versions.json.
_BAREMETAL_3P = "baremetal_3p"
_UBUNTU_PINNED_PACKAGES = ["moby-engine", "moby-cli"]

# By distro family; each package is its own install command, as in bash.
_MOBY_PACKAGES = {
    "ubuntu": ["moby-engine", "moby-cli", "moby-buildx"],
    "azurelinux": ["moby-engine", "moby-cli", "docker-buildx"],
    "rhel": ["moby-engine", "moby-cli", "moby-buildx"],
}

# bash: up to 30 `[ -S socket ]` checks, sleeping 1s after each miss.
# Module-level so the tests can point it at a temporary folder.
_CONTAINERD_SOCKET = "/run/containerd/containerd.sock"
_SOCKET_WAIT_TRIES = 30
_SOCKET_WAIT_SECONDS = 1

_TOOLKIT_SCRIPT = "install_nvidia_container_toolkit.sh"

# Bash word splitting and awk's default fields split on these only
# (str.split() would also split on \r, \f and \v).
_BLANKS = re.compile(r"[ \t\n]+")


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def _lines(output: str) -> list[str]:
    """Lines as grep/awk/cut see them: split on "\\n" only, no empty last line."""
    if not output:
        return []
    lines = output.split("\n")
    if output.endswith("\n"):
        lines.pop()
    return lines


def _awk_field(line: str, number: int) -> str:
    """awk's $<number> with the default separator ("" if missing)."""
    fields = [field for field in _BLANKS.split(line) if field]
    return fields[number - 1] if len(fields) >= number else ""


def first_word(text: str) -> str | None:
    """The first word of an unquoted bash expansion, or None if there is none."""
    words = [word for word in _BLANKS.split(text) if word]
    return words[0] if words else None


def docker_version(output: str) -> str | None:
    """DOCKER from `docker --version` output.

        docker_version=$(docker --version | awk -F' ' '{print $3}')
        write_component_version "DOCKER" ${docker_version::-1}

    "Docker version 29.1.4, build 0e5a6ab" -> "29.1.4,"  -> "29.1.4".
    """
    # $(...) strips trailing newlines; ${v::-1} then drops one character.
    captured = "\n".join(_awk_field(line, 3) for line in _lines(output)).rstrip("\n")
    if not captured:
        return None
    return first_word(captured[:-1])


def grep_awk_version(output: str, pattern: str) -> str | None:
    """`... | grep <pattern> | awk -F' ' '{print $2}'`, first word.

        "moby-engine/now 29.1.4-ubuntu24.04u1 amd64 [installed,local]"  (apt)
        "moby-engine.x86_64   28.5.1-1.el9   @packages-microsoft-com-prod" (dnf)
    """
    values = [_awk_field(line, 2) for line in _lines(output) if pattern in line]
    return first_word("\n".join(values))


def _cut_fields_3_4(line: str) -> str:
    """`cut -d'-' -f3,4` on one line (a line without "-" passes through)."""
    if "-" not in line:
        return line
    return "-".join(line.split("-")[2:4])


def rpm_moby_version(output: str) -> str | None:
    """`rpm -qa | grep moby | cut -d'-' -f3,4`, first word.

    "moby-engine-25.0.3-11.azl3.x86_64" -> "25.0.3-11.azl3.x86_64". Every
    moby-* package matches, and the first one rpm lists wins.
    """
    values = [_cut_fields_3_4(line) for line in _lines(output) if "moby" in line]
    return first_word("\n".join(values))


def rpm_moby_source_line(output: str) -> str | None:
    """The `rpm -qa` line rpm_moby_version() takes its value from.

    The first "moby" line whose cut result isn't blank (blank results vanish
    in bash's word splitting).
    """
    for line in _lines(output):
        if "moby" in line and first_word(_cut_fields_3_4(line)) is not None:
            return line
    return None


_apt_or_dnf_moby_version = partial(grep_awk_version, pattern="moby-engine")

# How each family reports the installed moby-engine version: (command, parser).
_MOBY_VERSION_QUERY = {
    "ubuntu": (["apt", "list", "--installed"], _apt_or_dnf_moby_version),
    "azurelinux": (["rpm", "-qa"], rpm_moby_version),
    "rhel": (["dnf", "list", "installed"], _apt_or_dnf_moby_version),
}


def moby_version_query(distro: Distro):
    """(command, parser) for the installed moby-engine version."""
    return _MOBY_VERSION_QUERY.get(distro.family, _MOBY_VERSION_QUERY["rhel"])


def wants_container_toolkit(env: dict[str, str]) -> bool:
    """bash's [[ "${GPU:-NVIDIA}" == "NVIDIA" ]] (unset or empty is NVIDIA)."""
    return (env.get("GPU") or "NVIDIA") == "NVIDIA"


# --------------------------------------------------------------------------- #
# Steps: each returns a result, or False/None after the failure was logged.
# --------------------------------------------------------------------------- #
def _moby_installs(env: dict[str, str], distro: Distro) -> list[list[str]] | None:
    """The package lists to install, one per bash install command."""
    node_type = env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE
    if distro.name == _UBUNTU_DOCKER_IO:
        log_info(_OP, "Ubuntu 26.04: installing Canonical's docker.io and "
                      "docker-buildx (no Microsoft moby packages), on every "
                      "node type", distro=str(distro), node_type=node_type,
                 packages="docker.io docker-buildx")
        return [list(_UBUNTU_DOCKER_IO_PACKAGES)]
    if distro.is_ubuntu and node_type == _BAREMETAL_3P:
        cfg = require_config("moby", env, _OP, "version")
        if cfg is None:
            return None
        moby = str(cfg["version"])
        log_info(_OP, "Ubuntu baremetal_3p: pinning moby-engine and moby-cli "
                      "to the versions.json version (no moby-buildx)",
                 distro=str(distro), node_type=node_type, moby_version=moby)
        return [[f"{package}={moby}"] for package in _UBUNTU_PINNED_PACKAGES]
    packages = _MOBY_PACKAGES.get(distro.family, _MOBY_PACKAGES["rhel"])
    log_info(_OP, f"Installing {', '.join(packages)}", distro=str(distro),
             family=distro.family, node_type=node_type)
    return [[package] for package in packages]


def _install_container_toolkit(env: dict[str, str], gpu: str) -> bool:
    """The nested bash component, on NVIDIA images only."""
    if not wants_container_toolkit(env):
        log_info(_OP, f"Skipping the NVIDIA Container Toolkit (GPU is {gpu}, "
                      "not NVIDIA)", gpu=gpu)
        return True
    log_info(_OP, "Installing the NVIDIA Container Toolkit (GPU is NVIDIA)",
             gpu=gpu, script=_TOOLKIT_SCRIPT)
    return run([os.path.join(env.get("COMPONENT_DIR", ""), _TOOLKIT_SCRIPT)], _OP, env=env)


def _is_socket(path: str) -> bool:
    """bash's `[ -S path ]` (follows symlinks)."""
    try:
        return stat.S_ISSOCK(os.stat(path).st_mode)
    except OSError:
        return False


def _wait_for_containerd_socket() -> None:
    """bash's 30 x 1s wait for the socket; logs the outcome, never fails."""
    for attempt in range(1, _SOCKET_WAIT_TRIES + 1):
        if _is_socket(_CONTAINERD_SOCKET):
            log_info(_OP, f"containerd socket ready after {attempt - 1}s",
                     socket=_CONTAINERD_SOCKET, waited_s=attempt - 1)
            return
        log_debug(_OP, f"Waiting for containerd socket... ({attempt}/{_SOCKET_WAIT_TRIES})")
        time.sleep(_SOCKET_WAIT_SECONDS)
    log_warn(_OP, f"containerd socket did not appear after {_SOCKET_WAIT_TRIES} "
                  "checks; continuing like bash",
             socket=_CONTAINERD_SOCKET, waited_s=_SOCKET_WAIT_TRIES)


def _restart_containerd(env: dict[str, str]) -> bool:
    """NVIDIA: restart containerd for the runtime config, wait, list plugins."""
    log_info(_OP, "Restarting containerd to load the NVIDIA runtime config",
             service="containerd")
    if not systemctl("restart", "containerd", op=_OP, env=env):
        return False
    _wait_for_containerd_socket()
    # Lists containerd's (snapshotter) plugins into the log.
    return run(["ctr", "plugin", "ls"], _OP, env=env)


def _record_docker_version(env: dict[str, str]) -> str | None:
    rc, output = run_capture(["docker", "--version"], _OP, env=env)
    docker = docker_version(output) if rc == 0 else None
    if docker is None:
        log_error(_OP, "could not determine the docker version from "
                       f"`docker --version`: {output.strip()!r}", exit_code=rc)
        return None
    return docker if record_version("DOCKER", docker, _OP) else None


def _docker_io_version(env: dict[str, str]) -> str | None:
    """Ubuntu 26.04's MOBY_ENGINE value: the installed docker.io version."""
    version = installed_package_version("docker.io", env, _OP)
    if version is not None:
        log_info(_OP, "Ubuntu 26.04: MOBY_ENGINE records the docker.io package "
                      "version, as bash does", package="docker.io", version=version)
    return version


def _moby_engine_version(env: dict[str, str], distro: Distro) -> str | None:
    """The installed moby-engine version, parsed from the family's package listing."""
    command, parse = moby_version_query(distro)
    rc, output = run_capture(command, _OP, env=env)
    log_debug(_OP, f"`{' '.join(command)}` output:\n{output}")
    moby = parse(output) if rc == 0 else None
    if moby is None:
        log_error(_OP, "could not determine the installed moby-engine version "
                       f"from `{' '.join(command)}`", distro=str(distro), exit_code=rc)
        return None
    if distro.is_azurelinux:
        source = rpm_moby_source_line(output) or ""
        if not source.startswith("moby-engine-"):
            log_warn(_OP, "MOBY_ENGINE is taken from the first `rpm -qa | grep "
                          "moby` match, which is not moby-engine; recording its "
                          "version anyway, like bash",
                     matched_package=source, version=moby)
    return moby


def _record_moby_version(env: dict[str, str], distro: Distro) -> str | None:
    if distro.name == _UBUNTU_DOCKER_IO:
        moby = _docker_io_version(env)
    else:
        moby = _moby_engine_version(env, distro)
    return moby if moby is not None and record_version("MOBY_ENGINE", moby, _OP) else None


def install(env: dict[str, str]) -> int:
    """Install moby/Docker (+ the NVIDIA Container Toolkit on NVIDIA), start
    the services, and record the DOCKER and MOBY_ENGINE versions."""
    distro = Distro.from_env(env)
    gpu = env.get("GPU") or "NVIDIA"
    toolkit = wants_container_toolkit(env)
    log_info(_OP, "Installing Docker (moby)", distro=str(distro), gpu=gpu,
             node_type=env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE,
             container_toolkit=toolkit)

    installs = _moby_installs(env, distro)
    if installs is None:
        return 3
    if not all(install_packages(packages, env, _OP) for packages in installs):
        return 3
    if not _install_container_toolkit(env, gpu):
        return 3
    log_info(_OP, "Enabling and restarting the docker service", service="docker")
    if not (systemctl("enable", "docker", op=_OP, env=env)
            and systemctl("restart", "docker", op=_OP, env=env)):
        return 3
    if toolkit and not _restart_containerd(env):
        return 3

    docker = _record_docker_version(env)
    if docker is None:
        return 3
    moby = _record_moby_version(env, distro)
    if moby is None:
        return 3
    log_info(_OP, "Docker installed", distro=str(distro), docker_version=docker,
             moby_engine_version=moby, container_toolkit=toolkit)
    return 0
