"""install_waagent.py — Python port of components/install_waagent.sh.

WALinuxAgent (waagent) is the Azure guest agent: it provisions the VM, runs
VM extensions and, on InfiniBand SKUs, sets up IPoIB. This step updates the
agent, tunes /etc/waagent.conf, records the agent's versions and restarts it.

Bash takes one of four branches (install_branch()):

  * azurelinux — `dnf update -y WALinuxAgent`, then nine waagent.conf keys
    with OS.EnableRDMA=y for every SKU.
  * el8 / el9 / default — build the GitHub release (versions.json "waagent")
    with `setup.py install --register-service`. The RHEL family first makes
    pip, setuptools and `distro` available for python3; EL8 repeats that
    inside the source tree, EL9 does it for python3.12, installs with
    python3.12 and points waagent.service's ExecStart at python3.12. Then the
    waagent.conf keys, with OS.EnableRDMA=y only if the target uses IPoIB.

Every branch then records WAAGENT and WAAGENT_EXTENSIONS from
`waagent --version`, reloads systemd, restarts the agent (walinuxagent on
Ubuntu, waagent elsewhere) and sleeps 60s so the restarted agent doesn't take
the package manager lock from the next step.

utils/plans/steps.py makes install() the action of the "install-waagent"
Step; deleting `action=` there runs the bash script again.

Deliberate differences from bash
--------------------------------
  * A missing waagent version or sha256 fails the step before anything runs
    (bash would do the pip set-up, then download "vnull.tar.gz").
  * Bash downloads and unpacks in its cwd; we use _WORK_DIR, emptied first.
    As in bash, only WALinuxAgent-<version>/ is removed afterwards.
  * `waagent --version` runs once, not twice, and the step fails if it fails
    or its output lacks either version (bash records empty versions).
  * Distro tests go through Distro: bash's `== azurelinux3.0` and `rhel8*`
    become is_azurelinux and major "8" (the same for every distro built).
  * The EL9 systemctl calls don't run inside the source tree (bash's pushd);
    systemctl doesn't use its cwd.

Kept bash quirks
----------------
  * EL8 is an exact match for almalinux8.10 and rocky8.10, so e.g.
    almalinux8.9 takes the default branch.
  * Azure Linux sets Extensions.GoalStatePeriod twice and ignores the waagent
    version versions.json pins for it (a warning says so).
  * update_waagent_conf's grep/sed treat "." in a key as "any character", and
    `echo >>` glues a new key onto a last line that has no newline (warned).
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime

from utils.component_config import DEFAULT_NODE_TYPE, record_version, require_config
from utils.distro import Distro
from utils.download import fetch_verified
from utils.files import copy_file, edit_file, extract_tarball, remove_paths
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.package_installer import update_packages
from utils.process import run_all, run_capture
from utils.services import systemctl
from utils.sku import sku_network_mode, sku_uses_ipoib

_OP = "install-waagent"

_WAAGENT_CONF = "/etc/waagent.conf"
_WAAGENT_SERVICE_FILE = "/usr/lib/systemd/system/waagent.service"
_WORK_DIR = "/tmp/waagent-build"
_DOWNLOAD_URL = "https://github.com/Azure/WALinuxAgent/archive/refs/tags/v{version}.tar.gz"
_WAIT_SECONDS = 60

# Bash's branches (see install_branch()).
_AZURE_LINUX = "azurelinux"
_EL8 = "el8"
_EL9 = "el9"
_DEFAULT = "default"

# The python each source branch runs setup.py with.
_SETUP_PYTHON = {_EL8: "python3", _EL9: "python3.12", _DEFAULT: "python3"}

# The agent's systemd unit, by distro family; other distros restart nothing.
_AGENT_SERVICE = {"ubuntu": "walinuxagent", "rhel": "waagent", "azurelinux": "waagent"}

# The EL9 sed edit of waagent.service. Apart from the leading ^ the pattern
# has no regex characters, so it is a plain match at the start of a line.
_EXEC_START_OLD = "ExecStart=/usr/bin/python3 -u /usr/sbin/waagent -daemon"
_EXEC_START_NEW = "ExecStart=/usr/bin/python3.12 -u /usr/sbin/waagent -daemon"

# waagent.conf settings, (key, value) in bash's order. The repeated
# GoalStatePeriod is bash's; the second write changes nothing.
_AZURE_LINUX_CONF = (
    ("OS.EnableRDMA", "y"),
    ("Extensions.GoalStatePeriod", "300"),
    ("Extensions.GoalStatePeriod", "300"),
    ("Extensions.InitialGoalStatePeriod", "6"),
    ("OS.EnableFirewallPeriod", "300"),
    ("OS.RemovePersistentNetRulesPeriod", "300"),
    ("OS.RootDeviceScsiTimeoutPeriod", "300"),
    ("OS.MonitorDhcpClientRestartPeriod", "60"),
    ("Provisioning.MonitorHostNamePeriod", "60"),
)
# The source branches: OS.EnableRDMA=y goes between these, on IPoIB targets.
_CONF_BEFORE_RDMA = (
    ("Extensions.GoalStatePeriod", "300"),
    ("Extensions.InitialGoalStatePeriod", "6"),
    ("OS.EnableFirewallPeriod", "300"),
)
_RDMA_SETTING = ("OS.EnableRDMA", "y")
_CONF_AFTER_RDMA = (
    ("OS.RemovePersistentNetRulesPeriod", "300"),
    ("OS.RootDeviceScsiTimeoutPeriod", "300"),
    ("OS.MonitorDhcpClientRestartPeriod", "60"),
    ("Provisioning.MonitorHostName", "y"),
    ("Provisioning.MonitorHostNamePeriod", "60"),
)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def install_branch(distro: Distro) -> str:
    """Which branch bash takes: azurelinux, el8, el9 or default."""
    if distro.is_azurelinux:
        return _AZURE_LINUX
    if distro.name in ("almalinux8.10", "rocky8.10") or (distro.is_rhel and distro.major == "8"):
        return _EL8
    if distro.is_rhel_family and distro.major == "9":
        return _EL9
    return _DEFAULT


def pip_setup_commands(python: str) -> list[list[str]]:
    """The three commands that make pip, setuptools and `distro` available."""
    return [
        [python, "-m", "ensurepip", "--upgrade"],
        [python, "-m", "pip", "install", "--upgrade", "pip", "setuptools"],
        [python, "-m", "pip", "install", "distro"],
    ]


def setup_commands(branch: str) -> list[list[str]]:
    """What bash runs inside WALinuxAgent-<version>/ for a source branch."""
    python = _SETUP_PYTHON[branch]
    setup = [[python, "setup.py", "install", "--register-service"]]
    if branch == _DEFAULT:
        return setup
    return pip_setup_commands(python) + setup


def version_command(branch: str) -> list[str]:
    """How bash runs `waagent --version`: with python3.12 on EL9."""
    if branch == _EL9:
        return ["python3.12", "-u", "/usr/sbin/waagent", "--version"]
    return ["waagent", "--version"]


def agent_service(distro: Distro) -> str | None:
    """The agent's systemd unit bash restarts, or None if it restarts none."""
    return _AGENT_SERVICE.get(distro.family)


def conf_key_pattern(key: str) -> str:
    """Bash's grep/sed pattern "^$key=" as a Python regex ("." stays a wildcard)."""
    return "^" + re.escape(key).replace(r"\.", ".") + "="


def update_waagent_conf_text(text: str, key: str, value: str) -> tuple[str, int]:
    """Bash's update_waagent_conf on the file's text: (new text, lines replaced).

    Like sed, every line matching "^key=" is replaced whole and a missing
    final newline stays missing. 0 replaced means "key=value\\n" was appended
    (bash's `echo >>`).
    """
    # A function replacement is used literally (no backslash processing).
    pattern = re.compile(conf_key_pattern(key) + ".*", re.MULTILINE)
    new_text, count = pattern.subn(lambda _match: f"{key}={value}", text)
    if count:
        return new_text, count
    return text + f"{key}={value}\n", 0


def retarget_exec_start(text: str) -> tuple[str, int]:
    """The EL9 sed edit of waagent.service: (new text, lines changed).

    An already-edited line ("python3.12 -u") doesn't match, so reruns are no-ops.
    """
    pattern = re.compile("^" + re.escape(_EXEC_START_OLD), re.MULTILINE)
    return pattern.subn(lambda _match: _EXEC_START_NEW, text)


def parse_waagent_version(output: str) -> tuple[str, str]:
    """(WAAGENT, WAAGENT_EXTENSIONS) from `waagent --version` output.

        WALinuxAgent-2.15.0.1 running on ubuntu 24.04
        Python: 3.12.3
        Goal state agent: 2.15.0.1

    WAAGENT is bash's `head -n 1 | awk '{print $1}' | awk -F- '{print $2}'`;
    WAAGENT_EXTENSIONS is `sed '3q;d' | awk '{print $4}'`. A missing piece
    gives "", as awk does.
    """
    lines = output.split("\n")
    # split() with no argument skips leading blanks, like awk's fields.
    first_words = lines[0].split()
    dash_parts = first_words[0].split("-") if first_words else []
    agent = dash_parts[1] if len(dash_parts) > 1 else ""
    third_words = lines[2].split() if len(lines) > 2 else []
    extensions = third_words[3] if len(third_words) > 3 else ""
    return agent, extensions


def unused_azure_linux_entry(env: dict[str, str], distro: Distro):
    """versions.json's waagent entry for this Azure Linux, or None.

    Bash never reads it, so install() only warns that it is ignored. Read
    from the raw JSON because config_for() would fall back to "common";
    invalid JSON counts as no entry.
    """
    try:
        versions = json.loads(env.get("COMPONENT_VERSIONS") or "{}")
    except ValueError:
        return None
    waagent = versions.get("waagent") if isinstance(versions, dict) else None
    entry = waagent.get(distro.name) if isinstance(waagent, dict) else None
    return entry or None


def backup_timestamp() -> str:
    """The suffix bash's `date +%F_%T` gives, e.g. "2026-10-29_15:07:42"."""
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


# --------------------------------------------------------------------------- #
# File edits
# --------------------------------------------------------------------------- #
def _set_conf(key: str, value: str) -> bool:
    """Set one waagent.conf key (bash's update_waagent_conf) and log what changed.

    A missing file is created, as bash's `echo >>` does.
    """
    existed = os.path.exists(_WAAGENT_CONF)

    def edit(text: str) -> str:
        new_text, replaced = update_waagent_conf_text(text, key, value)
        action = "replaced" if replaced else ("appended" if existed else "created")
        if action == "appended" and text and not text.endswith("\n"):
            log_warn(_OP, f"{_WAAGENT_CONF} did not end with a newline, so {key} was "
                          "glued onto its last line, as bash's `echo >>` does",
                     path=_WAAGENT_CONF, key=key, value=value)
        log_info(_OP, f"Setting {key}={value} in {_WAAGENT_CONF} ({action})",
                 path=_WAAGENT_CONF, key=key, value=value, action=action,
                 lines_replaced=replaced)
        return new_text
    return edit_file(_WAAGENT_CONF, edit, _OP, create=True)


def _apply_conf(settings) -> bool:
    return all(_set_conf(key, value) for key, value in settings)


def _retarget_exec_start_text(text: str) -> str:
    """The edit_file transform for waagent.service; warns when nothing matches."""
    new_text, changed = retarget_exec_start(text)
    if changed:
        log_info(_OP, f"Editing {_WAAGENT_SERVICE_FILE}: ExecStart now runs waagent "
                      "with python3.12", path=_WAAGENT_SERVICE_FILE, lines_changed=changed)
    else:
        log_warn(_OP, f"No '{_EXEC_START_OLD}' line in {_WAAGENT_SERVICE_FILE}; "
                      "ExecStart left unchanged (bash's sed does the same)",
                 path=_WAAGENT_SERVICE_FILE, lines_changed=0)
    return new_text


def _retarget_service_file() -> bool:
    """Back up waagent.service and point its ExecStart at python3.12 (EL9)."""
    backup = f"{_WAAGENT_SERVICE_FILE}.bak.{backup_timestamp()}"
    return (copy_file(_WAAGENT_SERVICE_FILE, backup, _OP)      # contents + mode, like `cp`
            and edit_file(_WAAGENT_SERVICE_FILE, _retarget_exec_start_text, _OP))


# --------------------------------------------------------------------------- #
# Steps: each returns True, or False after the failure was logged.
# --------------------------------------------------------------------------- #
def _update_from_distro_repo(env: dict[str, str], distro: Distro) -> bool:
    """Bash's azurelinux branch: dnf update, then the Azure Linux conf keys."""
    log_info(_OP, "Updating WALinuxAgent from the Azure Linux repository",
             distro=str(distro), method="dnf update", source="distro repository")
    if not update_packages(["WALinuxAgent"], env, _OP):
        return False
    entry = unused_azure_linux_entry(env, distro)
    if entry is not None:
        log_warn(_OP, f"versions.json has a waagent entry for {distro} that is not used: "
                      "Azure Linux takes WALinuxAgent from the distro repository, as in bash",
                 distro=str(distro), unused_entry=entry)
    log_info(_OP, "Azure Linux: OS.EnableRDMA=y is set for every SKU and node type",
             distro=str(distro), sku=env.get("SKU", ""),
             node_type=env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE)
    return _apply_conf(_AZURE_LINUX_CONF)


def _download_release(version: str, sha256: str, env: dict[str, str]) -> str | None:
    """Fetch and unpack the release into _WORK_DIR; the source dir, or None."""
    remove_paths(_WORK_DIR, op=_OP)
    tarball = fetch_verified(_DOWNLOAD_URL.format(version=version), sha256, _OP,
                             dest_dir=_WORK_DIR)
    if tarball is None:
        return None
    if not extract_tarball(os.path.basename(tarball), _OP, cwd=_WORK_DIR, env=env):
        return None
    source_dir = os.path.join(_WORK_DIR, f"WALinuxAgent-{version}")
    if not os.path.isdir(source_dir):
        log_error(_OP, f"{tarball} did not unpack into {source_dir}")
        return None
    return source_dir


def _run_agent_with_python312(env: dict[str, str]) -> bool:
    """EL9: setup.py registered a python3 unit; rewrite it for python3.12."""
    if not (systemctl("stop", "waagent", op=_OP, env=env)
            and systemctl("disable", "waagent", op=_OP, env=env)
            and _retarget_service_file()
            and systemctl("daemon-reexec", op=_OP, env=env)
            and systemctl("enable", "waagent", op=_OP, env=env)):
        return False
    log_info(_OP, "waagent service re-enabled with its python3.12 ExecStart",
             service="waagent", action="enabled")
    return True


def _conf_for_target(env: dict[str, str]) -> tuple:
    """The source branches' conf settings, with OS.EnableRDMA=y on IPoIB targets."""
    sku = env.get("SKU", "")
    node_type = env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE
    uses_ipoib = sku_uses_ipoib(sku, node_type)
    message = ("Target uses IPoIB: OS.EnableRDMA=y will be set" if uses_ipoib
               else "Target doesn't use IPoIB: OS.EnableRDMA is left as it is")
    log_info(_OP, message, sku=sku, node_type=node_type,
             network_mode=sku_network_mode(sku, node_type), uses_ipoib=uses_ipoib)
    return _CONF_BEFORE_RDMA + ((_RDMA_SETTING,) if uses_ipoib else ()) + _CONF_AFTER_RDMA


def _install_from_source(env: dict[str, str], distro: Distro, branch: str) -> bool:
    """Bash's el8/el9/default branches: build the release, then the conf keys."""
    cfg = require_config("waagent", env, _OP, "version", "sha256")
    if cfg is None:
        return False
    version = cfg["version"]
    log_info(_OP, f"Installing WALinuxAgent {version} from source",
             version=version, source=_DOWNLOAD_URL.format(version=version),
             distro=str(distro), handler=branch, sku=env.get("SKU", ""),
             node_type=env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE)

    if distro.is_rhel_family and not run_all(pip_setup_commands("python3"), _OP, env=env):
        return False
    source_dir = _download_release(version, cfg["sha256"], env)
    if source_dir is None:
        return False
    if not run_all(setup_commands(branch), _OP, cwd=source_dir, env=env):
        return False
    if branch == _EL9 and not _run_agent_with_python312(env):
        return False
    if not _apply_conf(_conf_for_target(env)):
        return False
    remove_paths(source_dir, op=_OP)      # the tarball stays, as in bash
    return True


def _record_versions(env: dict[str, str], branch: str) -> tuple[str, str] | None:
    """Run `waagent --version` and record WAAGENT and WAAGENT_EXTENSIONS."""
    command = version_command(branch)
    rc, output = run_capture(command, _OP, env=env)
    if rc != 0:
        return None
    log_debug(_OP, f"`{' '.join(command)}` output:\n{output}")
    versions = parse_waagent_version(output)
    if not all(versions):
        log_error(_OP, f"unexpected `{' '.join(command)}` output: {output.strip()!r}",
                  waagent=versions[0], waagent_extensions=versions[1])
        return None
    if not all(record_version(component, version, _OP)
               for component, version in zip(("WAAGENT", "WAAGENT_EXTENSIONS"), versions)):
        return None
    return versions


def _restart_agent(env: dict[str, str], distro: Distro) -> bool:
    """Bash's `systemctl daemon-reload`, then restart the agent's unit, if it has one."""
    if not systemctl("daemon-reload", op=_OP, env=env):
        return False
    service = agent_service(distro)
    if service is None:
        log_info(_OP, f"No agent restart rule for {distro}; not restarting (same as bash)",
                 distro=str(distro))
        return True
    log_info(_OP, f"Restarting {service}", service=service, action="restarted")
    return systemctl("restart", service, op=_OP, env=env)


def install(env: dict[str, str]) -> int:
    """Install/update WALinuxAgent, configure it, record its versions, restart it."""
    distro = Distro.from_env(env)
    branch = install_branch(distro)
    if branch == _AZURE_LINUX:
        installed = _update_from_distro_repo(env, distro)
    else:
        installed = _install_from_source(env, distro, branch)
    if not installed:
        return 3
    versions = _record_versions(env, branch)
    if versions is None:
        return 3
    if not _restart_agent(env, distro):
        return 3

    log_info(_OP, f"Waiting {_WAIT_SECONDS}s so waagent doesn't take the package "
                  "manager locks", seconds=_WAIT_SECONDS)
    time.sleep(_WAIT_SECONDS)
    agent, extensions = versions
    log_info(_OP, f"WALinuxAgent {agent} installed and configured",
             waagent=agent, waagent_extensions=extensions, distro=str(distro),
             conf=_WAAGENT_CONF, service=agent_service(distro) or "")
    return 0
