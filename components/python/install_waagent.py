"""install_waagent.py — Python port of components/install_waagent.sh.

What this step does
-------------------
WALinuxAgent (waagent) is the Azure guest agent: it provisions the VM, runs
VM extensions and, on InfiniBand SKUs, sets up IPoIB (IP over InfiniBand).
This step updates the agent and tunes its config file, /etc/waagent.conf.

  * Azure Linux 3.0: update the WALinuxAgent package with dnf, then set nine
    keys in /etc/waagent.conf (OS.EnableRDMA=y always).
  * Every other distro: build and install the agent from the GitHub release
    tarball (version and sha256 from versions.json "waagent"):
      - RHEL family (Alma/Rocky/RHEL): first make sure pip, setuptools and the
        `distro` module are available for python3.
      - EL8 (almalinux8.10, rocky8.10, rhel8*): the same pip set-up again, then
        `python3 setup.py install --register-service`.
      - EL9 (almalinux9*, rocky9*, rhel9*): the pip set-up and setup.py install
        with python3.12, then point waagent.service's ExecStart at python3.12
        (backing the unit file up first) and re-enable the service.
      - Anything else (Ubuntu): `python3 setup.py install --register-service`.
    Then set the waagent.conf keys. OS.EnableRDMA=y is only set when the
    target uses IPoIB (see utils/sku.py: sku_uses_ipoib).
  * Record the agent version (WAAGENT) and its extension handler version
    (WAAGENT_EXTENSIONS) from `waagent --version`.
  * Reload systemd, restart the agent (walinuxagent on Ubuntu, waagent on
    Alma/Rocky/RHEL/Azure Linux), then wait 60 seconds so the restarted agent
    doesn't grab the package manager lock from the next build step.

Setting a waagent.conf key (bash's update_waagent_conf):

    if grep -q "^$key=" /etc/waagent.conf; then
        sed -i "s/^$key=.*/$key=$value/" /etc/waagent.conf
    else
        echo "$key=$value" >> /etc/waagent.conf
    fi

The rest of the original bash, summarised (see components/install_waagent.sh):

    if [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
        dnf update -y WALinuxAgent
        update_waagent_conf ... (9 calls, OS.EnableRDMA=y first)
    else
        [RHEL family] python3 -m ensurepip --upgrade; pip install --upgrade
                      pip setuptools; pip install distro
        download_and_verify .../WALinuxAgent/archive/refs/tags/v${V}.tar.gz ${SHA}
        tar -xvf v${V}.tar.gz; pushd WALinuxAgent-${V}/
        [EL8]  pip set-up again; python3 setup.py install --register-service
        [EL9]  pip set-up with python3.12; python3.12 setup.py install ...;
               systemctl stop/disable waagent; cp unit file to .bak.$(date +%F_%T);
               sed -i 's|^ExecStart=/usr/bin/python3 -u ...|...python3.12 -u ...|';
               systemctl daemon-reexec; systemctl enable waagent
        [else] python3 setup.py install --register-service
        popd
        update_waagent_conf ... (OS.EnableRDMA=y only if sku_uses_ipoib)
        rm -rf WALinuxAgent-${V}
    fi
    write_component_version "WAAGENT"            <line 1, field 1, part after "-">
    write_component_version "WAAGENT_EXTENSIONS" <line 3, field 4>
    systemctl daemon-reload
    systemctl restart walinuxagent | waagent
    sleep 60

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-waagent" Step. The build then calls install(env) instead of running
the bash script. Deleting `action=` runs the bash script again, which is why
the .sh file is kept.

The waagent.conf and waagent.service edits are plain Python (small, tested
text helpers that mirror grep/sed exactly). dnf, pip, setup.py, tar and
systemctl run as programs with exec_program; `waagent --version` runs with
run_capture because its output is parsed. The release tarball is unpacked
with the `tar` command, as in bash.

install() returns 0 for success or 3 for failure. It logs problems and returns
3; it does not let exceptions escape to the build loop.

Deliberate differences from bash
--------------------------------
  * A missing waagent version or sha256 fails the step before anything is
    installed (bash would try to download "vnull.tar.gz"). On Azure Linux the
    config isn't used, exactly as in bash (versions.json has an azurelinux3.0
    entry that the script never reads).
  * Work directory: bash downloads and unpacks in its current directory (the
    distro build dir). We use _WORK_DIR, emptied first so a leftover from an
    earlier run can't get in the way. As in bash, only the unpacked
    WALinuxAgent-<version> folder is removed afterwards; the tarball stays.
  * `waagent --version` runs once, not twice. If it fails, or its output
    doesn't contain both versions, the step fails. Bash would silently record
    empty versions.
"""

# Lets type hints such as `str | None` work on Python versions older than 3.10.
from __future__ import annotations

import os
import re
import shutil
import time
from datetime import datetime

from utils.component_config import DEFAULT_NODE_TYPE, config_for, write_component_version
from utils.download import ChecksumError, download_and_verify
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.process import exec_program, run_capture
from utils.sku import sku_network_mode, sku_uses_ipoib

# The Step's `op` label in build_config.py. Every log line from this module uses it.
_OP = "install-waagent"

_AZURE_LINUX = "azurelinux3.0"

_WAAGENT_CONF = "/etc/waagent.conf"
_WAAGENT_SERVICE_FILE = "/usr/lib/systemd/system/waagent.service"

# Where the release tarball is downloaded and unpacked (bash used its cwd).
_WORK_DIR = "/tmp/waagent-build"

_DOWNLOAD_URL = "https://github.com/Azure/WALinuxAgent/archive/refs/tags/v{version}.tar.gz"

# How long bash waits after restarting the agent.
_WAIT_SECONDS = 60

# The EL9 waagent.service edit. The sed pattern contains no regex special
# characters (apart from the leading ^), so it is a plain text match at the
# start of a line. The rest of the line is kept, as sed keeps it.
_EXEC_START_OLD = "ExecStart=/usr/bin/python3 -u /usr/sbin/waagent -daemon"
_EXEC_START_NEW = "ExecStart=/usr/bin/python3.12 -u /usr/sbin/waagent -daemon"

# waagent.conf settings, as (key, value), in bash's exact order.
# Azure Linux sets the GoalStatePeriod twice; that is copied from bash (the
# second call just replaces the value with the same one).
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

# The other distros: the settings before and after the optional
# OS.EnableRDMA=y (written only when the target uses IPoIB).
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

# The three setup.py paths (see install_handler()).
_HANDLER_EL8 = "el8"
_HANDLER_EL9 = "el9"
_HANDLER_DEFAULT = "default"


# --------------------------------------------------------------------------- #
# Pure helpers: no side effects, so tests call them directly.
# --------------------------------------------------------------------------- #
def is_rhel_family(distribution: str) -> bool:
    """Bash's `*"almalinux"* || *"rocky"* || *"rhel"*` test ("contains")."""
    return any(name in distribution for name in ("almalinux", "rocky", "rhel"))


def is_el9(distribution: str) -> bool:
    """Bash's `almalinux9* || rocky9* || rhel9*` test ("starts with")."""
    return distribution.startswith(("almalinux9", "rocky9", "rhel9"))


def install_handler(distribution: str) -> str:
    """Which setup.py path bash takes for this distro: el8, el9 or default.

    EL8 is an exact match for almalinux8.10 and rocky8.10 (so e.g. almalinux8.9
    would take the default path, as in bash), plus anything starting "rhel8".
    """
    if distribution in ("almalinux8.10", "rocky8.10") or distribution.startswith("rhel8"):
        return _HANDLER_EL8
    if is_el9(distribution):
        return _HANDLER_EL9
    return _HANDLER_DEFAULT


def pip_setup_commands(python: str) -> list[list[str]]:
    """The three commands that make pip, setuptools and `distro` available."""
    return [
        [python, "-m", "ensurepip", "--upgrade"],
        [python, "-m", "pip", "install", "--upgrade", "pip", "setuptools"],
        [python, "-m", "pip", "install", "distro"],
    ]


def setup_commands(handler: str) -> list[list[str]]:
    """The commands bash runs inside WALinuxAgent-<version>/ for a handler.

    (The EL9 service edit that follows setup.py is done separately.)
    """
    setup = ["setup.py", "install", "--register-service"]
    if handler == _HANDLER_EL8:
        return pip_setup_commands("python3") + [["python3", *setup]]
    if handler == _HANDLER_EL9:
        return pip_setup_commands("python3.12") + [["python3.12", *setup]]
    return [["python3", *setup]]


def conf_key_pattern(key: str) -> str:
    """Translate bash's grep/sed pattern "^$key=" into a Python regex.

    In the bash pattern, the "." in a key such as "OS.EnableRDMA" is a regex
    wildcard (it matches any character), and the keys contain only letters and
    dots. re.escape() escapes the dots; we then turn them back into wildcards
    so the pattern matches exactly what grep and sed match.
    """
    return "^" + re.escape(key).replace(r"\.", ".") + "="


def update_waagent_conf_text(text: str, key: str, value: str) -> tuple[str, int]:
    """Mirror bash's update_waagent_conf on the file's text.

    Returns (new text, number of lines replaced). 0 replaced means the line
    "key=value\\n" was appended (bash's `echo "$key=$value" >>`). Like sed,
    every line starting with "key=" is replaced as a whole, and a file
    without a final newline is left without one.
    """
    # re.MULTILINE makes ^ match at the start of every line, not only the
    # start of the text. "." doesn't match "\n", so ".*" stops at the end of
    # the line, like sed's ".*". subn() returns the new text and how many
    # replacements it made.
    # A function as the replacement makes re.subn use its result literally.
    pattern = re.compile(conf_key_pattern(key) + ".*", re.MULTILINE)
    new_text, count = pattern.subn(lambda _match: f"{key}={value}", text)
    if count:
        return new_text, count
    return text + f"{key}={value}\n", 0


def retarget_exec_start(text: str) -> tuple[str, int]:
    """Mirror the EL9 sed edit of waagent.service. Returns (new text, lines changed).

    sed -i 's|^ExecStart=/usr/bin/python3 -u /usr/sbin/waagent -daemon|
              ExecStart=/usr/bin/python3.12 -u /usr/sbin/waagent -daemon|'

    An already-edited line ("python3.12 -u") doesn't match, because the text
    after "python3" must be " -u". So a rerun changes nothing.
    """
    pattern = re.compile("^" + re.escape(_EXEC_START_OLD), re.MULTILINE)
    return pattern.subn(lambda _match: _EXEC_START_NEW, text)


def parse_waagent_version(output: str) -> tuple[str, str]:
    """Extract (WAAGENT, WAAGENT_EXTENSIONS) from `waagent --version` output.

    Typical output:

        WALinuxAgent-2.15.0.1 running on ubuntu 24.04
        Python: 3.12.3
        Goal state agent: 2.15.0.1

    WAAGENT mirrors `head -n 1 | awk -F' ' '{print $1}' | awk -F- '{print $2}'`:
    the first word of line 1, then the part after its first "-" ("2.15.0.1").
    WAAGENT_EXTENSIONS mirrors `sed '3q;d' | awk -F' ' '{print $4}'`: the
    fourth word of line 3 ("2.15.0.1"). A missing piece gives "", as awk does.
    """
    lines = output.split("\n")

    # str.split() with no argument splits on runs of spaces/tabs and ignores
    # leading ones, like awk's default field splitting (-F' ').
    first_words = lines[0].split()
    agent = ""
    if first_words:
        # split("-") keeps empty parts, like awk -F-; [1] is awk's $2.
        dash_parts = first_words[0].split("-")
        if len(dash_parts) > 1:
            agent = dash_parts[1]

    extensions = ""
    if len(lines) > 2:
        third_words = lines[2].split()
        if len(third_words) > 3:
            extensions = third_words[3]
    return agent, extensions


def version_command(distribution: str) -> list[str]:
    """How bash runs `waagent --version`: with python3.12 on EL9."""
    if is_el9(distribution):
        return ["python3.12", "-u", "/usr/sbin/waagent", "--version"]
    return ["waagent", "--version"]


def restart_service(distribution: str) -> str | None:
    """The agent's systemd service bash restarts, or None if it restarts none."""
    if "ubuntu" in distribution:
        return "walinuxagent"
    if (distribution.startswith(("almalinux", "rocky", "rhel"))
            or distribution == _AZURE_LINUX):
        return "waagent"
    return None


def backup_timestamp() -> str:
    """The suffix bash's `date +%F_%T` gives, e.g. "2026-10-29_15:07:42".

    A separate function so tests can replace it with a fixed value.
    """
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


# --------------------------------------------------------------------------- #
# Steps. Each returns True on success, or False after logging why not.
# --------------------------------------------------------------------------- #
def _run_all(commands, env: dict[str, str], cwd: str | None = None) -> bool:
    """Run commands in order, stopping at the first failure (like `set -e`).

    exec_program already logs a failed command with its output; we add the
    consequence.
    """
    for command in commands:
        rc = exec_program(command, _OP, cwd=cwd, env=env)
        if rc != 0:
            log_error(_OP, f"`{' '.join(command)}` failed; stopping the waagent install",
                      command=command[0], exit_code=rc)
            return False
    return True


def _read_text(path: str) -> str:
    # newline="" keeps "\r\n" as it is (no newline translation), and
    # surrogateescape round-trips any non-UTF-8 bytes unchanged, so writing the
    # text back only changes what sed would change.
    with open(path, encoding="utf-8", errors="surrogateescape", newline="") as handle:
        return handle.read()


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as handle:
        handle.write(text)


def _set_conf(key: str, value: str) -> bool:
    """Set one waagent.conf key (bash's update_waagent_conf) and log what changed."""
    try:
        try:
            text = _read_text(_WAAGENT_CONF)
            existed = True
        except FileNotFoundError:
            # Bash's grep fails on a missing file, so it takes the `echo >>`
            # branch, which creates the file.
            text, existed = "", False
        new_text, replaced = update_waagent_conf_text(text, key, value)
        _write_text(_WAAGENT_CONF, new_text)
    except OSError as exc:
        log_error(_OP, f"could not set {key}={value} in {_WAAGENT_CONF}: {exc}",
                  path=_WAAGENT_CONF, key=key, value=value)
        return False

    if replaced:
        action = "replaced"
    elif existed:
        action = "appended"
    else:
        action = "created"
    log_info(_OP, f"Set {key}={value} in {_WAAGENT_CONF} ({action})",
             path=_WAAGENT_CONF, key=key, value=value, action=action,
             lines_replaced=replaced)
    return True


def _apply_conf(settings) -> bool:
    for key, value in settings:
        if not _set_conf(key, value):
            return False
    return True


def _retarget_service_file() -> bool:
    """Back up waagent.service and point its ExecStart at python3.12 (EL9)."""
    backup = f"{_WAAGENT_SERVICE_FILE}.bak.{backup_timestamp()}"
    try:
        # shutil.copy copies the contents and the permission bits, like `cp`.
        shutil.copy(_WAAGENT_SERVICE_FILE, backup)
        log_info(_OP, f"Backed up {_WAAGENT_SERVICE_FILE} to {backup}",
                 path=_WAAGENT_SERVICE_FILE, backup=backup)
        text = _read_text(_WAAGENT_SERVICE_FILE)
        new_text, changed = retarget_exec_start(text)
        _write_text(_WAAGENT_SERVICE_FILE, new_text)
    except OSError as exc:
        log_error(_OP, f"could not edit {_WAAGENT_SERVICE_FILE}: {exc}",
                  path=_WAAGENT_SERVICE_FILE, backup=backup)
        return False

    if changed:
        log_info(_OP, f"Edited {_WAAGENT_SERVICE_FILE} in place: ExecStart now runs "
                      "waagent with python3.12",
                 path=_WAAGENT_SERVICE_FILE, action="edited", lines_changed=changed)
    else:
        # sed is silent when nothing matches; the unit keeps whatever it had.
        log_warn(_OP, f"No '{_EXEC_START_OLD}' line in {_WAAGENT_SERVICE_FILE}; "
                      "ExecStart left unchanged (bash's sed does the same)",
                 path=_WAAGENT_SERVICE_FILE, lines_changed=0)
    return True


def _install_from_source(env: dict[str, str], distribution: str) -> bool:
    """Everything bash's non-Azure-Linux branch does, in order."""
    config = config_for("waagent", env)
    # isinstance() guards against a missing (None) or oddly shaped config.
    if not isinstance(config, dict) or not config.get("version") or not config.get("sha256"):
        log_error(_OP, "could not resolve waagent version and sha256 from versions.json",
                  distro=distribution)
        return False
    version = config["version"]
    url = _DOWNLOAD_URL.format(version=version)
    handler = install_handler(distribution)

    log_info(_OP, f"Installing WALinuxAgent {version} from source",
             version=version, source=url, distro=distribution, handler=handler,
             sku=env.get("SKU", ""),
             node_type=env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE)

    # ---- 1. pip set-up for python3 on the RHEL family ---------------------
    if is_rhel_family(distribution):
        log_info(_OP, "RHEL family: making pip, setuptools and distro available "
                      "for python3", distro=distribution)
        if not _run_all(pip_setup_commands("python3"), env):
            return False

    # ---- 2. Download and unpack the release ------------------------------
    try:
        shutil.rmtree(_WORK_DIR, ignore_errors=True)
        tarball = download_and_verify(url, config["sha256"], dest_dir=_WORK_DIR)
    # download() raises OSError (network/file errors); a bad hash raises ChecksumError.
    except (OSError, ChecksumError) as exc:
        log_error(_OP, f"download/verify of WALinuxAgent {version} failed: {exc}",
                  source=url)
        return False

    if not _run_all([["tar", "-xvf", os.path.basename(tarball)]], env, cwd=_WORK_DIR):
        return False
    source_dir = os.path.join(_WORK_DIR, f"WALinuxAgent-{version}")
    if not os.path.isdir(source_dir):
        # Bash's `pushd WALinuxAgent-${V}/` would fail here.
        log_error(_OP, f"{tarball} did not unpack into {source_dir}")
        return False

    # ---- 3. setup.py install (per distro) ---------------------------------
    commands = setup_commands(handler)
    log_info(_OP, f"Using the {handler} setup path: {' '.join(commands[-1])}",
             handler=handler, python=commands[-1][0], distro=distribution)
    if not _run_all(commands, env, cwd=source_dir):
        return False

    if handler == _HANDLER_EL9:
        # The unit file setup.py registered runs waagent with python3; on EL9
        # the agent was installed for python3.12, so the unit is rewritten.
        if not _run_all([["systemctl", "stop", "waagent"],
                         ["systemctl", "disable", "waagent"]], env, cwd=source_dir):
            return False
        if not _retarget_service_file():
            return False
        if not _run_all([["systemctl", "daemon-reexec"],
                         ["systemctl", "enable", "waagent"]], env, cwd=source_dir):
            return False
        log_info(_OP, "waagent service re-enabled with its python3.12 ExecStart",
                 service="waagent", action="enabled")

    # ---- 4. waagent.conf ---------------------------------------------------
    sku = env.get("SKU", "")
    node_type = env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE
    uses_ipoib = sku_uses_ipoib(sku, node_type)
    mode = sku_network_mode(sku, node_type)
    if uses_ipoib:
        log_info(_OP, "Target uses IPoIB: OS.EnableRDMA=y will be set",
                 sku=sku, node_type=node_type, network_mode=mode, uses_ipoib=True)
    else:
        log_info(_OP, "Target doesn't use IPoIB: OS.EnableRDMA is left as it is",
                 sku=sku, node_type=node_type, network_mode=mode, uses_ipoib=False)

    settings = _CONF_BEFORE_RDMA + ((_RDMA_SETTING,) if uses_ipoib else ()) + _CONF_AFTER_RDMA
    if not _apply_conf(settings):
        return False

    # Bash's `rm -rf WALinuxAgent-${V}`. A leftover is harmless (the next run
    # empties _WORK_DIR first), so a failed cleanup doesn't fail the step.
    shutil.rmtree(source_dir, ignore_errors=True)
    log_debug(_OP, f"Removed {source_dir}; {tarball} is kept, as in bash")
    return True


def _record_versions(env: dict[str, str], distribution: str) -> tuple[str, str] | None:
    """Run `waagent --version` and record WAAGENT and WAAGENT_EXTENSIONS."""
    command = version_command(distribution)
    rc, output = run_capture(command, _OP, env=env)
    if rc != 0:
        log_error(_OP, "could not read the installed waagent version",
                  command=" ".join(command), exit_code=rc)
        return None
    log_debug(_OP, f"`{' '.join(command)}` output:\n{output}")

    agent, extensions = parse_waagent_version(output)
    if not agent or not extensions:
        log_error(_OP, f"unexpected `{' '.join(command)}` output: {output.strip()!r}",
                  waagent=agent, waagent_extensions=extensions)
        return None

    for component, version in (("WAAGENT", agent), ("WAAGENT_EXTENSIONS", extensions)):
        try:
            write_component_version(component, version)
        except OSError as exc:
            log_error(_OP, f"could not record {component}={version}: {exc}",
                      component=component, version=version)
            return None
        log_info(_OP, f"Recorded {component}={version}", component=component,
                 version=version)
    return agent, extensions


def install(env: dict[str, str]) -> int:
    """Install/update WALinuxAgent, configure it, record its versions, restart it.

    `env` is the build environment made by component_env() in build_config.py.
    Uses DISTRIBUTION, SKU, TARGET_NODE_TYPE and the values config_for() reads.

    Returns 0 on success and 3 on failure.
    """
    distribution = env.get("DISTRIBUTION", "")

    # ---- 1. Install and configure -----------------------------------------
    if distribution == _AZURE_LINUX:
        log_info(_OP, "Updating WALinuxAgent from the Azure Linux repository",
                 distro=distribution, method="dnf update", source="distro repository")
        if not _run_all([["dnf", "update", "-y", "WALinuxAgent"]], env):
            return 3
        # Unlike the other distros, bash doesn't check sku_uses_ipoib here.
        log_info(_OP, "Azure Linux: OS.EnableRDMA=y is set for every SKU and node type",
                 distro=distribution, sku=env.get("SKU", ""),
                 node_type=env.get("TARGET_NODE_TYPE") or DEFAULT_NODE_TYPE)
        if not _apply_conf(_AZURE_LINUX_CONF):
            return 3
    elif not _install_from_source(env, distribution):
        return 3

    # ---- 2. Record versions -------------------------------------------------
    versions = _record_versions(env, distribution)
    if versions is None:
        return 3

    # ---- 3. Reload systemd and restart the agent ---------------------------
    if not _run_all([["systemctl", "daemon-reload"]], env):
        return 3
    service = restart_service(distribution)
    if service is None:
        log_info(_OP, f"No agent restart rule for {distribution!r}; not restarting "
                      "(same as bash)", distro=distribution)
    else:
        log_info(_OP, f"Restarting {service}", service=service, action="restarted")
        if not _run_all([["systemctl", "restart", service]], env):
            return 3

    log_info(_OP, f"Waiting {_WAIT_SECONDS}s so waagent doesn't take the package "
                  "manager locks", seconds=_WAIT_SECONDS)
    time.sleep(_WAIT_SECONDS)

    agent, extensions = versions
    log_info(_OP, f"WALinuxAgent {agent} installed and configured",
             waagent=agent, waagent_extensions=extensions, distro=distribution,
             conf=_WAAGENT_CONF, service=service or "")
    return 0
