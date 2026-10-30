"""install_monitoring_tools.py — Python port of components/install_monitoring_tools.sh.

What this step does
-------------------
Installs Moneo (Azure's HPC monitoring tool) into /opt/azurehpc/tools/Moneo:

  1. Resolve the "moneo" version and sha256 from versions.json.
  2. Make sure pip is usable (per distro, exact matches as in bash):
       azurelinux3.0             `dnf install -y python3-pip`
       ubuntu24.04, ubuntu26.04  `apt-get install -y --only-upgrade python3-pip`
       anything else             `python3 -m pip install --upgrade pip`
  3. Download the release tarball v<version>.tar.gz into /opt/azurehpc/tools
     and check its sha256.
  4. In /opt/azurehpc/tools: create Moneo/, extract the tarball into it with
     `tar -xvf v<version>.tar.gz --strip-components=1 -C Moneo`, and
     `chmod 777 Moneo`.
  5. In Moneo/linux_service: run `./configure_service.sh` (Moneo's own script;
     it must be executable, which it is in the release tarball, mode 755).
  6. Append the `moneo` alias to /etc/bash.bashrc unless that exact line is
     already there.
  7. Record MONEO=<version> in the component-versions file.

The tarball is left in /opt/azurehpc/tools, as in bash.

The original bash:

    moneo_metadata=$(get_component_config "moneo")
    MONEO_VERSION=$(jq -r '.version' <<< $moneo_metadata)
    MONEO_SHA256=$(jq -r '.sha256' <<< $moneo_metadata)

    if [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
        dnf install -y python3-pip
    elif [[ $DISTRIBUTION == "ubuntu24.04" ]] || [[ $DISTRIBUTION == "ubuntu26.04" ]]; then
        apt-get install -y --only-upgrade python3-pip
    else
        python3 -m pip install --upgrade pip
    fi

    MONITOR_DIR=/opt/azurehpc/tools
    TARBALL="v${MONEO_VERSION}.tar.gz"
    MONEO_DOWNLOAD_URL=https://github.com/Azure/Moneo/archive/refs/tags/${TARBALL}
    download_and_verify ${MONEO_DOWNLOAD_URL} ${MONEO_SHA256} $MONITOR_DIR

    pushd $MONITOR_DIR
        mkdir Moneo && tar -xvf $TARBALL --strip-components=1 -C Moneo
        chmod 777 Moneo
        pushd Moneo/linux_service
            ./configure_service.sh
        popd
    popd

    if ! grep -qxF "alias moneo='python3 /opt/azurehpc/tools/Moneo/moneo.py'" /etc/bash.bashrc; then
        echo "alias moneo='python3 /opt/azurehpc/tools/Moneo/moneo.py'" >> /etc/bash.bashrc
    fi

    write_component_version "MONEO" ${MONEO_VERSION}

Bash details mirrored
---------------------
  * `mkdir Moneo && tar ...`: `set -e` does not stop the script when the
    first command of an `&&` list fails. So if Moneo/ already exists (a
    rerun), mkdir fails, tar is skipped, and the step carries on with the old
    contents. install() does the same, with a warning. A failing tar does
    stop bash and returns 3 here.
  * `./configure_service.sh` is run directly, so it needs its exec bit (bash
    fails with "Permission denied" otherwise). The release tarball has it,
    and real `tar` keeps it. If the script can't be run, the error says
    whether it exists and is executable.
  * `grep -qxF` compares whole lines byte for byte, split only on "\\n" (a
    line ending in "\\r\\n" doesn't match). A missing or unreadable bashrc
    counts as "not there", and `echo ... >>` then appends the line (creating
    the file if needed). ensure_line() does exactly that.

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets this module's install() as the `action=` of the
"install-monitoring-tools" Step. Deleting `action=` makes the build run the
bash script again, which is why the .sh file is kept. install() returns 0 for
success or 3 for failure; it logs problems and never lets an exception escape.
Every command runs with the build environment (`env`), in bash's order and
working directory.

Deliberate differences from bash
--------------------------------
  * A missing moneo version or sha256 fails before anything is installed;
    bash would carry on and fail at the download of "vnull.tar.gz".
  * Bash's download_and_verify downloads into the current directory and then
    moves the file into /opt/azurehpc/tools; we download straight there.
  * python3-pip is installed with install_packages(), which writes
    `apt-get install -y -o Dpkg::Use-Pty=0 --only-upgrade python3-pip` /
    `dnf install -y python3-pip` (the documented normalisation).
"""

from __future__ import annotations

import os

from utils.component_config import config_for, write_component_version
from utils.download import download_and_verify
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import exec_program

# The Step's `op` label in plans/steps.py; used on every log line.
_OP = "install-monitoring-tools"

# Paths. Module-level so tests can point them at a temporary directory.
_MONITOR_DIR = "/opt/azurehpc/tools"
_MONEO_DIR_NAME = "Moneo"                       # under _MONITOR_DIR
_SERVICE_DIR = os.path.join("Moneo", "linux_service")
_CONFIGURE_SCRIPT = "./configure_service.sh"    # run from _SERVICE_DIR
_BASHRC = "/etc/bash.bashrc"

_URL = "https://github.com/Azure/Moneo/archive/refs/tags/{tarball}"
_ALIAS = "alias moneo='python3 /opt/azurehpc/tools/Moneo/moneo.py'"

# Step 2, bash's if/elif/else (exact matches). On these distros python3-pip
# comes from the package manager, with these install options; on any other,
# pip upgrades itself.
_PIP_PACKAGE = "python3-pip"
_PIP_PACKAGE_OPTIONS = {
    "azurelinux3.0": [],
    "ubuntu24.04": ["--only-upgrade"],
    "ubuntu26.04": ["--only-upgrade"],
}
_PIP_SELF_UPGRADE = ["python3", "-m", "pip", "install", "--upgrade", "pip"]


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
def pip_package_options(distribution: str) -> list[str] | None:
    """The install_packages() options for python3-pip on `distribution`, or
    None where bash runs `python3 -m pip install --upgrade pip` instead."""
    options = _PIP_PACKAGE_OPTIONS.get(distribution)
    # list(...) returns a copy, so a caller can't change the constant.
    return None if options is None else list(options)


def ensure_line(path: str, line: str) -> bool:
    """Append `line` to `path` unless a line equal to it is already there.

    Mirrors  if ! grep -qxF "<line>" <path>; then echo "<line>" >> <path>; fi
    Returns True if the line was appended, False if it was already present.
    Raises OSError if the append fails (bash's `>>` would stop the script).
    """
    wanted = line.encode("utf-8")
    try:
        with open(path, "rb") as handle:
            # grep splits on b"\n" only; "-x" means the whole line must match.
            if wanted in handle.read().split(b"\n"):
                return False
    except FileNotFoundError:
        pass
    except OSError as exc:
        # grep exits 2, `!` makes that true, so bash goes on to append.
        log_warn(_OP, f"could not read {path} ({exc}); appending the line, as "
                      "bash does when grep fails", path=path)
    with open(path, "ab") as handle:
        handle.write(wanted + b"\n")
    return True


# --------------------------------------------------------------------------- #
# Steps. Each returns True on success or False after logging why it failed.
# --------------------------------------------------------------------------- #
def _run(command: list[str], env: dict[str, str], what: str,
         cwd: str | None = None) -> bool:
    """Run one command; log the consequence if it fails (exec_program logs the rest)."""
    rc = exec_program(command, _OP, cwd=cwd, env=env)
    if rc != 0:
        log_error(_OP, f"{what} failed; Moneo is not installed",
                  command=" ".join(command), exit_code=rc, cwd=cwd or "")
        return False
    return True


def install_deps(env: dict[str, str]) -> bool:
    """Make sure pip is usable before installing Moneo (step 2)."""
    distribution = env.get("DISTRIBUTION", "")
    options = pip_package_options(distribution)
    if options is None:
        log_info(_OP, "Preparing pip: upgrading it with pip itself",
                 distro=distribution, command=" ".join(_PIP_SELF_UPGRADE))
        return _run(_PIP_SELF_UPGRADE, env, "the pip dependency command")
    log_info(_OP, f"Preparing pip: installing {_PIP_PACKAGE} from the distro",
             distro=distribution, package=_PIP_PACKAGE, options=" ".join(options))
    if not install_packages([_PIP_PACKAGE], env, _OP, options=options):
        log_error(_OP, f"installing {_PIP_PACKAGE} failed; Moneo is not installed",
                  package=_PIP_PACKAGE)
        return False
    return True


def _unpack(tarball_name: str, env: dict[str, str]) -> bool:
    """`mkdir Moneo && tar -xvf $TARBALL --strip-components=1 -C Moneo`, then
    `chmod 777 Moneo`, all in _MONITOR_DIR."""
    moneo_dir = os.path.join(_MONITOR_DIR, _MONEO_DIR_NAME)
    try:
        os.mkdir(moneo_dir)              # like `mkdir` without -p: fails if it exists
        created = True
    except FileExistsError:
        created = False
    except OSError as exc:
        # bash skips tar, then `chmod 777 Moneo` fails and stops the script.
        log_error(_OP, f"could not create {moneo_dir}: {exc}", path=moneo_dir)
        return False

    if created:
        log_info(_OP, f"Created {moneo_dir}", path=moneo_dir, action="created")
        if not _run(["tar", "-xvf", tarball_name, "--strip-components=1",
                     "-C", _MONEO_DIR_NAME], env, "extracting the Moneo tarball",
                    cwd=_MONITOR_DIR):
            return False
        log_info(_OP, f"Extracted {tarball_name} into {moneo_dir}",
                 tarball=tarball_name, path=moneo_dir)
    else:
        log_warn(_OP, f"{moneo_dir} already exists (rerun?); not extracting "
                      f"{tarball_name} and using its current contents, as bash "
                      "does (`mkdir Moneo && tar ...`)", path=moneo_dir)

    try:
        os.chmod(moneo_dir, 0o777)
    except OSError as exc:
        log_error(_OP, f"could not chmod 777 {moneo_dir}: {exc}", path=moneo_dir)
        return False
    log_info(_OP, f"chmod 777 {moneo_dir}", path=moneo_dir, mode="777")
    return True


def _configure_service(env: dict[str, str]) -> bool:
    """`pushd Moneo/linux_service; ./configure_service.sh; popd`."""
    service_dir = os.path.join(_MONITOR_DIR, _SERVICE_DIR)
    if not os.path.isdir(service_dir):
        # bash: `pushd` fails and set -e stops the script.
        log_error(_OP, f"{service_dir} does not exist; the Moneo tarball has no "
                       "linux_service folder", path=service_dir)
        return False

    script = os.path.join(service_dir, os.path.basename(_CONFIGURE_SCRIPT))
    log_info(_OP, "Running Moneo's configure_service.sh", path=script,
             cwd=service_dir)
    rc = exec_program([_CONFIGURE_SCRIPT], _OP, cwd=service_dir, env=env)
    if rc != 0:
        exists = os.path.isfile(script)
        executable = exists and os.access(script, os.X_OK)
        if not exists:
            why = "it does not exist"
        elif not executable:
            why = ("it is not executable (bash runs ./configure_service.sh, which "
                   "needs the exec bit)")
        else:
            why = "it exited with an error"
        log_error(_OP, f"{_CONFIGURE_SCRIPT} failed in {service_dir}: {why}",
                  path=script, exit_code=rc, exists=exists, executable=executable)
        return False
    log_info(_OP, "Configured the Moneo services", path=script)
    return True


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def install(env: dict[str, str]) -> int:
    """Install Moneo and record its version. Returns 0 or 3."""
    distribution = env.get("DISTRIBUTION", "")
    cfg = config_for("moneo", env)
    if not isinstance(cfg, dict) or not cfg.get("version") or not cfg.get("sha256"):
        log_error(_OP, "could not resolve moneo version and sha256 from versions.json",
                  distro=distribution)
        return 3
    version = str(cfg["version"])
    tarball_name = f"v{version}.tar.gz"
    url = _URL.format(tarball=tarball_name)
    log_info(_OP, f"Installing Moneo {version}", version=version, source=url,
             distro=distribution, path=os.path.join(_MONITOR_DIR, _MONEO_DIR_NAME))

    try:
        # ---- 2. pip -------------------------------------------------------------
        if not install_deps(env):
            return 3

        # ---- 3. download + verify ------------------------------------------------
        try:
            tarball = download_and_verify(url, cfg["sha256"], dest_dir=_MONITOR_DIR)
        except Exception as exc:  # network, checksum or disk errors
            log_error(_OP, f"download/verify of the Moneo tarball failed: {exc}",
                      url=url)
            return 3
        log_info(_OP, f"Downloaded {tarball}", path=str(tarball), url=url)

        # ---- 4./5. extract, chmod, configure ------------------------------------
        if not _unpack(tarball_name, env):
            return 3
        if not _configure_service(env):
            return 3

        # ---- 6. alias -------------------------------------------------------------
        if ensure_line(_BASHRC, _ALIAS):
            log_info(_OP, f"Appended the moneo alias to {_BASHRC}", path=_BASHRC,
                     action="appended", line=_ALIAS)
        else:
            log_info(_OP, f"{_BASHRC} already has the moneo alias", path=_BASHRC,
                     action="unchanged")

        # ---- 7. version -------------------------------------------------------------
        write_component_version("MONEO", version)
    except OSError as exc:
        log_error(_OP, f"file operation failed: {exc}",
                  path=getattr(exc, "filename", None) or "")
        return 3
    log_info(_OP, f"Recorded MONEO version {version}", component="MONEO",
             version=version)

    log_info(_OP, f"Moneo {version} installed", version=version,
             path=os.path.join(_MONITOR_DIR, _MONEO_DIR_NAME),
             tarball_left=str(tarball))
    return 0
