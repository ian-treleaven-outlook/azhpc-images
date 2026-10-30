"""install_monitoring_tools.py — Python port of components/install_monitoring_tools.sh.

Installs Moneo (Azure's HPC monitoring tool) into /opt/azurehpc/tools/Moneo:

  1. Make sure pip is usable (_PIP_PACKAGE_OPTIONS, bash's exact distro
     matches): python3-pip from dnf on azurelinux3.0, `--only-upgrade` from
     apt on ubuntu24.04/26.04, else `python3 -m pip install --upgrade pip`.
  2. Download v<version>.tar.gz ("moneo" in versions.json) into
     /opt/azurehpc/tools and verify it. The tarball is left there, as in bash.
  3. Unpack it into Moneo/ and `chmod 777 Moneo`.
  4. Run Moneo's own linux_service/configure_service.sh.
  5. Add the `moneo` alias to /etc/bash.bashrc unless the line is there.
  6. Record MONEO=<version>.

Kept bash quirks
----------------
  * `mkdir Moneo && tar ...`: `set -e` ignores a failure that isn't the last
    command of an `&&` list, so on a rerun (Moneo/ exists) bash skips tar and
    carries on with the old contents. So does this step, with a warning. A
    failing tar stops the step.
  * `./configure_service.sh` is run directly, so it needs its exec bit (the
    release tarball has it, and tar keeps it). If it can't run, the error
    says whether it exists and is executable.
  * `grep -qxF` compares whole lines byte for byte, split only on "\\n" (a
    "\\r\\n" line doesn't match); a missing or unreadable bashrc counts as
    "not there", and `echo ... >>` then appends (creating the file).

Deliberate differences
----------------------
  * A missing moneo version or sha256 fails before anything is installed
    (bash would fail later, downloading "vnull.tar.gz").
  * The tarball is downloaded straight into /opt/azurehpc/tools (bash
    downloads into its current folder, then moves it there).
  * python3-pip goes through install_packages(): `apt-get install -y -o
    Dpkg::Use-Pty=0 --only-upgrade python3-pip` / `dnf install -y python3-pip`.
"""

from __future__ import annotations

import os

from utils.component_config import require_config, write_component_version
from utils.distro import Distro
from utils.download import fetch_verified
from utils.files import extract_tarball, write_file
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import run

_OP = "install-monitoring-tools"

# Paths are module constants so tests can point them at a temporary directory.
_MONITOR_DIR = "/opt/azurehpc/tools"
_MONEO_DIR_NAME = "Moneo"                       # under _MONITOR_DIR
_SERVICE_DIR = os.path.join("Moneo", "linux_service")
_CONFIGURE_SCRIPT = "./configure_service.sh"    # run from _SERVICE_DIR
_BASHRC = "/etc/bash.bashrc"

_URL = "https://github.com/Azure/Moneo/archive/refs/tags/{tarball}"
_ALIAS = "alias moneo='python3 /opt/azurehpc/tools/Moneo/moneo.py'"

# Where python3-pip comes from the package manager, and with which options;
# on any other distro pip upgrades itself.
_PIP_PACKAGE = "python3-pip"
_PIP_PACKAGE_OPTIONS = {
    "azurelinux3.0": [],
    "ubuntu24.04": ["--only-upgrade"],
    "ubuntu26.04": ["--only-upgrade"],
}
_PIP_SELF_UPGRADE = ["python3", "-m", "pip", "install", "--upgrade", "pip"]


# ---- Pure helpers -------------------------------------------------------------
def tarball_name(version: str) -> str:
    return f"v{version}.tar.gz"


def pip_package_options(distribution: str) -> list[str] | None:
    """install_packages() options for python3-pip, or None where bash runs
    `python3 -m pip install --upgrade pip` instead."""
    options = _PIP_PACKAGE_OPTIONS.get(distribution)
    return None if options is None else list(options)


def has_line(path: str, line: str) -> bool:
    """bash `grep -qxF "<line>" <path>`: True if a whole line equals `line`."""
    try:
        with open(path, "rb") as handle:
            return line.encode("utf-8") in handle.read().split(b"\n")
    except FileNotFoundError:
        return False
    except OSError as exc:
        # grep exits 2 and `if !` treats that as "not there".
        log_warn(_OP, f"could not read {path} ({exc}); appending the line, as "
                      "bash does when grep fails", path=path)
        return False


# ---- Steps ----------------------------------------------------------------------
# TODO(ADO 1001): toolkit candidate: ensure_line(path, line, op) -> bool
# (bash `grep -qxF line file || echo line >> file`).
def add_line_once(path: str, line: str) -> bool:
    """`if ! grep -qxF "<line>" <path>; then echo "<line>" >> <path>; fi`."""
    if has_line(path, line):
        log_info(_OP, f"{path} already has the line", path=path, action="unchanged",
                 line=line)
        return True
    return write_file(path, line + "\n", _OP, append=True)


def install_deps(env: dict[str, str]) -> bool:
    """Step 1: make sure pip is usable."""
    distro = Distro.from_env(env)
    options = pip_package_options(distro.name)
    if options is None:
        log_info(_OP, "Preparing pip: upgrading it with pip itself", distro=distro.name)
        return run(_PIP_SELF_UPGRADE, _OP, env=env)
    log_info(_OP, f"Preparing pip: installing {_PIP_PACKAGE} from the distro",
             distro=distro.name, options=" ".join(options))
    return install_packages([_PIP_PACKAGE], env, _OP, options=options)


def _unpack(tarball: str, env: dict[str, str]) -> bool:
    """Step 3: `mkdir Moneo && tar -xvf $TARBALL --strip-components=1 -C Moneo`,
    then `chmod 777 Moneo`, in _MONITOR_DIR."""
    moneo_dir = os.path.join(_MONITOR_DIR, _MONEO_DIR_NAME)
    try:
        os.mkdir(moneo_dir)              # like `mkdir` without -p: fails if it exists
    except FileExistsError:
        log_warn(_OP, f"{moneo_dir} already exists (rerun?); not extracting {tarball} "
                      "and using its current contents, as bash does", path=moneo_dir)
    except OSError as exc:
        # bash skips tar, then `chmod 777 Moneo` fails and stops the script.
        log_error(_OP, f"could not create {moneo_dir}: {exc}", path=moneo_dir)
        return False
    else:
        if not extract_tarball(tarball, _OP, dest=_MONEO_DIR_NAME, strip_components=1,
                               cwd=_MONITOR_DIR, env=env):
            return False
    try:
        os.chmod(moneo_dir, 0o777)
    except OSError as exc:
        log_error(_OP, f"could not chmod 777 {moneo_dir}: {exc}", path=moneo_dir)
        return False
    return True


def _configure_service(env: dict[str, str]) -> bool:
    """Step 4: `pushd Moneo/linux_service; ./configure_service.sh; popd`."""
    service_dir = os.path.join(_MONITOR_DIR, _SERVICE_DIR)
    if not os.path.isdir(service_dir):
        # bash: `pushd` fails and set -e stops the script.
        log_error(_OP, f"{service_dir} does not exist; the Moneo tarball has no "
                       "linux_service folder", path=service_dir)
        return False
    if run([_CONFIGURE_SCRIPT], _OP, cwd=service_dir, env=env):
        return True
    script = os.path.join(service_dir, os.path.basename(_CONFIGURE_SCRIPT))
    exists = os.path.isfile(script)
    executable = exists and os.access(script, os.X_OK)
    why = ("it does not exist" if not exists else
           "it is not executable (bash runs ./configure_service.sh, which needs "
           "the exec bit)" if not executable else "it exited with an error")
    log_error(_OP, f"{_CONFIGURE_SCRIPT} failed in {service_dir}: {why}",
              path=script, exists=exists, executable=executable)
    return False


def _record_version(version: str) -> bool:
    try:
        write_component_version("MONEO", version)
    except OSError as exc:
        log_error(_OP, f"could not record MONEO={version}: {exc}")
        return False
    log_info(_OP, f"Recorded MONEO version {version}", component="MONEO",
             version=version)
    return True


def install(env: dict[str, str]) -> int:
    """Install Moneo and record its version. Returns 0 or 3."""
    cfg = require_config("moneo", env, _OP, "version", "sha256")
    if cfg is None:
        return 3
    version = str(cfg["version"])
    url = _URL.format(tarball=tarball_name(version))
    moneo_dir = os.path.join(_MONITOR_DIR, _MONEO_DIR_NAME)
    log_info(_OP, f"Installing Moneo {version}", version=version, source=url,
             distro=env.get("DISTRIBUTION", ""), path=moneo_dir)

    if not install_deps(env):
        return 3
    if fetch_verified(url, cfg["sha256"], _OP, _MONITOR_DIR) is None:
        return 3
    if not _unpack(tarball_name(version), env):
        return 3
    if not _configure_service(env):
        return 3
    if not add_line_once(_BASHRC, _ALIAS):
        return 3
    if not _record_version(version):
        return 3

    log_info(_OP, f"Moneo {version} installed", version=version, path=moneo_dir,
             tarball_left=os.path.join(_MONITOR_DIR, tarball_name(version)))
    return 0
