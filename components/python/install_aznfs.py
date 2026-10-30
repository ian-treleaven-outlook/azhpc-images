"""install_aznfs.py — Python port of components/install_aznfs.sh.

What this step does
-------------------
Installs the AZNFS mount helper (Azure Blob NFS mounts). Where Microsoft's
package repository (PMC) publishes an aznfs package, it comes from there:

  * Ubuntu (`$DISTRIBUTION == *"ubuntu"*`)       `apt install -y aznfs`
  * AlmaLinux / Rocky / RHEL (contains
    "almalinux", "rocky" or "rhel")              `dnf install -y aznfs`
  * anything else (Azure Linux 3.0 today)        the upstream AZNFS-mount
    installer script: download the release's aznfs_install.sh, check its
    sha256, replace the first "yum" on each line with "dnf", run it with
    bash, then delete it.

AZNFS_NONINTERACTIVE_INSTALL=1 is set for every command, so the package's
setup never stops at a TTY prompt during a Packer build. No version is
recorded (bash records none either).

Upstream (#614, the Ubuntu 26.04 support) turned the Azure Linux branch into
the `else` branch for every distro without a PMC package, and added the
`rm -f aznfs_install.sh` at the end.

The original bash:

    export AZNFS_NONINTERACTIVE_INSTALL=1

    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        apt install -y aznfs
    elif [[ $DISTRIBUTION == *"almalinux"* || $DISTRIBUTION == *"rocky"* || $DISTRIBUTION == *"rhel"* ]]; then
        dnf install -y aznfs
    else
        aznfs_metadata=$(get_component_config "aznfs")
        AZNFS_VERSION=$(jq -r '.version' <<< $aznfs_metadata)
        AZNFS_SHA256=$(jq -r '.sha256' <<< $aznfs_metadata)
        AZNFS_DOWNLOAD_URL=https://github.com/Azure/AZNFS-mount/releases/download/${AZNFS_VERSION}/aznfs_install.sh

        download_and_verify $AZNFS_DOWNLOAD_URL $AZNFS_SHA256
        sed -i 's/yum/dnf/' aznfs_install.sh
        bash aznfs_install.sh
        rm -f aznfs_install.sh
    fi

`sed 's/yum/dnf/'` has no /g flag, so it changes only the FIRST "yum" on each
line (and "yum" inside longer words too, e.g. "yummy" -> "dnfmy").
sed_yum_to_dnf() does exactly that; it was compared with real `sed -i` in WSL
on lines with zero, one and several matches, a CRLF line, an empty line and a
last line without a newline.

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets this module's install() as the `action=` of the
"install-aznfs" Step. Deleting `action=` makes the build run the bash script
again, which is why the .sh file is kept. install() returns 0 for success or 3
for failure; it logs problems and never lets an exception escape.

Bash's download_and_verify has no folder argument here, so the installer goes
into the current directory, is run from there and is deleted afterwards. The
port does the same (download_and_verify(dest_dir=".")).

Deliberate differences from bash
--------------------------------
  * A missing aznfs version or sha256 fails before anything is downloaded;
    bash would try to download ".../download/null/aznfs_install.sh".
  * AZNFS_NONINTERACTIVE_INSTALL=1 is passed in the environment of this step's
    commands instead of being exported into the whole build process (bash's
    export also only reaches this script and its children).
  * A leftover aznfs_install.sh from an earlier failed run is overwritten
    (with a warning). Bash's wget would save the new download as
    aznfs_install.sh.1 and then check the old, already-patched file, which
    fails the checksum.
  * Like bash, a failed installer leaves aznfs_install.sh in place (`set -e`
    stops before the `rm -f`); the port logs its path.
"""

from __future__ import annotations

import os
from pathlib import Path

from utils.component_config import config_for
from utils.download import download_and_verify
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.process import exec_program

# The Step's `op` label in plans/steps.py; used on every log line.
_OP = "install-aznfs"

# Substring matches, as in bash ($DISTRIBUTION == *"ubuntu"*).
_APT_DISTROS = ("ubuntu",)
_DNF_DISTROS = ("almalinux", "rocky", "rhel")

# The distro the installer branch was written for; any other distro reaching
# it is allowed (bash's `else`) but warned about.
_INSTALLER_DISTRO = "azurelinux"

_INSTALLER_NAME = "aznfs_install.sh"
_INSTALLER_URL = ("https://github.com/Azure/AZNFS-mount/releases/download/"
                  "{version}/" + _INSTALLER_NAME)

# Where bash's download_and_verify (no folder argument) puts the file: the
# current directory. Module-level so tests can point it at a temporary dir.
_DOWNLOAD_DIR = "."

_NONINTERACTIVE = {"AZNFS_NONINTERACTIVE_INSTALL": "1"}


# --------------------------------------------------------------------------- #
# Pure helpers: they only compute values, so tests call them directly.
# --------------------------------------------------------------------------- #
def install_branch(distribution: str) -> str:
    """Which branch of bash's if/elif/else runs: "apt", "dnf" or "installer"."""
    if any(name in distribution for name in _APT_DISTROS):
        return "apt"
    if any(name in distribution for name in _DNF_DISTROS):
        return "dnf"
    return "installer"


def sed_yum_to_dnf(data: bytes) -> tuple[bytes, int]:
    """`sed 's/yum/dnf/'` on the file's bytes: the first "yum" on each line.

    Returns (new bytes, number of lines changed). Splitting on b"\\n" and
    joining again keeps a missing final newline missing, as sed does.
    """
    lines = data.split(b"\n")
    changed = sum(1 for line in lines if b"yum" in line)
    return b"\n".join(line.replace(b"yum", b"dnf", 1) for line in lines), changed


# --------------------------------------------------------------------------- #
# The branches. Each returns 0 or 3 and logs its own failures.
# --------------------------------------------------------------------------- #
def _install_from_pmc(manager: str, run_env: dict[str, str], distribution: str) -> int:
    """`apt install -y aznfs` or `dnf install -y aznfs`, from Microsoft's repo."""
    command = [manager, "install", "-y", "aznfs"]
    log_info(_OP, f"Installing aznfs from the PMC package repo with {manager}",
             distro=distribution, branch=manager, command=" ".join(command))
    rc = exec_program(command, _OP, env=run_env)
    if rc != 0:
        log_error(_OP, f"'{' '.join(command)}' failed; AZNFS is not installed",
                  command=" ".join(command), exit_code=rc, distro=distribution)
        return 3
    log_info(_OP, "AZNFS installed", distro=distribution, branch=manager,
             package="aznfs")
    return 0


def _install_from_upstream_installer(env: dict[str, str], run_env: dict[str, str],
                                     distribution: str) -> int:
    """Download aznfs_install.sh, patch yum -> dnf, run it, delete it."""
    cfg = config_for("aznfs", env)
    if not isinstance(cfg, dict) or not cfg.get("version") or not cfg.get("sha256"):
        log_error(_OP, "could not resolve aznfs version and sha256 from versions.json",
                  distro=distribution, arch=env.get("ARCHITECTURE", ""))
        return 3
    version = str(cfg["version"])
    url = _INSTALLER_URL.format(version=version)

    if _INSTALLER_DISTRO not in distribution:
        log_warn(_OP, f"{distribution!r} has no PMC aznfs package in bash's list; "
                      "using the upstream installer (bash's `else` branch), which "
                      "detects the distro itself", distro=distribution)
    log_info(_OP, f"Installing AZNFS {version} with the upstream installer",
             distro=distribution, branch="installer", version=version, source=url)

    leftover = os.path.join(_DOWNLOAD_DIR, _INSTALLER_NAME)
    if os.path.exists(leftover):
        log_warn(_OP, f"{leftover} already exists (left by an earlier failed run?); "
                      "overwriting it (bash's wget would save aznfs_install.sh.1 "
                      "and check the old file)", path=leftover)

    try:
        script = Path(download_and_verify(url, cfg["sha256"], dest_dir=_DOWNLOAD_DIR))
    except Exception as exc:  # network, checksum or disk errors
        log_error(_OP, f"download/verify of the AZNFS installer failed: {exc}", url=url)
        return 3

    # sed -i 's/yum/dnf/' aznfs_install.sh
    try:
        patched, changed = sed_yum_to_dnf(script.read_bytes())
        script.write_bytes(patched)
    except OSError as exc:
        log_error(_OP, f"could not patch {script}: {exc}", path=str(script))
        return 3
    log_info(_OP, f"Patched {script}: first 'yum' on each line -> 'dnf' "
                  "(sed -i 's/yum/dnf/')", path=str(script), action="replaced",
             lines_changed=changed)
    if changed == 0:
        log_warn(_OP, f"{script} has no 'yum' to replace; running it unchanged",
                 path=str(script))
    log_debug(_OP, f"Patched installer {script}:\n"
                   f"{patched.decode('utf-8', errors='replace')}", path=str(script))

    rc = exec_program(["bash", str(script)], _OP, env=run_env)
    if rc != 0:
        log_error(_OP, f"the AZNFS installer failed; {script} is left in place, "
                       "as in bash", path=str(script), exit_code=rc, version=version)
        return 3

    # rm -f aznfs_install.sh: a missing file is fine, any other error fails.
    try:
        script.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        log_error(_OP, f"could not remove {script}: {exc}", path=str(script))
        return 3
    log_info(_OP, f"Removed {script}", path=str(script), action="removed")

    log_info(_OP, "AZNFS installed", distro=distribution, branch="installer",
             version=version)
    return 0


def install(env: dict[str, str]) -> int:
    """Install AZNFS from PMC or the upstream installer. Returns 0 or 3."""
    distribution = env.get("DISTRIBUTION", "")
    branch = install_branch(distribution)
    # bash: export AZNFS_NONINTERACTIVE_INSTALL=1 (see "Deliberate differences").
    run_env = {**env, **_NONINTERACTIVE}
    log_info(_OP, f"Installing AZNFS for {distribution}", distro=distribution,
             branch=branch, noninteractive=True)

    if branch == "installer":
        return _install_from_upstream_installer(env, run_env, distribution)
    return _install_from_pmc(branch, run_env, distribution)
