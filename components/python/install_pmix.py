"""install_pmix.py — Python port of components/install_pmix.sh.

What this step does
-------------------
PMIx (Process Management Interface for Exascale) is the library that MPI
programs and the Slurm workload manager use to start and wire up parallel jobs.
This step installs the PMIx package from Microsoft's Slurm package repository
(or from the distro on Azure Linux), plus the hwloc/libevent development
packages, and records the PMIx version in the component-versions file.

There is one handler per distro family, like install_doca.py:

  * Ubuntu        add the Microsoft Slurm apt repo + a pin file, make sure the
                  Microsoft package-signing key is installed, then
                  `apt install` and `apt-mark hold` pmix=<version>.
  * Azure Linux   `dnf install` pmix from the distro repos. The version
                  recorded is the one dnf actually installed, NOT the one in
                  versions.json.
  * RHEL family   (AlmaLinux, Rocky, RHEL) add the Microsoft Slurm yum repo,
                  make sure the Microsoft repo package is installed, enable
                  CRB (9) or PowerTools (8), then `dnf install
                  pmix-<version>.el<major>`.

The original bash, for comparison (abridged):

    pmix_metadata=$(get_component_config "pmix")
    PMIX_VERSION=$(jq -r '.version' <<< $pmix_metadata)

    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        UBUNTU_VERSION=$(cat /etc/os-release | grep VERSION_ID | cut -d= -f2 | cut -d\\" -f2)
        if [ $UBUNTU_VERSION == 24.04 ]; then
            REPO=slurm-ubuntu-noble
            SIGNED_BY="/usr/share/keyrings/microsoft-prod.gpg"
        elif [ $UBUNTU_VERSION == 22.04 ]; then
            REPO=slurm-ubuntu-jammy
            SIGNED_BY="/etc/apt/trusted.gpg.d/microsoft-prod.gpg"
        else echo "$DISTRIBUTION not supported for pmix installation."
        fi
        echo "deb [arch=$ARCHITECTURE_DISTRO signed-by=$SIGNED_BY] https://packages.microsoft.com/repos/$REPO/ insiders main" > /etc/apt/sources.list.d/slurm.list
        cp ${COMPONENT_DIR}/slurm-repo/slurm-u.pin /etc/apt/preferences.d/slurm-repository-pin-990
        if [ ! -e /etc/apt/sources.list.d/microsoft-prod.list ]; then
            curl -sSL -O https://packages.microsoft.com/config/ubuntu/$UBUNTU_VERSION/packages-microsoft-prod.deb
            dpkg -i packages-microsoft-prod.deb
            rm packages-microsoft-prod.deb
        fi
        apt update
        apt install -y pmix=${PMIX_VERSION} libevent-dev libhwloc-dev
        apt-mark hold pmix=${PMIX_VERSION} libevent-dev libhwloc-dev
    elif [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
        dnf -y install pmix pmix-devel pmix-tools
        dnf -y install hwloc-devel libevent-devel munge-devel
        if [ "$ARCHITECTURE" = "aarch64" ]; then postfix="aarch64"; else postfix="x86_64"; fi
        PMIX_VERSION=$(dnf list installed | grep -i pmix.${postfix} | sed 's/.*[[:space:]]\\([0-9.]*-[0-9]*\\)\\..*/\\1/')
    else
        OS_MAJOR_VERSION=$(sed -n 's/^VERSION_ID="\\([0-9]\\+\\).*/\\1/p' /etc/os-release)
        cp ${COMPONENT_DIR}/slurm-repo/slurm-el${OS_MAJOR_VERSION}.repo /etc/yum.repos.d/slurm.repo
        if [ ! -e /etc/yum.repos.d/microsoft-prod.repo ];then
            curl -sSL -O https://packages.microsoft.com/config/rhel/${OS_MAJOR_VERSION}/packages-microsoft-prod.rpm
            rpm -i packages-microsoft-prod.rpm
            rm packages-microsoft-prod.rpm
        fi
        if [[ $OS_MAJOR_VERSION == "9" ]]; then dnf config-manager --set-enabled crb
        elif [[ $OS_MAJOR_VERSION == "8" ]]; then dnf config-manager --set-enabled powertools
        fi
        dnf update -y
        dnf -y install pmix-${PMIX_VERSION}.el${OS_MAJOR_VERSION} hwloc-devel libevent-devel munge-devel
    fi

    write_component_version "PMIX" ${PMIX_VERSION}

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-pmix" Step. Deleting `action=` from that Step makes the build run the
bash script again, which is why the .sh file is kept.

install() receives the build environment (a dict) and returns 0 for success or
3 for failure. It logs problems and never lets an exception escape.

Deliberate differences from bash
--------------------------------
  * OS versions come from DISTRIBUTION, not from re-reading /etc/os-release.
    DISTRIBUTION is built from the same file as $ID$VERSION_ID (see
    set_properties.sh / _detect_distribution), so "ubuntu24.04" gives "24.04"
    and "almalinux9.8" / "rocky8.10" give major "9" / "8". This keeps the
    handlers testable without a fake /etc/os-release.
  * An Ubuntu version other than 22.04/24.04 fails (returns 3) before touching
    the system. Bash prints a message and carries on, writing a broken repo line
    (empty repo name and signed-by path) into slurm.list.
  * A missing pmix version in versions.json fails up front on every distro
    (bash would continue with an empty version). Azure Linux ignores that
    version, as in bash, but versions.json always has one for it.
  * On Azure Linux, an installed version that can't be parsed out of
    `dnf list installed` fails instead of recording an empty or garbage value,
    and so does a failing `dnf list installed` (bash, without pipefail, would
    ignore its exit code and record whatever grep/sed produced).
  * The Microsoft repo package (.deb/.rpm) is downloaded into a scratch work
    directory instead of the distro build directory bash runs in. Like bash,
    there is no checksum: it's fetched with curl from packages.microsoft.com.
"""

from __future__ import annotations

import os
import re
import shutil

from utils.component_config import config_for, write_component_version
from utils.logger import log_error, log_info
from utils.process import exec_program, run_capture

# The Step's `op` label in build_config.py; used on every log line.
_OP = "install-pmix"

# The only distro handled by the Azure Linux branch (an exact match, as in bash).
_AZURE_LINUX = "azurelinux3.0"

# ---- Ubuntu paths ---------------------------------------------------------
# Module-level constants so the tests can point them at a temporary directory.
_SLURM_APT_LIST = "/etc/apt/sources.list.d/slurm.list"
_SLURM_APT_PIN = "/etc/apt/preferences.d/slurm-repository-pin-990"
_MS_PROD_APT_LIST = "/etc/apt/sources.list.d/microsoft-prod.list"

# ---- RHEL-family paths ----------------------------------------------------
_SLURM_YUM_REPO = "/etc/yum.repos.d/slurm.repo"
_MS_PROD_YUM_REPO = "/etc/yum.repos.d/microsoft-prod.repo"

# Scratch directory used as the working directory for the curl downloads of the
# Microsoft repo package (bash downloads into whatever directory it runs in).
_WORK_DIR = "/tmp/azhpc-pmix"

# Ubuntu version -> (Microsoft Slurm repo name, keyring that signs it).
_UBUNTU_REPOS = {
    "24.04": ("slurm-ubuntu-noble", "/usr/share/keyrings/microsoft-prod.gpg"),
    "22.04": ("slurm-ubuntu-jammy", "/etc/apt/trusted.gpg.d/microsoft-prod.gpg"),
}

# RHEL major version -> the extra repo that provides the -devel packages.
# Other majors enable nothing, exactly like bash's if/elif.
_RHEL_EXTRA_REPOS = {"9": "crb", "8": "powertools"}

# The sed expression bash uses to cut the version out of a `dnf list installed`
# line, as a Python regex:
#     sed 's/.*[[:space:]]\([0-9.]*-[0-9]*\)\..*/\1/'
# [[:space:]] is written out as an explicit character class, and [0-9] is used
# instead of \d (which in Python also matches non-ASCII digits).
_DNF_VERSION_RE = re.compile(r".*[ \t\n\r\f\v]([0-9.]*-[0-9]*)\..*")


# --------------------------------------------------------------------------- #
# Pure helpers: they only transform strings, so tests call them directly.
# --------------------------------------------------------------------------- #
def ubuntu_version(distribution: str) -> str:
    """Return the Ubuntu version from DISTRIBUTION, e.g. "ubuntu24.04" -> "24.04".

    Bash reads VERSION_ID from /etc/os-release; DISTRIBUTION is $ID$VERSION_ID,
    so removing the "ubuntu" prefix gives the same value.
    """
    # str.removeprefix() returns the string unchanged if the prefix is absent.
    return distribution.removeprefix("ubuntu")


def rhel_major_version(distribution: str) -> str | None:
    """Return the OS major version from DISTRIBUTION, e.g. "rocky8.10" -> "8".

    Mirrors bash's  sed -n 's/^VERSION_ID="\\([0-9]\\+\\).*/\\1/p' /etc/os-release,
    which keeps the leading digits of VERSION_ID. DISTRIBUTION is $ID$VERSION_ID,
    so we skip the letters of the ID and take the first run of digits.
    Returns None if there are no digits.
    """
    # re.match() only matches at the start of the string. [^0-9]* skips the
    # non-digit ID ("almalinux"); ([0-9]+) captures the digits after it ("9").
    match = re.match(r"[^0-9]*([0-9]+)", distribution)
    return match.group(1) if match else None


def apt_source_line(architecture_distro: str, repo: str, signed_by: str) -> str:
    """The exact line bash's `echo ... > slurm.list` writes (echo adds the newline)."""
    return (f"deb [arch={architecture_distro} signed-by={signed_by}] "
            f"https://packages.microsoft.com/repos/{repo}/ insiders main\n")


def azure_linux_pmix_version(dnf_output: str, architecture: str) -> str | None:
    """Extract the installed PMIx version from `dnf list installed` output.

    Mirrors:
        dnf list installed | grep -i pmix.${postfix} \\
            | sed 's/.*[[:space:]]\\([0-9.]*-[0-9]*\\)\\..*/\\1/'

    e.g. "pmix.x86_64   5.0.1-1.azl3   @azurelinux-official-base"  ->  "5.0.1-1"

    Details that are mirrored on purpose:
      * postfix is "aarch64" on aarch64, otherwise "x86_64".
      * grep -i is case-insensitive, and the "." in "pmix.x86_64" is a regex
        "any character", so it is not escaped here either. It does not match
        "pmix-devel.x86_64" or "pmix-tools.x86_64" (there's no single character
        between "pmix" and the arch), but it does match e.g. "openpmix.x86_64".
      * If several lines match, bash's unquoted ${PMIX_VERSION} is split into
        words and write_component_version records only its first argument,
        i.e. the version from the first matching line. We return that too.

    Returns None when no line matches, or when the first matching line doesn't
    fit the sed pattern (sed would then pass the whole line through and bash
    would record a garbage word such as the package name).
    """
    postfix = "aarch64" if architecture == "aarch64" else "x86_64"
    grep = re.compile(f"pmix.{postfix}", re.IGNORECASE)

    # split("\n") (not splitlines()) because grep only treats "\n" as a line
    # break; splitlines() would also split on characters such as "\r" or "\f".
    for line in dnf_output.split("\n"):
        if grep.search(line):
            match = _DNF_VERSION_RE.match(line)
            # Only the first matching line matters (see the docstring), so we
            # return straight away. group(1) is the text inside the (...).
            if match and match.group(1):
                return match.group(1)
            return None
    return None


# --------------------------------------------------------------------------- #
# Small helpers with side effects.
# --------------------------------------------------------------------------- #
def _run_all(commands: list[list[str]], env: dict[str, str]) -> bool:
    """Run commands in order, stopping at the first failure (like `set -e`)."""
    for command in commands:
        if exec_program(command, _OP, env=env) != 0:
            return False
    return True


def _install_ms_repo_package(url: str, installer: list[str],
                             env: dict[str, str]) -> bool:
    """Download a packages-microsoft-prod package, install it, delete it.

    Mirrors bash's  curl -sSL -O <url>; dpkg -i <file> (or rpm -i <file>);
    rm <file>  — run inside our scratch work directory. `installer` is the
    command before the file name, e.g. ["dpkg", "-i"].
    """
    # The file name curl -O saves to: the last part of the URL.
    filename = url.rsplit("/", 1)[-1]
    os.makedirs(_WORK_DIR, exist_ok=True)
    try:
        if exec_program(["curl", "-sSL", "-O", url], _OP, cwd=_WORK_DIR, env=env) != 0:
            return False
        if exec_program([*installer, filename], _OP, cwd=_WORK_DIR, env=env) != 0:
            return False
        # Mirrors `rm <file>`, which would stop the bash script if it failed.
        os.remove(os.path.join(_WORK_DIR, filename))
    finally:
        # Our own addition: remove the scratch directory whatever happened.
        shutil.rmtree(_WORK_DIR, ignore_errors=True)
    return True


# --------------------------------------------------------------------------- #
# Per-distro handlers. Each returns the version to record, or None on failure.
# They may raise OSError from file operations; install() catches it.
# --------------------------------------------------------------------------- #
def _install_ubuntu(env: dict[str, str], version: str) -> str | None:
    distribution = env.get("DISTRIBUTION", "")
    ubuntu = ubuntu_version(distribution)
    if ubuntu not in _UBUNTU_REPOS:
        log_error(_OP, f"{distribution} not supported for pmix installation.")
        return None
    repo, signed_by = _UBUNTU_REPOS[ubuntu]

    # open(..., "w") replaces the file's contents, like bash's `>`.
    with open(_SLURM_APT_LIST, "w", encoding="utf-8") as handle:
        handle.write(apt_source_line(env.get("ARCHITECTURE_DISTRO", ""), repo, signed_by))

    # copyfile copies the contents only, like plain `cp` onto a file.
    pin = os.path.join(env.get("COMPONENT_DIR", ""), "slurm-repo", "slurm-u.pin")
    shutil.copyfile(pin, _SLURM_APT_PIN)

    # The Microsoft repo package (pre-installed on CycleCloud images) sets up
    # the gpg key that signs the Slurm repo. Generic marketplace images may lack it.
    if not os.path.exists(_MS_PROD_APT_LIST):
        url = (f"https://packages.microsoft.com/config/ubuntu/{ubuntu}/"
               "packages-microsoft-prod.deb")
        if not _install_ms_repo_package(url, ["dpkg", "-i"], env):
            return None

    packages = [f"pmix={version}", "libevent-dev", "libhwloc-dev"]
    if not _run_all([
        ["apt", "update"],
        ["apt", "install", "-y", *packages],
        # Hold the packages so routine upgrades don't replace them. Bash passes
        # "pmix=<version>" to apt-mark too (not just "pmix"), so we do the same.
        ["apt-mark", "hold", *packages],
    ], env):
        return None
    return version


def _install_azurelinux(env: dict[str, str], version: str) -> str | None:
    if not _run_all([
        ["dnf", "-y", "install", "pmix", "pmix-devel", "pmix-tools"],
        ["dnf", "-y", "install", "hwloc-devel", "libevent-devel", "munge-devel"],
    ], env):
        return None

    # On Azure Linux bash records the version dnf actually installed, replacing
    # the versions.json value. `version` is deliberately unused here.
    rc, output = run_capture(["dnf", "list", "installed"], _OP, env=env)
    if rc != 0:
        log_error(_OP, "could not list installed packages with dnf")
        return None
    installed = azure_linux_pmix_version(output, env.get("ARCHITECTURE", ""))
    if installed is None:
        log_error(_OP, "could not find the installed pmix version in `dnf list installed`")
        return None
    return installed


def _install_rhel(env: dict[str, str], version: str) -> str | None:
    distribution = env.get("DISTRIBUTION", "")
    major = rhel_major_version(distribution)
    if major is None:
        log_error(_OP, f"could not determine the OS major version from {distribution!r}")
        return None

    # If there is no slurm-el<major>.repo (an unsupported major version), this
    # raises FileNotFoundError, just as bash's cp would stop the script.
    repo_file = os.path.join(env.get("COMPONENT_DIR", ""), "slurm-repo",
                             f"slurm-el{major}.repo")
    shutil.copyfile(repo_file, _SLURM_YUM_REPO)

    if not os.path.exists(_MS_PROD_YUM_REPO):
        url = (f"https://packages.microsoft.com/config/rhel/{major}/"
               "packages-microsoft-prod.rpm")
        if not _install_ms_repo_package(url, ["rpm", "-i"], env):
            return None

    commands = []
    if major in _RHEL_EXTRA_REPOS:
        commands.append(["dnf", "config-manager", "--set-enabled", _RHEL_EXTRA_REPOS[major]])
    commands += [
        ["dnf", "update", "-y"],
        ["dnf", "-y", "install", f"pmix-{version}.el{major}",
         "hwloc-devel", "libevent-devel", "munge-devel"],
    ]
    if not _run_all(commands, env):
        return None
    return version


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def install(env: dict[str, str]) -> int:
    """Install PMIx for this distro and record its version. Returns 0 or 3."""
    cfg = config_for("pmix", env)
    # isinstance() guards against a config that isn't a JSON object.
    if not isinstance(cfg, dict) or not cfg.get("version"):
        log_error(_OP, "could not resolve pmix version from versions.json")
        return 3
    version = str(cfg["version"])

    distribution = env.get("DISTRIBUTION", "")
    if "ubuntu" in distribution:
        handler = _install_ubuntu
    elif distribution == _AZURE_LINUX:
        handler = _install_azurelinux
    else:  # RHEL family: AlmaLinux, Rocky Linux, RHEL, etc.
        handler = _install_rhel

    log_info(_OP, f"Installing PMIx on {distribution}")
    try:
        recorded = handler(env, version)
        if recorded is None:
            return 3
        write_component_version("PMIX", recorded)
    except OSError as exc:
        # File copies/writes under /etc and the versions file under /opt.
        log_error(_OP, f"file operation failed: {exc}")
        return 3

    log_info(_OP, f"Installed PMIx {recorded}")
    return 0
