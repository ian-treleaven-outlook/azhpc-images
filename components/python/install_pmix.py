"""install_pmix.py — Python port of components/install_pmix.sh.

What this step does
-------------------
PMIx (Process Management Interface for Exascale) is the library that MPI
programs and the Slurm workload manager use to start and wire up parallel jobs.
This step installs the PMIx package from Microsoft's Slurm package repository
(or from the distro on Azure Linux), plus the hwloc/libevent development
packages, and records the PMIx version in the component-versions file.

There is one handler per distro family, like install_doca.py:

  * Ubuntu 26.04  nothing: it uses the PMIx 5 stack bundled with HPC-X
                  (install_mpis builds Open MPI with its internal PMIx and
                  records that version). No version is recorded here.
  * Ubuntu        add the Microsoft Slurm apt repo + a pin file, make sure the
                  Microsoft package-signing key is installed, then
                  `apt install` and `apt-mark hold` pmix=<version>.
  * Azure Linux   `dnf install` pmix from the distro repos. The version
                  recorded is the one dnf actually installed, NOT the one in
                  versions.json.
  * RHEL family   (AlmaLinux, Rocky, RHEL) add the Microsoft Slurm yum repo,
                  make sure the Microsoft repo package is installed, enable
                  the repo with the -devel packages (RHEL: the RHUI CodeReady
                  Builder repo; otherwise CRB on 9, PowerTools on 8), then
                  `dnf install pmix-<version>.el<major>`.

The original bash, for comparison (abridged):

    if [[ "$DISTRIBUTION" == "ubuntu26.04" ]]; then
        echo "Ubuntu 26.04 uses the PMIx 5 stack bundled with HPC-X."
        exit 0
    fi

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
            apt install -y ./packages-microsoft-prod.deb
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
        if [[ $DISTRIBUTION == rhel* ]]; then
            dnf config-manager --set-enabled codeready-builder-for-rhel-${OS_MAJOR_VERSION}-${ARCHITECTURE}-rhui-rpms
        elif [[ $OS_MAJOR_VERSION == "9" ]]; then dnf config-manager --set-enabled crb
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
  * Package installs go through install_packages() (utils/package_installer.py),
    which spells them `apt-get install -y -o Dpkg::Use-Pty=0` / `dnf install -y`
    (bash: `apt install -y` / `dnf -y install`). `rpm -i`, `apt update`,
    `apt-mark hold`, `dnf config-manager` and `dnf update` are run as in bash.
    (A RHEL-like distro install_packages() doesn't know, e.g. CentOS, now
    fails at the pmix install; bash's else branch would use dnf.)
  * OS versions come from DISTRIBUTION, not from re-reading /etc/os-release.
    DISTRIBUTION is built from the same file as $ID$VERSION_ID (see
    set_properties.sh / _detect_distribution), so "ubuntu24.04" gives "24.04"
    and "almalinux9.8" / "rocky8.10" give major "9" / "8". This keeps the
    handlers testable without a fake /etc/os-release.
  * An Ubuntu version other than 22.04/24.04 (26.04 returns before this)
    fails (returns 3) before touching the system. Bash prints a message and
    carries on, writing a broken repo line (empty repo name and signed-by
    path) into slurm.list.
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
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import exec_program, run_capture

# The Step's `op` label in build_config.py; used on every log line.
_OP = "install-pmix"

# The only distro handled by the Azure Linux branch (an exact match, as in bash).
_AZURE_LINUX = "azurelinux3.0"
# Uses HPC-X's bundled PMIx 5, so this step does nothing there (exact match).
_UBUNTU_26 = "ubuntu26.04"

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

# RHEL major version -> the extra repo that provides the -devel packages, on
# AlmaLinux/Rocky. Other majors enable nothing, exactly like bash's if/elif.
# RHEL itself uses rhel_codeready_repo() instead.
_RHEL_EXTRA_REPOS = {"9": "crb", "8": "powertools"}

# Azure Linux: the two `dnf -y install` commands, one list per command.
_AZURE_LINUX_PACKAGES = [
    ["pmix", "pmix-devel", "pmix-tools"],
    ["hwloc-devel", "libevent-devel", "munge-devel"],
]

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


def ubuntu_packages(version: str) -> list[str]:
    """What bash installs and then holds on Ubuntu, pmix pinned to `version`.

    bash: apt install -y pmix=${PMIX_VERSION} libevent-dev libhwloc-dev
    """
    return [f"pmix={version}", "libevent-dev", "libhwloc-dev"]


def rhel_packages(version: str, major: str) -> list[str]:
    """What bash installs on the RHEL family, e.g. pmix-4.2.9-2.el9 + -devels.

    bash: dnf -y install pmix-${PMIX_VERSION}.el${OS_MAJOR_VERSION} hwloc-devel \\
              libevent-devel munge-devel
    """
    return [f"pmix-{version}.el{major}", "hwloc-devel", "libevent-devel", "munge-devel"]


def rhel_extra_repo(distribution: str, major: str, architecture: str) -> str | None:
    """The repo bash enables for the -devel packages, or None.

    bash: if [[ $DISTRIBUTION == rhel* ]]; then
              codeready-builder-for-rhel-${OS_MAJOR_VERSION}-${ARCHITECTURE}-rhui-rpms
          elif [[ $OS_MAJOR_VERSION == "9" ]]; then crb
          elif [[ $OS_MAJOR_VERSION == "8" ]]; then powertools
    e.g. ("rhel9.6", "9", "x86_64") -> "codeready-builder-for-rhel-9-x86_64-rhui-rpms".
    """
    if distribution.startswith("rhel"):
        return f"codeready-builder-for-rhel-{major}-{architecture}-rhui-rpms"
    return _RHEL_EXTRA_REPOS.get(major)


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
        i.e. the version from the first matching line. We return that too
        (install() logs a warning when there are several).

    Returns None when no line matches, or when the first matching line doesn't
    fit the sed pattern (sed would then pass the whole line through and bash
    would record a garbage word such as the package name).
    """
    grep = _pmix_grep(architecture)

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


def _pmix_grep(architecture: str) -> re.Pattern:
    """bash's  grep -i pmix.${postfix}  as a compiled regex (see above)."""
    postfix = "aarch64" if architecture == "aarch64" else "x86_64"
    return re.compile(f"pmix.{postfix}", re.IGNORECASE)


def pmix_match_count(dnf_output: str, architecture: str) -> int:
    """How many `dnf list installed` lines bash's grep -i pmix.<arch> matches.

    More than one means bash records only the first line's version (see
    azure_linux_pmix_version), which install() warns about.
    """
    grep = _pmix_grep(architecture)
    return sum(1 for line in dnf_output.split("\n") if grep.search(line))


# --------------------------------------------------------------------------- #
# Small helpers with side effects.
# --------------------------------------------------------------------------- #
def _run_all(commands: list[list[str]], env: dict[str, str]) -> bool:
    """Run commands in order, stopping at the first failure (like `set -e`)."""
    for command in commands:
        rc = exec_program(command, _OP, env=env)
        if rc != 0:
            log_error(_OP, f"`{' '.join(command)}` failed; stopping, as bash's set -e would",
                      command=command[0], exit_code=rc)
            return False
    return True


def _install_ms_repo_package(url: str, env: dict[str, str], *, with_rpm: bool) -> bool:
    """Download a packages-microsoft-prod package, install it, delete it.

    Mirrors bash's  curl -sSL -O <url>; <install>; rm <file>  — run inside our
    scratch work directory. The install is
      * Ubuntu: `apt install -y ./packages-microsoft-prod.deb`, through
        install_packages() (apt needs the "./" to treat the argument as a
        local file, not a package name);
      * RHEL family (with_rpm=True): `rpm -i packages-microsoft-prod.rpm`,
        run as is.
    """
    # The file name curl -O saves to: the last part of the URL.
    filename = url.rsplit("/", 1)[-1]
    log_info(_OP, f"Installing the Microsoft repo package {filename} (sets up the "
                  "packages.microsoft.com signing key)", url=url,
             installer="rpm -i" if with_rpm else "install_packages")
    os.makedirs(_WORK_DIR, exist_ok=True)
    try:
        if exec_program(["curl", "-sSL", "-O", url], _OP, cwd=_WORK_DIR, env=env) != 0:
            log_error(_OP, f"could not download {url}", url=url)
            return False
        if with_rpm:
            if exec_program(["rpm", "-i", filename], _OP, cwd=_WORK_DIR, env=env) != 0:
                log_error(_OP, f"could not install {filename}", command="rpm")
                return False
        # install_packages() logs its own failure.
        elif not install_packages([f"./{filename}"], env, _OP, cwd=_WORK_DIR):
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
        log_error(_OP, f"{distribution} not supported for pmix installation.",
                  distro=distribution, supported=sorted(_UBUNTU_REPOS))
        return None
    repo, signed_by = _UBUNTU_REPOS[ubuntu]
    log_info(_OP, f"Ubuntu {ubuntu}: Microsoft Slurm repo {repo}", handler="ubuntu",
             ubuntu_version=ubuntu, repo=repo, signed_by=signed_by)

    # open(..., "w") replaces the file's contents, like bash's `>`.
    action = "replaced" if os.path.lexists(_SLURM_APT_LIST) else "created"
    line = apt_source_line(env.get("ARCHITECTURE_DISTRO", ""), repo, signed_by)
    with open(_SLURM_APT_LIST, "w", encoding="utf-8") as handle:
        handle.write(line)
    log_info(_OP, f"Wrote {_SLURM_APT_LIST} ({action})", path=_SLURM_APT_LIST,
             action=action, content=line.strip())

    # copyfile copies the contents only, like plain `cp` onto a file.
    pin = os.path.join(env.get("COMPONENT_DIR", ""), "slurm-repo", "slurm-u.pin")
    action = "replaced" if os.path.lexists(_SLURM_APT_PIN) else "created"
    shutil.copyfile(pin, _SLURM_APT_PIN)
    log_info(_OP, f"Copied {pin} to {_SLURM_APT_PIN} ({action})", path=_SLURM_APT_PIN,
             source=pin, action=action)

    # The Microsoft repo package (pre-installed on CycleCloud images) sets up
    # the gpg key that signs the Slurm repo. Generic marketplace images may lack it.
    if not os.path.exists(_MS_PROD_APT_LIST):
        url = (f"https://packages.microsoft.com/config/ubuntu/{ubuntu}/"
               "packages-microsoft-prod.deb")
        if not _install_ms_repo_package(url, env, with_rpm=False):
            return None
    else:
        log_info(_OP, f"{_MS_PROD_APT_LIST} exists; the Microsoft repo package is "
                      "already installed", path=_MS_PROD_APT_LIST)

    packages = ubuntu_packages(version)
    if not _run_all([["apt", "update"]], env):
        return None
    if not install_packages(packages, env, _OP):
        return None
    # Hold the packages so routine upgrades don't replace them. Bash passes
    # "pmix=<version>" to apt-mark too (not just "pmix"), so we do the same.
    if not _run_all([["apt-mark", "hold", *packages]], env):
        return None
    return version


def _install_azurelinux(env: dict[str, str], version: str) -> str | None:
    log_info(_OP, "Azure Linux: installing pmix from the distro repos; recording the "
                  "installed version, not versions.json's", handler="azurelinux",
             versions_json_version=version)
    for packages in _AZURE_LINUX_PACKAGES:
        if not install_packages(packages, env, _OP):
            return None

    # On Azure Linux bash records the version dnf actually installed, replacing
    # the versions.json value. `version` is deliberately unused here.
    rc, output = run_capture(["dnf", "list", "installed"], _OP, env=env)
    if rc != 0:
        log_error(_OP, "could not list installed packages with dnf")
        return None
    installed = azure_linux_pmix_version(output, env.get("ARCHITECTURE", ""))
    if installed is None:
        log_error(_OP, "could not find the installed pmix version in `dnf list installed`",
                  arch=env.get("ARCHITECTURE", ""))
        return None
    log_info(_OP, f"dnf installed pmix {installed}", installed_version=installed)
    # BASH QUIRK (kept): with several matching lines, bash records only the
    # first line's version. Warn so the log shows a choice was made.
    matches = pmix_match_count(output, env.get("ARCHITECTURE", ""))
    if matches > 1:
        log_warn(_OP, f"{matches} lines of `dnf list installed` match "
                      "pmix.<arch>; recording the first one's version, as bash does",
                 matches=matches, recorded=installed)
    return installed


def _install_rhel(env: dict[str, str], version: str) -> str | None:
    distribution = env.get("DISTRIBUTION", "")
    major = rhel_major_version(distribution)
    if major is None:
        log_error(_OP, f"could not determine the OS major version from {distribution!r}")
        return None
    architecture = env.get("ARCHITECTURE", "")
    extra_repo = rhel_extra_repo(distribution, major, architecture)
    log_info(_OP, f"RHEL family: {distribution} (EL{major}), extra repo "
                  f"{extra_repo or 'none'}", handler="rhel", distro=distribution,
             os_major=major, extra_repo=extra_repo or "none", arch=architecture)

    # If there is no slurm-el<major>.repo (an unsupported major version), this
    # raises FileNotFoundError, just as bash's cp would stop the script.
    repo_file = os.path.join(env.get("COMPONENT_DIR", ""), "slurm-repo",
                             f"slurm-el{major}.repo")
    action = "replaced" if os.path.lexists(_SLURM_YUM_REPO) else "created"
    shutil.copyfile(repo_file, _SLURM_YUM_REPO)
    log_info(_OP, f"Copied {repo_file} to {_SLURM_YUM_REPO} ({action})",
             path=_SLURM_YUM_REPO, source=repo_file, action=action)

    if not os.path.exists(_MS_PROD_YUM_REPO):
        url = (f"https://packages.microsoft.com/config/rhel/{major}/"
               "packages-microsoft-prod.rpm")
        if not _install_ms_repo_package(url, env, with_rpm=True):
            return None
    else:
        log_info(_OP, f"{_MS_PROD_YUM_REPO} exists; the Microsoft repo package is "
                      "already installed", path=_MS_PROD_YUM_REPO)

    commands = []
    if extra_repo:
        commands.append(["dnf", "config-manager", "--set-enabled", extra_repo])
    commands.append(["dnf", "update", "-y"])
    if not _run_all(commands, env):
        return None
    if not install_packages(rhel_packages(version, major), env, _OP):
        return None
    return version


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def install(env: dict[str, str]) -> int:
    """Install PMIx for this distro and record its version. Returns 0 or 3."""
    distribution = env.get("DISTRIBUTION", "")
    # Checked first, as in bash: no config is needed for Ubuntu 26.04.
    if distribution == _UBUNTU_26:
        log_info(_OP, "Ubuntu 26.04 uses the PMIx 5 stack bundled with HPC-X.",
                 distro=distribution, skipped=True)
        return 0

    cfg = config_for("pmix", env)
    # isinstance() guards against a config that isn't a JSON object.
    if not isinstance(cfg, dict) or not cfg.get("version"):
        log_error(_OP, "could not resolve pmix version from versions.json",
                  distro=distribution)
        return 3
    version = str(cfg["version"])

    if "ubuntu" in distribution:
        handler = _install_ubuntu
    elif distribution == _AZURE_LINUX:
        handler = _install_azurelinux
    else:  # RHEL family: AlmaLinux, Rocky Linux, RHEL, etc.
        handler = _install_rhel

    log_info(_OP, f"Installing PMIx on {distribution}", distro=distribution,
             version=version, handler=handler.__name__.removeprefix("_install_"))
    try:
        recorded = handler(env, version)
        if recorded is None:
            return 3
        write_component_version("PMIX", recorded)
        log_info(_OP, f"Recorded PMIX={recorded}", component="PMIX", version=recorded)
    except OSError as exc:
        # File copies/writes under /etc and the versions file under /opt.
        log_error(_OP, f"file operation failed: {exc}", path=getattr(exc, "filename", None))
        return 3

    log_info(_OP, f"Installed PMIx {recorded}")
    return 0
