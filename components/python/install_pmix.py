"""install_pmix.py — Python port of components/install_pmix.sh.

PMIx (Process Management Interface for Exascale) is the library MPI programs
and Slurm use to start and wire up parallel jobs. This step installs the PMIx
package (from Microsoft's Slurm repository, or from the distro on Azure
Linux) plus the hwloc/libevent development packages, and records PMIX in the
component-versions file. One handler per bash branch:

  * Ubuntu 26.04  nothing: it uses the PMIx 5 stack bundled with HPC-X
                  (install_mpis records that version).
  * Ubuntu        the Microsoft Slurm apt repo + pin file, the Microsoft repo
                  package if missing, then `apt update`, install and
                  `apt-mark hold` of pmix=<version> and the -dev packages.
  * Azure Linux   pmix from the distro repos; the version recorded is the one
                  dnf installed, NOT versions.json's.
  * RHEL family   the Microsoft Slurm yum repo, the Microsoft repo package if
                  missing, the repo with the -devel packages (RHEL: RHUI
                  CodeReady Builder; otherwise CRB on 9, PowerTools on 8),
                  `dnf update -y`, then pmix-<version>.el<major>.

utils/plans/steps.py sets install() as the `action=` of the "install-pmix"
Step; deleting `action=` runs the bash script again.

Deliberate differences
----------------------
  * Package installs go through install_packages(), spelled `apt-get install
    -y -o Dpkg::Use-Pty=0` / `dnf install -y` (bash: `apt install -y` /
    `dnf -y install`). `rpm -i`, `apt update`, `apt-mark hold`,
    `dnf config-manager` and `dnf update` run as in bash. A RHEL-like distro
    install_packages() doesn't know (e.g. CentOS) fails at the pmix install;
    bash's else branch would use dnf.
  * OS versions come from DISTRIBUTION ($ID$VERSION_ID, see
    set_properties.sh), not from re-reading /etc/os-release: "ubuntu24.04"
    gives "24.04", "rocky8.10" gives major "8".
  * An Ubuntu version other than 22.04/24.04 fails before touching the
    system; bash prints a message and writes a broken repo line.
  * A missing pmix version in versions.json fails up front on every distro
    (Azure Linux ignores it, as in bash, but versions.json always has one).
  * On Azure Linux, a failing `dnf list installed` or a version that can't be
    parsed out of it fails instead of recording an empty or garbage value
    (bash, without pipefail, records whatever grep/sed produced).
  * The Microsoft repo package (.deb/.rpm) is downloaded into a scratch
    directory, removed afterwards, instead of the distro build directory.
    Like bash there is no checksum: curl fetches it from packages.microsoft.com.
  * write_file creates /etc/apt/sources.list.d if it is missing.
"""

from __future__ import annotations

import os
import re
import shutil

from utils.component_config import require_config, write_component_version
from utils.distro import Distro
from utils.files import remove_paths, write_file
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import run, run_all, run_capture

_OP = "install-pmix"

# bash matches these exactly.
_AZURE_LINUX = "azurelinux3.0"
_UBUNTU_26 = "ubuntu26.04"

# System paths, module level so tests can point them at a temporary directory.
_SLURM_APT_LIST = "/etc/apt/sources.list.d/slurm.list"
_SLURM_APT_PIN = "/etc/apt/preferences.d/slurm-repository-pin-990"
_MS_PROD_APT_LIST = "/etc/apt/sources.list.d/microsoft-prod.list"
_SLURM_YUM_REPO = "/etc/yum.repos.d/slurm.repo"
_MS_PROD_YUM_REPO = "/etc/yum.repos.d/microsoft-prod.repo"
# Where the Microsoft repo package is downloaded (bash: its current directory).
_WORK_DIR = "/tmp/azhpc-pmix"

# Ubuntu version -> (Microsoft Slurm repo name, keyring that signs it).
_UBUNTU_REPOS = {
    "24.04": ("slurm-ubuntu-noble", "/usr/share/keyrings/microsoft-prod.gpg"),
    "22.04": ("slurm-ubuntu-jammy", "/etc/apt/trusted.gpg.d/microsoft-prod.gpg"),
}

# EL major version -> the repo with the -devel packages, on AlmaLinux/Rocky.
# Other majors enable nothing, like bash's if/elif. RHEL uses rhel_extra_repo().
_RHEL_EXTRA_REPOS = {"9": "crb", "8": "powertools"}

# Azure Linux: bash's two `dnf -y install` commands, one list per command.
_AZURE_LINUX_PACKAGES = [
    ["pmix", "pmix-devel", "pmix-tools"],
    ["hwloc-devel", "libevent-devel", "munge-devel"],
]

# bash's  sed 's/.*[[:space:]]\([0-9.]*-[0-9]*\)\..*/\1/'  as a Python regex.
# [[:space:]] is spelled out, and [0-9] is used because \d also matches
# non-ASCII digits.
_DNF_VERSION_RE = re.compile(r".*[ \t\n\r\f\v]([0-9.]*-[0-9]*)\..*")


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
def apt_source_line(architecture_distro: str, repo: str, signed_by: str) -> str:
    """The exact line bash's `echo ... > slurm.list` writes (echo adds the newline)."""
    return (f"deb [arch={architecture_distro} signed-by={signed_by}] "
            f"https://packages.microsoft.com/repos/{repo}/ insiders main\n")


def ubuntu_packages(version: str) -> list[str]:
    """What bash installs and then holds on Ubuntu, pmix pinned to `version`."""
    return [f"pmix={version}", "libevent-dev", "libhwloc-dev"]


def rhel_packages(version: str, major: str) -> list[str]:
    """What bash installs on the RHEL family, e.g. pmix-4.2.9-2.el9 + -devels."""
    return [f"pmix-{version}.el{major}", "hwloc-devel", "libevent-devel", "munge-devel"]


def rhel_extra_repo(distro: Distro, architecture: str) -> str | None:
    """The repo bash enables for the -devel packages, or None.

    bash: rhel* -> codeready-builder-for-rhel-<major>-<arch>-rhui-rpms;
    otherwise major 9 -> crb, major 8 -> powertools.
    """
    if distro.is_rhel:
        return f"codeready-builder-for-rhel-{distro.major}-{architecture}-rhui-rpms"
    return _RHEL_EXTRA_REPOS.get(distro.major)


def azure_linux_pmix_version(dnf_output: str, architecture: str) -> str | None:
    """The installed PMIx version in `dnf list installed` output, or None.

    bash: dnf list installed | grep -i pmix.${postfix} | sed '<_DNF_VERSION_RE>'
    e.g. "pmix.x86_64   5.0.1-1.azl3   @azurelinux-official-base" -> "5.0.1-1"

    Mirrored on purpose:
      * postfix is "aarch64" on aarch64, otherwise "x86_64".
      * grep -i is case-insensitive and its "." matches any character, so
        "openpmix.x86_64" matches too (but not "pmix-devel.x86_64").
      * With several matching lines bash's unquoted ${PMIX_VERSION} splits
        into words and write_component_version records the first, i.e. the
        first line's version (install() warns when there are several).

    None when no line matches, or when the first match doesn't fit the sed
    pattern (sed would pass the line through and bash record a garbage word).
    """
    grep = _pmix_grep(architecture)
    # split("\n"), not splitlines(): grep only breaks lines at "\n".
    for line in dnf_output.split("\n"):
        if grep.search(line):
            match = _DNF_VERSION_RE.match(line)
            return match.group(1) if match and match.group(1) else None
    return None


def _pmix_grep(architecture: str) -> re.Pattern:
    """bash's  grep -i pmix.${postfix}  as a compiled regex (see above)."""
    postfix = "aarch64" if architecture == "aarch64" else "x86_64"
    return re.compile(f"pmix.{postfix}", re.IGNORECASE)


def pmix_match_count(dnf_output: str, architecture: str) -> int:
    """How many `dnf list installed` lines bash's grep -i pmix.<arch> matches."""
    grep = _pmix_grep(architecture)
    return sum(1 for line in dnf_output.split("\n") if grep.search(line))


# --------------------------------------------------------------------------- #
# Repository setup.
# --------------------------------------------------------------------------- #
# TODO(ADO 1001): toolkit candidate: copy_file(src, dest, op) -> bool (bash `cp`,
# logging created/replaced); a dozen ports copy files with shutil.
def _copy_file(source: str, dest: str) -> bool:
    """bash `cp source dest` onto a file: copy the contents only."""
    action = "replaced" if os.path.lexists(dest) else "created"
    try:
        shutil.copyfile(source, dest)
    except OSError as exc:
        log_error(_OP, f"could not copy {source} to {dest}: {exc}", path=dest, source=source)
        return False
    log_info(_OP, f"Copied {source} to {dest} ({action})", path=dest, source=source,
             action=action)
    return True


def _ensure_ms_repo_package(marker: str, url: str, env: dict[str, str], *,
                            with_rpm: bool) -> bool:
    """Install packages-microsoft-prod unless `marker` (its repo file) exists.

    The package (pre-installed on CycleCloud images) sets up the key that
    signs the Slurm repo; generic marketplace images may lack it.
    """
    if os.path.exists(marker):
        log_info(_OP, f"{marker} exists; the Microsoft repo package is already installed",
                 path=marker)
        return True
    return _install_ms_repo_package(url, env, with_rpm=with_rpm)


def _install_ms_repo_package(url: str, env: dict[str, str], *, with_rpm: bool) -> bool:
    """bash: curl -sSL -O <url>; <install it>; rm <file>  (in a scratch dir).

    The install is `rpm -i <file>` on the RHEL family, and on Ubuntu
    `apt install -y ./<file>` through install_packages() (the "./" makes apt
    treat it as a local file).
    """
    filename = url.rsplit("/", 1)[-1]
    log_info(_OP, f"Installing the Microsoft repo package {filename} (sets up the "
                  "packages.microsoft.com signing key)", url=url,
             installer="rpm -i" if with_rpm else "install_packages")
    os.makedirs(_WORK_DIR, exist_ok=True)
    try:
        if not run(["curl", "-sSL", "-O", url], _OP, cwd=_WORK_DIR, env=env):
            return False
        if with_rpm:
            installed = run(["rpm", "-i", filename], _OP, cwd=_WORK_DIR, env=env)
        else:
            installed = install_packages([f"./{filename}"], env, _OP, cwd=_WORK_DIR)
        if not installed:
            return False
        # bash's `rm <file>` stops the script if the file is gone.
        os.remove(os.path.join(_WORK_DIR, filename))
    finally:
        remove_paths(_WORK_DIR, op=_OP)
    return True


# --------------------------------------------------------------------------- #
# Per-distro handlers: the version to record, or None on failure.
# --------------------------------------------------------------------------- #
def _install_ubuntu(env: dict[str, str], distro: Distro, version: str) -> str | None:
    if distro.version not in _UBUNTU_REPOS:
        log_error(_OP, f"{distro} not supported for pmix installation.",
                  distro=str(distro), supported=sorted(_UBUNTU_REPOS))
        return None
    repo, signed_by = _UBUNTU_REPOS[distro.version]
    log_info(_OP, f"Ubuntu {distro.version}: Microsoft Slurm repo {repo}", handler="ubuntu",
             ubuntu_version=distro.version, repo=repo, signed_by=signed_by)
    line = apt_source_line(env.get("ARCHITECTURE_DISTRO", ""), repo, signed_by)
    pin = os.path.join(env.get("COMPONENT_DIR", ""), "slurm-repo", "slurm-u.pin")
    ms_package = (f"https://packages.microsoft.com/config/ubuntu/{distro.version}/"
                  "packages-microsoft-prod.deb")
    packages = ubuntu_packages(version)
    ok = (write_file(_SLURM_APT_LIST, line, _OP)
          and _copy_file(pin, _SLURM_APT_PIN)
          and _ensure_ms_repo_package(_MS_PROD_APT_LIST, ms_package, env, with_rpm=False)
          and run(["apt", "update"], _OP, env=env)
          and install_packages(packages, env, _OP)
          # Held so routine upgrades don't replace them. bash holds
          # "pmix=<version>" (not just "pmix"), and so do we.
          and run(["apt-mark", "hold", *packages], _OP, env=env))
    return version if ok else None


def _install_azurelinux(env: dict[str, str], distro: Distro, version: str) -> str | None:
    # bash records the version dnf installed; versions.json's is unused here.
    log_info(_OP, "Azure Linux: installing pmix from the distro repos; recording the "
                  "installed version, not versions.json's", handler="azurelinux",
             versions_json_version=version)
    for packages in _AZURE_LINUX_PACKAGES:
        if not install_packages(packages, env, _OP):
            return None
    return _installed_azure_linux_version(env)


def _installed_azure_linux_version(env: dict[str, str]) -> str | None:
    """The pmix version `dnf list installed` reports, parsed as bash does."""
    architecture = env.get("ARCHITECTURE", "")
    rc, output = run_capture(["dnf", "list", "installed"], _OP, env=env)
    if rc != 0:
        return None
    installed = azure_linux_pmix_version(output, architecture)
    if installed is None:
        log_error(_OP, "could not find the installed pmix version in `dnf list installed`",
                  arch=architecture)
        return None
    log_info(_OP, f"dnf installed pmix {installed}", installed_version=installed)
    matches = pmix_match_count(output, architecture)
    if matches > 1:
        # BASH QUIRK (kept): bash records only the first matching line's version.
        log_warn(_OP, f"{matches} lines of `dnf list installed` match "
                      "pmix.<arch>; recording the first one's version, as bash does",
                 matches=matches, recorded=installed)
    return installed


def _install_rhel(env: dict[str, str], distro: Distro, version: str) -> str | None:
    major = distro.major
    if not major.isdigit():
        log_error(_OP, f"could not determine the OS major version from {str(distro)!r}")
        return None
    extra_repo = rhel_extra_repo(distro, env.get("ARCHITECTURE", ""))
    log_info(_OP, f"RHEL family: {distro} (EL{major}), extra repo {extra_repo or 'none'}",
             handler="rhel", distro=str(distro), os_major=major,
             extra_repo=extra_repo or "none", arch=env.get("ARCHITECTURE", ""))
    # No slurm-el<major>.repo (an unsupported major) fails the copy, as bash's cp would.
    repo_file = os.path.join(env.get("COMPONENT_DIR", ""), "slurm-repo",
                             f"slurm-el{major}.repo")
    ms_package = f"https://packages.microsoft.com/config/rhel/{major}/packages-microsoft-prod.rpm"
    enable = [["dnf", "config-manager", "--set-enabled", extra_repo]] if extra_repo else []
    ok = (_copy_file(repo_file, _SLURM_YUM_REPO)
          and _ensure_ms_repo_package(_MS_PROD_YUM_REPO, ms_package, env, with_rpm=True)
          and run_all([*enable, ["dnf", "update", "-y"]], _OP, env=env)
          and install_packages(rhel_packages(version, major), env, _OP))
    return version if ok else None


def _handler_for(distro: Distro):
    """bash's if/elif/else: Ubuntu, Azure Linux 3.0, else the RHEL family."""
    if distro.is_ubuntu:
        return _install_ubuntu
    if distro.name == _AZURE_LINUX:
        return _install_azurelinux
    return _install_rhel


def install(env: dict[str, str]) -> int:
    """Install PMIx for this distro and record its version. Returns 0 or 3."""
    distro = Distro.from_env(env)
    # Checked first, as in bash: no config is needed for Ubuntu 26.04.
    if distro.name == _UBUNTU_26:
        log_info(_OP, "Ubuntu 26.04 uses the PMIx 5 stack bundled with HPC-X.",
                 distro=str(distro), skipped=True)
        return 0
    cfg = require_config("pmix", env, _OP, "version")
    if cfg is None:
        return 3
    version = str(cfg["version"])
    handler = _handler_for(distro)
    log_info(_OP, f"Installing PMIx on {distro}", distro=str(distro), version=version,
             handler=handler.__name__.removeprefix("_install_"))
    try:
        recorded = handler(env, distro, version)
        if recorded is None:
            return 3
        write_component_version("PMIX", recorded)
    except OSError as exc:
        # The versions file under /opt, or removing the downloaded package.
        log_error(_OP, f"file operation failed: {exc}", path=getattr(exc, "filename", None))
        return 3
    log_info(_OP, f"Recorded PMIX={recorded}", component="PMIX", version=recorded)
    return 0
