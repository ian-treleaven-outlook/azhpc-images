"""install_lustre_client.py — Python port of components/install_lustre_client.sh.

What this step does
-------------------
Installs the Azure Managed Lustre (AMLFS) client: the Lustre userspace tools
and a DKMS kernel module built for the image's kernel, from Microsoft's AMLFS
package repo. Azure VMs reach Lustre servers over TCP, not InfiniBand, so DKMS
is configured to build Lustre without o2ib (the InfiniBand LNet driver) and to
skip that driver's artifact. DKMS must then report the module installed for
the running kernel (utils/dkms.py), and the version is recorded as LUSTRE.

This step must stay immediately before install-mpifileutils, which builds
against the Lustre client.

Kernel 7.0 is skipped (#598): when KERNEL_VERSION (if set and not empty, as
Packer passes it) or else `uname -r` is "7.0", "7.0.*" or "7.0-*", nothing is
installed or recorded and the step succeeds, even on a distro without a
"lustre" entry in versions.json (Ubuntu 26.04 today).

Per distro family (bash: `$DISTRIBUTION == *"ubuntu"*`, else RHEL family):

  * Ubuntu  Write the AMLFS apt source (codename from /etc/lsb-release,
            signed-by key per Ubuntu version from /etc/os-release, as bash
            reads them), `apt-get update`, install lustre-client-<v> first
            (it owns /etc/sysconfig/dkms-lustre; a pre-created conffile makes
            noninteractive dpkg stop at a prompt), then configure DKMS:
              22.04  the legacy /etc/dkms/lustre-client-modules-<v>.conf
                     (a MAKE= line with --with-o2ib=no), skipping record 11;
              other  /etc/sysconfig/dkms-lustre (LUSTRE_DKMS_CONFIGURE_EXTRA).
            Then install amlfs-lustre-client-dkms-<v> and check the DKMS
            module lustre-client-modules.
  * RHEL    (AlmaLinux, Rocky, RHEL) The version's "-" become "_" (and that
            is what gets recorded). Import the Microsoft key, write
            /etc/yum.repos.d/amlfs.repo for el<major>, write
            /etc/sysconfig/lustre (`echo --with-o2ib=no`), append the record-3
            skip to /etc/dkms/lustre-client-<v>.conf, `dnf install`
            lustre-client-dkms and -devel, and check the DKMS module
            lustre-client.

The configure_* functions below are bash's helper functions of the same
names. Every generated file was rendered by the real bash in WSL and is
compared byte for byte (tests/python/fixtures/install_lustre_client/).

Upstream removed the LU-20071 timer patch for RHEL 9.8 kernels (#610) and
the Ubuntu 22.04 AMD dkms 3.4.1 upgrade (#633/#643); this port has neither.

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets install() as the "install-lustre" Step's
`action=`; deleting `action=` runs the bash script again, which is why it is
kept.

Bash issues, kept and warned about
----------------------------------
  * configure_lustre_dkms_skip_artifact APPENDS to
    /etc/dkms/<module>-<version>.conf and nothing empties it first on the RHEL
    family, so a rerun on the same machine appends the same lines again.
    Warned when the file already holds the text being appended.
  * distros/azurelinux3.0/install.sh never runs install_lustre_client.sh, but
    the build plan's install-lustre Step has no distro gate, so on Azure Linux
    it runs and takes bash's RHEL-family branch. Warned at the start (gating
    is left alone; versions.json has no Azure Linux "lustre" entry, so the
    step then fails on the missing version, as bash would fail at dnf).

Deliberate differences from bash
--------------------------------
  * A missing "lustre" version fails before changing anything (after the
    kernel 7.0 skip); bash would install "lustre-client-null". Bash looks the
    version up before the kernel check; this port after it, which changes
    nothing else (the lookup has no side effects).
  * Missing ARCHITECTURE_DISTRO, an empty DISTRIB_CODENAME, or an Ubuntu
    version other than 22.04/24.04 (no SIGNED_BY) fails before the apt source
    is written; bash would write a source line with blank values. An empty
    OS major version on the RHEL family fails before the repo is written;
    bash would write "baseurl=.../amlfs-el".
  * DISTRIB_CODENAME is read by a small parser (plain, "double"- or
    'single'-quoted values; the last assignment wins) instead of `source`-ing
    /etc/lsb-release. That covers how Ubuntu writes it.
  * Package installs go through install_packages() (one call per bash
    command, options kept): `apt-get install -y -o Dpkg::Use-Pty=0 ...` /
    `dnf install -y ...`. On a distribution it doesn't know, bash's else
    branch would try dnf; install_packages() logs an error and the step
    returns 3.
"""

from __future__ import annotations

import os
import platform
import re

from utils.component_config import record_version, require_config
from utils.distro import Distro
from utils.dkms import check_dkms_status
from utils.files import write_file
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages, refresh_package_index
from utils.process import run

_OP = "install-lustre"

# System paths. Module constants so tests can point them at temporary dirs.
_LSB_RELEASE = "/etc/lsb-release"
_OS_RELEASE = "/etc/os-release"
_APT_SOURCE = "/etc/apt/sources.list.d/amlfs.list"
_YUM_REPO = "/etc/yum.repos.d/amlfs.repo"
_DKMS_CONF_DIR = "/etc/dkms"
_UBUNTU_DKMS_SYSCONFIG = "/etc/sysconfig/dkms-lustre"
_EL_SYSCONFIG = "/etc/sysconfig/lustre"

# Ubuntu apt signing key per version (bash's SIGNED_BY if/elif).
_SIGNED_BY = {
    "24.04": "/usr/share/keyrings/microsoft-prod.gpg",
    "22.04": "/etc/apt/trusted.gpg.d/microsoft-prod.gpg",
}
# The Ubuntu version that gets the legacy DKMS config.
_LEGACY_DKMS_UBUNTU = "22.04"

_MICROSOFT_KEY_URL = "https://packages.microsoft.com/keys/microsoft.asc"

# bash: dnf install -y --disableexcludes=main --refresh "${LUSTRE_PACKAGES[@]}"
_EL_DNF_OPTIONS = ["--disableexcludes=main", "--refresh"]

# Kernels whose version is exactly this, or starts with it plus "." or "-",
# get no Lustre client (bash: == "7.0" || == 7.0.* || == 7.0-*).
_SKIPPED_KERNEL = "7.0"

# The DKMS module names checked after installation.
_UBUNTU_DKMS_MODULE = "lustre-client-modules"
_EL_DKMS_MODULE = "lustre-client"

# The skip-artifact records and the comments bash writes above them.
_UBUNTU_SKIP_SLOT = 11
_UBUNTU_SKIP_DESCRIPTION = ("Ubuntu 22.04 ko2iblnd is record 11; --with-o2ib=no "
                            "intentionally skips it.")
_EL_SKIP_SLOT = 3
_EL_SKIP_DESCRIPTION = "EL ko2iblnd is record 3; --with-o2ib=no intentionally skips it."

# ---- File contents from the quoted heredocs (<<'EOF': nothing is expanded),
# byte for byte. A test checks each against install_lustre_client.sh.

# configure_lustre_dkms_no_o2ib (RHEL family, /etc/sysconfig/lustre).
_NO_O2IB_CONF = (
    "# Azure clients do not have IB line of sight to Lustre servers, so use TCP LNet.\n"
    "# EL DKMS captures stdout from sourcing this file into DKMS_CONFIG_OPTS.\n"
    "echo --with-o2ib=no\n"
)

# configure_lustre_dkms_no_o2ib_with_tr_workaround (Ubuntu 24.04,
# /etc/sysconfig/dkms-lustre).
_NO_O2IB_TR_CONF = (
    "# Azure clients do not have IB line of sight to Lustre servers, so use TCP LNet.\n"
    'LUSTRE_DKMS_CONFIGURE_EXTRA="--with-o2ib=no"\n'
)

# configure_legacy_lustre_dkms_no_o2ib (Ubuntu 22.04). "$kernel_source_dir" is
# literal text: the heredoc is quoted, and DKMS expands it when it sources the
# file to build the module.
_LEGACY_DKMS_CONF = (
    "# Azure clients do not have IB line of sight to Lustre servers, so use TCP LNet.\n"
    "# Ubuntu 22.04's published AMLFS Lustre DKMS package has a static dkms.conf with\n"
    "# ko2iblnd in record 11, so configure without o2ib and skip only that artifact.\n"
    'MAKE="sh autogen.sh && ./configure --with-linux=$kernel_source_dir '
    "--with-linux-obj=$kernel_source_dir --disable-server --disable-quilt "
    "--disable-dependency-tracking --disable-doc --disable-utils --disable-iokit "
    "--disable-snmp --disable-tests --enable-quota --with-kmp-moddir=updates "
    '--with-o2ib=no --enable-gss && make"\n'
)


# --------------------------------------------------------------------------- #
# Pure helpers: they only transform values, so tests call them directly.
# --------------------------------------------------------------------------- #
def _lines(text: str) -> list[str]:
    """The lines grep/sed/cut see: split on "\\n", without the empty piece
    after a final newline."""
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def _cut_field2(line: str, delimiter: str) -> str:
    """`cut -d<delimiter> -f2` on one line (a line without it is printed whole)."""
    parts = line.split(delimiter)
    return parts[1] if len(parts) > 1 else line


def ubuntu_version_from_os_release(text: str) -> str:
    """UBUNTU_VERSION from /etc/os-release's text, as bash computes it:
    $(cat /etc/os-release | grep VERSION_ID | cut -d= -f2 | cut -d\\" -f2).

        'VERSION_ID="22.04"\\n' -> "22.04"
    """
    values = [_cut_field2(_cut_field2(line, "="), '"')
              for line in _lines(text) if "VERSION_ID" in line]
    return "\n".join(values).rstrip("\n")


def os_major_from_os_release(text: str) -> str:
    """OS_MAJOR_VERSION from /etc/os-release's text, as bash computes it:
    $(sed -n 's/^VERSION_ID="\\([0-9]\\+\\).*/\\1/p' /etc/os-release).

        'VERSION_ID="9.8"\\n' -> "9"      'VERSION_ID=9.8' -> ""
    """
    values = []
    for line in _lines(text):
        match = re.match(r'VERSION_ID="([0-9]+)', line)
        if match:
            values.append(match.group(1))
    return "\n".join(values).rstrip("\n")


def lsb_codename(text: str) -> str:
    """DISTRIB_CODENAME as `source /etc/lsb-release` would set it, or "".

    Handles plain, "double"- or 'single'-quoted values; the last assignment
    wins, as when sourcing.
    """
    codename = ""
    for line in _lines(text):
        match = re.fullmatch(r"\s*DISTRIB_CODENAME=(.*?)\s*", line)
        if not match:
            continue
        value = match.group(1)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        codename = value
    return codename


def apt_source_problem(codename: str, ubuntu_version: str, arch: str) -> str | None:
    """Why the apt source line can't be written (bash would write blanks), or None."""
    if not codename:
        return f"no DISTRIB_CODENAME in {_LSB_RELEASE}"
    if ubuntu_version not in _SIGNED_BY:
        return (f"no apt signing key known for Ubuntu {ubuntu_version!r} "
                "(bash sets SIGNED_BY only for 22.04 and 24.04)")
    if not arch:
        return "ARCHITECTURE_DISTRO is not set"
    return None


def underscore_version(version: str) -> str:
    """${LUSTRE_VERSION//-/_}: every "-" becomes "_"."""
    return version.replace("-", "_")


def skips_kernel(kernel: str) -> bool:
    """True for a kernel version that gets no Lustre client.

    bash: [[ "$k" == "7.0" || "$k" == 7.0.* || "$k" == 7.0-* ]]

        "7.0", "7.0.0-1004-azure", "7.0-rc1" -> True
        "7.00", "17.0", "7.1.0", " 7.0", ""  -> False
    """
    return (kernel == _SKIPPED_KERNEL
            or kernel.startswith(_SKIPPED_KERNEL + ".")
            or kernel.startswith(_SKIPPED_KERNEL + "-"))


def amlfs_apt_source(arch: str, signed_by: str, codename: str) -> str:
    """The line bash's `echo ... | tee` writes to /etc/apt/sources.list.d/amlfs.list."""
    return (f"deb [arch={arch} signed-by={signed_by}] "
            f"https://packages.microsoft.com/repos/amlfs-{codename}/ {codename} main\n")


def amlfs_yum_repo(os_major: str) -> str:
    """The six `echo -e ... >> ${REPO_PATH}` lines (no escapes to expand)."""
    return (
        "[amlfs]\n"
        "name=Azure Lustre Packages\n"
        f"baseurl=https://packages.microsoft.com/yumrepos/amlfs-el{os_major}\n"
        "enabled=1\n"
        "gpgcheck=1\n"
        f"gpgkey={_MICROSOFT_KEY_URL}\n"
    )


def skip_artifact_block(slot: int, description: str) -> str:
    """configure_lustre_dkms_skip_artifact's unquoted heredoc, expanded.

    `^$` matches no kernel name, so DKMS never builds/installs that record.
    """
    return f'# {description}\nBUILD_EXCLUSIVE_KERNEL[{slot}]="^$"\n'


def dkms_conf_path(module: str, module_version: str) -> str:
    """/etc/dkms/<module>-<version>.conf, DKMS's per-module override file."""
    return os.path.join(_DKMS_CONF_DIR, f"{module}-{module_version}.conf")


def _kernel_release() -> str:
    """The running kernel's release, like `uname -r`. Tests replace this."""
    return platform.release()


def current_kernel_version(env: dict[str, str]) -> tuple[str, str]:
    """CURRENT_KERNEL_VERSION=${KERNEL_VERSION:-$(uname -r)}, and where it came from.

    Returns (version, source), source being "KERNEL_VERSION" or "uname -r". An
    empty KERNEL_VERSION counts as unset, as with bash's `:-`.
    """
    requested = env.get("KERNEL_VERSION", "")
    if requested:
        return requested, "KERNEL_VERSION"
    return _kernel_release(), "uname -r"


# --------------------------------------------------------------------------- #
# bash's DKMS configuration functions. Each returns True, or False if a file
# couldn't be written (logged by write_file).
# --------------------------------------------------------------------------- #
def configure_lustre_dkms_no_o2ib(config_file: str) -> bool:
    """The RHEL-family LNet-over-TCP DKMS config (_NO_O2IB_CONF)."""
    return write_file(config_file, _NO_O2IB_CONF, _OP)


def configure_lustre_dkms_no_o2ib_with_tr_workaround(config_file: str) -> bool:
    """The Ubuntu 24.04 LUSTRE_DKMS_CONFIGURE_EXTRA config (_NO_O2IB_TR_CONF)."""
    return write_file(config_file, _NO_O2IB_TR_CONF, _OP)


def configure_lustre_dkms_skip_artifact(module: str, module_version: str, slot: int,
                                        description: str) -> bool:
    """Append a BUILD_EXCLUSIVE_KERNEL[<slot>]="^$" record skip to
    /etc/dkms/<module>-<version>.conf (bash `cat >>`)."""
    path = dkms_conf_path(module, module_version)
    block = skip_artifact_block(slot, description)
    _warn_if_rerun_append(path, block)
    if not write_file(path, block, _OP, append=True):
        return False
    log_info(_OP, f"DKMS will skip artifact record {slot} of {module} "
                  "(the o2ib/ko2iblnd module, not built without o2ib)",
             module=module, slot=slot, path=path)
    return True


def configure_legacy_lustre_dkms_no_o2ib(module_version: str) -> bool:
    """The Ubuntu 22.04 legacy DKMS override (_LEGACY_DKMS_CONF), then the
    record-11 skip appended to the same file."""
    module = _UBUNTU_DKMS_MODULE
    return (write_file(dkms_conf_path(module, module_version), _LEGACY_DKMS_CONF, _OP)
            and configure_lustre_dkms_skip_artifact(
                module, module_version, _UBUNTU_SKIP_SLOT, _UBUNTU_SKIP_DESCRIPTION))


def _warn_if_rerun_append(path: str, text: str) -> None:
    """Warn when `path` already contains `text`, which is about to be appended."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            already = text in handle.read()
    except OSError:
        return
    if already:
        # Kept bash issue: the lines are appended on every run, never replaced.
        log_warn(_OP, f"{path} already contains these lines; appending them again, "
                      "as bash does on a rerun (DKMS then sees them twice)",
                 path=path, appended=" | ".join(text.strip().split("\n")))


# --------------------------------------------------------------------------- #
# The steps.
# --------------------------------------------------------------------------- #
def _read_system_file(path: str) -> str | None:
    """The text of /etc/lsb-release or /etc/os-release, or None (logged)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError as exc:
        log_error(_OP, f"could not read {path}: {exc}", path=path)
        return None


def _skips_this_kernel(env: dict[str, str], distro: Distro) -> bool:
    """Whether this build's kernel is a 7.0 kernel, which gets no Lustre client (logged)."""
    kernel, source = current_kernel_version(env)
    skipped = skips_kernel(kernel)
    if skipped:
        log_info(_OP, f"Skipping Lustre client installation for kernel version {kernel}",
                 kernel_version=kernel, kernel_source=source, distro=distro.name,
                 skipped=True)
    else:
        log_info(_OP, f"Kernel version {kernel} is not {_SKIPPED_KERNEL}; installing "
                      "the Lustre client", kernel_version=kernel, kernel_source=source,
                 skipped=False)
    return skipped


def _write_amlfs_apt_source(env: dict[str, str]) -> str | None:
    """Write the AMLFS apt source; the Ubuntu version (e.g. "22.04"), or None (logged)."""
    lsb_release = _read_system_file(_LSB_RELEASE)     # bash: source /etc/lsb-release
    if lsb_release is None:
        return None
    os_release = _read_system_file(_OS_RELEASE)
    if os_release is None:
        return None
    codename = lsb_codename(lsb_release)
    ubuntu_version = ubuntu_version_from_os_release(os_release)
    arch = env.get("ARCHITECTURE_DISTRO", "")
    signed_by = _SIGNED_BY.get(ubuntu_version, "")
    log_info(_OP, f"Ubuntu {ubuntu_version} ({codename})", distro=env.get("DISTRIBUTION", ""),
             codename=codename, ubuntu_version=ubuntu_version, arch=arch,
             signed_by=signed_by)
    problem = apt_source_problem(codename, ubuntu_version, arch)
    if problem:
        log_error(_OP, problem, ubuntu_version=ubuntu_version)
        return None
    if not write_file(_APT_SOURCE, amlfs_apt_source(arch, signed_by, codename), _OP):
        return None
    return ubuntu_version


def _configure_ubuntu_dkms(ubuntu_version: str, version: str) -> bool:
    """22.04: the legacy DKMS config; any other Ubuntu: /etc/sysconfig/dkms-lustre."""
    if ubuntu_version == _LEGACY_DKMS_UBUNTU:
        log_info(_OP, "Ubuntu 22.04: legacy DKMS config (--with-o2ib=no, skip record 11)")
        return configure_legacy_lustre_dkms_no_o2ib(version)
    log_info(_OP, "Using LUSTRE_DKMS_CONFIGURE_EXTRA=--with-o2ib=no "
                  f"in {_UBUNTU_DKMS_SYSCONFIG}")
    return configure_lustre_dkms_no_o2ib_with_tr_workaround(_UBUNTU_DKMS_SYSCONFIG)


def _install_ubuntu(env: dict[str, str], version: str, kernel: str) -> str | None:
    """The Ubuntu branch; the version to record, or None (logged)."""
    ubuntu_version = _write_amlfs_apt_source(env)
    if ubuntu_version is None:
        return None
    # The userspace package goes before the DKMS config: it owns
    # /etc/sysconfig/dkms-lustre, and a pre-created conffile makes
    # noninteractive dpkg stop at a prompt.
    installed = (refresh_package_index(env, _OP)
                 and install_packages([f"lustre-client-{version}"], env, _OP)
                 and _configure_ubuntu_dkms(ubuntu_version, version)
                 and install_packages([f"amlfs-lustre-client-dkms-{version}"], env, _OP)
                 and check_dkms_status([_UBUNTU_DKMS_MODULE], _OP, env, kernel=kernel))
    return version if installed else None


def _install_rhel(env: dict[str, str], version: str, kernel: str) -> str | None:
    """The RHEL-family branch; the underscore version to record, or None (logged)."""
    version_us = underscore_version(version)
    os_release = _read_system_file(_OS_RELEASE)
    if os_release is None:
        return None
    os_major = os_major_from_os_release(os_release)
    log_info(_OP, f"RHEL family: el{os_major}, Lustre version {version_us}",
             distro=env.get("DISTRIBUTION", ""), os_major=os_major, version=version_us)
    if not os_major:
        log_error(_OP, f"could not read a numeric VERSION_ID from {_OS_RELEASE}",
                  path=_OS_RELEASE)
        return None
    packages = [f"lustre-client-dkms-{version_us}", f"lustre-client-{version_us}-devel"]
    installed = (run(["rpm", "--import", _MICROSOFT_KEY_URL], _OP, env=env)
                 and write_file(_YUM_REPO, amlfs_yum_repo(os_major), _OP)
                 and configure_lustre_dkms_no_o2ib(_EL_SYSCONFIG)
                 and configure_lustre_dkms_skip_artifact(
                     _EL_DKMS_MODULE, version_us, _EL_SKIP_SLOT, _EL_SKIP_DESCRIPTION)
                 and install_packages(packages, env, _OP, options=_EL_DNF_OPTIONS)
                 and check_dkms_status([_EL_DKMS_MODULE], _OP, env, kernel=kernel))
    return version_us if installed else None


# Keyed by Distro.family; anything else takes bash's `else` (RHEL family).
_INSTALLERS = {"ubuntu": _install_ubuntu}


def install(env: dict[str, str]) -> int:
    """Install the AMLFS Lustre client and record its version. Returns 0 or 3."""
    distro = Distro.from_env(env)
    if distro.is_azurelinux:
        log_warn(_OP, "distros/azurelinux3.0/install.sh does not run "
                      "install_lustre_client.sh, but the build plan runs this step "
                      "on every distro; continuing with bash's RHEL-family branch",
                 distro=distro.name)
    if _skips_this_kernel(env, distro):
        return 0

    cfg = require_config("lustre", env, _OP, "version")
    if cfg is None:
        return 3
    version = str(cfg["version"])
    kernel = _kernel_release()          # the DKMS check uses the running kernel
    log_info(_OP, f"Installing the Lustre client {version}", version=version,
             distro=distro.name, arch=env.get("ARCHITECTURE", ""),
             gpu=env.get("GPU", ""), kernel=kernel,
             family="ubuntu" if distro.is_ubuntu else "rhel")

    installer = _INSTALLERS.get(distro.family, _install_rhel)
    recorded = installer(env, version, kernel)
    if recorded is None or not record_version("LUSTRE", recorded, _OP):
        return 3
    log_info(_OP, "Lustre client installed", version=recorded, distro=distro.name,
             kernel=kernel)
    return 0
