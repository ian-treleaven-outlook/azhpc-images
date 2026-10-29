"""install_lustre_client.py — Python port of components/install_lustre_client.sh.

What this step does
-------------------
Installs the Azure Managed Lustre (AMLFS) client: the Lustre userspace tools
and a DKMS kernel module built for the image's kernel, from Microsoft's AMLFS
package repo. Azure VMs reach Lustre servers over TCP, not InfiniBand, so DKMS
is configured to build Lustre without o2ib (the InfiniBand LNet driver) and to
skip that driver's artifact. Afterwards DKMS must report the module installed
for the running kernel (utils/dkms.py, a port of bash check_dkms_status), and
the version is recorded as LUSTRE in the component-versions file.

This step must stay immediately before install-mpifileutils, which builds
against the Lustre client.

Per distro family (bash tests `$DISTRIBUTION == *"ubuntu"*`, else RHEL family):

  * Ubuntu  Read DISTRIB_CODENAME from /etc/lsb-release and the Ubuntu version
            (VERSION_ID) from /etc/os-release, as bash does. Write the AMLFS
            apt source (signed-by key per version), `apt-get update`, install
            lustre-client-<v> first (it owns /etc/sysconfig/dkms-lustre; a
            pre-created conffile makes noninteractive dpkg stop at a prompt),
            then the DKMS configuration:
              22.04  on AMD GPUs, upgrade dkms to NVIDIA's 3.4.1 .deb if the
                     installed one is older; then write the legacy
                     /etc/dkms/lustre-client-modules-<v>.conf (a MAKE= line
                     configuring --with-o2ib=no) and skip artifact record 11.
              24.04  write /etc/sysconfig/dkms-lustre (LUSTRE_DKMS_CONFIGURE_EXTRA).
            Then install amlfs-lustre-client-dkms-<v> and check the DKMS module
            lustre-client-modules.
  * RHEL    (AlmaLinux, Rocky, RHEL) The version's "-" become "_". Import the
            Microsoft key, write /etc/yum.repos.d/amlfs.repo for el<major>
            (major from VERSION_ID in /etc/os-release, as bash does), write
            /etc/sysconfig/lustre (`echo --with-o2ib=no`), append the record-3
            skip to /etc/dkms/lustre-client-<v>.conf, and, when the running
            kernel's timer.h no longer has from_timer() (RHEL 9.8 kernels
            5.14.0-687+), copy the LU-20071 patch and append PATCH[0]. Then
            `dnf install` lustre-client-dkms and -devel and check the DKMS
            module lustre-client. The underscore version is recorded.

The original bash (helper functions are mirrored one-to-one below, each with
its bash body in its docstring):

    lustre_metadata=$(get_component_config "lustre")
    LUSTRE_VERSION=$(jq -r '.version' <<< $lustre_metadata)
    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        source /etc/lsb-release
        UBUNTU_VERSION=$(cat /etc/os-release | grep VERSION_ID | cut -d= -f2 | cut -d\\" -f2)
        if [ $UBUNTU_VERSION == 24.04 ]; then
            SIGNED_BY="/usr/share/keyrings/microsoft-prod.gpg"
        elif [ $UBUNTU_VERSION == 22.04 ]; then
            SIGNED_BY="/etc/apt/trusted.gpg.d/microsoft-prod.gpg"
        fi
        echo "deb [arch=$ARCHITECTURE_DISTRO signed-by=$SIGNED_BY] https://packages.microsoft.com/repos/amlfs-${DISTRIB_CODENAME}/ ${DISTRIB_CODENAME} main" | tee /etc/apt/sources.list.d/amlfs.list
        apt-get update
        LUSTRE_CLIENT_PACKAGE="lustre-client-${LUSTRE_VERSION}"
        LUSTRE_PACKAGE="amlfs-lustre-client-dkms-${LUSTRE_VERSION}"
        apt-get install -y "${LUSTRE_CLIENT_PACKAGE}"
        if [[ $UBUNTU_VERSION == 22.04 ]]; then
            install_cuda_dkms_3_4_1_for_jammy_amd
            configure_legacy_lustre_dkms_no_o2ib "${LUSTRE_VERSION}"
        else
            configure_lustre_dkms_no_o2ib_with_tr_workaround /etc/sysconfig/dkms-lustre
        fi
        apt-get install -y "${LUSTRE_PACKAGE}"
        check_dkms_status lustre-client-modules
    else
        LUSTRE_VERSION_UNDERSCORE=${LUSTRE_VERSION//-/_}
        OS_MAJOR_VERSION=$(sed -n 's/^VERSION_ID="\\([0-9]\\+\\).*/\\1/p' /etc/os-release)
        REPO_PATH=/etc/yum.repos.d/amlfs.repo
        rpm --import https://packages.microsoft.com/keys/microsoft.asc
        echo -e "[amlfs]" > ${REPO_PATH}
        echo -e "name=Azure Lustre Packages" >> ${REPO_PATH}
        ... (baseurl, enabled, gpgcheck, gpgkey; see amlfs_yum_repo())
        LUSTRE_PACKAGES=("lustre-client-dkms-${LUSTRE_VERSION_UNDERSCORE}"
                         "lustre-client-${LUSTRE_VERSION_UNDERSCORE}-devel")
        configure_lustre_dkms_no_o2ib /etc/sysconfig/lustre
        configure_lustre_dkms_skip_artifact lustre-client "${LUSTRE_VERSION_UNDERSCORE}" 3 \\
            "EL ko2iblnd is record 3; --with-o2ib=no intentionally skips it."
        configure_lustre_dkms_lu20071_patch "${LUSTRE_VERSION_UNDERSCORE}"
        dnf install -y --disableexcludes=main --refresh "${LUSTRE_PACKAGES[@]}"
        check_dkms_status lustre-client
        LUSTRE_VERSION=${LUSTRE_VERSION_UNDERSCORE}
    fi
    write_component_version "LUSTRE" ${LUSTRE_VERSION}

Every generated file was rendered with the real bash functions/commands in WSL
and compared byte for byte (fixtures in tests/python/fixtures/install_lustre_client/).

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-lustre" Step. Deleting `action=` makes the build run the bash script
again, which is why the .sh file is kept. install() returns 0 for success or 3
for failure; it logs problems and never lets an exception escape.

Bash issues, kept and warned about (log_warn when they apply)
-------------------------------------------------------------
  * configure_lustre_dkms_skip_artifact and configure_lustre_dkms_lu20071_patch
    APPEND to /etc/dkms/<module>-<version>.conf and nothing empties it first
    (on Ubuntu 22.04 the legacy config rewrites it just before, on the RHEL
    family nothing does). A rerun on the same machine appends the same lines
    again. Warned when the file already holds the text being appended.
  * distros/azurelinux3.0/install.sh never runs install_lustre_client.sh, but
    the build plan's install-lustre Step has no distro gate, so on Azure Linux
    it runs and takes bash's RHEL-family branch. Warned at the start (gating
    is left alone; versions.json has no Azure Linux "lustre" entry, so the
    step then fails on the missing version, as bash would fail at dnf).
  * In configure_lustre_dkms_lu20071_patch, an unreadable timer.h makes bash's
    `grep -q ... && return 0` fail (without stopping `set -e`), so the patch is
    applied. Kept, and warned.

Deliberate differences from bash
--------------------------------
  * A missing "lustre" version fails before changing anything; bash would
    install "lustre-client-null".
  * Missing ARCHITECTURE_DISTRO, an empty DISTRIB_CODENAME, or an Ubuntu
    version other than 22.04/24.04 (no SIGNED_BY) fails before the apt source
    is written; bash would write a source line with blank values. An empty OS
    major version on the RHEL family fails before the repo is written; bash
    would write "baseurl=.../amlfs-el".
  * The jammy-AMD dkms .deb is downloaded into this step's own work directory
    (_WORK_DIR, emptied before and removed after) instead of bash's current
    directory (the distro build dir), and installed by absolute path.
  * Before `dkms --version` runs, a missing dkms command is detected with
    shutil.which() (no "command not found" error is logged); bash's
    `2>/dev/null || true` gives the same empty answer.
  * The DISTRIB_CODENAME assignment in /etc/lsb-release is read by a small
    parser (plain, "double"- or 'single'-quoted values; the last assignment
    wins) instead of `source`-ing the file. That covers how Ubuntu writes it.
"""

from __future__ import annotations

import os
import platform
import re
import shutil

from utils.component_config import config_for, write_component_version
from utils.dkms import check_dkms_status
from utils.download import download_and_verify
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.process import exec_program, run_capture

# The Step's `op` label in build_config.py; used on every log line.
_OP = "install-lustre"

# --------------------------------------------------------------------------- #
# System paths. Module constants so tests can point them at temporary dirs.
# --------------------------------------------------------------------------- #
_LSB_RELEASE = "/etc/lsb-release"
_OS_RELEASE = "/etc/os-release"
_APT_SOURCE = "/etc/apt/sources.list.d/amlfs.list"
_YUM_REPO = "/etc/yum.repos.d/amlfs.repo"
_DKMS_CONF_DIR = "/etc/dkms"
_UBUNTU_DKMS_SYSCONFIG = "/etc/sysconfig/dkms-lustre"
_EL_SYSCONFIG = "/etc/sysconfig/lustre"
_MODULES_DIR = "/lib/modules"
# This step's own download directory (see "Deliberate differences").
_WORK_DIR = "/tmp/lustre-build"

# Ubuntu apt signing key per version (bash's SIGNED_BY if/elif).
_SIGNED_BY = {
    "24.04": "/usr/share/keyrings/microsoft-prod.gpg",
    "22.04": "/etc/apt/trusted.gpg.d/microsoft-prod.gpg",
}

_MICROSOFT_KEY_URL = "https://packages.microsoft.com/keys/microsoft.asc"

# install_cuda_dkms_3_4_1_for_jammy_amd's .deb and the minimum dkms version.
_JAMMY_DKMS_URL = ("https://developer.download.nvidia.com/compute/cuda/repos/"
                   "ubuntu2204/x86_64/dkms_3.4.1-1ubuntu1_all.deb")
_JAMMY_DKMS_SHA256 = "16ce508e74cbe8426fe19c1c56de5ea6e9f3dbe05d85ba5cbf5a8a271d34c2be"
_JAMMY_DKMS_MIN_VERSION = "3.4.1"

# The LU-20071 patch, under COMPONENT_DIR, and its name inside the DKMS dir.
_LU20071_PATCH_SOURCE = "patches/lustre-client-lu-20071-timer-container-of.patch"
_LU20071_PATCH_NAME = "lu-20071-timer-container-of.patch"

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

# configure_lustre_dkms_lu20071_patch's appended line.
_LU20071_PATCH_LINE = f'PATCH[0]="{_LU20071_PATCH_NAME}"\n'


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
    """`cut -d<delimiter> -f2` on one line.

    A line without the delimiter is printed whole (cut has no -s here).
    """
    parts = line.split(delimiter)
    return parts[1] if len(parts) > 1 else line


def ubuntu_version_from_os_release(text: str) -> str:
    """UBUNTU_VERSION as bash computes it from /etc/os-release's text.

    Mirrors  $(cat /etc/os-release | grep VERSION_ID | cut -d= -f2 | cut -d\\" -f2):
    every line CONTAINING "VERSION_ID", then field 2 split on "=", then field 2
    split on '"'; $(...) drops trailing newlines.

        'VERSION_ID="22.04"\\n' -> "22.04"
    """
    values = [_cut_field2(_cut_field2(line, "="), '"')
              for line in _lines(text) if "VERSION_ID" in line]
    return "\n".join(values).rstrip("\n")


def os_major_from_os_release(text: str) -> str:
    """OS_MAJOR_VERSION as bash computes it from /etc/os-release's text.

    Mirrors  $(sed -n 's/^VERSION_ID="\\([0-9]\\+\\).*/\\1/p' /etc/os-release):
    the leading digits of each quoted VERSION_ID line.

        'VERSION_ID="9.8"\\n' -> "9"      'VERSION_ID=9.8' -> ""
    """
    values = []
    for line in _lines(text):
        match = re.match(r'VERSION_ID="([0-9]+)', line)
        if match:
            values.append(match.group(1))
    return "\n".join(values).rstrip("\n")


def lsb_codename(text: str) -> str:
    """DISTRIB_CODENAME as `source /etc/lsb-release` would set it.

    Handles the assignments Ubuntu writes: DISTRIB_CODENAME=jammy, with or
    without "double" or 'single' quotes. The last assignment wins, as when
    sourcing. "" when there is none.
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


def dkms_version_from_output(output: str) -> str:
    """The version bash parses from `dkms --version`.

    Mirrors  $(dkms --version 2>/dev/null | sed -n 's/^dkms-\\(.*\\)$/\\1/p'):
    whatever follows "dkms-" on each line that starts with it.

        "dkms-3.0.10\\n" -> "3.0.10"      "dkms: 3.4.1\\n" -> ""
    """
    values = [line[len("dkms-"):] for line in _lines(output) if line.startswith("dkms-")]
    return "\n".join(values).rstrip("\n")


def underscore_version(version: str) -> str:
    """${LUSTRE_VERSION//-/_}: every "-" becomes "_"."""
    return version.replace("-", "_")


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


def _command_exists(name: str, env: dict[str, str]) -> bool:
    """`command -v <name>` on the build env's PATH. Tests replace this."""
    return shutil.which(name, path=env.get("PATH")) is not None


# --------------------------------------------------------------------------- #
# File writing (raises OSError; install() turns that into a failure).
# --------------------------------------------------------------------------- #
def _write_file(path: str, content: str, *, append: bool = False) -> str:
    """Write (`cat >`, `tee`) or append (`cat >>`) `content`; log what happened.

    Creates the parent directory first (bash's `mkdir -p "$(dirname ...)"`).
    Returns "created", "replaced" or "appended".
    """
    existed = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    if not existed:
        action = "created"
    else:
        action = "appended" if append else "replaced"
    log_info(_OP, f"{action.capitalize()} {path}", path=path, action=action)
    log_debug(_OP, f"Wrote to {path}:\n{content}", path=path)
    return action


def _warn_if_rerun_append(path: str, text: str) -> None:
    """Warn when `path` already contains `text`, which is about to be appended."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            already = text in handle.read()
    except FileNotFoundError:
        return
    if already:
        # BASH ISSUE (kept): the lines are appended on every run, never replaced.
        log_warn(_OP, f"{path} already contains these lines; appending them again, "
                      "as bash does on a rerun (DKMS then sees them twice)",
                 path=path, appended=" | ".join(text.strip().split("\n")))


# --------------------------------------------------------------------------- #
# The bash helper functions, one to one.
# --------------------------------------------------------------------------- #
def configure_lustre_dkms_no_o2ib(config_file: str) -> None:
    """Write the RHEL-family LNet-over-TCP DKMS config.

        mkdir -p "$(dirname "${config_file}")"
        cat > "${config_file}" <<'EOF'
        ...  (_NO_O2IB_CONF)
        EOF
    """
    _write_file(config_file, _NO_O2IB_CONF)


def install_cuda_dkms_3_4_1_for_jammy_amd(env: dict[str, str]) -> bool:
    """On Ubuntu 22.04 AMD builds, make sure dkms is at least 3.4.1.

        [[ "${DISTRIBUTION}" == "ubuntu22.04" && "${GPU:-}" == "AMD" ]] || return 0
        current_version=$(dkms --version 2>/dev/null | sed -n 's/^dkms-\\(.*\\)$/\\1/p' || true)
        if [[ -n "${current_version}" ]] && dpkg --compare-versions "${current_version}" ge 3.4.1; then
            return 0
        fi
        rm -f "./${dkms_deb}"
        download_and_verify "${dkms_url}" "${dkms_sha256}"
        apt-get install -y "./${dkms_deb}"
        rm -f "./${dkms_deb}"

    Returns True on success (including "not needed"), False after logging.
    """
    distribution = env.get("DISTRIBUTION", "")
    gpu = env.get("GPU", "")
    if not (distribution == "ubuntu22.04" and gpu == "AMD"):
        log_info(_OP, "dkms 3.4.1 workaround not needed (only for ubuntu22.04 "
                      "with AMD GPUs)", distro=distribution, gpu=gpu)
        return True

    current = ""
    # Bash sends "command not found" to /dev/null and gets ""; check first so
    # a missing dkms isn't logged as a failed command.
    if not _command_exists("dkms", env):
        log_info(_OP, "dkms is not installed yet")
    else:
        # The exit code is ignored, as in bash (`|| true`, and sed's status wins).
        _rc, output = run_capture(["dkms", "--version"], _OP, env=env,
                                  failure_is_error=False)
        current = dkms_version_from_output(output)

    if current:
        rc, _out = run_capture(["dpkg", "--compare-versions", current, "ge",
                                _JAMMY_DKMS_MIN_VERSION], _OP, env=env,
                               failure_is_error=False)
        if rc == 0:
            log_info(_OP, f"dkms {current} is at least {_JAMMY_DKMS_MIN_VERSION}; "
                          "keeping it", dkms_version=current)
            return True
        log_info(_OP, f"dkms {current} is older than {_JAMMY_DKMS_MIN_VERSION}",
                 dkms_version=current)

    log_info(_OP, f"Ubuntu 22.04 AMD: installing dkms {_JAMMY_DKMS_MIN_VERSION} "
                  "from NVIDIA's CUDA repo before the Lustre DKMS package",
             dkms_version=current, source=_JAMMY_DKMS_URL)
    shutil.rmtree(_WORK_DIR, ignore_errors=True)          # bash: rm -f "./${dkms_deb}"
    try:
        try:
            deb = download_and_verify(_JAMMY_DKMS_URL, _JAMMY_DKMS_SHA256,
                                      dest_dir=_WORK_DIR)
        except Exception as exc:  # network, checksum or disk errors
            log_error(_OP, f"could not download/verify the dkms .deb: {exc}",
                      url=_JAMMY_DKMS_URL)
            return False
        if exec_program(["apt-get", "install", "-y", str(deb)], _OP, env=env) != 0:
            log_error(_OP, "could not install the dkms 3.4.1 .deb", path=str(deb))
            return False
        log_info(_OP, f"Installed dkms {_JAMMY_DKMS_MIN_VERSION}", package=str(deb))
        return True
    finally:
        shutil.rmtree(_WORK_DIR, ignore_errors=True)      # bash: rm -f "./${dkms_deb}"


def configure_lustre_dkms_skip_artifact(module: str, module_version: str, slot: int,
                                        description: str) -> None:
    """Append a BUILD_EXCLUSIVE_KERNEL[<slot>]="^$" record skip.

        local dkms_conf=/etc/dkms/${module}-${module_version}.conf
        mkdir -p "$(dirname "${dkms_conf}")"
        cat >> "${dkms_conf}" <<EOF
        # ${description}
        BUILD_EXCLUSIVE_KERNEL[${slot}]="^$"
        EOF
    """
    path = dkms_conf_path(module, module_version)
    block = skip_artifact_block(slot, description)
    _warn_if_rerun_append(path, block)
    _write_file(path, block, append=True)
    log_info(_OP, f"DKMS will skip artifact record {slot} of {module} "
                  "(the o2ib/ko2iblnd module, not built without o2ib)",
             module=module, slot=slot, path=path)


def configure_lustre_dkms_lu20071_patch(module_version: str, env: dict[str, str],
                                        kernel: str) -> None:
    """Add the LU-20071 patch when the running kernel no longer has from_timer().

        local kernel_header=/lib/modules/$(uname -r)/build/include/linux/timer.h
        # RHEL 9.8 kernels 5.14.0-687+ dropped from_timer(); LU-20071 rewires
        # Lustre's cfs_from_timer wrapper to timer_container_of.
        [[ -f "${kernel_header}" ]] || return 0
        grep -q 'from_timer' "${kernel_header}" && return 0
        mkdir -p "${patch_dir}"
        cp "${patch_file}" "${patch_dir}/lu-20071-timer-container-of.patch"
        mkdir -p "$(dirname "${dkms_conf}")"
        cat >> "${dkms_conf}" <<'EOF'
        PATCH[0]="lu-20071-timer-container-of.patch"
        EOF
    """
    module = _EL_DKMS_MODULE
    header = os.path.join(_MODULES_DIR, kernel, "build", "include", "linux", "timer.h")
    if not os.path.isfile(header):
        log_info(_OP, f"LU-20071 patch not needed: no kernel header {header}",
                 kernel=kernel, path=header, lu20071_patch=False)
        return
    try:
        with open(header, "rb") as handle:
            has_from_timer = b"from_timer" in handle.read()
    except OSError as exc:
        # BASH ISSUE (kept): grep fails on an unreadable file, so `&& return 0`
        # doesn't run and bash goes on to apply the patch.
        log_warn(_OP, f"could not read {header} ({exc}); applying the LU-20071 "
                      "patch, as bash does when grep fails", path=header)
        has_from_timer = False
    if has_from_timer:
        log_info(_OP, "LU-20071 patch not needed: the kernel still has from_timer()",
                 kernel=kernel, path=header, lu20071_patch=False)
        return

    log_info(_OP, "LU-20071 patch needed: the kernel's timer.h has no from_timer() "
                  "(RHEL 9.8 kernels 5.14.0-687+)",
             kernel=kernel, path=header, lu20071_patch=True)
    patch_file = os.path.join(env.get("COMPONENT_DIR", ""), _LU20071_PATCH_SOURCE)
    patch_dir = os.path.join(_DKMS_CONF_DIR, module, "patches")
    target = os.path.join(patch_dir, _LU20071_PATCH_NAME)
    os.makedirs(patch_dir, exist_ok=True)
    existed = os.path.exists(target)
    shutil.copyfile(patch_file, target)
    action = "replaced" if existed else "created"
    log_info(_OP, f"Copied {patch_file} to {target} ({action})",
             source=patch_file, path=target, action=action)

    path = dkms_conf_path(module, module_version)
    _warn_if_rerun_append(path, _LU20071_PATCH_LINE)
    _write_file(path, _LU20071_PATCH_LINE, append=True)


def configure_lustre_dkms_no_o2ib_with_tr_workaround(config_file: str) -> None:
    """Write the Ubuntu 24.04 LUSTRE_DKMS_CONFIGURE_EXTRA config.

        mkdir -p "$(dirname "${config_file}")"
        cat > "${config_file}" <<'EOF'
        ...  (_NO_O2IB_TR_CONF)
        EOF
    """
    _write_file(config_file, _NO_O2IB_TR_CONF)


def configure_legacy_lustre_dkms_no_o2ib(module_version: str) -> None:
    """Write the Ubuntu 22.04 legacy DKMS override, then skip record 11.

        mkdir -p /etc/dkms
        cat > "${dkms_conf}" <<'EOF'
        ...  (_LEGACY_DKMS_CONF)
        EOF
        configure_lustre_dkms_skip_artifact "${module}" "${module_version}" 11 \\
            "Ubuntu 22.04 ko2iblnd is record 11; --with-o2ib=no intentionally skips it."
    """
    module = _UBUNTU_DKMS_MODULE
    _write_file(dkms_conf_path(module, module_version), _LEGACY_DKMS_CONF)
    configure_lustre_dkms_skip_artifact(module, module_version, _UBUNTU_SKIP_SLOT,
                                        _UBUNTU_SKIP_DESCRIPTION)


# --------------------------------------------------------------------------- #
# Per-distro handlers. Each returns the version to record, or None on failure
# (already logged). File errors raise OSError, handled in install().
# --------------------------------------------------------------------------- #
def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _run(command: list[str], env: dict[str, str]) -> bool:
    """Run one command; True if it succeeded (exec_program logs a failure)."""
    return exec_program(command, _OP, env=env) == 0


def _install_ubuntu(env: dict[str, str], version: str, kernel: str) -> str | None:
    codename = lsb_codename(_read(_LSB_RELEASE))       # bash: source /etc/lsb-release
    ubuntu_version = ubuntu_version_from_os_release(_read(_OS_RELEASE))
    arch = env.get("ARCHITECTURE_DISTRO", "")
    signed_by = _SIGNED_BY.get(ubuntu_version, "")
    log_info(_OP, f"Ubuntu {ubuntu_version} ({codename})", distro=env.get("DISTRIBUTION", ""),
             codename=codename, ubuntu_version=ubuntu_version, arch=arch,
             signed_by=signed_by)
    # Deliberate differences: bash would write blanks into the apt source line.
    if not codename:
        log_error(_OP, f"no DISTRIB_CODENAME in {_LSB_RELEASE}", path=_LSB_RELEASE)
        return None
    if not signed_by:
        log_error(_OP, f"no apt signing key known for Ubuntu {ubuntu_version!r} "
                       "(bash sets SIGNED_BY only for 22.04 and 24.04)",
                  ubuntu_version=ubuntu_version)
        return None
    if not arch:
        log_error(_OP, "ARCHITECTURE_DISTRO is not set")
        return None

    _write_file(_APT_SOURCE, amlfs_apt_source(arch, signed_by, codename))
    if not _run(["apt-get", "update"], env):
        return None

    client_package = f"lustre-client-{version}"
    dkms_package = f"amlfs-lustre-client-dkms-{version}"
    # Install the userspace package first because it owns /etc/sysconfig/dkms-lustre.
    # Pre-creating that conffile makes noninteractive dpkg stop at a prompt.
    if not _run(["apt-get", "install", "-y", client_package], env):
        return None
    log_info(_OP, f"Installed {client_package}", package=client_package)

    if ubuntu_version == "22.04":
        log_info(_OP, "Ubuntu 22.04: legacy DKMS config (--with-o2ib=no, skip record 11)")
        if not install_cuda_dkms_3_4_1_for_jammy_amd(env):
            return None
        configure_legacy_lustre_dkms_no_o2ib(version)
    else:
        log_info(_OP, "Using LUSTRE_DKMS_CONFIGURE_EXTRA=--with-o2ib=no "
                      f"in {_UBUNTU_DKMS_SYSCONFIG}")
        configure_lustre_dkms_no_o2ib_with_tr_workaround(_UBUNTU_DKMS_SYSCONFIG)

    if not _run(["apt-get", "install", "-y", dkms_package], env):
        return None
    log_info(_OP, f"Installed {dkms_package}", package=dkms_package)

    if not check_dkms_status([_UBUNTU_DKMS_MODULE], _OP, env, kernel=kernel):
        return None
    return version


def _install_rhel(env: dict[str, str], version: str, kernel: str) -> str | None:
    version_us = underscore_version(version)
    os_major = os_major_from_os_release(_read(_OS_RELEASE))
    log_info(_OP, f"RHEL family: el{os_major}, Lustre version {version_us}",
             distro=env.get("DISTRIBUTION", ""), os_major=os_major,
             version=version_us)
    if not os_major:
        # Deliberate difference: bash would write "baseurl=.../amlfs-el".
        log_error(_OP, f"could not read a numeric VERSION_ID from {_OS_RELEASE}",
                  path=_OS_RELEASE)
        return None

    if not _run(["rpm", "--import", _MICROSOFT_KEY_URL], env):
        return None
    _write_file(_YUM_REPO, amlfs_yum_repo(os_major))

    packages = [f"lustre-client-dkms-{version_us}", f"lustre-client-{version_us}-devel"]
    configure_lustre_dkms_no_o2ib(_EL_SYSCONFIG)
    configure_lustre_dkms_skip_artifact(_EL_DKMS_MODULE, version_us, _EL_SKIP_SLOT,
                                        _EL_SKIP_DESCRIPTION)
    configure_lustre_dkms_lu20071_patch(version_us, env, kernel)

    if not _run(["dnf", "install", "-y", "--disableexcludes=main", "--refresh",
                 *packages], env):
        return None
    log_info(_OP, f"Installed {' '.join(packages)}", packages=" ".join(packages))

    if not check_dkms_status([_EL_DKMS_MODULE], _OP, env, kernel=kernel):
        return None
    return version_us


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def install(env: dict[str, str]) -> int:
    """Install the AMLFS Lustre client and record its version. Returns 0 or 3."""
    distribution = env.get("DISTRIBUTION", "")
    kernel = _kernel_release()
    is_ubuntu = "ubuntu" in distribution

    if "azurelinux" in distribution:
        # BASH ISSUE (kept): see the module docstring.
        log_warn(_OP, "distros/azurelinux3.0/install.sh does not run "
                      "install_lustre_client.sh, but the build plan runs this step "
                      "on every distro; continuing with bash's RHEL-family branch",
                 distro=distribution)

    cfg = config_for("lustre", env)
    if not isinstance(cfg, dict) or not cfg.get("version"):
        log_error(_OP, "could not resolve lustre version from versions.json",
                  distro=distribution, arch=env.get("ARCHITECTURE", ""))
        return 3
    version = str(cfg["version"])

    log_info(_OP, f"Installing the Lustre client {version}", version=version,
             distro=distribution, arch=env.get("ARCHITECTURE", ""),
             gpu=env.get("GPU", ""), kernel=kernel,
             family="ubuntu" if is_ubuntu else "rhel")

    handler = _install_ubuntu if is_ubuntu else _install_rhel
    try:
        recorded = handler(env, version, kernel)
    except OSError as exc:
        log_error(_OP, f"file operation failed: {exc}",
                  path=getattr(exc, "filename", None) or "")
        return 3
    if recorded is None:
        log_error(_OP, "Lustre client installation failed", distro=distribution,
                  version=version)
        return 3

    try:
        write_component_version("LUSTRE", recorded)
    except OSError as exc:
        log_error(_OP, f"could not record LUSTRE={recorded}: {exc}",
                  component="LUSTRE", version=recorded)
        return 3
    log_info(_OP, f"Recorded LUSTRE version {recorded}", component="LUSTRE",
             version=recorded)

    log_info(_OP, "Lustre client installed", version=recorded, distro=distribution,
             kernel=kernel,
             dkms_module=_UBUNTU_DKMS_MODULE if is_ubuntu else _EL_DKMS_MODULE)
    return 0
