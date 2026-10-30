"""install_rocm.py — Python port of components/install_rocm.sh.

What this step does
-------------------
ROCm is AMD's GPU software stack (kernel driver, runtime, compilers, math and
communication libraries). This step installs it on AMD GPU images (MI200 /
MI300X / MI300HF) and sets up the machine so the GPUs work after boot:

  1. Install ROCm, with one branch per distro family:
       * Ubuntu 26.04  ROCm 10 from AMD's apt repos: download and dearmor the
                       two AMD signing keys into /etc/apt/keyrings, write three
                       apt source files (ROCm core, the amdgpu driver version
                       from versions.json's "amdgpu" entry, and RVS), set up
                       the amdgpu DKMS patch (below), install the amdgpu DKMS
                       driver and check DKMS built it, then install the ROCm
                       core SDK and RVS. The ROCm version recorded is the one
                       the SDK reports in /opt/rocm/core/.info/version, and the
                       AMDGPU and RVS package versions are recorded too.
                       /opt/rocm/lib is added to the dynamic linker's path.
       * other Ubuntu  download + verify the `amdgpu-install` .deb from
                       versions.json, install it with apt, set up the amdgpu
                       DKMS patch, then run `amdgpu-install -y
                       --usecase=graphics` and apt-get install an explicit list
                       of ROCm packages (the `rocm` metapackage minus MIVisionX
                       and rocDecode, whose FFmpeg/Qt/... dependencies have
                       CVEs that block publishing). Ubuntu 22.04 also gets
                       libstdc++-12-dev. Then install rocm-bandwidth-test.
       * Azure Linux   install the AMD repo package, the GPU kernel drivers for
                       the running kernel and the amdgpu packages, append an
                       [amd_rocm] repo to /etc/yum.repos.d/amd_rocm.repo for
                       the ROCm version, then dnf-install the ROCm packages.
     The amdgpu DKMS patch (Ubuntu only) backports a kernel API change in
     pci_resize_resource() so the amdgpu module builds on newer kernels. It is
     copied to /etc/dkms/amdgpu/patches/, and /etc/dkms/amdgpu.conf tells DKMS
     to apply it only when the driver source doesn't already have AMD's own
     fix (or this backport).
  2. Write a udev rules file that gives every user access to the GPUs
     (/dev/kfd and the DRM render nodes), and reload/trigger udev.
  3. Record the ROCM version in the component-versions file.
  4. Ubuntu only: blacklist the amdgpu module (append to
     /etc/modprobe.d/blacklist.conf) and rebuild the initramfs, so amdgpu is
     NOT loaded automatically at boot.
  5. Install /etc/init.d/initamdgpu.sh plus a systemd unit (rocmstartup.service)
     that runs it at boot. The script waits (up to ~15 minutes) for all the
     GPUs to appear on the PCI bus, then loads amdgpu itself with the right
     options for the GPU model. Then start and enable the service.

The original bash, for comparison (abridged; the full file texts are the
constants below):

    rocm_metadata=$(get_component_config "rocm")
    rocm_version=$(jq -r '.version' <<< $rocm_metadata)

    configure_amdgpu_dkms_patches() {
        mkdir -p /etc/dkms/amdgpu/patches
        cp "${COMPONENT_DIR}/patches/amdgpu-pci-resize-resource-exclude-bars.patch" \\
            /etc/dkms/amdgpu/patches/pci-resize-resource-exclude-bars.patch
        cat > /etc/dkms/amdgpu.conf <<'EOF'
        ...  (_AMDGPU_DKMS_CONF)
        EOF
    }

    if [[ $DISTRIBUTION == "ubuntu26.04" ]]; then
        driver_metadata=$(get_component_config "amdgpu")
        driver_version=$(jq -er '.version' <<< "$driver_metadata")
        driver_url="https://repo.radeon.com/amdgpu/${driver_version}/ubuntu"
        rocm_package="amdrocm-core-sdk${rocm_version}"
        apt install -y ca-certificates wget gnupg dkms "linux-headers-$(uname -r)"
        install -d -m 0755 /etc/apt/keyrings
        wget -O /tmp/amdrocm-packages.key https://stable.repo.amd.com/rocm/gpg/packages.gpg
        gpg --batch --yes --dearmor -o /etc/apt/keyrings/amdrocm.gpg /tmp/amdrocm-packages.key
        wget -O /tmp/amdgpu-packages.key https://repo.radeon.com/rocm/rocm.gpg.key
        gpg --batch --yes --dearmor -o /etc/apt/keyrings/amdgpu.gpg /tmp/amdgpu-packages.key
        chmod 0644 /etc/apt/keyrings/amdrocm.gpg /etc/apt/keyrings/amdgpu.gpg
        rm -f /tmp/amdrocm-packages.key /tmp/amdgpu-packages.key
        cat > /etc/apt/sources.list.d/amdrocm-stable.sources <<EOF ... EOF
        cat > /etc/apt/sources.list.d/amdgpu.list <<EOF ... ${driver_url} ... EOF
        cat > /etc/apt/sources.list.d/rvs.list <<EOF ... EOF
        apt update
        configure_amdgpu_dkms_patches
        apt install -y amdgpu-dkms amdgpu-dkms-firmware
        check_dkms_status amdgpu
        apt install -y "$rocm_package" amdrocm10-rvs
        rocm_version=$(cat /opt/rocm/core/.info/version)
        write_component_version "AMDGPU" "$(dpkg-query -W -f='${Version}' amdgpu-dkms)"
        rvs_version=$(dpkg-query -W -f='${Version}' amdrocm10-rvs)
        write_component_version "RVS" "$rvs_version"
        echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf
        ldconfig
    elif [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        rocm_url=$(jq -r '.url' <<< $rocm_metadata)
        rocm_sha256=$(jq -r '.sha256' <<< $rocm_metadata)
        DEBPACKAGE=$(basename ${rocm_url})
        download_and_verify ${rocm_url} ${rocm_sha256}
        apt install -y ./${DEBPACKAGE}
        configure_amdgpu_dkms_patches
        # (commented-out if/else and TODOs: go back to the full `rocm` install
        #  once ROCm 7.0 is used again)
        amdgpu-install -y --usecase=graphics
        apt-get install -y rocm-utils rocm-developer-tools rocm-openmp-sdk \\
            rocm-opencl-sdk rocm-ml-sdk migraphx migraphx-dev rpp rpp-dev
        if [[ $DISTRIBUTION == "ubuntu22.04" ]]; then
            apt-get install -y libstdc++-12-dev
        fi
        apt install -y rocm-bandwidth-test
        rm -f ./${DEBPACKAGE}
    elif [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
        dnf install -y azurelinux-repos-amd
        dnf -y install kernel-drivers-gpu-$(uname -r)
        dnf -y install amdgpu amdgpu-firmware amdgpu-headers
        cat <<EOF >> /etc/yum.repos.d/amd_rocm.repo
        [amd_rocm] ... baseurl=https://repo.radeon.com/rocm/azurelinux3/${rocm_version}/main/ ...
        EOF
        dnf repolist --refresh
        dnf install -y rocm-dev rocm-validation-suite rocm-bandwidth-test
        dnf install -y rocm-smi-lib rocm-core rocm-device-libs rocm-llvm rocm-validation-suite
        dnf install -y rocm-bandwidth-test
    fi

    cat <<'EOF' > /etc/udev/rules.d/99-amdgpu-permissive.rules
    ...
    EOF
    udevadm control --reload-rules && sudo udevadm trigger

    write_component_version "ROCM" ${rocm_version}

    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        echo blacklist amdgpu | tee -a /etc/modprobe.d/blacklist.conf
        update-initramfs -c -k $(uname -r)
    fi

    cat <<'EOF' > /tmp/tempinit.sh
    ... (the GPU-wait-and-modprobe script) ...
    EOF
    cp /tmp/tempinit.sh /etc/init.d/initamdgpu.sh
    chmod +x /etc/init.d/initamdgpu.sh
    rm /tmp/tempinit.sh

    echo -e '[Unit]\\n\\nDescription=Runs /etc/init.d/initamdgpu.sh\\n\\n' | tee rocmstartup.service
    echo -e '[Service]\\n\\nExecStart=/etc/init.d/initamdgpu.sh\\n\\n' | tee -a rocmstartup.service
    echo -e '[Install]\\n\\nWantedBy=multi-user.target' | tee -a rocmstartup.service
    mv rocmstartup.service /etc/systemd/system/rocmstartup.service
    systemctl start rocmstartup
    systemctl enable rocmstartup

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-rocm" Step (AMD images only). Deleting `action=` from that Step makes
the build run the bash script again, which is why the .sh file is kept.

Ordering: build_plan() runs this step BEFORE install-pmix / install-mpis,
exactly as distros/*/install.sh runs install_rocm.sh before them, so the
HPC-X step can rebuild UCX with ROCm support. install() logs this reason.

install() returns 0 for success or 3 for failure. It logs problems and returns
3; it never lets an exception escape to the build loop.

Behaviour kept from bash
------------------------
  * Commands, arguments and order are the same. The first failing command
    stops the step (bash's `set -e`), except for one quirk: in
    `udevadm control --reload-rules && sudo udevadm trigger`, `set -e` ignores
    a failure of the command before `&&`. If the reload fails, bash skips the
    trigger and carries on. We do the same, with a warning. (The rules still
    apply from the next boot, which is when the image is used.)
  * `sudo` is kept on `udevadm trigger`, exactly as bash runs it.
  * Appends stay appends (amd_rocm.repo, blacklist.conf), so a rerun adds the
    blocks again, as bash does; a warning is logged when the text is already
    there. Every other file (DKMS patch and config, apt sources, keyrings,
    rocm.conf, udev rules, init script, service unit) is overwritten.
  * Ubuntu (26.04 included) blacklists amdgpu, so nothing loads it at boot
    except the rocmstartup service's script, once all the GPUs have appeared.
    A warning says so in the log.
  * `systemctl start rocmstartup` runs on the build VM too, as in bash. The
    script then polls the PCI bus for AMD GPUs for up to ~15 minutes, and a
    build VM may have none; a warning is logged before starting it. (Upstream
    #653 briefly made Packer wait for rocmstartup after the reboot, then
    reverted that in the same PR; install_rocm.sh's start + enable is
    unchanged.)
  * Ubuntu 26.04 records ROCM from /opt/rocm/core/.info/version (what the SDK
    says it is, e.g. "10.0.0"), not versions.json's "10.0", as bash does. Both
    are logged. Its RVS package name has a hard-coded "10" (amdrocm10-rvs),
    as in bash, even though the SDK package name follows versions.json.
  * The amdgpu DKMS patch and config are written on every Ubuntu, just before
    the step that installs amdgpu-dkms (apt on 26.04, amdgpu-install on the
    others), as in bash.
  * Ubuntu 22.04 and 24.04 run the same amdgpu-install + explicit package
    list. Bash keeps the old ubuntu24.04 if/else (and a "bundled userspace"
    TODO for ROCm 7.0) commented out; see the comment in _install_ubuntu.

Deliberate differences from bash
--------------------------------
  * Unsupported distros fail (return 3) before changing anything. Bash's
    if/elif has no other branch: it would install no ROCm at all, yet still
    write the udev rules, install the boot service and record ROCM, so the
    image would claim a ROCm it doesn't have. (versions.json has ROCm entries
    for Rocky Linux, but the bash never installs them.)
  * Missing config fails up front: a missing ROCm version on every distro
    (bash would use "null", e.g. apt install amdrocm-core-sdknull); a missing
    url/sha256 on Ubuntu other than 26.04 (bash would try to download
    "null"); a missing "amdgpu" version on Ubuntu 26.04 (bash's `jq -e` stops
    there too, before any change). Azure Linux and Ubuntu 26.04 use only the
    ROCm version, as in bash, so url/sha256 aren't required there.
  * On Ubuntu, a missing COMPONENT_DIR or DKMS patch file fails up front.
    Bash's `cp` would fail too, but only after the repos/.deb were set up.
  * An empty `uname -r` fails instead of producing package/initramfs names
    with a hole in them.
  * Ubuntu 26.04: a missing or empty /opt/rocm/core/.info/version, or a
    failing/empty `dpkg-query` for amdgpu-dkms or amdrocm10-rvs, fails the
    step. Bash stops on a failing `cat` and RVS query (assignments under
    `set -e`), but for AMDGPU the query runs inside a command argument, which
    `set -e` doesn't check, and empty values would be recorded.
  * The amdgpu-install .deb (Ubuntu 22.04/24.04) is downloaded into a scratch
    work directory (_WORK_DIR) instead of the distro build directory bash
    runs in; the directory is emptied before use and removed afterwards.
  * The init script is written straight to /etc/init.d/initamdgpu.sh instead
    of via /tmp/tempinit.sh + cp + rm; the resulting file is the same. Its
    mode then gets the execute bits added, as `chmod +x` does (0644 -> 0755
    under the build's umask 022).
  * The unit file is written straight to /etc/systemd/system instead of being
    built in the current directory and moved there. Its text is byte-for-byte
    what the three `echo -e ... | tee` lines produce (checked against real
    bash), and its mode is set to 0644, which is what `mv` of a freshly
    tee-created file gives under umask 022.
"""

# Lets type hints such as `str | None` work on older Python versions.
from __future__ import annotations

import os
import platform
import shutil
import stat

from utils.component_config import config_for, write_component_version
from utils.dkms import check_dkms_status
from utils.download import download_and_verify
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.process import exec_program, run_capture

# The Step's `op` label in build_config.py; used on every log line.
_OP = "install-rocm"

# Bash's distro branches that are exact matches on these values.
_AZURE_LINUX = "azurelinux3.0"
_UBUNTU_2604 = "ubuntu26.04"
_UBUNTU_2204 = "ubuntu22.04"

# ---- Paths (module-level so tests can point them at a temporary directory) --
# Scratch directory for the amdgpu-install .deb (bash uses its current dir).
_WORK_DIR = "/tmp/azhpc-rocm"
_AMD_ROCM_REPO_FILE = "/etc/yum.repos.d/amd_rocm.repo"
_UDEV_RULES_FILE = "/etc/udev/rules.d/99-amdgpu-permissive.rules"
_BLACKLIST_FILE = "/etc/modprobe.d/blacklist.conf"
_INIT_SCRIPT = "/etc/init.d/initamdgpu.sh"
_SERVICE_FILE = "/etc/systemd/system/rocmstartup.service"
_SERVICE_NAME = "rocmstartup"

# The amdgpu DKMS patch: its source (relative to COMPONENT_DIR), where DKMS
# looks for it, the name it gets there, and the DKMS config that selects it.
_DKMS_PATCH_SOURCE = "patches/amdgpu-pci-resize-resource-exclude-bars.patch"
_DKMS_PATCH_DIR = "/etc/dkms/amdgpu/patches"
_DKMS_PATCH_NAME = "pci-resize-resource-exclude-bars.patch"
_DKMS_CONF_FILE = "/etc/dkms/amdgpu.conf"

# Ubuntu 26.04 apt setup. The keyring paths also appear, literally, in the apt
# source texts below (as in bash), so tests that move these files don't change
# what the source files say.
_KEYRINGS_DIR = "/etc/apt/keyrings"
_AMDROCM_KEY_URL = "https://stable.repo.amd.com/rocm/gpg/packages.gpg"
_AMDGPU_KEY_URL = "https://repo.radeon.com/rocm/rocm.gpg.key"
_AMDROCM_KEY_DOWNLOAD = "/tmp/amdrocm-packages.key"
_AMDGPU_KEY_DOWNLOAD = "/tmp/amdgpu-packages.key"
_AMDROCM_KEYRING = "/etc/apt/keyrings/amdrocm.gpg"
_AMDGPU_KEYRING = "/etc/apt/keyrings/amdgpu.gpg"
_AMDROCM_SOURCES_FILE = "/etc/apt/sources.list.d/amdrocm-stable.sources"
_AMDGPU_LIST_FILE = "/etc/apt/sources.list.d/amdgpu.list"
_RVS_LIST_FILE = "/etc/apt/sources.list.d/rvs.list"
_ROCM_INFO_VERSION_FILE = "/opt/rocm/core/.info/version"
_LD_SO_CONF_FILE = "/etc/ld.so.conf.d/rocm.conf"
_ROCM_LIB_DIR = "/opt/rocm/lib"

# Ubuntu 26.04 package names. Bash hard-codes the "10" in the RVS package even
# though the SDK package name follows versions.json's ROCm version.
_AMDGPU_DKMS_PACKAGE = "amdgpu-dkms"
_RVS_PACKAGE = "amdrocm10-rvs"

# Ubuntu 22.04 / 24.04: the `rocm` metapackage contents minus MIVisionX (and
# rocDecode, which only MIVisionX pulls in). Bash's comment: those pull FFmpeg
# (plus Qt, cJSON and mbedTLS on 24.04) packages with publishing-blocking CVEs
# whose Ubuntu fixes need Pro/ESM. Same order as bash.
_ROCM_PACKAGES = ["rocm-utils", "rocm-developer-tools", "rocm-openmp-sdk",
                  "rocm-opencl-sdk", "rocm-ml-sdk", "migraphx", "migraphx-dev",
                  "rpp", "rpp-dev"]

# ---- File contents (exact bytes bash writes) --------------------------------
# `cat <<'EOF'` has a QUOTED delimiter, so bash writes the text literally: no
# $variable or $(command) expansion. Each heredoc ends with a newline after its
# last line. The backslash after the opening """ stops the Python string from
# starting with an empty line.
_UDEV_RULES = """\
KERNEL=="kfd", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="renderD*", MODE="0666"
"""

# /etc/dkms/amdgpu.conf (quoted heredoc, so written literally). DKMS reads this
# file as shell when it builds amdgpu: it adds the patch to DKMS's PATCH list
# unless the driver source already has AMD's native fix (..._ALIGN) or our
# backport (..._EXCLUDE_BARS) in all three places.
_AMDGPU_DKMS_CONF = """\
_amdgpu_source="$source_tree/$module-$module_version"
_amdgpu_pci_resize_fixed=false
# AMD's native fix uses ALIGN; EXCLUDE_BARS identifies our backport.
for _amdgpu_flag in HAVE_PCI_RESIZE_RESOURCE_ALIGN HAVE_PCI_RESIZE_RESOURCE_EXCLUDE_BARS; do
    if grep -Fq "$_amdgpu_flag" "$_amdgpu_source/amd/dkms/configure" &&
       grep -Fq "$_amdgpu_flag" "$_amdgpu_source/amd/dkms/config/config.h.in" &&
       grep -Fq "$_amdgpu_flag" "$_amdgpu_source/amd/amdgpu/amdgpu_device.c"; then
        _amdgpu_pci_resize_fixed=true
        break
    fi
done
if [[ "$_amdgpu_pci_resize_fixed" == false ]]; then
    PATCH+=("pci-resize-resource-exclude-bars.patch")
fi
"""

# Ubuntu 26.04 apt sources. The heredocs are unquoted (<<EOF), but only
# amdgpu.list has a ${variable} in it (see amdgpu_list_text()).
_AMDROCM_SOURCES = """\
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/amdrocm.gpg
"""
_RVS_LIST = ("deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] "
             "https://stable.repo.amd.com/rocm/extras/rvs/packages/ubuntu2604/ stable main\n")

# `echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf`.
_LD_SO_CONF = f"{_ROCM_LIB_DIR}\n"

# PCI IDs used by the script: 1002:740c is MI200 (16 per node), 1002:74b5 is
# MI300X and 1002:74bd is MI300HF (8 per node). The $(...) and $((...)) are
# meant for /bin/sh at boot time; they are NOT expanded when written (quoted
# heredoc), so a plain (non-f) Python string reproduces them exactly.
_INIT_SCRIPT_TEXT = """\
#!/bin/sh
at_count=0
while [ $at_count -le 90 ]
do
    if [ $(lspci -d 1002:74b5 | wc -l) -eq 8 -o $(lspci -d 1002:74bd | wc -l) -eq 8 -o $(lspci -d 1002:740c | wc -l) -eq 16 ]; then
       echo Required number of GPUs found
       at_count=91
       sleep 120s
       echo doing Modprobe for amdgpu
       if [ $(lspci -d 1002:740c | wc -l) -eq 16 ]; then
          sudo modprobe amdgpu
       else
          sudo modprobe -r hyperv_drm
          sudo modprobe amdgpu ip_block_mask=0x7f
       fi
    else
       sleep 10
       at_count=$(($at_count + 1))
    fi
done

exit 0
"""

# What the three `echo -e '...' | tee [-a] rocmstartup.service` lines write.
# `echo -e` turns each \n into a newline, and echo adds one more newline at the
# end of each line, which is why there are three blank lines between sections.
# Verified byte for byte against real bash.
_SERVICE_UNIT = (
    "[Unit]\n\nDescription=Runs /etc/init.d/initamdgpu.sh\n\n\n"
    "[Service]\n\nExecStart=/etc/init.d/initamdgpu.sh\n\n\n"
    "[Install]\n\nWantedBy=multi-user.target\n"
)

# `echo blacklist amdgpu | tee -a ...` appends this line.
_BLACKLIST_LINE = "blacklist amdgpu\n"

# Mode of a file created by tee and then moved with mv, under umask 022.
_SERVICE_FILE_MODE = 0o644


# --------------------------------------------------------------------------- #
# Pure helpers: no side effects, so tests call them directly.
# --------------------------------------------------------------------------- #
def amd_rocm_repo_text(version: str) -> str:
    """The [amd_rocm] block bash appends to /etc/yum.repos.d/amd_rocm.repo.

    The heredoc delimiter is unquoted (`<<EOF`), so bash expands
    ${rocm_version}; an f-string does the same here. The double quotes around
    the name are part of the file text.
    """
    return (
        "[amd_rocm]\n"
        'name="AMD ROCM packages repo for Azure Linux 3.0"\n'
        f"baseurl=https://repo.radeon.com/rocm/azurelinux3/{version}/main/\n"
        "enabled=1\n"
        "repo_gpgcheck=0\n"
        "gpgcheck=0\n"
        "sslverify=0\n"
    )


def amdgpu_repo_url(driver_version: str) -> str:
    """bash: driver_url="https://repo.radeon.com/amdgpu/${driver_version}/ubuntu"."""
    return f"https://repo.radeon.com/amdgpu/{driver_version}/ubuntu"


def amdgpu_list_text(driver_url: str) -> str:
    """/etc/apt/sources.list.d/amdgpu.list (unquoted heredoc: ${driver_url}
    is expanded). "resolute" is Ubuntu 26.04's codename."""
    return (f"deb [arch=amd64 signed-by=/etc/apt/keyrings/amdgpu.gpg] "
            f"{driver_url} resolute main\n")


def ubuntu2604_package(rocm_version: str) -> str:
    """bash: rocm_package="amdrocm-core-sdk${rocm_version}", e.g. amdrocm-core-sdk10.0."""
    return f"amdrocm-core-sdk{rocm_version}"


def azure_linux_commands(kernel: str) -> tuple[list[list[str]], list[list[str]]]:
    """The Azure Linux dnf commands: (before the repo file, after it).

    `kernel` is the `uname -r` value used in the kernel-drivers-gpu package name.
    """
    before = [
        ["dnf", "install", "-y", "azurelinux-repos-amd"],
        ["dnf", "-y", "install", f"kernel-drivers-gpu-{kernel}"],
        ["dnf", "-y", "install", "amdgpu", "amdgpu-firmware", "amdgpu-headers"],
    ]
    after = [
        ["dnf", "repolist", "--refresh"],
        ["dnf", "install", "-y", "rocm-dev", "rocm-validation-suite", "rocm-bandwidth-test"],
        ["dnf", "install", "-y", "rocm-smi-lib", "rocm-core", "rocm-device-libs",
         "rocm-llvm", "rocm-validation-suite"],
        ["dnf", "install", "-y", "rocm-bandwidth-test"],
    ]
    return before, after


# --------------------------------------------------------------------------- #
# Small helpers with side effects.
# --------------------------------------------------------------------------- #
def _run_all(commands, env: dict[str, str], cwd: str | None = None) -> bool:
    """Run commands in order, stopping at the first failure (like `set -e`).

    exec_program() already logs each command, its exit code, and on failure
    its last output lines; here we only add what the failure means.
    """
    for command in commands:
        rc = exec_program(command, _OP, cwd=cwd, env=env)
        if rc != 0:
            log_error(_OP, f"ROCm install stopped: `{' '.join(command)}` failed",
                      exit_code=rc)
            return False
    return True


def _write(path: str, text: str, *, append: bool) -> None:
    """Write (bash `>`) or append (bash `>>` / `tee -a`) text to a file.

    Mode "w" empties an existing file first (and keeps its permissions); mode
    "a" adds to the end. Both create a missing file. Raises OSError on failure;
    install() catches it.
    """
    existed = os.path.exists(path)
    with open(path, "a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)
    if append:
        action = "appended"
    else:
        action = "replaced" if existed else "created"
    log_info(_OP, f"{'Appended to' if append else 'Wrote'} {path}",
             path=path, action=action)
    log_debug(_OP, f"Contents written to {path}:\n{text}")


def _has_line(path: str, line: str) -> bool:
    """True if the file at `path` already has `line` as one of its lines.

    A missing or unreadable file counts as "no"; the write that follows will
    report any real problem.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            # splitlines() drops the "\\n" at the end of each line.
            return line in handle.read().splitlines()
    except OSError:
        return False


def _remove_if_present(path: str) -> None:
    """bash's `rm -f <path>`: delete the file; a missing file is fine."""
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    log_info(_OP, f"Removed {path}", path=path, action="removed")


def _dpkg_version(package: str, env: dict[str, str]) -> str | None:
    """`dpkg-query -W -f='${Version}' <package>`: the installed version, or
    None (after logging) when the query fails or prints nothing."""
    # In bash the single quotes only stop ${Version} being expanded; dpkg-query
    # receives -f=${Version} literally, which is what this argument is.
    rc, output = run_capture(["dpkg-query", "-W", "-f=${Version}", package], _OP, env=env)
    # "$(...)" strips trailing newlines; dpkg-query prints none with this format.
    version = output.rstrip("\n") if rc == 0 else ""
    if not version:
        log_error(_OP, f"could not read the installed {package} version from dpkg-query",
                  package=package, exit_code=rc)
        return None
    return version


# --------------------------------------------------------------------------- #
# The amdgpu DKMS patch (every Ubuntu).
# --------------------------------------------------------------------------- #
def _configure_amdgpu_dkms_patches(component_dir: str) -> None:
    """bash's configure_amdgpu_dkms_patches: patch file + /etc/dkms/amdgpu.conf.

    Raises OSError if a file can't be copied or written; install() catches it.
    """
    source = os.path.join(component_dir, _DKMS_PATCH_SOURCE)
    target = os.path.join(_DKMS_PATCH_DIR, _DKMS_PATCH_NAME)
    os.makedirs(_DKMS_PATCH_DIR, exist_ok=True)       # mkdir -p
    existed = os.path.exists(target)
    # `cp` without -p: copies the bytes; a new file gets the default mode.
    shutil.copyfile(source, target)
    action = "replaced" if existed else "created"
    log_info(_OP, f"Copied the amdgpu DKMS patch to {target} ({action})",
             source=source, path=target, action=action, patch=_DKMS_PATCH_NAME)
    _write(_DKMS_CONF_FILE, _AMDGPU_DKMS_CONF, append=False)
    log_info(_OP, "amdgpu DKMS builds will apply the pci_resize_resource patch "
                  "unless the driver source already has the fix",
             path=_DKMS_CONF_FILE, patch=_DKMS_PATCH_NAME)


# --------------------------------------------------------------------------- #
# Per-distro ROCm installs. _install_ubuntu2604 returns the ROCm version to
# record (None on failure); the others return True/False. They may raise
# OSError from file operations; install() catches it.
# --------------------------------------------------------------------------- #
def _install_ubuntu2604(env: dict[str, str], rocm_version: str, driver_version: str,
                        kernel: str, component_dir: str) -> str | None:
    """Ubuntu 26.04: ROCm 10 + amdgpu DKMS from AMD's apt repos.

    Returns the ROCm version to record (read back from the installed SDK).
    """
    driver_url = amdgpu_repo_url(driver_version)
    rocm_package = ubuntu2604_package(rocm_version)
    log_info(_OP, "Ubuntu 26.04: installing ROCm from AMD's apt repos",
             handler="ubuntu26.04", rocm_package=rocm_package,
             amdgpu_version=driver_version, amdgpu_repo=driver_url,
             rvs_package=_RVS_PACKAGE)

    # ---- Signing keys -------------------------------------------------------
    if not _run_all([
        ["apt", "install", "-y", "ca-certificates", "wget", "gnupg", "dkms",
         f"linux-headers-{kernel}"],
        ["install", "-d", "-m", "0755", _KEYRINGS_DIR],
        ["wget", "-O", _AMDROCM_KEY_DOWNLOAD, _AMDROCM_KEY_URL],
        ["gpg", "--batch", "--yes", "--dearmor", "-o", _AMDROCM_KEYRING,
         _AMDROCM_KEY_DOWNLOAD],
        ["wget", "-O", _AMDGPU_KEY_DOWNLOAD, _AMDGPU_KEY_URL],
        ["gpg", "--batch", "--yes", "--dearmor", "-o", _AMDGPU_KEYRING,
         _AMDGPU_KEY_DOWNLOAD],
        ["chmod", "0644", _AMDROCM_KEYRING, _AMDGPU_KEYRING],
    ], env):
        return None
    log_info(_OP, "Installed the AMD apt signing keys",
             keyrings=[_AMDROCM_KEYRING, _AMDGPU_KEYRING],
             key_urls=[_AMDROCM_KEY_URL, _AMDGPU_KEY_URL])
    _remove_if_present(_AMDROCM_KEY_DOWNLOAD)
    _remove_if_present(_AMDGPU_KEY_DOWNLOAD)

    # ---- apt sources (all `cat >`, so overwritten) --------------------------
    _write(_AMDROCM_SOURCES_FILE, _AMDROCM_SOURCES, append=False)
    _write(_AMDGPU_LIST_FILE, amdgpu_list_text(driver_url), append=False)
    _write(_RVS_LIST_FILE, _RVS_LIST, append=False)
    if not _run_all([["apt", "update"]], env):
        return None

    # ---- amdgpu driver (DKMS) -----------------------------------------------
    _configure_amdgpu_dkms_patches(component_dir)
    if not _run_all([["apt", "install", "-y", _AMDGPU_DKMS_PACKAGE,
                      "amdgpu-dkms-firmware"]], env):
        return None
    # check_dkms_status logs the status lines and, on failure, the make.log tails.
    if not check_dkms_status(["amdgpu"], _OP, env, kernel=kernel):
        log_error(_OP, "ROCm install stopped: the amdgpu DKMS module is not "
                       "installed for the running kernel", kernel=kernel)
        return None

    # ---- ROCm SDK + RVS -----------------------------------------------------
    if not _run_all([["apt", "install", "-y", rocm_package, _RVS_PACKAGE]], env):
        return None

    # bash: rocm_version=$(cat /opt/rocm/core/.info/version). "$(...)" strips
    # trailing newlines only.
    try:
        with open(_ROCM_INFO_VERSION_FILE, encoding="utf-8") as handle:
            installed = handle.read().rstrip("\n")
    except OSError as exc:
        log_error(_OP, f"could not read the installed ROCm version: {exc}",
                  path=_ROCM_INFO_VERSION_FILE)
        return None
    if not installed:
        log_error(_OP, f"{_ROCM_INFO_VERSION_FILE} is empty; no ROCm version to record",
                  path=_ROCM_INFO_VERSION_FILE)
        return None
    log_info(_OP, f"Installed ROCm SDK reports version {installed}; recording that "
                  f"instead of versions.json's {rocm_version}, as bash does",
             path=_ROCM_INFO_VERSION_FILE, installed_version=installed,
             config_version=rocm_version)

    amdgpu_version = _dpkg_version(_AMDGPU_DKMS_PACKAGE, env)
    if amdgpu_version is None:
        return None
    write_component_version("AMDGPU", amdgpu_version)
    log_info(_OP, f"Recorded AMDGPU={amdgpu_version}", component="AMDGPU",
             version=amdgpu_version, package=_AMDGPU_DKMS_PACKAGE)
    rvs_version = _dpkg_version(_RVS_PACKAGE, env)
    if rvs_version is None:
        return None
    write_component_version("RVS", rvs_version)
    log_info(_OP, f"Recorded RVS={rvs_version}", component="RVS",
             version=rvs_version, package=_RVS_PACKAGE)

    # The SDK ships no ld.so.conf entry, so libamdhip64 can't be found by
    # anything linking ROCm (e.g. Open MPI's accelerator component) without it.
    _write(_LD_SO_CONF_FILE, _LD_SO_CONF, append=False)
    if not _run_all([["ldconfig"]], env):
        return None
    log_info(_OP, f"Added {_ROCM_LIB_DIR} to the dynamic linker path",
             path=_LD_SO_CONF_FILE, lib_dir=_ROCM_LIB_DIR)
    return installed


def _install_ubuntu(env: dict[str, str], config: dict, distribution: str,
                    component_dir: str) -> bool:
    """Ubuntu other than 26.04: amdgpu-install .deb + explicit package list."""
    url, sha256 = config["url"], config["sha256"]

    # Start from an empty work directory so leftovers from an earlier failed
    # run can't be picked up.
    shutil.rmtree(_WORK_DIR, ignore_errors=True)
    os.makedirs(_WORK_DIR, exist_ok=True)
    try:
        try:
            deb = download_and_verify(url, sha256, dest_dir=_WORK_DIR)
        except Exception as exc:  # network errors, checksum mismatch, ...
            log_error(_OP, f"download/verify of the amdgpu-install package failed: {exc}",
                      url=url)
            return False
        log_info(_OP, f"Downloaded and verified {deb.name}", path=str(deb))

        # `./<file>` makes apt treat the argument as a local .deb file rather
        # than a package name; it's relative to the work directory (cwd).
        if not _run_all([["apt", "install", "-y", f"./{deb.name}"]], env, cwd=_WORK_DIR):
            return False

        _configure_amdgpu_dkms_patches(component_dir)

        # Bash keeps its old `if ubuntu24.04 ... else amdgpu-install -y
        # --usecase=graphics,rocm` commented out, with TODOs to go back to the
        # full `rocm` usecase (or the bundled amdgpu-dkms + rocm packages) once
        # ROCm 7.0 is used again. Until then both 22.04 and 24.04 install the
        # graphics usecase plus the explicit package list without MIVisionX.
        log_info(_OP, "Installing ROCm without MIVisionX/rocDecode (their "
                      "dependencies have CVEs that block publishing)",
                 usecase="graphics", packages=_ROCM_PACKAGES)
        commands = [
            ["amdgpu-install", "-y", "--usecase=graphics"],
            ["apt-get", "install", "-y", *_ROCM_PACKAGES],
        ]
        if distribution == _UBUNTU_2204:
            # Bash: rocdecode-dev used to pull this in; ROCm's clang picks the
            # newest GCC (12), so hipcc needs its C++ headers.
            log_info(_OP, "Ubuntu 22.04: also installing libstdc++-12-dev for hipcc",
                     package="libstdc++-12-dev")
            commands.append(["apt-get", "install", "-y", "libstdc++-12-dev"])
        commands.append(["apt", "install", "-y", "rocm-bandwidth-test"])
        if not _run_all(commands, env, cwd=_WORK_DIR):
            return False

        # Bash's `rm -f ./<deb>`.
        os.remove(deb)
    finally:
        # Our own addition: remove the scratch directory whatever happened.
        shutil.rmtree(_WORK_DIR, ignore_errors=True)
    return True


def _install_azurelinux(env: dict[str, str], config: dict, kernel: str) -> bool:
    version = config["version"]
    before, after = azure_linux_commands(kernel)
    if not _run_all(before, env):
        return False

    # Appended (bash `cat <<EOF >>`), so a rerun adds a second [amd_rocm] block.
    if _has_line(_AMD_ROCM_REPO_FILE, "[amd_rocm]"):
        log_warn(_OP, f"{_AMD_ROCM_REPO_FILE} already has an [amd_rocm] section; "
                      "appending another one, as bash does on a rerun",
                 path=_AMD_ROCM_REPO_FILE)
    _write(_AMD_ROCM_REPO_FILE, amd_rocm_repo_text(version), append=True)
    log_info(_OP, "Added the AMD ROCm dnf repo", repo="amd_rocm", version=version)

    return _run_all(after, env)


# --------------------------------------------------------------------------- #
# Steps shared by every supported distro.
# --------------------------------------------------------------------------- #
def _setup_udev(env: dict[str, str]) -> bool:
    """Write the permissive GPU udev rules, then reload and trigger udev."""
    _write(_UDEV_RULES_FILE, _UDEV_RULES, append=False)

    # `udevadm control --reload-rules && sudo udevadm trigger` under `set -e`:
    # a failure BEFORE `&&` doesn't stop bash; it only skips the trigger.
    rc = exec_program(["udevadm", "control", "--reload-rules"], _OP, env=env)
    if rc != 0:
        log_warn(_OP, "udev rules reload failed; skipping `udevadm trigger` and "
                      "continuing, as bash does (rules apply from next boot)",
                 exit_code=rc)
        return True
    if not _run_all([["sudo", "udevadm", "trigger"]], env):
        return False
    log_info(_OP, "Reloaded udev rules and triggered device events",
             path=_UDEV_RULES_FILE)
    return True


def _blacklist_amdgpu(env: dict[str, str], kernel: str) -> bool:
    """Ubuntu only: stop amdgpu auto-loading and rebuild the initramfs."""
    # BASH BEHAVIOUR (kept): with amdgpu blacklisted, the only thing that loads
    # it is the rocmstartup service's script, after all the GPUs have shown up
    # on the PCI bus. Until then the GPUs have no driver.
    log_warn(_OP, "Blacklisting amdgpu: it will not auto-load at boot; only "
                  f"{_SERVICE_NAME}.service loads it, once all GPUs are present",
             path=_BLACKLIST_FILE, service=_SERVICE_NAME)
    # `tee -a` appends, so a rerun adds the same line again.
    if _has_line(_BLACKLIST_FILE, _BLACKLIST_LINE.rstrip("\n")):
        log_warn(_OP, f"{_BLACKLIST_FILE} already has 'blacklist amdgpu'; "
                      "appending it again, as bash does on a rerun",
                 path=_BLACKLIST_FILE)
    _write(_BLACKLIST_FILE, _BLACKLIST_LINE, append=True)
    if not _run_all([["update-initramfs", "-c", "-k", kernel]], env):
        return False
    log_info(_OP, "Blacklisted amdgpu and created a new initramfs", kernel=kernel)
    return True


def _install_boot_service(env: dict[str, str]) -> bool:
    """Install the init script and the systemd unit that runs it, start + enable."""
    log_info(_OP, "Writing gpu mode probe in init.d", path=_INIT_SCRIPT)
    _write(_INIT_SCRIPT, _INIT_SCRIPT_TEXT, append=False)
    # `chmod +x` adds the execute bits (user, group, other) to the current
    # mode. stat.S_IMODE() keeps only the permission bits of st_mode.
    mode = stat.S_IMODE(os.stat(_INIT_SCRIPT).st_mode) | 0o111
    os.chmod(_INIT_SCRIPT, mode)
    log_info(_OP, f"Made {_INIT_SCRIPT} executable", path=_INIT_SCRIPT,
             mode=oct(mode))

    _write(_SERVICE_FILE, _SERVICE_UNIT, append=False)
    os.chmod(_SERVICE_FILE, _SERVICE_FILE_MODE)

    # BASH BEHAVIOUR (kept): the service is started on the build VM, not just
    # enabled for the image's first boot. Its script waits up to ~15 minutes
    # (91 checks, 10 s apart) for AMD GPUs that a build VM may not have.
    log_warn(_OP, f"Starting {_SERVICE_NAME} on the build VM, as bash does: its "
                  "script polls for AMD GPUs for up to ~15 minutes and the build "
                  "VM may have none", service=_SERVICE_NAME, script=_INIT_SCRIPT)
    if not _run_all([["systemctl", "start", _SERVICE_NAME],
                     ["systemctl", "enable", _SERVICE_NAME]], env):
        return False
    log_info(_OP, f"Started and enabled {_SERVICE_NAME}.service",
             service=_SERVICE_NAME, path=_SERVICE_FILE)
    return True


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def _check_config(env: dict[str, str], distribution: str, is_ubuntu: bool,
                  config) -> tuple[str | None, str | None]:
    """Check the inputs bash will need. Returns (amdgpu driver version, error).

    The driver version is only resolved on Ubuntu 26.04; error is None when
    everything needed is there.
    """
    # isinstance() guards against a missing (None) or oddly shaped config.
    if not isinstance(config, dict) or not config.get("version"):
        return None, "could not resolve rocm version from versions.json"
    driver_version = None
    if distribution == _UBUNTU_2604:
        # bash: `jq -er '.version'` stops the script when it's missing.
        driver = config_for("amdgpu", env)
        if not isinstance(driver, dict) or not driver.get("version"):
            return None, "could not resolve amdgpu driver version from versions.json"
        driver_version = str(driver["version"])
    elif is_ubuntu and not (config.get("url") and config.get("sha256")):
        return None, "rocm url/sha256 missing from versions.json"
    if is_ubuntu:
        component_dir = env.get("COMPONENT_DIR", "")
        patch = os.path.join(component_dir, _DKMS_PATCH_SOURCE)
        if not component_dir or not os.path.isfile(patch):
            return None, f"amdgpu DKMS patch not found at {patch!r} (COMPONENT_DIR unset?)"
    return driver_version, None


def install(env: dict[str, str]) -> int:
    """Install ROCm and its boot-time GPU setup. Returns 0 or 3.

    `env` is the build environment made by component_env() in build_config.py.
    Uses DISTRIBUTION, COMPONENT_DIR (Ubuntu's DKMS patch) and the values
    config_for() needs to find "rocm" (and "amdgpu" on Ubuntu 26.04).
    """
    # ---- 1. Checks before any change is made --------------------------------
    distribution = env.get("DISTRIBUTION", "")
    is_ubuntu = "ubuntu" in distribution            # bash: == *"ubuntu"*
    if not is_ubuntu and distribution != _AZURE_LINUX:
        log_error(_OP, f"ROCm install is not supported on {distribution!r} "
                       "(bash would install nothing but still record ROCM)",
                  distro=distribution)
        return 3

    config = config_for("rocm", env)
    driver_version, problem = _check_config(env, distribution, is_ubuntu, config)
    if problem:
        log_error(_OP, problem, distro=distribution,
                  version=config.get("version") if isinstance(config, dict) else None)
        return 3
    version = str(config["version"])
    component_dir = env.get("COMPONENT_DIR", "")

    # platform.release() is Python's `uname -r`: the running kernel.
    kernel = platform.release()
    if not kernel:
        log_error(_OP, "could not determine the running kernel version (uname -r)")
        return 3

    if distribution == _UBUNTU_2604:
        source = f"apt ({amdgpu_repo_url(driver_version)} + stable.repo.amd.com)"
    else:
        source = config.get("url", "dnf (azurelinux-repos-amd + amd_rocm repo)")
    log_info(_OP, f"Installing ROCm {version} on {distribution}",
             version=version, distro=distribution, kernel=kernel, source=source,
             amdgpu_version=driver_version)
    log_info(_OP, "ROCm is installed before PMIx/MPI, as in install.sh, so the "
                  "HPC-X step can rebuild UCX with ROCm support")

    try:
        # ---- 2. ROCm itself, per distro ------------------------------------
        if distribution == _UBUNTU_2604:
            installed = _install_ubuntu2604(env, version, driver_version, kernel,
                                            component_dir)
            ok = installed is not None
            if ok:
                version = installed     # bash: rocm_version=$(cat .../version)
        elif is_ubuntu:
            log_info(_OP, "Ubuntu: installing ROCm with amdgpu-install",
                     handler="ubuntu", usecase="graphics")
            ok = _install_ubuntu(env, config, distribution, component_dir)
        else:
            log_info(_OP, "Azure Linux: installing ROCm from the AMD dnf repos",
                     handler="azurelinux")
            ok = _install_azurelinux(env, config, kernel)
        if not ok:
            return 3

        # ---- 3. GPU access for all users -----------------------------------
        if not _setup_udev(env):
            return 3

        # ---- 4. Record the version (same point as bash) --------------------
        write_component_version("ROCM", version)
        log_info(_OP, f"Recorded ROCM={version}", component="ROCM", version=version)

        # ---- 5. Ubuntu: don't auto-load amdgpu at boot ---------------------
        if is_ubuntu and not _blacklist_amdgpu(env, kernel):
            return 3

        # ---- 6. Boot-time GPU wait + modprobe service ----------------------
        if not _install_boot_service(env):
            return 3
    except OSError as exc:
        # File writes under /etc, the work dir, and the versions file in /opt.
        log_error(_OP, f"file operation failed: {exc}",
                  path=getattr(exc, "filename", None))
        return 3

    log_info(_OP, f"ROCm {version} installed and configured",
             version=version, distro=distribution,
             amdgpu_blacklisted=is_ubuntu, service=_SERVICE_NAME)
    return 0
