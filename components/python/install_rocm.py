"""install_rocm.py — Python port of components/install_rocm.sh.

What this step does
-------------------
ROCm is AMD's GPU software stack (kernel driver, runtime, compilers, math and
communication libraries). This step installs it on AMD GPU images (MI200 /
MI300X / MI300HF) and sets up the machine so the GPUs work after boot:

  1. Install ROCm, with one branch per distro family:
       * Ubuntu        download + verify the `amdgpu-install` .deb from
                       versions.json, install it with apt, then run
                       `amdgpu-install -y --usecase=graphics,rocm` and install
                       rocm-bandwidth-test.
       * Azure Linux   install the AMD repo package, the GPU kernel drivers for
                       the running kernel and the amdgpu packages, append an
                       [amd_rocm] repo to /etc/yum.repos.d/amd_rocm.repo for
                       the ROCm version, then dnf-install the ROCm packages.
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
    rocm_url=$(jq -r '.url' <<< $rocm_metadata)
    rocm_sha256=$(jq -r '.sha256' <<< $rocm_metadata)
    DEBPACKAGE=$(basename ${rocm_url})

    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        download_and_verify ${rocm_url} ${rocm_sha256}
        apt install -y ./${DEBPACKAGE}
        if [[ $DISTRIBUTION == "ubuntu24.04" ]]; then
            # TODO: go back to bundled userspace once we move back to ROCm 7.0
            # ... (commented-out apt install of amdgpu-dkms/rocm) ...
            amdgpu-install -y --usecase=graphics,rocm
        else
            amdgpu-install -y --usecase=graphics,rocm
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

Known ordering drift (not changed here): build_plan() runs this step AFTER
install-pmix / install-mpis, while distros/*/install.sh runs install_rocm.sh
BEFORE them (so HPC-X can rebuild UCX with ROCm support). ToDo.md item 1
tracks this; porting the script deliberately leaves the Step where it is.

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
    blocks again, as bash does. The udev rules, init script and service unit
    are overwritten.
  * Both Ubuntu versions run the same `amdgpu-install` command. Bash has an
    if/else for ubuntu24.04 with identical branches plus a commented-out TODO
    ("go back to bundled userspace once we move back to ROCm 7.0"); see the
    comment in _install_ubuntu.

Deliberate differences from bash
--------------------------------
  * Unsupported distros fail (return 3) before changing anything. Bash's
    if/elif has no other branch: it would install no ROCm at all, yet still
    write the udev rules, install the boot service and record ROCM, so the
    image would claim a ROCm it doesn't have. (versions.json has ROCm entries
    for Rocky Linux, but the bash never installs them.)
  * Missing config fails up front: a missing version on every distro, and a
    missing url/sha256 on Ubuntu (bash would try to download "null"). Azure
    Linux uses only the version, as in bash (its versions.json entry has no
    url), so url/sha256 aren't required there.
  * An empty `uname -r` fails instead of producing package/initramfs names
    with a hole in them.
  * The .deb is downloaded into a scratch work directory (_WORK_DIR) instead
    of the distro build directory bash runs in; the directory is emptied
    before use and removed afterwards.
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
from utils.download import download_and_verify
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.process import exec_program

# The Step's `op` label in build_config.py; used on every log line.
_OP = "install-rocm"

# Bash's Azure Linux branch is an exact match on this value.
_AZURE_LINUX = "azurelinux3.0"

# ---- Paths (module-level so tests can point them at a temporary directory) --
# Scratch directory for the amdgpu-install .deb (bash uses its current dir).
_WORK_DIR = "/tmp/azhpc-rocm"
_AMD_ROCM_REPO_FILE = "/etc/yum.repos.d/amd_rocm.repo"
_UDEV_RULES_FILE = "/etc/udev/rules.d/99-amdgpu-permissive.rules"
_BLACKLIST_FILE = "/etc/modprobe.d/blacklist.conf"
_INIT_SCRIPT = "/etc/init.d/initamdgpu.sh"
_SERVICE_FILE = "/etc/systemd/system/rocmstartup.service"
_SERVICE_NAME = "rocmstartup"

# ---- File contents (exact bytes bash writes) --------------------------------
# `cat <<'EOF'` has a QUOTED delimiter, so bash writes the text literally: no
# $variable or $(command) expansion. Each heredoc ends with a newline after its
# last line. The backslash after the opening """ stops the Python string from
# starting with an empty line.
_UDEV_RULES = """\
KERNEL=="kfd", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="renderD*", MODE="0666"
"""

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
    with open(path, "a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)
    log_info(_OP, f"{'Appended to' if append else 'Wrote'} {path}",
             path=path, action="appended" if append else "replaced")
    log_debug(_OP, f"Contents written to {path}:\n{text}")


# --------------------------------------------------------------------------- #
# Per-distro ROCm installs. Each returns True on success. They may raise
# OSError from file operations; install() catches it.
# --------------------------------------------------------------------------- #
def _install_ubuntu(env: dict[str, str], config: dict) -> bool:
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
        commands = [
            ["apt", "install", "-y", f"./{deb.name}"],
            # Bash has an ubuntu24.04 branch and an else branch that both run
            # this exact command. The 24.04 branch carries a commented-out
            # TODO to go back to the bundled userspace (apt install amdgpu-dkms
            # rocm, recording RCCL from dpkg) once ROCm 7.0 is used again.
            ["amdgpu-install", "-y", "--usecase=graphics,rocm"],
            ["apt", "install", "-y", "rocm-bandwidth-test"],
        ]
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
    if os.path.exists(_AMD_ROCM_REPO_FILE):
        log_warn(_OP, f"{_AMD_ROCM_REPO_FILE} already exists; appending another "
                      "[amd_rocm] block, as bash does", path=_AMD_ROCM_REPO_FILE)
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

    if not _run_all([["systemctl", "start", _SERVICE_NAME],
                     ["systemctl", "enable", _SERVICE_NAME]], env):
        return False
    log_info(_OP, f"Started and enabled {_SERVICE_NAME}.service",
             service=_SERVICE_NAME, path=_SERVICE_FILE)
    return True


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def install(env: dict[str, str]) -> int:
    """Install ROCm and its boot-time GPU setup. Returns 0 or 3.

    `env` is the build environment made by component_env() in build_config.py.
    Uses DISTRIBUTION plus the values config_for() needs to find "rocm".
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
    # isinstance() guards against a missing (None) or oddly shaped config.
    if not isinstance(config, dict) or not config.get("version"):
        log_error(_OP, "could not resolve rocm version from versions.json",
                  distro=distribution)
        return 3
    if is_ubuntu and not (config.get("url") and config.get("sha256")):
        log_error(_OP, "rocm url/sha256 missing from versions.json",
                  distro=distribution, version=config["version"])
        return 3
    version = str(config["version"])

    # platform.release() is Python's `uname -r`: the running kernel.
    kernel = platform.release()
    if not kernel:
        log_error(_OP, "could not determine the running kernel version (uname -r)")
        return 3

    log_info(_OP, f"Installing ROCm {version} on {distribution}",
             version=version, distro=distribution, kernel=kernel,
             source=config.get("url", "dnf (azurelinux-repos-amd + amd_rocm repo)"))

    try:
        # ---- 2. ROCm itself, per distro ------------------------------------
        if is_ubuntu:
            log_info(_OP, "Ubuntu: installing ROCm with amdgpu-install",
                     handler="ubuntu", usecase="graphics,rocm")
            ok = _install_ubuntu(env, config)
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
