"""setup_sku_customizations.py — Python port of components/setup_sku_customizations.sh.

What this step does
-------------------
Prepares a boot-time systemd service, sku-customizations.service, that tunes
the finished image for the VM size it lands on (NCCL topology files, NVMe
setup, and so on). The image is generic; the VM size is only known at boot.

  1. Copy our per-SKU scripts (components/customizations/*) to
     /opt/azurehpc/customizations, and the topology XML files (topology/*) to
     /opt/microsoft.
  2. Generate two scripts, which differ between NVIDIA and AMD images:
       * /usr/sbin/setup_sku_customizations.sh asks the Azure metadata service
         (IMDS) for the VM size and runs the matching customization script.
         On NVIDIA, NCCL_TOPO_FAMILY can be preset to skip the IMDS lookup.
       * /usr/sbin/remove_sku_customizations.sh undoes that when the service
         stops (and on NVIDIA also stops fabric manager and unloads
         nvidia_peermem).
     Any other GPU vendor is an error.
  3. Write /etc/systemd/system/sku-customizations.service, which runs the two
     scripts. On NVIDIA it also orders itself after nvidia-fabricmanager.
  4. NVIDIA only:
       * a drop-in for openibd.service that loads nvidia-peermem after the
         InfiniBand stack starts;
       * if the fabric manager unit file exists, delete its
         "After=multi-user.target" lines, so it can start automatically.
  5. Azure Linux 3.0 on GB200 only: add kernel command line options after
     "lockdown=integrity" in /etc/default/grub and /boot/grub2/grub.cfg, and
     stop and disable irqbalance.
  6. `systemctl enable sku-customizations`. It is NOT started during the
     build: that would be pointless before the reboot and may fail on
     general-purpose VM sizes (bash says the same in a comment).

The original bash, summarised (the full heredoc texts are the _..._SCRIPT,
_SERVICE_... and _PEERMEM_DROPIN constants below):

    mkdir -p /opt/azurehpc/customizations
    cp $COMPONENT_DIR/customizations/* /opt/azurehpc/customizations
    mkdir -p /opt/microsoft
    cp $TOP_DIR/topology/* /opt/microsoft

    if [ "$GPU" = "NVIDIA" ]; then
        cat <<EOF >/usr/sbin/setup_sku_customizations.sh    ... NVIDIA text
        chmod 755 /usr/sbin/setup_sku_customizations.sh
        cat <<EOF >/usr/sbin/remove_sku_customizations.sh   ... NVIDIA text
    elif [ "$GPU" = "AMD" ]; then
        cat <<EOF >/usr/sbin/setup_sku_customizations.sh    ... AMD text
        chmod 755 /usr/sbin/setup_sku_customizations.sh
        cat <<EOF >/usr/sbin/remove_sku_customizations.sh   ... AMD text
    else
        echo "GPU is not NVIDIA or AMD, exiting"; exit 1
    fi
    chmod 755 /usr/sbin/remove_sku_customizations.sh

    cat <<EOF >/etc/systemd/system/sku-customizations.service    ... [Unit] part
    if NVIDIA: cat <<EOF >>... After=/Wants=nvidia-fabricmanager.service
    cat <<EOF >>/etc/systemd/system/sku-customizations.service   ... blank line, [Service], [Install]

    if NVIDIA:
        mkdir -p /etc/systemd/system/openibd.service.d
        cat <<EOF >/etc/systemd/system/openibd.service.d/10-nvidia-peermem.conf ...
    if NVIDIA and [ -f /lib/systemd/system/nvidia-fabricmanager.service ]:
        sed -i '/^After=multi-user.target/d' $FMSYSPATH
        echo "Removed multi-user.target in Unit section"

    if [[ $DISTRIBUTION == "azurelinux3.0" && "$SKU" == "GB200" ]]; then
        sudo sed -i 's/lockdown=integrity/lockdown=integrity <options>/' /etc/default/grub
        sudo sed -i 's/lockdown=integrity/lockdown=integrity <options>/' /boot/grub2/grub.cfg
        sudo systemctl stop irqbalance
        sudo systemctl disable irqbalance
    fi

    systemctl enable sku-customizations

About the generated scripts: every `cat <<EOF` here is an *unquoted* heredoc,
so bash would expand `$...`. The bash source escapes every `$` as `\\$`, so
nothing is actually expanded and the files get the text with plain `$`. The
constants below hold that resulting text. For both vendors, every generated
file was rendered with real bash in WSL and compared byte for byte; the tests
compare against those renders (tests/python/fixtures/setup_sku_customizations).

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets this module's install() as the `action=` of the
"setup-sku-customizations" Step. The build then calls install(env) instead of
running the bash script. Deleting `action=` runs the bash script again, which
is why the .sh file is kept. The vendor comes from env["GPU"] ("NVIDIA" or
"AMD"), the SKU from env["SKU"] and the distro from env["DISTRIBUTION"].

install() returns 0 for success or 3 for failure. It logs problems and returns
3; it does not let exceptions escape to the build loop. Like bash's `set -e`,
the first failure stops the step.

VR200: like bash's install.sh, the Python build treats a VR200 SKU as GB200
(see SKU_ALIASES in utils/build_config.py), so SKU is GB200 here and the GB200
GRUB/irqbalance changes apply to VR200 images too.

Rerun behaviour (same as bash): the GB200 replacement text still contains
"lockdown=integrity", so running the step twice adds the options twice. A
warning is logged when the options are already in a file before the edit.

Which commands still run as programs
------------------------------------
Only `systemctl`. Copies, file writes, chmod and the sed edits are plain
Python. The sed edits were checked against real `sed` in WSL.

Deliberate differences from bash
--------------------------------
  * The GPU vendor is checked before anything is copied. Bash copies the
    customization and topology files first and only then exits 1 for an
    unknown vendor. The step fails either way.
  * The service file is built as one text (the same three parts, in the same
    order) and written once, instead of one `>` and two `>>` writes. The file
    content is identical.
  * `cp` is shutil.copy(), which also copies the permission bits. For new
    files this matches `cp`; an existing file in the destination gets the
    source's mode, where `cp` would have kept the old one.
  * Bash runs the GB200 commands with `sudo`. The build already runs as root
    (this step writes /usr/sbin and /etc), so the Python port edits the files
    directly and runs `systemctl` without sudo.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import stat

from utils.logger import log_debug, log_error, log_info, log_warn
from utils.process import exec_program

# The Step's `op` label in plans/steps.py. Every log line uses it.
_OP = "setup-sku-customizations"

# Every path the step reads or writes. They are module-level "constants"
# (capital names that the program never changes) so the tests can point them
# at a temporary folder instead of the real /opt, /usr, /etc, /lib and /boot.
_CUSTOMIZATIONS_DIR = "/opt/azurehpc/customizations"
_TOPOLOGY_DIR = "/opt/microsoft"
_SETUP_SCRIPT = "/usr/sbin/setup_sku_customizations.sh"
_REMOVE_SCRIPT = "/usr/sbin/remove_sku_customizations.sh"
_SERVICE_FILE = "/etc/systemd/system/sku-customizations.service"
_OPENIBD_DROPIN_DIR = "/etc/systemd/system/openibd.service.d"
_PEERMEM_DROPIN_NAME = "10-nvidia-peermem.conf"
_FABRICMANAGER_UNIT = "/lib/systemd/system/nvidia-fabricmanager.service"
_GRUB_DEFAULT = "/etc/default/grub"
_GRUB_CFG = "/boot/grub2/grub.cfg"

_SERVICE_NAME = "sku-customizations"

# The GB200 GRUB edit applies only to this exact distro and SKU, as in bash.
_GB200_DISTRO = "azurelinux3.0"
_GB200_SKU = "GB200"

# bash: sed -i 's/lockdown=integrity/lockdown=integrity <options>/' <file>
# The pattern has no characters that are special to sed, so a plain text
# replace of the first occurrence on each line does exactly the same.
_LOCKDOWN = "lockdown=integrity"
_GB200_CMDLINE = (
    "lockdown=integrity efi_mmap_nr_slack_slots=256 numa_balancing=disable "
    "iommu.passthrough=1 irqchip.gicv3_nolpi=y arm_smmu_v3.disable_msipolling=1 "
    "init_on_alloc=0 nvme_core.io_timeout=240 panic=-1")

# bash: sed -i '/^After=multi-user.target/d' <fabric manager unit>
# re.match only matches at the start of the string, like sed's "^". The "."
# matches any character in both sed and Python, so it is left unescaped.
_FM_AFTER_PATTERN = re.compile(r"After=multi-user.target")

# ---- Generated file texts, byte for byte what bash writes -----------------
# A triple-quoted string ("""...""") can span several lines. The backslash
# right after the opening quotes stops the text from starting with a newline.
# Careful when editing: blank lines, trailing spaces and the final newlines
# are all part of the files, and the tests check them against bash's output.

_NVIDIA_SETUP_SCRIPT = """\
#!/bin/bash

# NCCL_TOPO_FAMILY may be pre-set by the caller via the environment
# to skip the Azure IMDS lookup on non-Azure nodes.
if [[ -n "${NCCL_TOPO_FAMILY:-}" ]]; then
    topoFamily="$NCCL_TOPO_FAMILY"
else
    metadata_endpoint="http://169.254.169.254/metadata/instance?api-version=2019-06-04"
    vmSize=$(curl -H Metadata:true "$metadata_endpoint" | jq -r ".compute.vmSize")
    vmSize=$(echo "$vmSize" | awk '{print tolower($0)}')

    ## Derive topology family from VM size
    case $vmSize in
        standard_nc96ads_a100_v4)          topoFamily="ncv4" ;;
        standard_nd*v4)                    topoFamily="ndv4" ;;
        standard_nd40rs_v2)               topoFamily="ndv2" ;;
        standard_hb176*v4)                topoFamily="hbv4" ;;
        standard_nc80adis_h100_v5)        topoFamily="ncv5" ;;
        standard_nd96is*_h[1-2]00_v5)    topoFamily="ndv5" ;;
        standard_nd128is*_gb[2-3]00_v6)  topoFamily="gb-family" ;;
        standard_nc*_rtxpro6000bse_v6)    topoFamily="ncv6" ;;
        *) topoFamily="" ;;
    esac
fi

## Topo file setup based on family
case $topoFamily in
    ncv4)      /opt/azurehpc/customizations/ncv4.sh;;
    ndv4)      /opt/azurehpc/customizations/ndv4.sh;;
    ndv2)      /opt/azurehpc/customizations/ndv2.sh;;
    hbv4)      /opt/azurehpc/customizations/hbv4.sh;;
    ncv5)      /opt/azurehpc/customizations/ncv5.sh;;
    ndv5)      /opt/azurehpc/customizations/ndv5.sh;;
    gb-family) /opt/azurehpc/customizations/ndv6.sh;;
    ncv6)      /opt/azurehpc/customizations/ncv6.sh;;
    *)         echo "No SKU customization for topoFamily='$topoFamily'";;
esac
"""

# Ends with an empty line: the heredoc has a blank line before its EOF.
_NVIDIA_REMOVE_SCRIPT = """\
#!/bin/bash

# Stop nvidia fabric manager
if systemctl is-active --quiet nvidia-fabricmanager
then
    systemctl stop nvidia-fabricmanager
    systemctl disable nvidia-fabricmanager
fi

# Remove NVIDIA peer memory module
if lsmod | grep nvidia_peermem &> /dev/null
then
    rmmod nvidia_peermem
fi

# Clear topo and graph files
rm -rf /opt/microsoft/ncv4
rm -rf /opt/microsoft/ndv2
rm -rf /opt/microsoft/ndv4
rm -rf /opt/microsoft/ndv5
rm -rf /opt/microsoft/ncv6

# Clear contents of nccl.conf
cat /dev/null > /etc/nccl.conf

"""

# Note: the line after "case $vmSize in" is four spaces, not empty.
_AMD_SETUP_SCRIPT = """\
#!/bin/bash

metadata_endpoint="http://169.254.169.254/metadata/instance?api-version=2019-06-04"
vmSize=$(curl -H Metadata:true $metadata_endpoint | jq -r ".compute.vmSize")
vmSize=$(echo "$vmSize" | awk '{print tolower($0)}')

## Topo file setup based on SKU
case $vmSize in
    
    standard_nd*v4)
        /opt/azurehpc/customizations/ndv4_rocm.sh;;

    standard_nd96is*_mi300x_v5)
        /opt/azurehpc/customizations/ndv5_rocm.sh;;

    *) echo "No SKU customization for $vmSize";;
esac
"""

# Ends with an empty line, like the NVIDIA one.
_AMD_REMOVE_SCRIPT = """\
#!/bin/bash

# Clear topo and graph files
rm -rf /opt/microsoft/ncv4
rm -rf /opt/microsoft/ndv2
rm -rf /opt/microsoft/ndv4
rm -rf /opt/microsoft/ndv5

# Clear contents of nccl.conf
cat /dev/null > /etc/nccl.conf

"""

# The vendor's (setup script, remove script) texts. A dict maps keys to values;
# looking up a missing key with .get() returns None.
_SCRIPTS = {
    "NVIDIA": (_NVIDIA_SETUP_SCRIPT, _NVIDIA_REMOVE_SCRIPT),
    "AMD": (_AMD_SETUP_SCRIPT, _AMD_REMOVE_SCRIPT),
}

# The service file comes in three parts, as in bash (see service_unit()).
_SERVICE_BASE = """\
[Unit]
Description=Customizations based on SKU
After=network-online.target openibd.service rdma-hw.target
Wants=network-online.target
"""

_SERVICE_NVIDIA_DEPS = """\
After=nvidia-fabricmanager.service
Wants=nvidia-fabricmanager.service
"""

# Starts with an empty line, which separates [Unit] from [Service].
_SERVICE_TAIL = """\

[Service]
Type=oneshot
ExecStart=/usr/sbin/setup_sku_customizations.sh
ExecStop=/usr/sbin/remove_sku_customizations.sh
RemainAfterExit=true
StandardOutput=journal

[Install]
WantedBy=multi-user.target
"""

_PEERMEM_DROPIN = """\
[Service]
ExecStartPost=-/bin/sh -c 'if /usr/sbin/modinfo nvidia_peermem >/dev/null 2>&1; then /usr/sbin/modprobe nvidia-peermem && /usr/bin/test -d /sys/module/nvidia_peermem; fi'
"""


# ---- Pure helpers (no side effects; tested directly) ----------------------
def service_unit(gpu: str) -> str:
    """Return sku-customizations.service's text for a vendor.

    bash writes the [Unit] part with `>`, then (NVIDIA only) appends the fabric
    manager After=/Wants= lines with `>>`, then appends the rest with `>>`.
    """
    nvidia_deps = _SERVICE_NVIDIA_DEPS if gpu == "NVIDIA" else ""
    return _SERVICE_BASE + nvidia_deps + _SERVICE_TAIL


def delete_matching_lines(text: str, pattern: re.Pattern) -> tuple[str, int]:
    """Mirror `sed -i '/^<pattern>/d'`: drop every line that starts with a match.

    Returns the new text and how many lines were removed (for the log).
    """
    # splitlines(keepends=True) keeps each line's newline, so joining the kept
    # lines gives the file back unchanged apart from the deleted lines.
    lines = text.splitlines(keepends=True)
    kept = [line for line in lines if not pattern.match(line)]
    return "".join(kept), len(lines) - len(kept)


def replace_first_per_line(text: str, old: str, new: str) -> tuple[str, int]:
    """Mirror `sed -i 's/<old>/<new>/'` for a literal `old` with no sed specials.

    sed works line by line and, without the `g` flag, replaces only the first
    match on each line. Returns the new text and how many lines changed.
    """
    lines = text.splitlines(keepends=True)
    # str.replace(old, new, 1) replaces at most one occurrence.
    edited = [line.replace(old, new, 1) for line in lines]
    changed = sum(1 for before, after in zip(lines, edited) if before != after)
    return "".join(edited), changed


# ---- Steps with side effects ---------------------------------------------
def _mode(path: str) -> str:
    """A file's permission bits as an octal string such as "0755", for logging."""
    return f"{stat.S_IMODE(os.stat(path).st_mode):04o}"


def _run(command: list[str], env: dict[str, str]) -> bool:
    """Run one command; log and return False if it fails.

    exec_program() already logs the command, its output and its exit code;
    here we only add what the failure means for this step.
    """
    rc = exec_program(command, _OP, env=env)
    if rc != 0:
        log_error(_OP, f"'{' '.join(command)}' failed; SKU customizations are "
                       f"not fully set up", command=command[0], exit_code=rc)
        return False
    return True


def _copy_all(src_dir: str, dest_dir: str) -> bool:
    """bash: `mkdir -p <dest_dir>` then `cp <src_dir>/* <dest_dir>`."""
    # glob.glob, like the shell's `*`, skips names starting with "."; sorted()
    # mimics the shell's alphabetical order.
    entries = sorted(glob.glob(os.path.join(src_dir, "*")))
    if not entries:
        # The shell would pass a literal "*" to cp, which then fails.
        log_error(_OP, f"no files to copy in {src_dir}", source=src_dir)
        return False

    copied = 0
    skipped = []
    try:
        os.makedirs(dest_dir, exist_ok=True)              # mkdir -p
        for entry in entries:
            # cp without -r copies files (following symlinks) and refuses
            # folders: it prints "omitting directory", copies the rest and
            # then exits with an error, which stops bash (set -e).
            if os.path.isdir(entry):
                skipped.append(entry)
                continue
            shutil.copy(entry, dest_dir)
            copied += 1
    except OSError as exc:
        log_error(_OP, f"could not copy {src_dir}/* to {dest_dir}: {exc}",
                  source=src_dir, dest=dest_dir)
        return False

    log_info(_OP, f"Copied {copied} files from {src_dir} to {dest_dir}",
             source=src_dir, dest=dest_dir, count=copied)
    log_debug(_OP, f"Copied to {dest_dir}: "
                   f"{', '.join(os.path.basename(e) for e in entries if e not in skipped)}")
    if skipped:
        log_error(_OP, f"{src_dir} contains folders, which `cp` (no -r) refuses "
                       f"to copy, failing the bash step too",
                  source=src_dir, folders=", ".join(skipped))
        return False
    return True


def _write_file(path: str, text: str, mode: int | None = None) -> bool:
    """Write `text` to `path`, replacing any old content (bash's `cat <<EOF >path`).

    When `mode` is given, it is applied afterwards, like bash's `chmod 755`.
    Returns True on success, or False after logging why it failed.
    """
    # Checked first only so the log can say "created" or "replaced".
    action = "replaced" if os.path.exists(path) else "created"
    try:
        # Mode "w" creates the file or empties an existing one before writing.
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        if mode is not None:
            os.chmod(path, mode)
        actual_mode = _mode(path)
    except OSError as exc:
        log_error(_OP, f"could not write {path}: {exc}", path=path)
        return False
    log_info(_OP, f"Wrote {path}", path=path, action=action, mode=actual_mode)
    # The full text is bulky, so it goes to DEBUG only.
    log_debug(_OP, f"Contents of {path}:\n{text}", path=path)
    return True


def _edit_in_place(path: str, edit) -> int | None:
    """Apply `edit` (text -> (new text, lines changed)) to a file, like `sed -i`.

    Returns the number of lines changed, or None after logging a failure.
    sed -i fails when the file is missing, and so do we.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        new_text, changed = edit(text)
        # sed -i rewrites the file even when nothing matched; so do we.
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(new_text)
    except OSError as exc:
        log_error(_OP, f"could not edit {path}: {exc}", path=path)
        return None
    return changed


def _setup_nvidia_extras() -> bool:
    """NVIDIA only: the openibd peermem drop-in and the fabric manager fix."""
    dropin = os.path.join(_OPENIBD_DROPIN_DIR, _PEERMEM_DROPIN_NAME)
    try:
        os.makedirs(_OPENIBD_DROPIN_DIR, exist_ok=True)   # mkdir -p
    except OSError as exc:
        log_error(_OP, f"could not create {_OPENIBD_DROPIN_DIR}: {exc}",
                  path=_OPENIBD_DROPIN_DIR)
        return False
    if not _write_file(dropin, _PEERMEM_DROPIN):
        return False

    # A workaround to make the fabric manager service start automatically.
    # os.path.isfile is bash's `[ -f path ]` (true for a regular file, or a
    # symlink to one).
    if not os.path.isfile(_FABRICMANAGER_UNIT):
        log_info(_OP, f"{_FABRICMANAGER_UNIT} not found; no fabric manager "
                      f"ordering fix needed", path=_FABRICMANAGER_UNIT)
        return True
    removed = _edit_in_place(
        _FABRICMANAGER_UNIT,
        lambda text: delete_matching_lines(text, _FM_AFTER_PATTERN))
    if removed is None:
        return False
    log_info(_OP, f"Edited {_FABRICMANAGER_UNIT} in place: removed "
                  f"After=multi-user.target", path=_FABRICMANAGER_UNIT,
             lines_removed=removed)
    return True


def _add_gb200_options(path: str):
    """The edit for one GRUB file: warn on a rerun, then do bash's sed.

    Returns a function (text -> (new text, lines changed)) for _edit_in_place.
    """
    def edit(text: str) -> tuple[str, int]:
        # BASH QUIRK (kept): the replacement still starts with
        # "lockdown=integrity", so a second run matches again and adds the
        # options a second time. Warn when they are already there.
        if _GB200_CMDLINE in text:
            log_warn(_OP, f"GB200 kernel options are already in {path}; adding "
                          "them again, as bash's sed does on a rerun", path=path)
        return replace_first_per_line(text, _LOCKDOWN, _GB200_CMDLINE)
    return edit


def _setup_gb200(env: dict[str, str]) -> bool:
    """Azure Linux 3.0 + GB200: GRUB kernel options, then irqbalance off."""
    for path in (_GRUB_DEFAULT, _GRUB_CFG):
        changed = _edit_in_place(path, _add_gb200_options(path))
        if changed is None:
            return False
        if changed == 0:
            # Bash carries on silently here; the warning makes it visible.
            log_warn(_OP, f"{_LOCKDOWN} not found in {path}; GB200 kernel "
                          f"options were not added there", path=path)
        else:
            log_info(_OP, f"Edited {path} in place: added GB200 kernel options "
                          f"after {_LOCKDOWN}", path=path, lines_changed=changed,
                     options=_GB200_CMDLINE)

    # Recommended to disable irqbalance in GB200 images (bash comment).
    for action in ("stop", "disable"):
        if not _run(["systemctl", action, "irqbalance"], env):
            return False
    log_info(_OP, "Stopped and disabled irqbalance (GB200)", service="irqbalance")
    return True


def install(env: dict[str, str]) -> int:
    """Set up the boot-time SKU customization service. Returns 0 or 3."""
    gpu = env.get("GPU", "")
    sku = env.get("SKU", "")
    distro = env.get("DISTRIBUTION", "")
    component_dir = env.get("COMPONENT_DIR", "")
    top_dir = env.get("TOP_DIR", "")

    log_info(_OP, "Setting up SKU customizations", gpu=gpu, sku=sku, distro=distro,
             node_type=env.get("TARGET_NODE_TYPE", ""))

    scripts = _SCRIPTS.get(gpu)
    if scripts is None:
        # bash: echo "GPU is not NVIDIA or AMD, exiting"; exit 1
        log_error(_OP, "GPU is not NVIDIA or AMD", gpu=gpu)
        return 3
    if not component_dir or not top_dir:
        log_error(_OP, "COMPONENT_DIR and TOP_DIR must be set",
                  component_dir=component_dir, top_dir=top_dir)
        return 3
    # Unpack the (setup, remove) pair into two names.
    setup_text, remove_text = scripts
    log_info(_OP, f"Using the {gpu} setup/remove scripts", gpu=gpu)

    # ---- 1. Customization scripts and topology files ----------------------
    if not _copy_all(os.path.join(component_dir, "customizations"), _CUSTOMIZATIONS_DIR):
        return 3
    if not _copy_all(os.path.join(top_dir, "topology"), _TOPOLOGY_DIR):
        return 3

    # ---- 2. The setup/remove scripts and the service ----------------------
    if not _write_file(_SETUP_SCRIPT, setup_text, mode=0o755):
        return 3
    if not _write_file(_REMOVE_SCRIPT, remove_text, mode=0o755):
        return 3
    if not _write_file(_SERVICE_FILE, service_unit(gpu)):
        return 3

    # ---- 3. NVIDIA extras --------------------------------------------------
    if gpu == "NVIDIA" and not _setup_nvidia_extras():
        return 3

    # ---- 4. GB200 on Azure Linux 3.0 --------------------------------------
    gb200 = distro == _GB200_DISTRO and sku == _GB200_SKU
    if gb200:
        log_info(_OP, "Applying GB200 GRUB options and disabling irqbalance",
                 distro=distro, sku=sku)
        if not _setup_gb200(env):
            return 3
    else:
        log_info(_OP, f"GB200 GRUB/irqbalance changes not needed (only for "
                      f"{_GB200_SKU} on {_GB200_DISTRO})", distro=distro, sku=sku)

    # ---- 5. Enable (but don't start) the service --------------------------
    if not _run(["systemctl", "enable", _SERVICE_NAME], env):
        return 3
    log_info(_OP, f"Enabled {_SERVICE_NAME} (not started during the build; it "
                  f"runs at boot)", service=_SERVICE_NAME)

    log_info(_OP, "SKU customizations set up", gpu=gpu, sku=sku, distro=distro,
             service=_SERVICE_NAME, gb200_changes=gb200)
    return 0
