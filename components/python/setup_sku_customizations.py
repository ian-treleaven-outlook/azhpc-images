"""setup_sku_customizations.py — Python port of components/setup_sku_customizations.sh.

Prepares sku-customizations.service, a boot-time systemd service that tunes
the finished image for the VM size it lands on (NCCL topology files, NVMe
set-up, ...). The image is generic; the VM size is only known at boot.

  1. Copy components/customizations/* to /opt/azurehpc/customizations and
     topology/* to /opt/microsoft.
  2. Write the vendor's (NVIDIA or AMD) /usr/sbin/setup_sku_customizations.sh,
     which asks IMDS for the VM size and runs the matching customization,
     and remove_sku_customizations.sh, which undoes it; both mode 755.
  3. Write the service unit (NVIDIA also orders it after fabric manager).
  4. NVIDIA: an openibd.service drop-in that loads nvidia-peermem, and, if
     the fabric manager unit exists, delete its "After=multi-user.target"
     lines so it can start automatically.
  5. Azure Linux 3.0 on GB200 (VR200 is built as GB200, see SKU_ALIASES in
     utils/build_config.py): add kernel options after "lockdown=integrity" in
     /etc/default/grub and /boot/grub2/grub.cfg, and stop and disable
     irqbalance.
  6. `systemctl enable sku-customizations` — not started during the build
     (pointless before the reboot, and it may fail on general-purpose sizes).

The generated texts are the _..._SCRIPT, _SERVICE_... and _PEERMEM_DROPIN
constants. Bash writes them with unquoted heredocs but escapes every `$`, so
the files hold plain `$`; the tests compare them with real bash renders in
tests/python/fixtures/setup_sku_customizations.

utils/plans/steps.py makes install() the action of the
"setup-sku-customizations" Step; deleting `action=` runs the bash again.

Deliberate differences from bash
--------------------------------
  * The GPU vendor (and COMPONENT_DIR/TOP_DIR) are checked before anything is
    copied; bash copies first, then exits 1 for an unknown vendor.
  * The service file is written once, from the same three parts in the same
    order, instead of one `>` and two `>>`. The content is identical.
  * `cp` is copy_file() (shutil.copy), which also copies the mode: for an
    existing file in the destination, `cp` would keep the old mode.
  * The GB200 commands run without bash's `sudo`: the build runs as root.
  * Bash's `$DISTRIBUTION == "azurelinux3.0"` is Distro.is_azurelinux (the
    only Azure Linux built).
  * Files are written with write_file, which creates a missing parent folder.

Kept bash quirks
----------------
  * The GB200 options still start with "lockdown=integrity", so a rerun adds
    them again (warned). A GRUB file without "lockdown=integrity" is left
    alone, as sed does (warned); a missing one fails the step, as sed -i does.
  * `cp src/*` skips dot files and refuses folders: the files are copied,
    then the step fails.
"""

from __future__ import annotations

import glob
import os
import re

from utils.distro import Distro
from utils.files import copy_file, edit_file, make_dirs, write_file
from utils.logger import log_error, log_info, log_warn
from utils.services import systemctl

_OP = "setup-sku-customizations"

# Module-level so the tests can point them at a temporary folder.
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

_GB200_SKU = "GB200"

# bash: sed -i 's/lockdown=integrity/lockdown=integrity <options>/' <file>
# Nothing in the pattern is special to sed, so a plain replace of the first
# occurrence on each line does the same.
_LOCKDOWN = "lockdown=integrity"
_GB200_CMDLINE = (
    "lockdown=integrity efi_mmap_nr_slack_slots=256 numa_balancing=disable "
    "iommu.passthrough=1 irqchip.gicv3_nolpi=y arm_smmu_v3.disable_msipolling=1 "
    "init_on_alloc=0 nvme_core.io_timeout=240 panic=-1")

# bash: sed -i '/^After=multi-user.target/d' <fabric manager unit>
# re.match anchors at the line start like "^"; "." is a wildcard in both.
_FM_AFTER_PATTERN = re.compile(r"After=multi-user.target")

# ---- Generated file texts, byte for byte what bash writes -----------------
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

# (setup script, remove script) per GPU vendor; any other vendor is an error.
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
    edited = [line.replace(old, new, 1) for line in lines]
    changed = sum(1 for before, after in zip(lines, edited) if before != after)
    return "".join(edited), changed


# ---- Steps: each returns True, or False after the failure was logged -------
def _copy_all(src_dir: str, dest_dir: str) -> bool:
    """bash: `mkdir -p <dest_dir>` then `cp <src_dir>/* <dest_dir>`."""
    # Not copy_files(): `cp` without -r copies every file and only then fails
    # on folders, where copy_files() would stop at the first folder.
    # Like the shell's `*`: no dot files, alphabetical order.
    entries = sorted(glob.glob(os.path.join(src_dir, "*")))
    if not entries:
        # The shell would pass a literal "*" to cp, which then fails.
        log_error(_OP, f"no files to copy in {src_dir}", source=src_dir)
        return False
    files = [entry for entry in entries if not os.path.isdir(entry)]
    folders = [entry for entry in entries if os.path.isdir(entry)]
    if not (make_dirs(dest_dir, op=_OP)
            and all(copy_file(entry, dest_dir, _OP) for entry in files)):
        return False
    log_info(_OP, f"Copied {len(files)} files from {src_dir} to {dest_dir}",
             source=src_dir, dest=dest_dir, count=len(files))
    if folders:
        log_error(_OP, f"{src_dir} contains folders, which `cp` (no -r) refuses "
                       "to copy, failing the bash step too",
                  source=src_dir, folders=", ".join(folders))
        return False
    return True


def _copy_repo_files(env: dict[str, str]) -> bool:
    """The customization scripts and the topology files."""
    return (_copy_all(os.path.join(env["COMPONENT_DIR"], "customizations"),
                      _CUSTOMIZATIONS_DIR)
            and _copy_all(os.path.join(env["TOP_DIR"], "topology"), _TOPOLOGY_DIR))


def _write_service(gpu: str) -> bool:
    """The vendor's setup/remove scripts (mode 755) and the service unit."""
    setup_text, remove_text = _SCRIPTS[gpu]
    return (write_file(_SETUP_SCRIPT, setup_text, _OP, mode=0o755)
            and write_file(_REMOVE_SCRIPT, remove_text, _OP, mode=0o755)
            and write_file(_SERVICE_FILE, service_unit(gpu), _OP))


def _setup_nvidia_extras() -> bool:
    """NVIDIA only: the openibd peermem drop-in and the fabric manager fix."""
    dropin = os.path.join(_OPENIBD_DROPIN_DIR, _PEERMEM_DROPIN_NAME)
    if not write_file(dropin, _PEERMEM_DROPIN, _OP):
        return False
    # bash's `[ -f path ]`: a regular file, or a symlink to one.
    if not os.path.isfile(_FABRICMANAGER_UNIT):
        log_info(_OP, f"{_FABRICMANAGER_UNIT} not found; no fabric manager "
                      "ordering fix needed", path=_FABRICMANAGER_UNIT)
        return True
    def edit(text: str) -> str:
        new_text, removed = delete_matching_lines(text, _FM_AFTER_PATTERN)
        log_info(_OP, f"Removing After=multi-user.target from {_FABRICMANAGER_UNIT}",
                 path=_FABRICMANAGER_UNIT, lines_removed=removed)
        return new_text
    return edit_file(_FABRICMANAGER_UNIT, edit, _OP)


def _add_gb200_options(path: str) -> bool:
    """bash's GB200 sed on one GRUB file, warning on a rerun or no match."""
    def edit(text: str) -> str:
        # The options start with "lockdown=integrity", so a rerun adds them again.
        if _GB200_CMDLINE in text:
            log_warn(_OP, f"GB200 kernel options are already in {path}; adding "
                          "them again, as bash's sed does on a rerun", path=path)
        new_text, changed = replace_first_per_line(text, _LOCKDOWN, _GB200_CMDLINE)
        if changed == 0:
            log_warn(_OP, f"{_LOCKDOWN} not found in {path}; GB200 kernel "
                          "options were not added there", path=path)
        else:
            log_info(_OP, f"Adding GB200 kernel options after {_LOCKDOWN} in {path}",
                     path=path, lines_changed=changed, options=_GB200_CMDLINE)
        return new_text
    # A missing file fails here, as bash's sed -i does.
    return edit_file(path, edit, _OP)


def _setup_gb200(env: dict[str, str]) -> bool:
    """Azure Linux 3.0 + GB200: GRUB kernel options, then irqbalance off."""
    if not (_add_gb200_options(_GRUB_DEFAULT) and _add_gb200_options(_GRUB_CFG)
            and systemctl("stop", "irqbalance", op=_OP, env=env)
            and systemctl("disable", "irqbalance", op=_OP, env=env)):
        return False
    log_info(_OP, "Stopped and disabled irqbalance (GB200)", service="irqbalance")
    return True


def _inputs_ok(env: dict[str, str]) -> bool:
    """A known GPU vendor, and the repo folders to copy from."""
    gpu = env.get("GPU", "")
    if gpu not in _SCRIPTS:
        # bash: echo "GPU is not NVIDIA or AMD, exiting"; exit 1
        log_error(_OP, "GPU is not NVIDIA or AMD", gpu=gpu)
        return False
    if not env.get("COMPONENT_DIR") or not env.get("TOP_DIR"):
        log_error(_OP, "COMPONENT_DIR and TOP_DIR must be set",
                  component_dir=env.get("COMPONENT_DIR", ""), top_dir=env.get("TOP_DIR", ""))
        return False
    log_info(_OP, f"Using the {gpu} setup/remove scripts", gpu=gpu)
    return True


def install(env: dict[str, str]) -> int:
    """Set up the boot-time SKU customization service. Returns 0 or 3."""
    gpu, sku = env.get("GPU", ""), env.get("SKU", "")
    distro = Distro.from_env(env)
    log_info(_OP, "Setting up SKU customizations", gpu=gpu, sku=sku, distro=str(distro),
             node_type=env.get("TARGET_NODE_TYPE", ""))
    if not _inputs_ok(env):
        return 3
    if not _copy_repo_files(env) or not _write_service(gpu):
        return 3
    if gpu == "NVIDIA" and not _setup_nvidia_extras():
        return 3

    # Bash: [[ $DISTRIBUTION == "azurelinux3.0" && "$SKU" == "GB200" ]]
    gb200 = distro.is_azurelinux and sku == _GB200_SKU
    if gb200:
        log_info(_OP, "Applying GB200 GRUB options and disabling irqbalance",
                 distro=str(distro), sku=sku)
        if not _setup_gb200(env):
            return 3
    else:
        log_info(_OP, "GB200 GRUB/irqbalance changes not needed (only for "
                      f"{_GB200_SKU} on Azure Linux)", distro=str(distro), sku=sku)

    if not systemctl("enable", _SERVICE_NAME, op=_OP, env=env):
        return 3
    log_info(_OP, f"Enabled {_SERVICE_NAME} (not started during the build; it "
                  "runs at boot)", service=_SERVICE_NAME)
    log_info(_OP, "SKU customizations set up", gpu=gpu, sku=sku, distro=str(distro),
             service=_SERVICE_NAME, gb200_changes=gb200)
    return 0
