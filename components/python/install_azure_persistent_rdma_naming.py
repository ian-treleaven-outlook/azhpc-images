"""install_azure_persistent_rdma_naming.py — Python port of
components/install_azure_persistent_rdma_naming.sh.

Gives a VM's RDMA devices stable names (mlx5_ib<N> for InfiniBand ports,
mlx5_an<N> for Ethernet / accelerated networking ports), at boot and again
every minute:

  1. Build `rdma_rename` from the rdma-core release in versions.json
     ("rdma_core") and install it as /usr/sbin/rdma_rename_<version>. (The
     distro's rdma_rename can't be used: the NAME_FIXED mode is needed.)
  2. Write /usr/sbin/azure_persistent_rdma_naming.sh, which renames every
     device not already called mlx5_ib<N> / mlx5_an<N>, numbering on from the
     highest existing index.
  3. Write azure_persistent_rdma_naming.service (runs the script after udev
     settles and openibd starts) and its .timer (30s after boot, then 60s
     after each run).
  4. `systemctl daemon-reload`, enable the service and timer, start both.

Upstream #614 (Ubuntu 26.04 support) made the script rerunnable, added the
timer and deleted install_azure_persistent_rdma_naming_monitor.sh.

The generated files
-------------------
Bash writes the naming script with an unquoted heredoc: only
`${RDMA_CORE_VERSION}` expands, every other `$` is escaped as `\\$` in the bash
source. _NAMING_SCRIPT_TEMPLATE is the text as it lands in the file, with that
one placeholder left for naming_script() to fill. It and the two units are
compared byte for byte with fixtures rendered by real bash.

Kept bash quirks
----------------
  * `mkdir rdma-core && tar ...`: `set -e` ignores a failure that isn't the
    last command of an `&&` list, so if /tmp/rdma-core is left over from a
    failed run, bash skips tar and builds in the old folder. So does this
    step, with a warning. A failing tar stops the step.
  * The downloaded tarball is left in /tmp; only the source folder is removed.
  * No component version is recorded (bash records none); the rdma-core
    version is logged.

Deliberate differences
----------------------
  * A missing rdma_core version or sha256 fails before anything is downloaded
    (bash would fetch "vnull.tar.gz").
  * The tarball is downloaded straight into /tmp (bash downloads into its
    current folder, then moves it there).
  * `cp` is copy_file (shutil.copy), which also copies the built file's mode: a new file
    gets 0755 either way, but an existing rdma_rename_<version> gets the
    built mode too, where `cp` would have kept the old one.
"""

from __future__ import annotations

import os

from utils.component_config import require_config
from utils.download import fetch_verified
from utils.files import copy_file, extract_tarball, remove_paths, write_file
from utils.logger import log_error, log_info, log_warn
from utils.process import run
from utils.services import systemctl

_OP = "install-persistent-rdma-naming"

# Paths are module constants so tests can point them at a temporary folder.
_WORK_DIR = "/tmp"                       # download + build folder (bash: pushd /tmp)
_SRC_DIR_NAME = "rdma-core"              # unpacked source, under _WORK_DIR
_SBIN_DIR = "/usr/sbin"                  # where rdma_rename_<version> is installed
_NAMING_SCRIPT = "/usr/sbin/azure_persistent_rdma_naming.sh"
_SERVICE_NAME = "azure_persistent_rdma_naming.service"
_SERVICE_FILE = "/etc/systemd/system/azure_persistent_rdma_naming.service"
_TIMER_NAME = "azure_persistent_rdma_naming.timer"
_TIMER_FILE = "/etc/systemd/system/azure_persistent_rdma_naming.timer"

_URL = "https://github.com/linux-rdma/rdma-core/archive/refs/tags/{tarball}"

# bash's systemctl calls, in order.
_SYSTEMCTL_CALLS = (
    ("daemon-reload",),
    ("enable", _SERVICE_NAME),
    ("enable", _TIMER_NAME),
    ("start", _SERVICE_NAME),
    ("start", _TIMER_NAME),
)

# The naming script as bash writes it, ${RDMA_CORE_VERSION} still a placeholder.
# Careful when editing: indentation and several "empty" lines are TABs, as in
# the bash heredoc (the fixture test catches a stray change). Raw string, so
# `[\ \t]` and `\1` in the sed command reach the file unchanged.
_NAMING_SCRIPT_TEMPLATE = r"""#!/bin/bash

rdma_rename=/usr/sbin/rdma_rename_${RDMA_CORE_VERSION}

mapfile -t all_devices < <(ibdev2netdev -v | sort -n | awk '{print $2}')

next_index() {
	local prefix=$1
	local max_index=-1
	local dev suffix

	for dev in "${all_devices[@]}"; do
		suffix=${dev#$prefix}
		if [[ "$dev" == "$prefix"* && "$suffix" =~ ^[0-9]+$ && "$suffix" -gt "$max_index" ]]; then
			max_index=$suffix
		fi
	done

	echo $((max_index + 1))
}

an_index=$(next_index mlx5_an)
ib_index=$(next_index mlx5_ib)

for old_device in "${all_devices[@]}"; do

	case "$old_device" in
		mlx5_ib*|mlx5_an*) continue ;;
	esac

	link_layer=$(ibv_devinfo -d $old_device | sed -n 's/^[\ \t]*link_layer:[\ \t]*\([a-zA-Z]*\)$/\1/p')
	
	if [ "$link_layer" = "InfiniBand" ]; then
		
		$rdma_rename $old_device NAME_FIXED mlx5_ib${ib_index}
		ib_index=$(($ib_index + 1))
		
	elif [ "$link_layer" = "Ethernet" ]; then
	
		$rdma_rename $old_device NAME_FIXED mlx5_an${an_index}
		an_index=$(($an_index + 1))
		
	else
	
		echo "Unknown device type for $old_device."
		
	fi
	
done
"""
_VERSION_PLACEHOLDER = "${RDMA_CORE_VERSION}"

_SERVICE_UNIT = """\
[Unit]
Description=Azure persistent RDMA naming
After=network.target systemd-udev-settle.service openibd.service
Wants=systemd-udev-settle.service

[Service]
Type=oneshot
ExecStart=/usr/sbin/azure_persistent_rdma_naming.sh
StandardOutput=journal

[Install]
WantedBy=multi-user.target
"""

_TIMER_UNIT = """\
[Unit]
Description=Retry Azure persistent RDMA naming

[Timer]
OnBootSec=30s
OnUnitActiveSec=60s
AccuracySec=5s
Unit=azure_persistent_rdma_naming.service

[Install]
WantedBy=timers.target
"""


def tarball_name(version: str) -> str:
    return f"v{version}.tar.gz"


def rdma_rename_path(version: str) -> str:
    """Where the built rdma_rename is installed, e.g. /usr/sbin/rdma_rename_59.0."""
    return os.path.join(_SBIN_DIR, f"rdma_rename_{version}")


def naming_script(version: str) -> str:
    """The text of /usr/sbin/azure_persistent_rdma_naming.sh for `version`.

    The path inside is always /usr/sbin/... (what runs on the image), not the
    _SBIN_DIR tests may redirect.
    """
    return _NAMING_SCRIPT_TEMPLATE.replace(_VERSION_PLACEHOLDER, version)


def _unpack(src_dir: str, tarball: str, env: dict[str, str]) -> bool:
    """bash: `mkdir rdma-core && tar -xvf $TARBALL --strip-components=1 -C rdma-core`."""
    try:
        os.mkdir(src_dir)             # fails if it exists, like `mkdir` without -p
    except FileExistsError:
        if not os.path.isdir(src_dir):
            # bash would skip tar and then fail at `pushd rdma-core`.
            log_error(_OP, f"{src_dir} exists and is not a folder", path=src_dir)
            return False
        log_warn(_OP, f"{src_dir} already exists (left by an earlier failed run?); "
                      f"building in it without extracting {tarball} (same as bash)",
                 path=src_dir)
        return True
    except OSError as exc:
        log_error(_OP, f"could not create {src_dir}: {exc}", path=src_dir)
        return False
    return extract_tarball(tarball, _OP, dest=_SRC_DIR_NAME, strip_components=1,
                           cwd=_WORK_DIR, env=env)


def _build_rdma_rename(version: str, sha256: str, env: dict[str, str]) -> bool:
    """Step 1: download and build rdma-core, install rdma_rename_<version>."""
    tarball = tarball_name(version)
    if fetch_verified(_URL.format(tarball=tarball), sha256, _OP, _WORK_DIR) is None:
        return False
    src_dir = os.path.join(_WORK_DIR, _SRC_DIR_NAME)
    if not (_unpack(src_dir, tarball, env)
            and run(["bash", "build.sh"], _OP, cwd=src_dir, env=env)
            and copy_file(os.path.join(src_dir, "build", "bin", "rdma_rename"),
                          rdma_rename_path(version), _OP)):
        return False
    remove_paths(src_dir, op=_OP)
    log_info(_OP, f"Removed build folder {src_dir}; {tarball} is left in "
                  f"{_WORK_DIR} (same as bash)", path=src_dir)
    return True


def _write_units(version: str) -> bool:
    """Steps 2-3: the naming script, its service and the retry timer."""
    return (write_file(_NAMING_SCRIPT, naming_script(version), _OP, mode=0o755)
            and write_file(_SERVICE_FILE, _SERVICE_UNIT, _OP)
            and write_file(_TIMER_FILE, _TIMER_UNIT, _OP))


def _enable_and_start(env: dict[str, str]) -> bool:
    """Step 4: load the new units, enable both, start both (bash's order)."""
    for args in _SYSTEMCTL_CALLS:
        if not systemctl(*args, op=_OP, env=env):
            return False
    log_info(_OP, f"Enabled and started {_SERVICE_NAME} and {_TIMER_NAME}",
             service=_SERVICE_NAME, timer=_TIMER_NAME)
    return True


def install(env: dict[str, str]) -> int:
    """Install rdma_rename and the persistent RDMA naming services. Returns 0 or 3."""
    cfg = require_config("rdma_core", env, _OP, "version", "sha256")
    if cfg is None:
        return 3
    version = cfg["version"]
    log_info(_OP, "Installing persistent RDMA naming", rdma_core_version=version,
             source=_URL.format(tarball=tarball_name(version)),
             distro=env.get("DISTRIBUTION", ""), arch=env.get("ARCHITECTURE", ""))

    if not _build_rdma_rename(version, cfg["sha256"], env):
        return 3
    if not _write_units(version):
        return 3
    if not _enable_and_start(env):
        return 3

    log_info(_OP, "Persistent RDMA naming installed", rdma_core_version=version,
             rdma_rename=rdma_rename_path(version), naming_script=_NAMING_SCRIPT,
             service=_SERVICE_NAME, timer=_TIMER_NAME)
    return 0
