"""install_azure_persistent_rdma_naming.py — Python port of
components/install_azure_persistent_rdma_naming.sh.

What this step does
-------------------
Gives the RDMA (InfiniBand / RoCE) devices of a VM stable, predictable names:
mlx5_ib0, mlx5_ib1, ... for InfiniBand ports and mlx5_an0, ... for Ethernet
(accelerated networking) ports. It does that at boot, and again every minute,
with a systemd service and timer:

  1. Build `rdma_rename` from the rdma-core source release named in
     versions.json ("rdma_core"), and install it as
     /usr/sbin/rdma_rename_<version>. (The distro's rdma_rename can't be used:
     we need its NAME_FIXED mode, so we build a known version.)
  2. Write /usr/sbin/azure_persistent_rdma_naming.sh, a small script that walks
     the devices (`ibdev2netdev -v`), skips those already named mlx5_ib<N> /
     mlx5_an<N>, and renames the others with rdma_rename, numbering on from
     the highest existing index.
  3. Write azure_persistent_rdma_naming.service, which runs that script (after
     udev has settled and openibd has started), and
     azure_persistent_rdma_naming.timer, which runs the service 30s after boot
     and then 60s after each run.
  4. `systemctl daemon-reload`, enable the service and the timer, then start
     the service and the timer now.

Upstream (#614, the Ubuntu 26.04 support) rewrote the naming script so it can
run repeatedly, added the timer, and deleted the separate
install_azure_persistent_rdma_naming_monitor.sh (a looping monitor service that
restarted the naming service), which this step used to run.

The original bash, for comparison (the three heredoc texts are shortened here;
they are in full below as _NAMING_SCRIPT_TEMPLATE, _SERVICE_UNIT and
_TIMER_UNIT):

    set -ex
    source ${UTILS_DIR}/utilities.sh
    rdma_core_metadata=$(get_component_config "rdma_core")
    RDMA_CORE_VERSION=$(jq -r '.version' <<< $rdma_core_metadata)
    RDMA_CORE_SHA=$(jq -r '.sha256' <<< $rdma_core_metadata)

    TARBALL="v${RDMA_CORE_VERSION}.tar.gz"
    RDMA_CORE_DOWNLOAD_URL=https://github.com/linux-rdma/rdma-core/archive/refs/tags/${TARBALL}
    download_and_verify ${RDMA_CORE_DOWNLOAD_URL} ${RDMA_CORE_SHA} /tmp

    pushd /tmp
    mkdir rdma-core && tar -xvf $TARBALL --strip-components=1 -C rdma-core
    pushd rdma-core
    bash build.sh
    cp build/bin/rdma_rename /usr/sbin/rdma_rename_$RDMA_CORE_VERSION
    popd
    rm -rf rdma-core
    popd

    cat <<EOF >/usr/sbin/azure_persistent_rdma_naming.sh
    #!/bin/bash
    rdma_rename=/usr/sbin/rdma_rename_${RDMA_CORE_VERSION}
    ... (every other $ is written as \\$, so it stays literal) ...
    EOF
    chmod 755 /usr/sbin/azure_persistent_rdma_naming.sh

    cat <<EOF >/etc/systemd/system/azure_persistent_rdma_naming.service
    ...
    EOF

    cat <<EOF >/etc/systemd/system/azure_persistent_rdma_naming.timer
    ...
    EOF

    systemctl daemon-reload
    systemctl enable azure_persistent_rdma_naming.service
    systemctl enable azure_persistent_rdma_naming.timer
    systemctl start azure_persistent_rdma_naming.service
    systemctl start azure_persistent_rdma_naming.timer

About the generated script: `cat <<EOF` (an *unquoted* heredoc) expands
variables while writing. Only `${RDMA_CORE_VERSION}` is meant to expand; every
other `$` is escaped as `\\$` in the bash source so it reaches the file as a
plain `$`. Our template is the text as it lands in the file, with the one
`${RDMA_CORE_VERSION}` left in place and filled in by naming_script(). It and
the two units were compared byte for byte with the files real bash writes.

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets this module's install() as the `action=` of the
"install-persistent-rdma-naming" Step. The build then calls install(env)
instead of running the bash script. Deleting `action=` runs the bash script
again, which is why the .sh file is kept.

install() returns 0 for success or 3 for failure. It logs problems and returns
3; it does not let exceptions escape to the build loop. Like bash's `set -e`,
the first failure stops the step.

Which commands still run as programs
------------------------------------
  * `tar`: --strip-components has no simple tarfile equivalent, and the tar
    command keeps the archive's symlinks and modes exactly as bash.
  * `bash build.sh`: rdma-core's own build script (cmake + ninja/make).
  * `systemctl daemon-reload` / `enable` / `start`.
The copy, file writes, chmod and clean-up are plain Python.

A bash subtlety we mirror: `mkdir rdma-core && tar ...`
-------------------------------------------------------
`set -e` does not stop the script when a command that is *not the last* in an
`&&` list fails (checked in WSL). So if /tmp/rdma-core already exists (a run
that failed before its `rm -rf`), bash's `mkdir` fails, `tar` is skipped and
the build runs in the folder that is already there. install() does the same,
with a warning. A failing `tar` (the last command of the list) does stop bash,
and returns 3 here.

Deliberate differences from bash
--------------------------------
  * A missing rdma_core version or sha256 fails the step before anything is
    downloaded. Bash would have tried to download "vnull.tar.gz".
  * Bash downloads into its current folder and then moves the file into /tmp;
    we download straight into /tmp. As in bash, the tarball is left in /tmp.
  * `cp` is shutil.copy(), which copies the file and its permission bits. For
    a new file this matches `cp` (rdma_rename is built 0755). If an
    rdma_rename_<version> already existed, `cp` would have kept that file's
    old mode; shutil.copy sets the built file's mode.
  * Bash records no component version for this step, and neither do we. The
    rdma-core version used is logged instead.
"""

from __future__ import annotations

import os
import shutil
import stat

from utils.component_config import config_for
from utils.download import ChecksumError, download_and_verify
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.process import exec_program

# The Step's `op` label in plans/steps.py. Every log line uses it.
_OP = "install-persistent-rdma-naming"

# Where things go. They are module-level "constants" (capital names that the
# program never changes) so the tests can point them at a temporary folder.
_WORK_DIR = "/tmp"                       # download + build folder (bash: pushd /tmp)
_SRC_DIR_NAME = "rdma-core"              # unpacked source, under _WORK_DIR
_SBIN_DIR = "/usr/sbin"                  # where rdma_rename_<version> is installed
_NAMING_SCRIPT = "/usr/sbin/azure_persistent_rdma_naming.sh"
_SERVICE_NAME = "azure_persistent_rdma_naming.service"
_SERVICE_FILE = "/etc/systemd/system/azure_persistent_rdma_naming.service"
_TIMER_NAME = "azure_persistent_rdma_naming.timer"
_TIMER_FILE = "/etc/systemd/system/azure_persistent_rdma_naming.timer"

_URL = "https://github.com/linux-rdma/rdma-core/archive/refs/tags/{tarball}"

# The generated naming script, exactly as bash writes it, except that the one
# value bash expands, ${RDMA_CORE_VERSION}, is still a placeholder here.
#
# Careful when editing: the indentation is TAB characters and several "empty"
# lines contain tabs, exactly as in the bash heredoc. The tests compare this
# text with a file rendered by real bash, so a stray change shows up there.
#
# The r"""...""" prefix makes a "raw" string: backslashes are kept as typed, so
# `[\ \t]` and `\1` in the sed command reach the file unchanged. The text
# starts right after the opening quotes, so it begins with "#!/bin/bash".
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

# The placeholder in the template above that naming_script() fills in.
_VERSION_PLACEHOLDER = "${RDMA_CORE_VERSION}"

# The systemd unit, exactly as bash writes it (nothing in it is expanded).
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

# The retry timer, exactly as bash writes it (nothing in it is expanded either).
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


def rdma_rename_path(version: str) -> str:
    """Where the built rdma_rename is installed, e.g. /usr/sbin/rdma_rename_59.0."""
    # _SBIN_DIR is read when the function runs, so tests can redirect it.
    return os.path.join(_SBIN_DIR, f"rdma_rename_{version}")


def naming_script(version: str) -> str:
    """Return the text of /usr/sbin/azure_persistent_rdma_naming.sh.

    This is what bash's unquoted heredoc produces: ${RDMA_CORE_VERSION} becomes
    the version, and everything else stays as written. The path inside the
    script is always /usr/sbin/... (it's what runs on the finished image), so
    it doesn't use _SBIN_DIR, which tests may redirect.
    """
    return _NAMING_SCRIPT_TEMPLATE.replace(_VERSION_PLACEHOLDER, version)


def _mode(path: str) -> str:
    """A file's permission bits as an octal string such as "0755", for logging."""
    # S_IMODE keeps only the permission bits of st_mode (drops the file type).
    return f"{stat.S_IMODE(os.stat(path).st_mode):04o}"


def _run(command: list[str], env: dict[str, str], cwd: str | None = None) -> bool:
    """Run one command; log and return False if it fails.

    exec_program() already logs the command, its output and its exit code;
    here we only add what the failure means for this step.
    """
    rc = exec_program(command, _OP, cwd=cwd, env=env)
    if rc != 0:
        log_error(_OP, f"'{' '.join(command)}' failed; persistent RDMA naming "
                       f"is not installed", command=command[0], exit_code=rc)
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


def _unpack(work_dir: str, src_dir: str, tarball: str, env: dict[str, str]) -> bool:
    """bash: `mkdir rdma-core && tar -xvf $TARBALL --strip-components=1 -C rdma-core`."""
    try:
        # os.mkdir, unlike os.makedirs(exist_ok=True), fails if the folder
        # exists, like `mkdir` without -p.
        os.mkdir(src_dir)
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
        # Bash would skip tar here too, then fail at `pushd rdma-core`.
        log_error(_OP, f"could not create {src_dir}: {exc}", path=src_dir)
        return False

    # We only get here when mkdir succeeded.
    return _run(["tar", "-xvf", tarball, "--strip-components=1",
                 "-C", _SRC_DIR_NAME], env, cwd=work_dir)


def _build_rdma_rename(version: str, sha256: str, env: dict[str, str]) -> bool:
    """Download rdma-core, build it, install rdma_rename_<version>, clean up."""
    tarball = f"v{version}.tar.gz"
    url = _URL.format(tarball=tarball)
    try:
        download_and_verify(url, sha256, dest_dir=_WORK_DIR)
    # download() raises OSError (network/file errors); a bad hash raises ChecksumError.
    except (OSError, ChecksumError) as exc:
        log_error(_OP, f"rdma-core download/verify failed: {exc}", url=url)
        return False

    src_dir = os.path.join(_WORK_DIR, _SRC_DIR_NAME)
    if not _unpack(_WORK_DIR, src_dir, tarball, env):
        return False

    if not _run(["bash", "build.sh"], env, cwd=src_dir):
        return False
    built = os.path.join(src_dir, "build", "bin", "rdma_rename")
    log_info(_OP, "Built rdma-core", rdma_core_version=version, binary=built)

    target = rdma_rename_path(version)
    try:
        # shutil.copy copies the content and the permission bits (see docstring).
        shutil.copy(built, target)
        target_mode = _mode(target)
    except OSError as exc:
        log_error(_OP, f"could not copy {built} to {target}: {exc}",
                  source=built, path=target)
        return False
    log_info(_OP, f"Installed {target}", path=target, mode=target_mode,
             rdma_core_version=version)

    try:
        # bash: rm -rf rdma-core. The source tree is only needed for the build.
        shutil.rmtree(src_dir)
    except OSError as exc:
        log_error(_OP, f"could not remove {src_dir}: {exc}", path=src_dir)
        return False
    log_info(_OP, f"Removed build folder {src_dir}; {tarball} is left in "
                  f"{_WORK_DIR} (same as bash)", path=src_dir)
    return True


def install(env: dict[str, str]) -> int:
    """Install rdma_rename and the persistent RDMA naming services. Returns 0 or 3."""
    cfg = config_for("rdma_core", env)
    if not cfg or not cfg.get("version") or not cfg.get("sha256"):
        log_error(_OP, "could not resolve rdma_core version and sha256 from versions.json",
                  distro=env.get("DISTRIBUTION", ""), arch=env.get("ARCHITECTURE", ""))
        return 3
    version = cfg["version"]

    log_info(_OP, "Installing persistent RDMA naming",
             rdma_core_version=version,
             source=_URL.format(tarball=f"v{version}.tar.gz"),
             distro=env.get("DISTRIBUTION", ""), arch=env.get("ARCHITECTURE", ""))

    # ---- 1. rdma_rename_<version> -----------------------------------------
    if not _build_rdma_rename(version, cfg["sha256"], env):
        return 3

    # ---- 2. The naming script, its service and the retry timer -------------
    if not _write_file(_NAMING_SCRIPT, naming_script(version), mode=0o755):
        return 3
    if not _write_file(_SERVICE_FILE, _SERVICE_UNIT):
        return 3
    if not _write_file(_TIMER_FILE, _TIMER_UNIT):
        return 3

    # ---- 3. Load the new units, enable both, start both (bash's order) -----
    for command in (["systemctl", "daemon-reload"],
                    ["systemctl", "enable", _SERVICE_NAME],
                    ["systemctl", "enable", _TIMER_NAME],
                    ["systemctl", "start", _SERVICE_NAME],
                    ["systemctl", "start", _TIMER_NAME]):
        if not _run(command, env):
            return 3
    log_info(_OP, f"Enabled and started {_SERVICE_NAME} and {_TIMER_NAME}",
             service=_SERVICE_NAME, timer=_TIMER_NAME)

    log_info(_OP, "Persistent RDMA naming installed",
             rdma_core_version=version, rdma_rename=rdma_rename_path(version),
             naming_script=_NAMING_SCRIPT, service=_SERVICE_NAME, timer=_TIMER_NAME)
    return 0
