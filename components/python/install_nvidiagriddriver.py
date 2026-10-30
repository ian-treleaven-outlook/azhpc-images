"""install_nvidiagriddriver.py — Python port of components/install_nvidiagriddriver.sh.

What this step does
-------------------
Builds the GPU stack of an NCv6 image (NVIDIA vGPU, no InfiniBand). Unlike the
other NVIDIA SKUs (install_nvidiagpudriver) it installs Microsoft's GRID
driver instead of NVIDIA's Tesla driver, and has no nvidia-peermem, GDRCopy or
Fabric Manager. install() reads as the list of steps:

  1. Download and verify the GRID driver installer ("nvidia_grid" url/sha256
     in versions.json), run it (open kernel modules, DKMS) and record its
     version as NVIDIA_GRID.
  2. GRID licensing, per Azure's documentation: copy gridd.conf.template to
     gridd.conf, make sure IgnoreSP=FALSE and EnableUI=FALSE are set, and
     delete FeatureType=0 lines.
  3. Install cuda-toolkit-<cuda driver.version> from NVIDIA's CUDA apt repo.
     The bootstrap (install_cuda_repo_and_dkms.sh, upstream #643) adds that
     repo; we warn if no CUDA apt source is configured (bash doesn't check).
  4. Write /etc/profile.d/cuda.sh (mode 644): /usr/local/cuda/bin on PATH and
     /usr/local/cuda/lib64 on LD_LIBRARY_PATH for login shells.
  5. Record the CUDA version nvcc reports (via /etc/profile) as CUDA.
  6. Build the CUDA samples (install_cuda_samples.install, the port of
     install_cuda_samples.sh).
  7. Configure nvidia-persistenced (configure_nvidia_persistence.sh, bash).

Every command runs from _WORK_DIR, as bash runs everything from its cwd.

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets install() as the `action=` of the "install-nv-grid"
Step (NVIDIA NCv6 only). Deleting `action=` runs the bash script again, which
is why the .sh file is kept. install() returns 0 or 3 and never lets an
exception escape.

Behaviour kept from bash
------------------------
  * Commands, arguments and order are the same, and the first failure stops
    the step (`set -e`). The installer runs by its fixed name,
    NVIDIA-Linux-x86_64-<version>-grid-azure.run, not by the downloaded
    file's name (a mismatch is warned about; the run then fails).
  * The gridd.conf edits behave like
        grep -q '^Key=' f && sed -i 's/^Key=.*/Key=FALSE/' f || echo 'Key=FALSE' >> f
        sed -i '/^FeatureType=0/d' f
    line by line (the pure helpers below, applied with edit_file, a
    byte-preserving sed -i), including a file with no final newline.
    Checked against GNU grep/sed in WSL.
  * The CUDA version is parsed like `grep release | awk '{print $6}' |
    cut -c2-` plus the unquoted ${cuda_version} (first word recorded).
  * cuda.sh gets exactly the bytes bash's two `echo | tee` lines write (a
    test compares it with the .sh).
  * The toolkit package name, cuda.sh text, nvcc parser and CUDA apt source
    check are shared with install_nvidiagpudriver (utils/cuda.py).
  * Nothing checks the distro, like bash. Bash's apt only works on Ubuntu,
    so elsewhere we warn, carry on, and fail at the toolkit install, where
    bash's apt fails (without running install_packages(), which would pick
    that distro's dnf).

Deliberate differences from bash
--------------------------------
  * The toolkit install goes through install_packages(): `apt install -y`
    becomes `apt-get install -y -o Dpkg::Use-Pty=0`.
  * Missing GRID url/sha256/version or CUDA driver.version fails up front,
    before anything is installed. Bash would download "null" or
    `apt install cuda-toolkit-null` after installing the GRID driver.
  * An empty CUDA version (nvcc missing or unexpected output) or a failing
    `source /etc/profile; nvcc --version` fails the step; bash, without
    pipefail, records CUDA="".
  * The samples are built by install_cuda_samples.install(env), in its own
    work dir.
  * Work directory: bash works in the distro build dir and finally deletes
    *.run, *.tar.gz, *.rpm and EVERY subdirectory there (`rm -rf -- */`),
    whoever created them. We work in _WORK_DIR, emptied first, and remove
    only that, always (bash leaves its files behind when it fails). A
    warning records the bash behaviour.
  * cuda.sh is written in one go rather than by tee then tee -a; same file.
  * `cp gridd.conf.template gridd.conf` is copy_file (shutil.copy): the
    copy gets the template's permission bits, also when gridd.conf already
    existed (cp would keep that file's mode).

Bash quirks, kept and warned about
----------------------------------
  * `A && B || C` also runs the echo when the key IS present but sed fails
    (or grep can't read the file). We do the same: if edit_file can't read
    or rewrite the file (it logs an error), we warn and append "Key=FALSE"
    anyway, leaving the old line in place too. Only a failing append fails
    the step, as under `set -e`.
  * `echo >>` onto a file without a final newline glues "Key=FALSE" onto its
    last line; a later rewrite of that glued line can then drop it.
"""

from __future__ import annotations

import os
import re

from components.python import install_cuda_samples
from utils.component_config import record_version, require_config
from utils.cuda import (APT_SOURCES_DIR, CUDA_PROFILE, CUDA_PROFILE_TEXT,
                        NVCC_VERSION_COMMAND, check_cuda_apt_source, cuda_toolkit_package,
                        nvcc_cuda_version)
from utils.distro import Distro
from utils.download import fetch_verified
from utils.files import copy_file, edit_file, fresh_dir, remove_paths, write_file
from utils.logger import log_error, log_info, log_warn
from utils.package_installer import install_packages
from utils.process import run, run_capture

# The Step's `op` label in plans/steps.py.
_OP = "install-nv-grid"

# ---- Paths (module-level so tests can point them at a temporary directory) --
_WORK_DIR = "/tmp/azhpc-nvidia-grid"
_GRIDD_TEMPLATE = "/etc/nvidia/gridd.conf.template"
_GRIDD_CONF = "/etc/nvidia/gridd.conf"
_CUDA_PROFILE = CUDA_PROFILE
_APT_SOURCES_DIR = APT_SOURCES_DIR

_PERSISTENCE_SCRIPT = "configure_nvidia_persistence.sh"

# GRID licensing keys that must read FALSE (in bash's order), and the prefix
# of the lines that are deleted.
_GRIDD_FALSE_KEYS = ("IgnoreSP", "EnableUI")
_GRIDD_DELETE_PREFIX = "FeatureType=0"

_DRIVER_INSTALLER_ARGS = ["--silent", "--dkms", "--kernel-module-type=open"]


# --------------------------------------------------------------------------- #
# Pure helpers: they only transform values, so tests call them directly.
# --------------------------------------------------------------------------- #
def driver_installer_name(version: str) -> str:
    """The installer file name bash runs: NVIDIA-Linux-x86_64-<V>-grid-azure.run."""
    return f"NVIDIA-Linux-x86_64-{version}-grid-azure.run"


def split_lines(text: str) -> list[str]:
    """Lines as grep/sed see them, each keeping its "\\n" (only "\\n" ends a
    line, unlike str.splitlines()); the last may have none."""
    return re.findall(r"[^\n]*\n|[^\n]+$", text)


def set_key(text: str, key: str, value: str) -> tuple[str, int]:
    """sed 's/^<key>=.*/<key>=<value>/': (new text, lines changed).

    `.*` runs to the end of the line, so the whole line is replaced (a
    trailing "\\r" too); the line's newline, or lack of one, is kept.
    0 lines changed is bash's `grep -q '^<key>='` finding nothing.
    """
    changed = 0
    out = []
    for line in split_lines(text):
        if line.startswith(f"{key}="):
            line = f"{key}={value}" + ("\n" if line.endswith("\n") else "")
            changed += 1
        out.append(line)
    return "".join(out), changed


def delete_lines(text: str, prefix: str) -> tuple[str, int]:
    """sed '/^<prefix>/d': (new text, lines deleted)."""
    lines = split_lines(text)
    kept = [line for line in lines if not line.startswith(prefix)]
    return "".join(kept), len(lines) - len(kept)


# --------------------------------------------------------------------------- #
# 1. GRID driver
# --------------------------------------------------------------------------- #
def _install_grid_driver(env: dict[str, str], grid: dict[str, str]) -> bool:
    downloaded = fetch_verified(grid["url"], grid["sha256"], _OP, _WORK_DIR)
    if downloaded is None:
        return False
    installer = driver_installer_name(grid["version"])
    if downloaded.name != installer:
        # BASH QUIRK (kept): bash runs the installer by this fixed name, so a
        # URL with a different file name makes the next command fail.
        log_warn(_OP, f"the downloaded file is {downloaded.name}, but bash runs "
                      f"{installer}; running that name like bash",
                 downloaded=str(downloaded), installer=installer)
    log_info(_OP, f"Running the GRID driver installer {installer} "
                  "(open kernel modules, DKMS)", installer=installer,
             args=" ".join(_DRIVER_INSTALLER_ARGS))
    return (run(["bash", installer, *_DRIVER_INSTALLER_ARGS], _OP, cwd=_WORK_DIR, env=env)
            and record_version("NVIDIA_GRID", grid["version"], _OP))


# --------------------------------------------------------------------------- #
# 2. GRID licensing (gridd.conf)
# --------------------------------------------------------------------------- #
def _lacks_final_newline(path: str) -> bool:
    """True if `path` is a non-empty file whose last byte isn't a newline."""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except FileNotFoundError:
        return False                     # `>>` creates the file
    return bool(data) and not data.endswith(b"\n")


def _append_key(key: str, reason: str) -> bool:
    """echo '<key>=FALSE' >> gridd.conf"""
    log_info(_OP, f"Appending {key}=FALSE to {_GRIDD_CONF} ({reason})",
             path=_GRIDD_CONF, key=key, value="FALSE")
    try:
        unterminated = _lacks_final_newline(_GRIDD_CONF)
    except OSError as exc:
        log_error(_OP, f"could not append {key}=FALSE to {_GRIDD_CONF}: {exc}",
                  path=_GRIDD_CONF, key=key)
        return False
    if unterminated:
        # BASH QUIRK (kept): echo >> doesn't add a newline first.
        log_warn(_OP, f"{_GRIDD_CONF} does not end with a newline: {key}=FALSE "
                      "is glued onto its last line, as bash's echo >> does, so "
                      "it may not take effect", path=_GRIDD_CONF, key=key)
    return write_file(_GRIDD_CONF, f"{key}=FALSE\n", _OP, append=True)


def _ensure_key_false(key: str) -> bool:
    """grep -q '^Key=' f && sed -i 's/^Key=.*/Key=FALSE/' f || echo 'Key=FALSE' >> f"""
    replaced = []

    def rewrite(text: str) -> str:
        new_text, changed = set_key(text, key, "FALSE")
        replaced.append(changed)
        return new_text

    if not edit_file(_GRIDD_CONF, rewrite, _OP):
        # BASH QUIRK (kept): in `A && B || C`, C also runs when grep can't
        # read the file or sed fails (edit_file has logged which).
        log_warn(_OP, f"appending {key}=FALSE to {_GRIDD_CONF} anyway, as bash's "
                      "`grep && sed || echo` does (an existing line is kept)",
                 path=_GRIDD_CONF, key=key)
        return _append_key(key, "reading or rewriting the file failed")
    if not replaced[0]:
        return _append_key(key, f"no {key}= line")
    log_info(_OP, f"Set {key}=FALSE in {_GRIDD_CONF} ({replaced[0]} line(s) replaced)",
             path=_GRIDD_CONF, key=key, value="FALSE", lines=replaced[0])
    return True


def _delete_feature_type_lines() -> bool:
    """sed -i '/^FeatureType=0/d' gridd.conf"""
    deleted = []

    def drop(text: str) -> str:
        new_text, count = delete_lines(text, _GRIDD_DELETE_PREFIX)
        deleted.append(count)
        return new_text

    if not edit_file(_GRIDD_CONF, drop, _OP):
        return False
    log_info(_OP, f"Removed {deleted[0]} line(s) starting with {_GRIDD_DELETE_PREFIX} "
                  f"from {_GRIDD_CONF}", path=_GRIDD_CONF, lines=deleted[0])
    return True


def _configure_grid_licensing() -> bool:
    log_info(_OP, f"Configuring GRID licensing: {_GRIDD_CONF} from {_GRIDD_TEMPLATE}",
             path=_GRIDD_CONF, template=_GRIDD_TEMPLATE)
    if not copy_file(_GRIDD_TEMPLATE, _GRIDD_CONF, _OP):
        return False
    return (all(_ensure_key_false(key) for key in _GRIDD_FALSE_KEYS)
            and _delete_feature_type_lines())


# --------------------------------------------------------------------------- #
# 3-7. CUDA toolkit, cuda.sh, CUDA version, samples, persistence
# --------------------------------------------------------------------------- #
def _install_cuda_toolkit(env: dict[str, str], distro: Distro, toolkit: str) -> bool:
    if not distro.is_ubuntu:
        # install_packages() would pick dnf here; bash runs apt and fails.
        log_error(_OP, f"cannot install {toolkit}: install_nvidiagriddriver.sh "
                       f"installs it with apt, which {distro.name or '(unset)'} "
                       "doesn't have; failing here, where bash's apt fails",
                  package=toolkit, distro=distro.name)
        return False
    check_cuda_apt_source(_OP, _APT_SOURCES_DIR)
    log_info(_OP, f"Installing {toolkit} from the NVIDIA CUDA apt repository",
             package=toolkit)
    return install_packages([toolkit], env, _OP, cwd=_WORK_DIR)


def _record_cuda_version(env: dict[str, str]) -> str | None:
    """Ask nvcc, through /etc/profile, for the CUDA version, and record it."""
    rc, output = run_capture(list(NVCC_VERSION_COMMAND), _OP, cwd=_WORK_DIR, env=env)
    version = nvcc_cuda_version(output) if rc == 0 else None
    if version is None:
        log_error(_OP, "could not read the CUDA version from `nvcc --version`: "
                       f"{output.strip()!r}", exit_code=rc)
        return None
    return version if record_version("CUDA", version, _OP) else None


def _install_cuda_samples(env: dict[str, str]) -> bool:
    log_info(_OP, "Installing the CUDA samples (install_cuda_samples)")
    try:
        samples_rc = install_cuda_samples.install(env)
    except Exception as exc:     # its own contract says it shouldn't raise
        log_error(_OP, f"CUDA samples installation raised: {exc}")
        return False
    if samples_rc != 0:
        log_error(_OP, "CUDA samples installation failed", exit_code=samples_rc)
        return False
    return True


def _configure_persistence(env: dict[str, str]) -> bool:
    script = os.path.join(env.get("COMPONENT_DIR", ""), _PERSISTENCE_SCRIPT)
    log_info(_OP, "Configuring the nvidia-persistenced service", script=script)
    return run([script], _OP, cwd=_WORK_DIR, env=env)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _resolve_config(env: dict[str, str]) -> tuple[dict[str, str], str] | None:
    """(GRID url/sha256/version, CUDA driver version), or None (logged)."""
    grid = require_config("nvidia_grid", env, _OP, "url", "sha256", "version")
    if grid is None:
        return None
    cuda = require_config("cuda", env, _OP, "driver.version")
    if cuda is None:
        return None
    # str() handles a value written as a JSON number.
    return ({key: str(grid[key]) for key in ("url", "sha256", "version")},
            str(cuda["driver"]["version"]))


def _install_steps(env: dict[str, str], distro: Distro, grid: dict[str, str],
                   toolkit: str) -> str | None:
    """Steps 1-7 in the work dir; the recorded CUDA version, or None."""
    if not (_install_grid_driver(env, grid)
            and _configure_grid_licensing()
            and _install_cuda_toolkit(env, distro, toolkit)
            and write_file(_CUDA_PROFILE, CUDA_PROFILE_TEXT, _OP, mode=0o644)):
        return None
    cuda_version = _record_cuda_version(env)
    if cuda_version is None or not (_install_cuda_samples(env)
                                    and _configure_persistence(env)):
        return None
    return cuda_version


def install(env: dict[str, str]) -> int:
    """Install the GRID driver, CUDA toolkit, samples and persistence config.

    Returns 0 on success, 3 on failure.
    """
    distro = Distro.from_env(env)
    resolved = _resolve_config(env)
    if resolved is None:
        return 3
    grid, cuda_driver_version = resolved
    toolkit = cuda_toolkit_package(cuda_driver_version)
    log_info(_OP, f"Installing NVIDIA GRID driver {grid['version']} and {toolkit}",
             grid_version=grid["version"], source=grid["url"],
             cuda_toolkit_version=cuda_driver_version, distro=distro.name,
             sku=env.get("SKU", ""), node_type=env.get("TARGET_NODE_TYPE", ""),
             work_dir=_WORK_DIR)
    if not distro.is_ubuntu:
        log_warn(_OP, "install_nvidiagriddriver.sh assumes Ubuntu (apt) but "
                      "doesn't check the distro; continuing like bash, so the "
                      "step fails at the CUDA toolkit install", distro=distro.name)

    if not fresh_dir(_WORK_DIR, _OP):
        return 3
    try:
        cuda_version = _install_steps(env, distro, grid, toolkit)
    finally:
        remove_paths(_WORK_DIR, op=_OP)
    if cuda_version is None:
        return 3

    log_warn(_OP, "bash would now delete *.run, *.tar.gz, *.rpm and every "
                  "subdirectory of the distro build dir (rm -rf -- */); this port "
                  f"only removed its own work dir {_WORK_DIR}", work_dir=_WORK_DIR)
    log_info(_OP, "NVIDIA GRID driver installed", grid_version=grid["version"],
             cuda_toolkit=toolkit, cuda_version=cuda_version,
             gridd_conf=_GRIDD_CONF, cuda_profile=_CUDA_PROFILE, distro=distro.name)
    return 0