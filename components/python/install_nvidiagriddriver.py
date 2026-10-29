"""install_nvidiagriddriver.py — Python port of components/install_nvidiagriddriver.sh.

What this step does
-------------------
Builds the GPU stack of an NCv6 image (NVIDIA vGPU, no InfiniBand). Unlike the
other NVIDIA SKUs (install_nvidiagpudriver.sh) it installs Microsoft's GRID
driver instead of NVIDIA's Tesla driver, and has no nvidia-peermem, GDRCopy or
Fabric Manager. In order:

  1. Download the GRID driver installer ("nvidia_grid" in versions.json),
     verify its SHA256, run it (open kernel modules, DKMS) and record its
     version as NVIDIA_GRID.
  2. Configure GRID licensing, per Azure's documentation: copy
     /etc/nvidia/gridd.conf.template to /etc/nvidia/gridd.conf, make sure
     IgnoreSP=FALSE and EnableUI=FALSE are set, and delete FeatureType=0 lines.
  3. Add NVIDIA's CUDA apt repository (cuda-keyring) and install the CUDA
     toolkit whose version "cuda" -> driver.version gives (NCv6 has its own
     entry in versions.json; config_for resolves it).
  4. Write /etc/profile.d/cuda.sh, which puts /usr/local/cuda/bin on PATH and
     /usr/local/cuda/lib64 on LD_LIBRARY_PATH for login shells.
  5. Ask nvcc (via /etc/profile) for the installed CUDA version and record it
     as CUDA.
  6. Build and install the CUDA samples (install_cuda_samples.install).
  7. Configure the nvidia-persistenced service
     (components/configure_nvidia_persistence.sh, still bash).

The original bash, for comparison (comments trimmed):

    grid_metadata=$(get_component_config "nvidia_grid")
    GRID_DRIVER_URL=$(jq -r '.url' <<< $grid_metadata)
    GRID_DRIVER_SHA256=$(jq -r '.sha256' <<< $grid_metadata)
    GRID_DRIVER_VERSION=$(jq -r '.version' <<< $grid_metadata)

    download_and_verify $GRID_DRIVER_URL $GRID_DRIVER_SHA256
    bash NVIDIA-Linux-x86_64-${GRID_DRIVER_VERSION}-grid-azure.run --silent --dkms --kernel-module-type=open
    write_component_version "NVIDIA_GRID" ${GRID_DRIVER_VERSION}

    cp /etc/nvidia/gridd.conf.template /etc/nvidia/gridd.conf
    grep -q '^IgnoreSP=' /etc/nvidia/gridd.conf && sed -i 's/^IgnoreSP=.*/IgnoreSP=FALSE/' /etc/nvidia/gridd.conf || echo 'IgnoreSP=FALSE' >> /etc/nvidia/gridd.conf
    grep -q '^EnableUI=' /etc/nvidia/gridd.conf && sed -i 's/^EnableUI=.*/EnableUI=FALSE/' /etc/nvidia/gridd.conf || echo 'EnableUI=FALSE' >> /etc/nvidia/gridd.conf
    sed -i '/^FeatureType=0/d' /etc/nvidia/gridd.conf

    cuda_metadata=$(get_component_config "cuda")
    CUDA_DRIVER_VERSION=$(jq -r '.driver.version' <<< $cuda_metadata)
    CUDA_DRIVER_DISTRIBUTION=$(jq -r '.driver.distribution' <<< $cuda_metadata)
    wget https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DRIVER_DISTRIBUTION}/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i ./cuda-keyring_1.1-1_all.deb
    apt-get update
    apt install -y cuda-toolkit-${CUDA_DRIVER_VERSION//./-}

    echo 'export PATH="${PATH:+$PATH:}/usr/local/cuda/bin"' | tee /etc/profile.d/cuda.sh > /dev/null
    echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/cuda/lib64"' | tee -a /etc/profile.d/cuda.sh > /dev/null
    chmod 644 /etc/profile.d/cuda.sh

    cuda_version=$(source /etc/profile; nvcc --version | grep release | awk '{print $6}' | cut -c2-)
    write_component_version "CUDA" ${cuda_version}

    $COMPONENT_DIR/install_cuda_samples.sh
    $COMPONENT_DIR/configure_nvidia_persistence.sh

    rm -rf *.run *.tar.gz *.rpm
    (
        shopt -s dotglob nullglob
        rm -rf -- */ || true
    )

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-nv-grid" Step (NVIDIA NCv6 only). Deleting `action=` from that Step
makes the build run the bash script again, which is why the .sh file is kept.

install() receives the build environment (a dict) and returns 0 for success or
3 for failure. It logs problems and never lets an exception escape.

Behaviour kept from bash
------------------------
  * Commands, arguments and order are the same, and the first failing command
    stops the step (bash's `set -e`). The driver installer is run by its
    fixed name, NVIDIA-Linux-x86_64-<version>-grid-azure.run, not by the name
    of the downloaded file (we warn when the two differ: the run then fails).
  * The gridd.conf edits behave like the grep/sed/echo lines: every line that
    starts with "Key=" becomes "Key=FALSE" (sed's s/^Key=.*/ is per line), or
    "Key=FALSE" plus a newline is appended when there is no such line (echo
    >>, glued onto a last line that has no newline, with a warning; a later
    rewrite of that glued line can then drop it, exactly as in bash), and every
    line starting with "FeatureType=0" is deleted (FeatureType=01 too). A file
    without a final newline keeps it that way. Checked against real GNU
    grep/sed in WSL on representative input.
  * The CUDA version is parsed like `grep release | awk '{print $6}' |
    cut -c2-` followed by the unquoted ${cuda_version} (first word recorded);
    checked against the real tools in WSL.
  * /etc/profile.d/cuda.sh gets exactly the bytes the two echo | tee lines
    write (checked against bash in WSL; a test also compares it with the .sh).
  * Nothing checks the distro, like bash. The dpkg/apt commands only work on
    Ubuntu, so on any other distro we log a warning and carry on; the dpkg or
    apt command then fails the step, as it would fail the bash script.

Deliberate differences from bash
--------------------------------
  * Missing config fails up front, before anything is installed: the GRID
    url, sha256 and version, and the CUDA driver.version and
    driver.distribution. Bash would download "null", run a missing installer,
    or wget .../repos/null/... after the GRID driver was already installed.
  * An empty CUDA version (nvcc missing or unexpected output) or a failing
    `source /etc/profile; nvcc --version` fails the step. Bash, without
    pipefail and inside $(...), would record CUDA="".
  * Bash runs install_cuda_samples.sh; we call install_cuda_samples.install(env),
    its Python port (which downloads and builds in its own work dir).
  * Work directory: bash downloads and runs everything in its current
    directory (the distro build dir), then deletes *.run, *.tar.gz, *.rpm and
    EVERY subdirectory there (`rm -rf -- */`), whoever created them. We work
    in _WORK_DIR, emptied first, and remove only that, always (bash leaves its
    files behind when it fails). A warning records the bash behaviour.
  * /etc/profile.d/cuda.sh is written in one go rather than by tee then
    tee -a; the resulting file is the same.

Bash quirk, mirrored
--------------------
`grep -q ... && sed -i ... || echo ... >> gridd.conf` also runs the echo when
the key IS present but sed fails (and when grep can't read the file). We do
the same: if rewriting the file fails, we warn and append "Key=FALSE" anyway,
which leaves the old Key= line in place too. Only a failing append fails the
step, as in bash under `set -e`.
"""

# Lets type hints such as `str | None` work on older Python versions.
from __future__ import annotations

import os
import re
import shutil

from components.python import install_cuda_samples
from utils.component_config import config_for, write_component_version
from utils.download import download_and_verify
from utils.logger import log_debug, log_error, log_info, log_warn
from utils.process import exec_program, run_capture

# The Step's `op` label in build_config.py; used on every log line.
_OP = "install-nv-grid"

# ---- Paths (module-level so tests can point them at a temporary directory) --
# Scratch area for the driver installer and the cuda-keyring package.
_WORK_DIR = "/tmp/azhpc-nvidia-grid"
_GRIDD_TEMPLATE = "/etc/nvidia/gridd.conf.template"
_GRIDD_CONF = "/etc/nvidia/gridd.conf"
_CUDA_PROFILE = "/etc/profile.d/cuda.sh"

# The nested component script that configures nvidia-persistenced.
_PERSISTENCE_SCRIPT = "configure_nvidia_persistence.sh"

# GRID licensing settings that must read FALSE, in bash's order, and the
# prefix of the lines that are deleted.
_GRIDD_FALSE_KEYS = ("IgnoreSP", "EnableUI")
_GRIDD_DELETE_PREFIX = "FeatureType=0"

_DRIVER_INSTALLER_ARGS = ["--silent", "--dkms", "--kernel-module-type=open"]
_CUDA_KEYRING = "cuda-keyring_1.1-1_all.deb"

# The shell part of bash's  cuda_version=$(source /etc/profile; nvcc --version | ...).
# It runs in its own bash so /etc/profile (and so /etc/profile.d/cuda.sh) can
# put nvcc on PATH; the grep/awk/cut part is nvcc_release_version() below.
_NVCC_VERSION_SHELL = "source /etc/profile; nvcc --version"

# /etc/profile.d/cuda.sh, byte for byte: each `echo '...'` writes its
# single-quoted text plus a newline. The ${...} is literal text here; the
# shell expands it when a login shell sources the file. (Rendered with real
# bash in WSL and compared; a test also checks it against the .sh file.)
CUDA_PROFILE_CONTENT = (
    'export PATH="${PATH:+$PATH:}/usr/local/cuda/bin"\n'
    'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/cuda/lib64"\n'
)

# What awk's default field splitting and bash word splitting treat as
# separators: spaces, tabs and newlines.
_BLANKS = re.compile(r"[ \t\n]+")

# Read/write gridd.conf without changing any byte we don't edit:
# newline="" keeps "\r\n" as is, and surrogateescape round-trips any byte that
# isn't valid UTF-8.
_TEXT_IO = {"encoding": "utf-8", "errors": "surrogateescape", "newline": ""}


# --------------------------------------------------------------------------- #
# Pure helpers: they only transform values, so tests call them directly.
# --------------------------------------------------------------------------- #
def driver_installer_name(version: str) -> str:
    """The installer file name bash runs: NVIDIA-Linux-x86_64-<V>-grid-azure.run."""
    return f"NVIDIA-Linux-x86_64-{version}-grid-azure.run"


def cuda_keyring_url(repo_distribution: str) -> str:
    """The cuda-keyring .deb URL for a CUDA repo distribution such as ubuntu2404."""
    return ("https://developer.download.nvidia.com/compute/cuda/repos/"
            f"{repo_distribution}/x86_64/{_CUDA_KEYRING}")


def cuda_toolkit_package(version: str) -> str:
    """cuda-toolkit-${CUDA_DRIVER_VERSION//./-}: "13.2" -> "cuda-toolkit-13-2".

    ${var//./-} replaces EVERY "." with "-", like str.replace().
    """
    return "cuda-toolkit-" + version.replace(".", "-")


def split_lines(text: str) -> list[str]:
    """Split text into lines the way grep/sed see them, keeping each "\\n".

    Only "\\n" ends a line (str.splitlines() would also split on "\\r" and a
    few other characters). The last line may have no "\\n".
    """
    # [^\n]*\n is a complete line; [^\n]+$ is a last line without a newline.
    return re.findall(r"[^\n]*\n|[^\n]+$", text)


def has_key(text: str, key: str) -> bool:
    """grep -q '^<key>=': does any line start with "<key>="?"""
    return any(line.startswith(f"{key}=") for line in split_lines(text))


def set_key(text: str, key: str, value: str) -> tuple[str, int]:
    """sed 's/^<key>=.*/<key>=<value>/': rewrite every "<key>=..." line.

    `.*` runs to the end of the line, so the whole line is replaced (a
    trailing "\\r" too); the line's newline, or lack of one, is kept.
    Returns (new text, number of lines changed).
    """
    changed = 0
    out = []
    for line in split_lines(text):
        if line.startswith(f"{key}="):
            ending = "\n" if line.endswith("\n") else ""
            line = f"{key}={value}{ending}"
            changed += 1
        out.append(line)
    return "".join(out), changed


def delete_lines(text: str, prefix: str) -> tuple[str, int]:
    """sed '/^<prefix>/d': drop every line that starts with `prefix`.

    Returns (new text, number of lines deleted).
    """
    kept = [line for line in split_lines(text) if not line.startswith(prefix)]
    return "".join(kept), len(split_lines(text)) - len(kept)


def nvcc_release_version(output: str) -> str | None:
    """The CUDA version from `nvcc --version` output, or None.

    Mirrors  nvcc --version | grep release | awk '{print $6}' | cut -c2-
    followed by the unquoted ${cuda_version} (only the first word is
    recorded):

        "Cuda compilation tools, release 13.2, V13.2.51"
            -> $6 "V13.2.51" -> cut -c2- -> "13.2.51"

    A matching line with fewer than 6 fields gives an empty value, which
    word splitting then drops.
    """
    values = []
    for line in split_lines(output):
        line = line.rstrip("\n")
        if "release" not in line:
            continue
        fields = [field for field in _BLANKS.split(line) if field]
        sixth = fields[5] if len(fields) >= 6 else ""
        values.append(sixth[1:])          # cut -c2-: from the 2nd character
    words = [word for word in _BLANKS.split("\n".join(values)) if word]
    return words[0] if words else None


# --------------------------------------------------------------------------- #
# Helpers with side effects.
# --------------------------------------------------------------------------- #
def _run_all(commands: list[list[str]], env: dict[str, str]) -> bool:
    """Run commands in _WORK_DIR, stopping at the first failure (like `set -e`)."""
    for command in commands:
        if exec_program(command, _OP, cwd=_WORK_DIR, env=env) != 0:
            return False
    return True


def _record(component: str, version: str) -> bool:
    """Write one entry to the component-versions file; False (logged) on error."""
    try:
        write_component_version(component, version)
    except OSError as exc:
        log_error(_OP, f"could not record {component}={version}: {exc}",
                  component=component, version=version)
        return False
    log_info(_OP, f"Recorded {component} version {version}",
             component=component, version=version)
    return True


def _resolve_config(env: dict[str, str]) -> dict[str, str] | None:
    """The GRID and CUDA values bash reads with jq, or None (logged) if any is missing."""
    grid = config_for("nvidia_grid", env)
    grid = grid if isinstance(grid, dict) else {}
    cuda = config_for("cuda", env)
    driver = cuda.get("driver") if isinstance(cuda, dict) else None
    driver = driver if isinstance(driver, dict) else {}

    values = {
        "grid_url": grid.get("url"),
        "grid_sha256": grid.get("sha256"),
        "grid_version": grid.get("version"),
        "cuda_version": driver.get("version"),
        "cuda_repo": driver.get("distribution"),
    }
    names = {"grid_url": "nvidia_grid.url", "grid_sha256": "nvidia_grid.sha256",
             "grid_version": "nvidia_grid.version",
             "cuda_version": "cuda.driver.version",
             "cuda_repo": "cuda.driver.distribution"}
    missing = [names[key] for key, value in values.items() if not value]
    if missing:
        log_error(_OP, f"could not resolve {', '.join(missing)} from versions.json",
                  distro=env.get("DISTRIBUTION", ""), sku=env.get("SKU", ""),
                  missing=missing)
        return None
    # str() handles a value written as a JSON number.
    return {key: str(value) for key, value in values.items()}


def _lacks_final_newline(path: str) -> bool:
    """True if `path` is a non-empty file whose last byte isn't a newline."""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except FileNotFoundError:
        return False                     # `>>` creates the file
    return bool(data) and not data.endswith(b"\n")


def _append_key(key: str, reason: str) -> bool:
    """echo '<key>=FALSE' >> gridd.conf. False (logged) if that fails."""
    line = f"{key}=FALSE\n"
    try:
        unterminated = _lacks_final_newline(_GRIDD_CONF)
        with open(_GRIDD_CONF, "a", **_TEXT_IO) as handle:
            handle.write(line)
    except OSError as exc:
        log_error(_OP, f"could not append {key}=FALSE to {_GRIDD_CONF}: {exc}",
                  path=_GRIDD_CONF, key=key)
        return False
    if unterminated:
        # BASH QUIRK (kept): echo >> doesn't add a newline first, so the
        # setting is glued onto the file's unterminated last line.
        log_warn(_OP, f"{_GRIDD_CONF} did not end with a newline: {key}=FALSE "
                      "was glued onto its last line, as bash's echo >> does, "
                      "so it may not take effect", path=_GRIDD_CONF, key=key)
    log_info(_OP, f"Appended {key}=FALSE to {_GRIDD_CONF} ({reason})",
             path=_GRIDD_CONF, key=key, value="FALSE", action="appended")
    return True


def _ensure_key_false(key: str) -> bool:
    """grep -q '^Key=' f && sed -i 's/^Key=.*/Key=FALSE/' f || echo 'Key=FALSE' >> f"""
    try:
        with open(_GRIDD_CONF, **_TEXT_IO) as handle:
            text = handle.read()
    except OSError as exc:
        # grep can't read the file (exit 2), so bash's `||` runs the echo.
        log_warn(_OP, f"could not read {_GRIDD_CONF} ({exc}); appending "
                      f"{key}=FALSE anyway, as bash's `grep && sed || echo` does",
                 path=_GRIDD_CONF, key=key)
        return _append_key(key, "the file could not be read")

    if not has_key(text, key):
        return _append_key(key, f"no {key}= line")

    new_text, changed = set_key(text, key, "FALSE")
    try:
        with open(_GRIDD_CONF, "w", **_TEXT_IO) as handle:
            handle.write(new_text)
    except OSError as exc:
        # BASH QUIRK (kept): in `A && B || C`, C also runs when B fails, so a
        # failing sed is followed by the echo. The old line stays as well.
        log_warn(_OP, f"could not rewrite {key}= in {_GRIDD_CONF} ({exc}); "
                      f"appending {key}=FALSE instead, as bash's "
                      "`grep && sed || echo` does (the old line is kept)",
                 path=_GRIDD_CONF, key=key)
        return _append_key(key, "rewriting the existing line failed")
    log_info(_OP, f"Set {key}=FALSE in {_GRIDD_CONF} ({changed} line(s) replaced)",
             path=_GRIDD_CONF, key=key, value="FALSE", action="replaced",
             lines=changed)
    return True


def _configure_grid_licensing(env: dict[str, str]) -> bool:
    """cp the template, set the FALSE keys, delete FeatureType=0 lines."""
    log_info(_OP, f"Configuring GRID licensing: {_GRIDD_CONF} from {_GRIDD_TEMPLATE}",
             path=_GRIDD_CONF, template=_GRIDD_TEMPLATE)
    # cp itself (not shutil) so a new file gets cp's permissions.
    if not _run_all([["cp", _GRIDD_TEMPLATE, _GRIDD_CONF]], env):
        log_error(_OP, f"could not copy {_GRIDD_TEMPLATE} to {_GRIDD_CONF}",
                  path=_GRIDD_CONF)
        return False
    log_info(_OP, f"Created {_GRIDD_CONF} from the template (replaced if it existed)",
             path=_GRIDD_CONF, action="replaced")

    for key in _GRIDD_FALSE_KEYS:
        if not _ensure_key_false(key):
            return False

    # sed -i '/^FeatureType=0/d'
    try:
        with open(_GRIDD_CONF, **_TEXT_IO) as handle:
            text = handle.read()
        new_text, deleted = delete_lines(text, _GRIDD_DELETE_PREFIX)
        with open(_GRIDD_CONF, "w", **_TEXT_IO) as handle:
            handle.write(new_text)
    except OSError as exc:
        log_error(_OP, f"could not remove {_GRIDD_DELETE_PREFIX} lines from "
                       f"{_GRIDD_CONF}: {exc}", path=_GRIDD_CONF)
        return False
    log_info(_OP, f"Removed {deleted} line(s) starting with {_GRIDD_DELETE_PREFIX} "
                  f"from {_GRIDD_CONF}", path=_GRIDD_CONF, action="edited",
             lines=deleted)
    log_debug(_OP, f"{_GRIDD_CONF} now contains:\n{new_text}")
    return True


def _write_cuda_profile() -> bool:
    """The two echo | tee lines, then chmod 644. False (logged) on error."""
    existed = os.path.exists(_CUDA_PROFILE)
    try:
        with open(_CUDA_PROFILE, "w", encoding="utf-8") as handle:
            handle.write(CUDA_PROFILE_CONTENT)
        os.chmod(_CUDA_PROFILE, 0o644)
    except OSError as exc:
        log_error(_OP, f"could not write {_CUDA_PROFILE}: {exc}", path=_CUDA_PROFILE)
        return False
    log_info(_OP, f"{'Replaced' if existed else 'Created'} {_CUDA_PROFILE} (mode 644): "
                  "adds /usr/local/cuda/bin to PATH and /usr/local/cuda/lib64 "
                  "to LD_LIBRARY_PATH", path=_CUDA_PROFILE,
             action="replaced" if existed else "created", mode="644")
    return True


def _installed_cuda_version(env: dict[str, str]) -> str | None:
    """Ask nvcc, through /etc/profile, for the CUDA version. None (logged) on error."""
    rc, output = run_capture(["bash", "-c", _NVCC_VERSION_SHELL], _OP,
                             cwd=_WORK_DIR, env=env)
    version = nvcc_release_version(output) if rc == 0 else None
    if version is None:
        log_error(_OP, "could not read the CUDA version from `nvcc --version`: "
                       f"{output.strip()!r}", exit_code=rc)
    return version


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def install(env: dict[str, str]) -> int:
    """Install the GRID driver, CUDA toolkit, samples and persistence config.

    Returns 0 on success, 3 on failure.
    """
    distribution = env.get("DISTRIBUTION", "")
    values = _resolve_config(env)
    if values is None:
        return 3
    grid_version = values["grid_version"]
    toolkit = cuda_toolkit_package(values["cuda_version"])
    keyring_url = cuda_keyring_url(values["cuda_repo"])

    log_info(_OP, f"Installing NVIDIA GRID driver {grid_version} and {toolkit}",
             grid_version=grid_version, source=values["grid_url"],
             cuda_toolkit_version=values["cuda_version"],
             cuda_repo=values["cuda_repo"], distro=distribution,
             sku=env.get("SKU", ""), node_type=env.get("TARGET_NODE_TYPE", ""),
             work_dir=_WORK_DIR)
    if not distribution.startswith("ubuntu"):
        log_warn(_OP, "install_nvidiagriddriver.sh assumes Ubuntu (dpkg/apt) but "
                      "doesn't check the distro; continuing like bash, so the "
                      "dpkg/apt commands will fail", distro=distribution)

    try:
        # Start from an empty work directory (see the module docstring).
        shutil.rmtree(_WORK_DIR, ignore_errors=True)
        os.makedirs(_WORK_DIR, exist_ok=True)

        # ---- 1. GRID driver -------------------------------------------------
        try:
            downloaded = download_and_verify(values["grid_url"], values["grid_sha256"],
                                             dest_dir=_WORK_DIR)
        except Exception as exc:
            log_error(_OP, f"GRID driver download/verify failed: {exc}",
                      source=values["grid_url"])
            return 3
        installer = driver_installer_name(grid_version)
        if os.path.basename(str(downloaded)) != installer:
            # BASH QUIRK (kept): bash runs the installer by this fixed name, so
            # a URL with a different file name makes the next command fail.
            log_warn(_OP, f"the downloaded file is {os.path.basename(str(downloaded))}, "
                          f"but bash runs {installer}; running that name like bash",
                     downloaded=str(downloaded), installer=installer)
        log_info(_OP, f"Running the GRID driver installer {installer} "
                      "(open kernel modules, DKMS)", installer=installer,
                 args=" ".join(_DRIVER_INSTALLER_ARGS))
        if not _run_all([["bash", installer, *_DRIVER_INSTALLER_ARGS]], env):
            log_error(_OP, "GRID driver installation failed", installer=installer)
            return 3
        if not _record("NVIDIA_GRID", grid_version):
            return 3

        # ---- 2. GRID licensing ------------------------------------------------
        if not _configure_grid_licensing(env):
            return 3

        # ---- 3. CUDA toolkit ----------------------------------------------------
        log_info(_OP, f"Adding the NVIDIA CUDA apt repository ({values['cuda_repo']}) "
                      f"and installing {toolkit}", source=keyring_url,
                 package=toolkit, cuda_repo=values["cuda_repo"])
        if not _run_all([["wget", keyring_url],
                         ["dpkg", "-i", f"./{_CUDA_KEYRING}"],
                         ["apt-get", "update"],
                         ["apt", "install", "-y", toolkit]], env):
            log_error(_OP, f"CUDA toolkit installation failed ({toolkit})",
                      package=toolkit, distro=distribution)
            return 3

        # ---- 4. /etc/profile.d/cuda.sh -------------------------------------------
        if not _write_cuda_profile():
            return 3

        # ---- 5. CUDA version ------------------------------------------------------
        cuda_version = _installed_cuda_version(env)
        if cuda_version is None or not _record("CUDA", cuda_version):
            return 3

        # ---- 6. CUDA samples (Python port of install_cuda_samples.sh) -------------
        log_info(_OP, "Installing the CUDA samples (install_cuda_samples)")
        try:
            samples_rc = install_cuda_samples.install(env)
        except Exception as exc:     # its own contract says it shouldn't raise
            log_error(_OP, f"CUDA samples installation raised: {exc}")
            return 3
        if samples_rc != 0:
            log_error(_OP, "CUDA samples installation failed", exit_code=samples_rc)
            return 3

        # ---- 7. nvidia-persistenced (nested bash component) -----------------------
        script = os.path.join(env.get("COMPONENT_DIR", ""), _PERSISTENCE_SCRIPT)
        log_info(_OP, "Configuring the nvidia-persistenced service", script=script)
        if not _run_all([[script]], env):
            log_error(_OP, "nvidia-persistenced configuration failed", script=script)
            return 3
    except OSError as exc:
        # Creating the work dir (the other file operations log their own errors).
        log_error(_OP, f"file operation failed: {exc}",
                  path=getattr(exc, "filename", None))
        return 3
    finally:
        # Deliberate difference: only our own scratch area is removed.
        shutil.rmtree(_WORK_DIR, ignore_errors=True)

    # Bash's final cleanup ran in the distro build dir; see the module docstring.
    log_warn(_OP, "bash would now delete *.run, *.tar.gz, *.rpm and every "
                  "subdirectory of the distro build dir (rm -rf -- */); this port "
                  f"only removed its own work dir {_WORK_DIR}",
             work_dir=_WORK_DIR)
    log_info(_OP, "NVIDIA GRID driver installed", grid_version=grid_version,
             cuda_toolkit=toolkit, cuda_version=cuda_version,
             gridd_conf=_GRIDD_CONF, cuda_profile=_CUDA_PROFILE,
             distro=distribution)
    return 0
