"""install_health_checks.py — Python port of components/install_health_checks.sh.

What this step does
-------------------
Installs AzureHPC Node Health Checks (AzNHC) into
/opt/azurehpc/test/azurehpc-health-checks:

  1. Download the release tarball v<version>.tar.gz from GitHub (SHA256 checked)
     into /opt/azurehpc/test, and unpack it into azurehpc-health-checks/.
  2. Replace the upstream triggerGHR/triggerGHR.sh with our own
     components/trigger_aznhc_GHR.sh, convert the triggerGHR config files to
     Unix line endings, and make the two scripts we use executable.
  3. AMD: pull the ROCm health-check container image.
     NVIDIA: relax three test thresholds in the conf files, then pull the CUDA
     image.
  4. Record the version as AZ_HEALTH_CHECKS.

The original bash, for comparison:

    aznhc_metadata=$(get_component_config "aznhc")
    AZHC_VERSION=$(jq -r '.version' <<< $aznhc_metadata)
    AZHC_SHA=$(jq -r '.sha256' <<< $aznhc_metadata)
    DEST_TEST_DIR=/opt/azurehpc/test
    GPU_PLAT=$1
    TARBALL="v${AZHC_VERSION}.tar.gz"
    AZHC_DOWNLOAD_URL=https://github.com/Azure/azurehpc-health-checks/archive/refs/tags/${TARBALL}
    download_and_verify ${AZHC_DOWNLOAD_URL} ${AZHC_SHA} $DEST_TEST_DIR
    pushd $DEST_TEST_DIR
    mkdir azurehpc-health-checks && tar -xvf $TARBALL --strip-components=1 -C azurehpc-health-checks
    pushd azurehpc-health-checks
    rm ./triggerGHR/triggerGHR.sh
    cp $COMPONENT_DIR/trigger_aznhc_GHR.sh ./triggerGHR/triggerGHR.sh
    dos2unix ./triggerGHR/config/*
    chmod +x ./triggerGHR/triggerGHR.sh
    chmod +x ./dockerfile/pull-image-mcr.sh
    if [ "${GPU_PLAT}" = "AMD" ]; then
       ./dockerfile/pull-image-mcr.sh rocm
    else
       sed -i 's/\\* || check_gpu_bw 10/\\* || check_gpu_bw 9/' ./conf/nd40rs_v2.conf
       sed -i 's#\\* || check_nccl_allreduce 431.0 1 16G $AZ_NHC_ROOT/topofiles/ndv5-topo.xml#\\* || check_nccl_allreduce 431.0 1 16G#' ./conf/nd96isr_h200_v5.conf
       sed -i 's#\\* || check_nccl_allreduce 460.0 1 16G $AZ_NHC_ROOT/topofiles/ndv5-topo.xml#\\* || check_nccl_allreduce 460.0 1 16G#' ./conf/nd96isr_h100_v5.conf
       ./dockerfile/pull-image-mcr.sh cuda
    fi
    popd
    popd
    write_component_version "AZ_HEALTH_CHECKS" ${AZHC_VERSION}

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets this module's install() as the `action=` of the
"install-health-checks" Step. The build then calls install(env) instead of
running the bash script. Deleting `action=` runs the bash script again, which
is why the .sh file is kept.

The bash script gets the GPU vendor as its first argument ($1). The Step still
passes it (`args=(cfg.gpu_arg,)`, unchanged so the bash fallback keeps
working), but a Python action only receives `env`. So install() reads the
vendor from env["GPU"], which component_env() sets from the same
cfg.gpu_arg ("NVIDIA" or "AMD"). As in bash, "AMD" selects the ROCm branch and
any other value the CUDA branch.

install() returns 0 for success or 3 for failure. It logs problems and returns
3; it does not let exceptions escape to the build loop.

Which commands still run as programs
------------------------------------
  * `tar`: the --strip-components option has no simple tarfile equivalent,
    and the tar command keeps the archive's symlinks and modes exactly as bash.
  * `dos2unix`, so line-ending conversion is byte-for-byte what bash did.
  * The vendor's ./dockerfile/pull-image-mcr.sh.
The file copy, chmod and sed edits are plain Python.

A bash subtlety we mirror: `mkdir X && tar ...`
-----------------------------------------------
`set -e` does not stop the script when a command that is *not the last* in an
`&&` list fails. So if azurehpc-health-checks/ already exists (a rerun), bash's
`mkdir` fails, `tar` is skipped, and the script carries on with the folder that
is already there. install() does the same: it logs a warning, skips the
extraction and continues. A failing `tar` (the last command of the list) does
stop bash, and returns 3 here.

Deliberate differences from bash
--------------------------------
  * A missing aznhc version or sha256, or a missing GPU value, fails the step
    before anything is downloaded. Bash would have tried to download
    "vnull.tar.gz", or silently taken the CUDA branch with no vendor.
  * Bash downloads into its current folder and then moves the file into
    /opt/azurehpc/test; we download straight into /opt/azurehpc/test. As in
    bash, the tarball is left there afterwards.
  * The copied triggerGHR.sh is created with the default file mode and then
    made executable. With the usual umask (022) the result is 0755, as with
    bash's `cp` + `chmod +x`.
"""

from __future__ import annotations

import glob
import os
import re
import shutil

from utils.component_config import config_for, write_component_version
from utils.download import ChecksumError, download_and_verify
from utils.logger import log_error, log_info, log_warn
from utils.process import exec_program

# The Step's `op` label in plans/steps.py. Every log line uses it.
_OP = "install-health-checks"

# Where the tarball is downloaded and the repository unpacked.
_DEST_TEST_DIR = "/opt/azurehpc/test"
_REPO_DIR_NAME = "azurehpc-health-checks"

_URL = "https://github.com/Azure/azurehpc-health-checks/archive/refs/tags/{tarball}"

_TRIGGER_SCRIPT = os.path.join("triggerGHR", "triggerGHR.sh")
_PULL_SCRIPT = os.path.join("dockerfile", "pull-image-mcr.sh")

# The three NVIDIA `sed -i 's<old><new>'` edits, as (conf file, regex, replacement).
# sed patterns are regular expressions (POSIX "basic" syntax), translated here
# to Python's re syntax so they match exactly the same text:
#   * sed's `\*` is a literal "*"; in Python it is written `\*` too.
#   * `|` is a plain character in sed's basic syntax but means "or" in Python,
#     so it is escaped as `\|`.
#   * `.` in "431.0" and "ndv5-topo.xml" matches ANY character in both, and is
#     left unescaped on purpose to match sed exactly.
#   * `$` in the middle of a sed pattern is a literal "$", so it becomes `\$`.
#   * "10" also matches the start of "100", so "check_gpu_bw 100" would become
#     "check_gpu_bw 90". That is what sed does, so we do it too.
# The r"..." prefix makes a "raw" string: backslashes are kept as typed.
_NVIDIA_CONF_EDITS = [
    # sed -i 's/\* || check_gpu_bw 10/\* || check_gpu_bw 9/' ./conf/nd40rs_v2.conf
    ("nd40rs_v2.conf",
     r"\* \|\| check_gpu_bw 10",
     "* || check_gpu_bw 9"),
    # sed -i 's#\* || check_nccl_allreduce 431.0 1 16G $AZ_NHC_ROOT/topofiles/ndv5-topo.xml#\* || check_nccl_allreduce 431.0 1 16G#' ./conf/nd96isr_h200_v5.conf
    ("nd96isr_h200_v5.conf",
     r"\* \|\| check_nccl_allreduce 431.0 1 16G \$AZ_NHC_ROOT/topofiles/ndv5-topo.xml",
     "* || check_nccl_allreduce 431.0 1 16G"),
    # sed -i 's#\* || check_nccl_allreduce 460.0 1 16G $AZ_NHC_ROOT/topofiles/ndv5-topo.xml#\* || check_nccl_allreduce 460.0 1 16G#' ./conf/nd96isr_h100_v5.conf
    ("nd96isr_h100_v5.conf",
     r"\* \|\| check_nccl_allreduce 460.0 1 16G \$AZ_NHC_ROOT/topofiles/ndv5-topo.xml",
     "* || check_nccl_allreduce 460.0 1 16G"),
]


def sed_substitute(text: str, pattern: str, replacement: str) -> str:
    """Mirror `sed 's/<pattern>/<replacement>/'` on a whole file's text.

    sed works line by line and, without the `g` flag, replaces only the first
    match on each line. `replacement` is used literally (no backrefs).
    """
    # splitlines(keepends=True) keeps each line's "\n", so joining the lines
    # back gives the file unchanged apart from the edits (including a missing
    # final newline, which sed also preserves).
    lines = text.splitlines(keepends=True)
    # A function as the replacement makes re.sub use it literally, without
    # treating backslashes in it specially. count=1 means "first match only".
    return "".join(re.sub(pattern, lambda _match: replacement, line, count=1)
                   for line in lines)


def chmod_plus_x(path: str) -> None:
    """Mirror `chmod +x path`: add execute permission, as allowed by the umask.

    Without "u", "g" or "o", chmod's "+x" skips any bit the umask blocks. With
    the usual umask 022 this adds execute for everyone.
    """
    # Python can only read the umask by setting it, so set it and put it back.
    umask = os.umask(0)
    os.umask(umask)
    mode = os.stat(path).st_mode
    os.chmod(path, mode | (0o111 & ~umask))


def _run(command: list[str], env: dict[str, str], cwd: str) -> bool:
    """Run one command; log and return False if it fails."""
    rc = exec_program(command, _OP, cwd=cwd, env=env)
    if rc != 0:
        log_error(_OP, f"{' '.join(command)} failed with exit code {rc}")
        return False
    return True


def _prepare_repo(repo: str, env: dict[str, str]) -> bool:
    """Swap in our triggerGHR.sh, fix line endings, make scripts executable."""
    trigger = os.path.join(repo, _TRIGGER_SCRIPT)
    try:
        # bash: rm (no -f, so a missing file is an error) then cp.
        os.remove(trigger)
        shutil.copyfile(os.path.join(env["COMPONENT_DIR"], "trigger_aznhc_GHR.sh"),
                        trigger)
    except OSError as exc:
        log_error(_OP, f"could not replace {trigger}: {exc}")
        return False

    # bash: dos2unix ./triggerGHR/config/*
    # glob.glob, like the shell's `*`, skips names starting with "." and returns
    # nothing when there is no match; sorted() mimics the shell's order.
    configs = sorted(glob.glob(os.path.join(repo, "triggerGHR", "config", "*")))
    if not configs:
        log_error(_OP, f"no files found in {repo}/triggerGHR/config")
        return False
    if not _run(["dos2unix", *configs], env, cwd=repo):
        return False

    try:
        chmod_plus_x(trigger)
        chmod_plus_x(os.path.join(repo, _PULL_SCRIPT))
    except OSError as exc:
        log_error(_OP, f"could not make the scripts executable: {exc}")
        return False
    return True


def _edit_nvidia_confs(repo: str) -> bool:
    """Apply the three NVIDIA threshold edits (bash's three `sed -i`)."""
    for conf, pattern, replacement in _NVIDIA_CONF_EDITS:
        path = os.path.join(repo, "conf", conf)
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(sed_substitute(text, pattern, replacement))
        except OSError as exc:
            log_error(_OP, f"could not edit {path}: {exc}")
            return False
    return True


def install(env: dict[str, str]) -> int:
    """Install AzNHC and pull its container image. Returns 0 or 3."""
    cfg = config_for("aznhc", env)
    if not cfg or not cfg.get("version") or not cfg.get("sha256"):
        log_error(_OP, "could not resolve aznhc version and sha256 from versions.json")
        return 3
    version = cfg["version"]

    # The vendor bash receives as $1 (see "How it fits into the Python build").
    gpu = env.get("GPU", "")
    if not gpu:
        log_error(_OP, "GPU (NVIDIA or AMD) is not set")
        return 3

    log_info(_OP, f"Installing AzureHPC health checks {version} for {gpu}")

    tarball = f"v{version}.tar.gz"
    try:
        download_and_verify(_URL.format(tarball=tarball), cfg["sha256"],
                            dest_dir=_DEST_TEST_DIR)
    # download() raises OSError (network/file errors); a bad hash raises ChecksumError.
    except (OSError, ChecksumError) as exc:
        log_error(_OP, f"download/verify failed: {exc}")
        return 3

    # bash: mkdir azurehpc-health-checks && tar -xvf ... (see the docstring).
    repo = os.path.join(_DEST_TEST_DIR, _REPO_DIR_NAME)
    try:
        # os.mkdir, unlike os.makedirs(exist_ok=True), fails if it exists,
        # like `mkdir` without -p.
        os.mkdir(repo)
    except FileExistsError:
        log_warn(_OP, f"{repo} already exists; reusing it without extracting "
                      f"{tarball} (same as bash)")
    except OSError as exc:
        # Any other mkdir failure also skips tar in bash, but then its
        # `pushd azurehpc-health-checks` fails, so we fail here.
        log_error(_OP, f"could not create {repo}: {exc}")
        return 3
    else:
        # `else` runs only when the `try` raised nothing: mkdir succeeded.
        if not _run(["tar", "-xvf", tarball, "--strip-components=1",
                     "-C", _REPO_DIR_NAME], env, cwd=_DEST_TEST_DIR):
            return 3

    if not _prepare_repo(repo, env):
        return 3

    pull = "./" + _PULL_SCRIPT
    if gpu == "AMD":
        if not _run([pull, "rocm"], env, cwd=repo):
            return 3
    else:
        if not _edit_nvidia_confs(repo):
            return 3
        if not _run([pull, "cuda"], env, cwd=repo):
            return 3

    try:
        write_component_version("AZ_HEALTH_CHECKS", version)
    except OSError as exc:
        log_error(_OP, f"could not record AZ_HEALTH_CHECKS={version}: {exc}")
        return 3

    log_info(_OP, f"AzureHPC health checks {version} installed to {repo}")
    return 0
