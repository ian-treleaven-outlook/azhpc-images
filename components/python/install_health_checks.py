"""install_health_checks.py — Python port of components/install_health_checks.sh.

Installs AzureHPC Node Health Checks (AzNHC) into
/opt/azurehpc/test/azurehpc-health-checks: downloads the v<version> release
tarball (SHA256 checked) and unpacks it, swaps the upstream
triggerGHR/triggerGHR.sh for our components/trigger_aznhc_GHR.sh, converts the
triggerGHR configs to Unix line endings, then pulls the vendor's container
image with ./dockerfile/pull-image-mcr.sh. On NVIDIA it first relaxes three
test thresholds in the conf files (bash's three `sed -i`). Records
AZ_HEALTH_CHECKS.

The vendor: bash gets it as $1, and the Step still passes it for the bash
fallback, but a Python action only receives `env`, so install() reads
env["GPU"] ("NVIDIA" or "AMD", set by component_env() from the same
cfg.gpu_arg). As in bash, "AMD" selects ROCm and any other value CUDA.

Kept bash quirks:
  * `mkdir azurehpc-health-checks && tar ...`: under `set -e` a failing
    command that is not the last of an `&&` list does not stop the script, so
    on a rerun (the folder exists) bash skips tar and carries on with the
    folder already there. install() warns and does the same. A failing tar
    still fails the step.
  * `rm ./triggerGHR/triggerGHR.sh` has no -f: a missing file fails the step.
  * The sed edits match exactly what sed matches (see _NVIDIA_CONF_EDITS).
  * The tarball is left in /opt/azurehpc/test.

Deliberate differences:
  * A missing aznhc version/sha256, or no GPU, fails before downloading. Bash
    would fetch "vnull.tar.gz", or silently take the CUDA branch.
  * The tarball is downloaded straight into /opt/azurehpc/test rather than
    downloaded and then moved there.
  * The file copy (copy_file, which also copies the source's mode), chmod +x
    and sed edits (edit_file) are Python. tar, dos2unix and the vendor's pull
    script still run as programs, so they behave as in bash.
"""

from __future__ import annotations

import glob
import os
import re
from functools import partial

from utils.component_config import record_version, require_config
from utils.download import fetch_verified
from utils.files import copy_file, edit_file, extract_tarball
from utils.logger import log_error, log_info, log_warn
from utils.process import run

_OP = "install-health-checks"

_DEST_TEST_DIR = "/opt/azurehpc/test"
_REPO_DIR_NAME = "azurehpc-health-checks"
_URL = "https://github.com/Azure/azurehpc-health-checks/archive/refs/tags/{tarball}"

_TRIGGER_SCRIPT = os.path.join("triggerGHR", "triggerGHR.sh")
_PULL_SCRIPT = os.path.join("dockerfile", "pull-image-mcr.sh")

# The NVIDIA `sed -i 's<old><new>'` edits as (conf file, regex, replacement).
# sed's basic regular expressions, translated so Python matches the same text:
# `|` is literal in sed but "or" in Python, so it is escaped; a `$` mid-pattern
# is literal in sed, so it becomes `\$`. `.` is left unescaped because it
# matches any character in both. Like sed, "check_gpu_bw 10" also matches the
# start of "check_gpu_bw 100" (which becomes "check_gpu_bw 90").
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
    match on each line. `replacement` is used literally (no backrefs). A
    missing final newline stays missing, as with sed.
    """
    lines = text.splitlines(keepends=True)
    # A function as the replacement stops re.sub interpreting backslashes in it.
    return "".join(re.sub(pattern, lambda _match: replacement, line, count=1)
                   for line in lines)


def chmod_plus_x(path: str) -> None:
    """Mirror `chmod +x path`: add the execute bits the umask allows.

    Without "u", "g" or "o", chmod's "+x" skips bits the umask blocks; with
    the usual umask 022 that is execute for everyone.
    """
    # Python can only read the umask by setting it, so set it and put it back.
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(path, os.stat(path).st_mode | (0o111 & ~umask))


def _unpack(tarball: str, repo: str, env: dict[str, str]) -> bool:
    """bash `mkdir azurehpc-health-checks && tar ...` (see "Kept bash quirks")."""
    try:
        os.mkdir(repo)                   # not makedirs: an existing folder must fail
    except FileExistsError:
        log_warn(_OP, f"{repo} already exists; reusing it without extracting "
                      f"{tarball} (same as bash)", path=repo)
        return True
    except OSError as exc:
        # Bash would also skip tar here, but its `pushd` into the folder fails.
        log_error(_OP, f"could not create {repo}: {exc}", path=repo)
        return False
    return extract_tarball(tarball, _OP, dest=_REPO_DIR_NAME, strip_components=1,
                           cwd=_DEST_TEST_DIR, env=env)


def _replace_trigger_script(repo: str, env: dict[str, str]) -> bool:
    """bash `rm ./triggerGHR/triggerGHR.sh` (no -f) then `cp` ours over it."""
    trigger = os.path.join(repo, _TRIGGER_SCRIPT)
    try:
        os.remove(trigger)          # not remove_paths: plain `rm` fails the step
    except OSError as exc:
        log_error(_OP, f"could not remove {trigger}: {exc}", path=trigger)
        return False
    return copy_file(os.path.join(env["COMPONENT_DIR"], "trigger_aznhc_GHR.sh"),
                     trigger, _OP)


def _convert_trigger_configs(repo: str, env: dict[str, str]) -> bool:
    """bash `dos2unix ./triggerGHR/config/*`."""
    # Like the shell's `*`, glob skips dot files; sorted() gives the shell's order.
    configs = sorted(glob.glob(os.path.join(repo, "triggerGHR", "config", "*")))
    if not configs:
        log_error(_OP, f"no files found in {repo}/triggerGHR/config")
        return False
    return run(["dos2unix", *configs], _OP, cwd=repo, env=env)


def _make_scripts_executable(repo: str) -> bool:
    """bash `chmod +x` on triggerGHR.sh and pull-image-mcr.sh."""
    try:
        chmod_plus_x(os.path.join(repo, _TRIGGER_SCRIPT))
        chmod_plus_x(os.path.join(repo, _PULL_SCRIPT))
    except OSError as exc:
        log_error(_OP, f"could not make the scripts executable: {exc}")
        return False
    return True


def _edit_nvidia_confs(repo: str) -> bool:
    """Apply the three NVIDIA threshold edits (bash's three `sed -i`)."""
    return all(edit_file(os.path.join(repo, "conf", conf),
                         partial(sed_substitute, pattern=pattern, replacement=replacement),
                         _OP)
               for conf, pattern, replacement in _NVIDIA_CONF_EDITS)


def install(env: dict[str, str]) -> int:
    """Install AzNHC and pull its container image. Returns 0 or 3."""
    cfg = require_config("aznhc", env, _OP, "version", "sha256")
    if cfg is None:
        return 3
    gpu = env.get("GPU", "")
    if not gpu:
        log_error(_OP, "GPU (NVIDIA or AMD) is not set")
        return 3
    version = cfg["version"]
    image = "rocm" if gpu == "AMD" else "cuda"
    log_info(_OP, "Installing AzureHPC health checks", version=version, gpu=gpu,
             image=image)

    tarball = f"v{version}.tar.gz"
    if fetch_verified(_URL.format(tarball=tarball), cfg["sha256"], _OP,
                      _DEST_TEST_DIR) is None:
        return 3
    repo = os.path.join(_DEST_TEST_DIR, _REPO_DIR_NAME)
    if not (_unpack(tarball, repo, env)
            and _replace_trigger_script(repo, env)
            and _convert_trigger_configs(repo, env)
            and _make_scripts_executable(repo)):
        return 3
    if image == "cuda" and not _edit_nvidia_confs(repo):
        return 3
    if not run(["./" + _PULL_SCRIPT, image], _OP, cwd=repo, env=env):
        return 3
    if not record_version("AZ_HEALTH_CHECKS", version, _OP):
        return 3

    log_info(_OP, "AzureHPC health checks installed", version=version, path=repo)
    return 0
