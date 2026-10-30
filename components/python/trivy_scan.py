"""trivy_scan.py — Python port of components/trivy_scan.sh.

What this step does
-------------------
Near the end of an image build we scan the finished root filesystem for known
vulnerabilities with Trivy (an open-source scanner from Aqua Security) and keep
two reports in the image, which the Packer build also copies out:

  * /opt/azurehpc/trivy-report-rootfs.json     Trivy's own JSON report
                                                (fixable vulnerabilities only)
  * /opt/azurehpc/trivy-cyclonedx-rootfs.json  a CycloneDX SBOM (software bill
                                                of materials) of the same scan

Steps, in bash order:

  1. Ask the GitHub API for the latest Trivy release, e.g. "v0.58.1" -> "0.58.1".
  2. mkdir -p /opt/azurehpc.
  3. Download trivy_<version>_<Linux-64bit|Linux-ARM64>.tar.gz with wget,
     unpack it into a trivy/ directory, delete the tarball, chmod a+x the binary.
  4. Run the two scans. Each one is retried up to 10 times, under a 600-second
     `timeout`, waiting 30 seconds between attempts (the vulnerability database
     download from a container registry is sometimes flaky).
  5. rm -rf the trivy/ directory.

The original bash, for comparison (abridged; it runs with set -euxo pipefail):

    TRIVY_VERSION=$(curl -sfL -H "Accept: application/vnd.github+json" \\
        -H "X-GitHub-Api-Version: 2022-11-28" \\
        https://api.github.com/repos/aquasecurity/trivy/releases/latest \\
        | python3 -c "import sys,json;print(json.load(sys.stdin)['name'])")
    TRIVY_VERSION=${TRIVY_VERSION:1} # remove the leading 'v'
    if [[ "$ARCHITECTURE" == "aarch64" ]]; then TRIVY_ARCH="Linux-ARM64"
    else TRIVY_ARCH="Linux-64bit"; fi
    TRIVY_DB_REPOSITORIES="mcr.microsoft.com/mirror/ghcr/aquasecurity/trivy-db:2,..."
    declare -a SKIP_DIRS=("/var/lib/waagent" "/snap" "/mnt" "$(pwd)/trivy")

    retrycmd_if_failure() {
        retries=$1; wait_sleep=$2; timeout=$3; shift && shift && shift
        for i in $(seq 1 $retries); do
            timeout $timeout "${@}" && break || \\
            if [ $i -eq $retries ]; then
                echo Executed \\"$@\\" $i times; return 1
            else
                sleep $wait_sleep
            fi
        done
        echo Executed \\"$@\\" $i times;
    }

    mkdir -p "${TRIVY_REPORT_DIRNAME}"
    TARFILE="trivy_${TRIVY_VERSION}_${TRIVY_ARCH}.tar.gz"
    wget "https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/${TARFILE}" -O "${TARFILE}"
    mkdir -p trivy
    tar -xvzf "${TARFILE}" -C trivy
    rm "${TARFILE}"
    chmod a+x trivy/trivy

    retrycmd_if_failure 10 30 600 ./trivy/trivy --scanners vuln rootfs -f json \\
        --db-repository ${TRIVY_DB_REPOSITORIES} ${SKIP_FILES[@]/#/ --skip-files } \\
        ${SKIP_DIRS[@]/#/ --skip-dirs } --ignore-unfixed --list-all-pkgs=false \\
        -o "${TRIVY_REPORT_ROOTFS_JSON_PATH}" /
    retrycmd_if_failure 10 30 600 ./trivy/trivy --scanners vuln rootfs -f cyclonedx \\
        --db-repository ${TRIVY_DB_REPOSITORIES} ${SKIP_FILES[@]/#/ --skip-files } \\
        ${SKIP_DIRS[@]/#/ --skip-dirs } --ignore-unfixed \\
        -o "${TRIVY_CYCLONEDX_ROOTFS_JSON_PATH}" /

    rm -rf ./trivy

How the skip options expand in bash
-----------------------------------
`${SKIP_DIRS[@]/#/ --skip-dirs }` prefixes every element with " --skip-dirs ";
because it's unquoted, bash then splits each result into two words, giving
`--skip-dirs /var/lib/waagent --skip-dirs /snap ...`. SKIP_FILES is never
defined anywhere, and (checked with bash 5.3 under set -u) its expansion
produces no words at all, so there are no --skip-files options. skip_args()
below reproduces exactly that argument list.

How it fits into the Python build
---------------------------------
utils/plans/steps.py sets this module's install() as the `action=` of the
"trivy-scan" Step. Deleting `action=` from that Step makes the build run the
bash script again, which is why the .sh file is kept. (packer/build.pkr.hcl and
the distros/*/install.sh scripts still call the bash script directly.)

install() receives the build environment (a dict) and returns 0 for success or
3 for failure. It logs problems and never lets an exception escape.

Deliberate differences from bash
--------------------------------
  * Bash runs in the distro build directory and unpacks trivy/ there; we use a
    scratch work directory (_WORK_DIR) and run the scans with it as the current
    directory, so `./trivy/trivy` and the skipped "<work dir>/trivy" still
    refer to the unpacked copy. The work directory is removed even when a step
    fails (bash's `set -e` would leave trivy/ behind).
  * A release "name" that isn't a non-empty string (or leaves nothing once the
    first character is dropped) fails, instead of downloading a nonsense URL.
  * The tarball is unpacked with Python's tarfile module (filter="data") rather
    than `tar -xvzf`; the Trivy release archive holds only regular files.
  * Like bash, nothing is checksum-verified: the version is "latest" and the
    tarball comes straight from GitHub releases with wget.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import time

from utils.logger import log_error, log_info
from utils.process import exec_program, run_capture

# The Step's `op` label in plans/steps.py; used on every log line.
_OP = "trivy-scan"

# The GitHub API call bash makes to find the newest Trivy release.
_LATEST_RELEASE_COMMAND = [
    "curl", "-sfL",
    "-H", "Accept: application/vnd.github+json",
    "-H", "X-GitHub-Api-Version: 2022-11-28",
    "https://api.github.com/repos/aquasecurity/trivy/releases/latest",
]
_DOWNLOAD_URL = "https://github.com/aquasecurity/trivy/releases/download"

# Module-level constants so the tests can point them at a temporary directory.
_REPORT_DIR = "/opt/azurehpc"
_WORK_DIR = "/tmp/azhpc-trivy"

# Comma-separated list Trivy tries in order when downloading its database.
_DB_REPOSITORIES = ("mcr.microsoft.com/mirror/ghcr/aquasecurity/trivy-db:2,"
                    "ghcr.io/aquasecurity/trivy-db:2,"
                    "public.ecr.aws/aquasecurity/trivy-db")

# Directories excluded from the scan. The unpacked trivy/ directory is added
# at run time, because its path depends on the work directory.
_SKIP_DIRS = ["/var/lib/waagent", "/snap", "/mnt"]

# retrycmd_if_failure 10 30 600: attempts, seconds between them, timeout each.
_RETRIES = 10
_RETRY_WAIT_SECONDS = 30
_TIMEOUT_SECONDS = 600


# --------------------------------------------------------------------------- #
# Pure helpers: they only transform values, so tests call them directly.
# --------------------------------------------------------------------------- #
def version_from_release(release_json: str) -> str | None:
    """Return the Trivy version from the GitHub "latest release" JSON.

    Mirrors  python3 -c "...print(json.load(sys.stdin)['name'])"  followed by
    ${TRIVY_VERSION:1}, which drops the first character (the leading "v"):

        {"name": "v0.58.1", ...}  ->  "0.58.1"

    Like bash, the first character is dropped whatever it is. Returns None if
    the text isn't JSON, has no string "name", or leaves nothing behind.
    """
    try:
        release = json.loads(release_json)
    except json.JSONDecodeError:
        return None
    # isinstance() checks the type: the JSON could be a list, or name could be null.
    name = release.get("name") if isinstance(release, dict) else None
    if not isinstance(name, str):
        return None
    # name[1:] is a slice: everything from index 1 on, like bash's ${name:1}.
    return name[1:] or None


def trivy_arch(architecture: str) -> str:
    """The architecture label in Trivy's release file names."""
    return "Linux-ARM64" if architecture == "aarch64" else "Linux-64bit"


def tarball_name(version: str, architecture: str) -> str:
    """e.g. ("0.58.1", "x86_64") -> "trivy_0.58.1_Linux-64bit.tar.gz"."""
    return f"trivy_{version}_{trivy_arch(architecture)}.tar.gz"


def skip_args(skip_dirs: list[str]) -> list[str]:
    """The words bash produces for the (empty) SKIP_FILES and the SKIP_DIRS.

    e.g. ["/snap", "/mnt"] -> ["--skip-dirs", "/snap", "--skip-dirs", "/mnt"].
    SKIP_FILES is undefined in bash and expands to nothing (see the module
    docstring), so no --skip-files words are produced.
    """
    args = []
    for directory in skip_dirs:
        args += ["--skip-dirs", directory]
    return args


def scan_commands(trivy_dir: str, report_dir: str) -> list[list[str]]:
    """The two trivy commands bash runs, in order (JSON report, then CycloneDX).

    trivy_dir is the absolute path of the unpacked trivy/ directory; it is
    skipped so Trivy doesn't scan itself. The binary is run as ./trivy/trivy,
    so the commands must run with trivy_dir's parent as the current directory.
    """
    common = ["--db-repository", _DB_REPOSITORIES,
              *skip_args([*_SKIP_DIRS, trivy_dir]),
              "--ignore-unfixed"]
    # The * in [..., *common, ...] inserts common's items into the new list.
    return [
        ["./trivy/trivy", "--scanners", "vuln", "rootfs", "-f", "json", *common,
         "--list-all-pkgs=false",
         "-o", os.path.join(report_dir, "trivy-report-rootfs.json"), "/"],
        ["./trivy/trivy", "--scanners", "vuln", "rootfs", "-f", "cyclonedx", *common,
         "-o", os.path.join(report_dir, "trivy-cyclonedx-rootfs.json"), "/"],
    ]


# --------------------------------------------------------------------------- #
# Helpers with side effects.
# --------------------------------------------------------------------------- #
def retry_command(command: list[str], env: dict[str, str], *, cwd: str,
                  retries: int = _RETRIES, wait_seconds: int = _RETRY_WAIT_SECONDS,
                  timeout_seconds: int = _TIMEOUT_SECONDS) -> bool:
    """Port of bash's retrycmd_if_failure: run `timeout <secs> <command>` until
    it succeeds, at most `retries` times, sleeping between failed attempts.

    Returns True on success, False once every attempt has failed (bash returns
    1 there, which stops the script because of set -e). There is no sleep after
    the final failed attempt, as in bash.
    """
    timed = ["timeout", str(timeout_seconds), *command]
    # range(1, retries + 1) counts 1, 2, ..., retries, like `seq 1 $retries`.
    for attempt in range(1, retries + 1):
        if exec_program(timed, _OP, cwd=cwd, env=env) == 0:
            log_info(_OP, f'Executed "{" ".join(command)}" {attempt} times')
            return True
        if attempt < retries:
            time.sleep(wait_seconds)
    log_error(_OP, f'Executed "{" ".join(command)}" {retries} times; giving up')
    return False


def _download_trivy(version: str, architecture: str, env: dict[str, str]) -> bool:
    """wget the release tarball into the work dir, unpack it into trivy/,
    delete the tarball and make the binary executable.

    May raise OSError / tarfile.TarError; install() catches them.
    """
    tarfile_name = tarball_name(version, architecture)
    tarball = os.path.join(_WORK_DIR, tarfile_name)
    url = f"{_DOWNLOAD_URL}/v{version}/{tarfile_name}"
    # Relative -O name with the work dir as cwd, exactly like bash's command.
    if exec_program(["wget", url, "-O", tarfile_name], _OP, cwd=_WORK_DIR, env=env) != 0:
        return False

    trivy_dir = os.path.join(_WORK_DIR, "trivy")
    os.makedirs(trivy_dir, exist_ok=True)
    # tarfile.open() detects the gzip compression itself. filter="data" refuses
    # unsafe entries (absolute paths, links leaving the directory); the Trivy
    # archive contains only regular files, so it unpacks like `tar -xzf`.
    with tarfile.open(tarball) as archive:
        archive.extractall(trivy_dir, filter="data")
    os.remove(tarball)

    # chmod a+x: add the execute bit for user, group and others (0o111) to the
    # current mode. 0o... is Python's octal notation, as used by chmod.
    binary = os.path.join(trivy_dir, "trivy")
    os.chmod(binary, os.stat(binary).st_mode | 0o111)
    return True


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def install(env: dict[str, str]) -> int:
    """Download the latest Trivy and write the two rootfs scan reports.

    Returns 0 on success, 3 on failure.
    """
    # NOTE: this function only replaces trivy_scan.sh inside the Python build
    # (the "trivy-scan" Step). packer/build.pkr.hcl runs
    # `bash ../../components/trivy_scan.sh` itself after the build, and every
    # distros/*/install.sh still calls $COMPONENT_DIR/trivy_scan.sh directly,
    # so those scans use the bash script, not this port. A fix made here must
    # also be made in the .sh file until they are switched over.
    # ---- 1. Latest release version ----------------------------------------
    rc, output = run_capture(_LATEST_RELEASE_COMMAND, _OP, env=env)
    if rc != 0:
        log_error(_OP, "could not query the latest Trivy release from GitHub")
        return 3
    version = version_from_release(output)
    if version is None:
        log_error(_OP, f"unexpected GitHub release response: {output[:200]!r}")
        return 3
    log_info(_OP, f"Using Trivy {version}")

    architecture = env.get("ARCHITECTURE", "")
    trivy_dir = os.path.join(_WORK_DIR, "trivy")
    try:
        # ---- 2. Report directory (mkdir -p) --------------------------------
        os.makedirs(_REPORT_DIR, exist_ok=True)

        # ---- 3. Download and unpack ----------------------------------------
        os.makedirs(_WORK_DIR, exist_ok=True)
        if not _download_trivy(version, architecture, env):
            return 3

        # ---- 4. The two scans, each with retries ---------------------------
        for command in scan_commands(trivy_dir, _REPORT_DIR):
            if not retry_command(command, env, cwd=_WORK_DIR):
                return 3
    except (OSError, tarfile.TarError) as exc:
        log_error(_OP, f"file operation failed: {exc}")
        return 3
    finally:
        # ---- 5. Cleanup: bash's `rm -rf ./trivy`, plus our work directory.
        # A `finally` block runs whether the `try` returned, failed or raised.
        shutil.rmtree(_WORK_DIR, ignore_errors=True)

    log_info(_OP, f"Trivy reports written to {_REPORT_DIR}")
    return 0
