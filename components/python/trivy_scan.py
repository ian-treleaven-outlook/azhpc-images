"""trivy_scan.py — Python port of components/trivy_scan.sh.

Near the end of an image build, scans the finished root filesystem with Trivy
(Aqua Security's vulnerability scanner) and keeps two reports in the image,
which the Packer build also copies out:

  * /opt/azurehpc/trivy-report-rootfs.json     Trivy JSON, fixable vulns only
  * /opt/azurehpc/trivy-cyclonedx-rootfs.json  a CycloneDX SBOM of the same scan

Steps, in bash's order:

  1. Ask the GitHub API for the latest Trivy release ("v0.58.1" -> "0.58.1").
  2. mkdir -p /opt/azurehpc.
  3. wget trivy_<version>_<Linux-64bit|Linux-ARM64>.tar.gz, unpack it into
     trivy/, delete the tarball, chmod a+x the binary.
  4. Run the two scans, each under bash's retrycmd_if_failure 10 30 600: up to
     10 attempts under a 600 s `timeout`, 30 s apart (the vulnerability
     database download is sometimes flaky).
  5. rm -rf trivy/.

Only the Python build's "trivy-scan" Step runs this port: packer/build.pkr.hcl
and every distros/*/install.sh still call trivy_scan.sh directly, so a fix made
here must also be made in the .sh file until they are switched over.

Kept bash quirks
----------------
  * ${TRIVY_VERSION:1} drops the first character of the release name, "v" or
    not.
  * `${SKIP_DIRS[@]/#/ --skip-dirs }` is unquoted, so each directory becomes
    two words (`--skip-dirs /snap`). SKIP_FILES is never defined and (checked
    with bash 5.3 under set -u) expands to no words: no --skip-files options.
    skip_args() reproduces that list.
  * Like bash, nothing is checksum-verified: the version is "latest", fetched
    from GitHub releases with wget.

Deliberate differences
----------------------
  * Bash unpacks trivy/ in the distro build directory; this uses a scratch
    work directory (_WORK_DIR) as the scans' current directory, so
    `./trivy/trivy` and the skipped "<work dir>/trivy" still refer to it. It
    is removed even when a step fails (bash's `set -e` would leave trivy/).
  * A release "name" that isn't a non-empty string (or is empty once the
    first character is dropped) fails, instead of downloading a nonsense URL.
"""

from __future__ import annotations

import json
import os
import time

from utils.files import extract_tarball, remove_paths
from utils.logger import log_error, log_info
from utils.process import run, run_capture

_OP = "trivy-scan"

# The GitHub API call bash makes to find the newest Trivy release.
_LATEST_RELEASE_COMMAND = [
    "curl", "-sfL",
    "-H", "Accept: application/vnd.github+json",
    "-H", "X-GitHub-Api-Version: 2022-11-28",
    "https://api.github.com/repos/aquasecurity/trivy/releases/latest",
]
_DOWNLOAD_URL = "https://github.com/aquasecurity/trivy/releases/download"

# Module constants so the tests can point them at a temporary directory.
_REPORT_DIR = "/opt/azurehpc"
_WORK_DIR = "/tmp/azhpc-trivy"

# Comma-separated list Trivy tries in order when downloading its database.
_DB_REPOSITORIES = ("mcr.microsoft.com/mirror/ghcr/aquasecurity/trivy-db:2,"
                    "ghcr.io/aquasecurity/trivy-db:2,"
                    "public.ecr.aws/aquasecurity/trivy-db")

# Excluded from the scan; the unpacked trivy/ is added at run time.
_SKIP_DIRS = ["/var/lib/waagent", "/snap", "/mnt"]

# retrycmd_if_failure 10 30 600: attempts, seconds between them, timeout each.
_RETRIES = 10
_RETRY_WAIT_SECONDS = 30
_TIMEOUT_SECONDS = 600


# ---- Pure helpers -------------------------------------------------------------
def version_from_release(release_json: str) -> str | None:
    """The version in GitHub's "latest release" JSON: {"name": "v0.58.1"} -> "0.58.1".

    bash: `print(json.load(sys.stdin)['name'])` then ${TRIVY_VERSION:1}. None
    if the text isn't JSON, has no string "name", or leaves nothing behind.
    """
    try:
        release = json.loads(release_json)
    except json.JSONDecodeError:
        return None
    name = release.get("name") if isinstance(release, dict) else None
    if not isinstance(name, str):
        return None
    return name[1:] or None


def trivy_arch(architecture: str) -> str:
    """The architecture label in Trivy's release file names."""
    return "Linux-ARM64" if architecture == "aarch64" else "Linux-64bit"


def tarball_name(version: str, architecture: str) -> str:
    """e.g. ("0.58.1", "x86_64") -> "trivy_0.58.1_Linux-64bit.tar.gz"."""
    return f"trivy_{version}_{trivy_arch(architecture)}.tar.gz"


def skip_args(skip_dirs: list[str]) -> list[str]:
    """bash's SKIP_FILES (nothing) and SKIP_DIRS words: ["--skip-dirs", d, ...]."""
    args = []
    for directory in skip_dirs:
        args += ["--skip-dirs", directory]
    return args


def scan_commands(trivy_dir: str, report_dir: str) -> list[list[str]]:
    """The two trivy commands bash runs, in order (JSON report, then CycloneDX).

    trivy_dir (the unpacked trivy/, absolute) is skipped so Trivy doesn't scan
    itself. The binary is ./trivy/trivy, so run them from trivy_dir's parent.
    """
    common = ["--db-repository", _DB_REPOSITORIES,
              *skip_args([*_SKIP_DIRS, trivy_dir]),
              "--ignore-unfixed"]
    return [
        ["./trivy/trivy", "--scanners", "vuln", "rootfs", "-f", "json", *common,
         "--list-all-pkgs=false",
         "-o", os.path.join(report_dir, "trivy-report-rootfs.json"), "/"],
        ["./trivy/trivy", "--scanners", "vuln", "rootfs", "-f", "cyclonedx", *common,
         "-o", os.path.join(report_dir, "trivy-cyclonedx-rootfs.json"), "/"],
    ]


# ---- Steps ----------------------------------------------------------------------
def retry_command(command: list[str], env: dict[str, str], *, cwd: str,
                  retries: int = _RETRIES, wait_seconds: int = _RETRY_WAIT_SECONDS,
                  timeout_seconds: int = _TIMEOUT_SECONDS) -> bool:
    """bash's retrycmd_if_failure: `timeout <secs> <command>` until it succeeds,
    at most `retries` times, sleeping between failures (not after the last)."""
    timed = ["timeout", str(timeout_seconds), *command]
    for attempt in range(1, retries + 1):
        if run(timed, _OP, cwd=cwd, env=env):
            log_info(_OP, f'Executed "{" ".join(command)}" {attempt} times')
            return True
        if attempt < retries:
            time.sleep(wait_seconds)
    log_error(_OP, f'Executed "{" ".join(command)}" {retries} times; giving up')
    return False


def _latest_version(env: dict[str, str]) -> str | None:
    """Step 1: the latest Trivy release, from the GitHub API."""
    rc, output = run_capture(_LATEST_RELEASE_COMMAND, _OP, env=env)
    if rc != 0:
        log_error(_OP, "could not query the latest Trivy release from GitHub")
        return None
    version = version_from_release(output)
    if version is None:
        log_error(_OP, f"unexpected GitHub release response: {output[:200]!r}")
        return None
    log_info(_OP, f"Using Trivy {version}", version=version)
    return version


# TODO(ADO 1001): toolkit candidate: make_dirs(*paths, op) -> bool (bash `mkdir -p`,
# logging the path that failed).
def _make_dirs(*paths: str) -> bool:
    """bash `mkdir -p <paths>`."""
    try:
        for path in paths:
            os.makedirs(path, exist_ok=True)
    except OSError as exc:
        log_error(_OP, f"could not create {exc.filename}: {exc}", path=str(exc.filename))
        return False
    return True


def _download_trivy(version: str, architecture: str, env: dict[str, str]) -> bool:
    """Step 3: wget the tarball into the work dir, unpack it into trivy/, delete
    it and make the binary executable (bash's commands, run from the work dir)."""
    name = tarball_name(version, architecture)
    if not run(["wget", f"{_DOWNLOAD_URL}/v{version}/{name}", "-O", name], _OP,
               cwd=_WORK_DIR, env=env):
        return False
    trivy_dir = os.path.join(_WORK_DIR, "trivy")
    if not (_make_dirs(trivy_dir)
            and extract_tarball(name, _OP, dest="trivy", cwd=_WORK_DIR, env=env)):
        return False
    binary = os.path.join(trivy_dir, "trivy")
    try:
        os.remove(os.path.join(_WORK_DIR, name))                 # bash: rm (no -f)
        os.chmod(binary, os.stat(binary).st_mode | 0o111)        # chmod a+x
    except OSError as exc:
        log_error(_OP, f"could not prepare the trivy binary: {exc}", path=binary)
        return False
    return True


def _run_scans(env: dict[str, str]) -> bool:
    """Step 4: the two scans, each with retries, from the work dir."""
    for command in scan_commands(os.path.join(_WORK_DIR, "trivy"), _REPORT_DIR):
        if not retry_command(command, env, cwd=_WORK_DIR):
            return False
    return True


def install(env: dict[str, str]) -> int:
    """Download the latest Trivy and write the two rootfs scan reports; 0 or 3."""
    version = _latest_version(env)
    if version is None:
        return 3
    try:
        ok = (_make_dirs(_REPORT_DIR, _WORK_DIR)
              and _download_trivy(version, env.get("ARCHITECTURE", ""), env)
              and _run_scans(env))
    finally:
        remove_paths(_WORK_DIR, op=_OP)      # bash's `rm -rf ./trivy`, and the work dir
    if not ok:
        return 3
    log_info(_OP, f"Trivy reports written to {_REPORT_DIR}", path=_REPORT_DIR)
    return 0
