"""repo_source.py — Python port of write_repo_source_info in utils/utilities.sh.

Some components are built from a git checkout (NCCL's RDMA SHARP plugins,
nccl-tests, NVLOOM, nvbandwidth, RCCL, rccl-tests, perftest...). Upstream
records which commit and branch/tag each one was built from, next to the
component versions, so an image can be traced back to its sources:

    write_repo_source_info "NVLOOM" . --commit "${NVLOOM_COMMIT}" --ref "${NVLOOM_REF}"

writes two entries to /opt/azurehpc/component_versions.txt:

    "NVLOOM_REPO_COMMIT": "<commit>"
    "NVLOOM_REPO_REF":    "<ref>"

The original bash, summarised:

    if [[ -n "${specified_commit}" && "${specified_commit}" != "null" ]]; then
        repo_commit=${specified_commit}
    else
        repo_commit=$(git -C "${repo_dir}" rev-parse HEAD)
    fi
    write_component_version "${component}_REPO_COMMIT" "${repo_commit}"

    if [[ -n "${specified_ref}" && "${specified_ref}" != "null" ]]; then
        repo_ref=${specified_ref}
    else
        repo_ref=$(git -C "${repo_dir}" symbolic-ref --quiet --short HEAD 2>/dev/null || true)
        [[ -z "${repo_ref}" ]] && repo_ref="DETACHED"
    fi
    write_component_version "${component}_REPO_REF" "${repo_ref}"

"null" counts as "not given" because the values usually come from
`jq -r '.commit'`, which prints null when versions.json has no such key.

Deliberate difference: if `git rev-parse HEAD` fails, bash's `set -e` stops
the whole script; here the function logs the failure and returns False, and
the calling port returns 3. Nothing is recorded in that case.
"""

from __future__ import annotations

import os

from utils.component_config import write_component_version

from utils.logger import log_error, log_info
from utils.process import run, run_capture


def _given(value: str | None) -> bool:
    """True when bash would treat the value as set: non-empty and not "null"."""
    return bool(value) and value != "null"


def write_repo_source_info(component: str, repo_dir: str, op: str, *,
                           commit: str | None = None, ref: str | None = None,
                           env=None) -> bool:
    """Record the commit and ref `component` was built from.

    `repo_dir` is the git checkout; `op` is the calling step's log label.
    `commit` and `ref` are optional, as in bash; when missing they are read
    from the checkout. Returns True on success, False if git failed. Like
    write_component_version, it can raise OSError if the versions file can't
    be written; ports catch that with their other file errors.
    """
    if _given(commit):
        repo_commit = commit
    else:
        rc, out = run_capture(["git", "-C", repo_dir, "rev-parse", "HEAD"], op, env=env)
        if rc != 0:
            return False                     # run_capture has logged the error
        repo_commit = out.strip()

    if _given(ref):
        repo_ref = ref
    else:
        # `symbolic-ref` fails on a detached HEAD (e.g. after checking out a
        # commit); bash ignores that with `|| true`, so it isn't an error here.
        rc, out = run_capture(["git", "-C", repo_dir, "symbolic-ref", "--quiet",
                               "--short", "HEAD"], op, env=env, failure_is_error=False)
        repo_ref = out.strip() if rc == 0 else ""
        repo_ref = repo_ref or "DETACHED"

    write_component_version(f"{component}_REPO_COMMIT", repo_commit)
    write_component_version(f"{component}_REPO_REF", repo_ref)
    log_info(op, f"Recorded {component} source: {repo_commit} ({repo_ref})",
             component=component, repo_commit=repo_commit, repo_ref=repo_ref)
    return True


def configured_ref(config: dict, version: str) -> str:
    """The git ref to clone, as bash resolves it from a component's config:

        REF=$(jq -r '.ref // .branch // empty' <<< $metadata)
        REF=${REF:-v${VERSION}}

    jq's `//` skips a missing (null) or false value but keeps "" (so an empty
    "ref" hides "branch"); bash's `:-` then turns "" into v<version>.
    """
    ref = ""
    for key in ("ref", "branch"):
        value = config.get(key)
        if value is not None and value is not False:
            ref = str(value)
            break
    return ref or f"v{version}"


def git_clone(url: str, op: str, *, ref: str | None = None, commit: str | None = None,
              dest: str | None = None, cwd: str = ".", env=None) -> str | None:
    """`git clone [--branch <ref>] <url> [<dest>]` in `cwd`, then `git checkout <commit>`.

    Returns the checkout directory, or None if a git command failed or the
    clone didn't create the expected directory (logged). Bash's argument
    order varies between scripts (`git clone <url> --branch <ref>`); git
    treats them the same.
    """
    command = ["git", "clone"]
    if ref:
        command += ["--branch", ref]
    command.append(url)
    if dest:
        command.append(dest)
    if not run(command, op, cwd=cwd, env=env):
        return None
    name = dest or os.path.basename(url.rstrip("/")).removesuffix(".git")
    checkout = os.path.join(cwd, name)
    if not os.path.isdir(checkout):
        log_error(op, f"git clone did not create {checkout}", path=checkout, source=url)
        return None
    if commit and not run(["git", "checkout", commit], op, cwd=checkout, env=env):
        return None
    log_info(op, f"Cloned {url} into {checkout}", source=url, path=checkout,
             ref=ref or "", commit=commit or "")
    return checkout
