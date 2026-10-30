"""files.py — write, remove and unpack files, logging what changed.

Bash does these with `cat > file <<EOF`, `echo ... >> file`, `rm -rf` and
`tar -xvf`, and logs nothing about them. The image is made of these files, so
the Python build logs every one: which path, and whether it was created,
replaced or appended to. Each function returns True/False and logs its own
errors, so a port reads as a list of what it does:

    if not write_file(_SERVICE_FILE, service_text, _OP, mode=0o644):
        return 3
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from utils.logger import log_error, log_info
from utils.process import run


def write_file(path, text: str, op: str, *, mode: int | None = None,
               append: bool = False) -> bool:
    """Write `text` to `path` (bash `cat > path` / `>> path` with append=True).

    Creates missing parent directories, sets `mode` if given (bash `chmod`),
    and logs "Wrote <path> (created|replaced|appended)" with `path` and
    `action` fields. On an OSError it logs an ERROR and returns False.
    """
    path = Path(path)
    if append:
        action = "appended" if path.exists() else "created"
    else:
        action = "replaced" if path.exists() else "created"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a" if append else "w", encoding="utf-8") as out:
            out.write(text)
        if mode is not None:
            os.chmod(path, mode)
    except OSError as exc:
        log_error(op, f"could not write {path}: {exc}", path=str(path))
        return False
    log_info(op, f"Wrote {path} ({action})", path=str(path), action=action)
    return True


def remove_paths(*paths, op: str) -> None:
    """bash `rm -rf <paths>`: remove files or directory trees; missing is fine.

    Like `rm -rf`, it never fails the build: anything that can't be removed is
    logged as an ERROR and skipped.
    """
    for path in paths:
        path = Path(path)
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()
        except OSError as exc:
            log_error(op, f"could not remove {path}: {exc}", path=str(path))


def extract_tarball(tarball, op: str, *, dest=None, strip_components: int = 0,
                    cwd=None, env=None) -> bool:
    """Unpack `tarball` with the `tar` command, as bash does.

    Runs `tar -xf <tarball> [-C dest] [--strip-components=N]` (tar detects
    gzip/bzip2/xz itself, so bash's -z/-j and -v flags aren't needed). The tar
    command is used rather than Python's tarfile module because it keeps
    file modes, owners and symlinks exactly as bash's extraction would.
    """
    command = ["tar", "-xf", str(tarball)]
    if dest is not None:
        command += ["-C", str(dest)]
    if strip_components:
        command.append(f"--strip-components={strip_components}")
    return run(command, op, cwd=cwd, env=env)
