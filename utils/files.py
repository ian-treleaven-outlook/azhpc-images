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

import glob
import os
import shutil
import tempfile
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


# --- More bash file commands ------------------------------------------------
# Each mirrors one bash command, logs what it changed, and returns True/False
# (logging its own errors), so a port can write `if not copy_file(...): return 3`.

def make_dirs(*paths, op: str) -> bool:
    """bash `mkdir -p <paths>`."""
    for path in paths:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log_error(op, f"could not create directory {path}: {exc}", path=str(path))
            return False
    return True


def fresh_dir(path, op: str) -> bool:
    """bash `rm -rf <path> && mkdir -p <path>`: an empty directory to work in."""
    remove_paths(path, op=op)
    return make_dirs(path, op=op)


def make_temp_dir(op: str, *, prefix: str = "tmp") -> Path | None:
    """bash `mktemp -d`: a new private directory under /tmp, or None (logged)."""
    try:
        return Path(tempfile.mkdtemp(prefix=prefix))
    except OSError as exc:
        log_error(op, f"could not create a temporary directory: {exc}")
        return None


def touch(path, op: str) -> bool:
    """bash `touch <path>`: create it if missing (parents too), else update its time."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    except OSError as exc:
        log_error(op, f"could not touch {path}: {exc}", path=str(path))
        return False
    log_info(op, f"Touched {path}", path=str(path))
    return True


def copy_file(source, dest, op: str, *, mode: int | None = None) -> bool:
    """bash `cp <source> <dest>` (dest may be a directory), then `chmod` if mode.

    Copies contents and permission bits, like `cp`; logs created/replaced.
    """
    source, dest = Path(source), Path(dest)
    target = dest / source.name if dest.is_dir() else dest
    action = "replaced" if target.exists() else "created"
    try:
        shutil.copy(source, target)
        if mode is not None:
            os.chmod(target, mode)
    except OSError as exc:
        log_error(op, f"could not copy {source} to {target}: {exc}",
                  source=str(source), path=str(target))
        return False
    log_info(op, f"Copied {source} -> {target} ({action})",
             source=str(source), path=str(target), action=action)
    return True


def copy_files(pattern, dest_dir, op: str) -> bool:
    """bash `cp <glob> <dest_dir>/`: every file matching `pattern`.

    Like bash, a pattern that matches nothing is an error (cp gets the literal
    pattern and fails), so this logs an ERROR and returns False.
    """
    pattern = str(pattern)
    matches = sorted(glob.glob(pattern))
    if not matches:
        log_error(op, f"nothing matches {pattern} to copy", pattern=pattern)
        return False
    return all(copy_file(match, dest_dir, op) for match in matches)


def move_file(source, dest, op: str) -> bool:
    """bash `mv <source> <dest>` (dest may be a directory); logs created/replaced."""
    source, dest = Path(source), Path(dest)
    target = dest / source.name if dest.is_dir() else dest
    action = "replaced" if target.exists() else "created"
    try:
        shutil.move(str(source), str(target))
    except OSError as exc:
        log_error(op, f"could not move {source} to {target}: {exc}",
                  source=str(source), path=str(target))
        return False
    log_info(op, f"Moved {source} -> {target} ({action})",
             source=str(source), path=str(target), action=action)
    return True


def symlink(target, link, op: str, *, force: bool = False) -> bool:
    """bash `ln -s <target> <link>` (`ln -sf` with force=True)."""
    link = Path(link)
    try:
        if force and (link.is_symlink() or link.exists()):
            link.unlink()
        link.symlink_to(target)
    except OSError as exc:
        log_error(op, f"could not link {link} -> {target}: {exc}", path=str(link))
        return False
    log_info(op, f"Linked {link} -> {target}", path=str(link), target=str(target))
    return True


def file_has_line(path, line: str) -> bool:
    """bash `grep -qxF <line> <path>`: is `line` one of the file's lines?

    False if the file is missing or unreadable (grep's non-zero exit).
    """
    try:
        with open(path, encoding="utf-8", errors="surrogateescape", newline="") as fh:
            return any(existing.rstrip("\n") == line for existing in fh)
    except OSError:
        return False


def ensure_line(path, line: str, op: str) -> bool:
    """bash `grep -qxF <line> <path> || echo <line> >> <path>`."""
    if file_has_line(path, line):
        log_info(op, f"{path} already has the line", path=str(path), line=line)
        return True
    return write_file(path, line + "\n", op, append=True)


def edit_file(path, transform, op: str, *, create: bool = False) -> bool:
    """A `sed -i`: rewrite `path` with transform(text), keeping every other byte.

    `transform` takes and returns the whole file as a str. Line endings (\\r\\n
    stays \\r\\n) and bytes that aren't valid UTF-8 survive unchanged, as with
    sed. A missing file is an error unless create=True (then it starts empty).
    Logs whether the content changed.
    """
    path = Path(path)
    try:
        data = path.read_bytes() if (path.exists() or not create) else b""
    except OSError as exc:
        log_error(op, f"could not read {path}: {exc}", path=str(path))
        return False
    text = data.decode("utf-8", errors="surrogateescape")
    new_text = transform(text)
    changed = new_text != text
    if changed or not path.exists():
        try:
            path.write_bytes(new_text.encode("utf-8", errors="surrogateescape"))
        except OSError as exc:
            log_error(op, f"could not write {path}: {exc}", path=str(path))
            return False
    log_info(op, f"Edited {path} ({'changed' if changed else 'unchanged'})",
             path=str(path), changed=changed)
    return True