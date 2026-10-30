"""download.py — download a file and verify its SHA256 checksum.

Native Python port of download_and_verify / verify_checksum in
utils/utilities.sh. It uses hashlib and urllib instead of shelling out to
sha256sum, awk, and wget, so there are no pipes or subprocesses involved and the
checksum logic is unit-testable.
"""

from __future__ import annotations

import hashlib
import shutil
import time
import urllib.request
from pathlib import Path

from utils.logger import log_info, log_error

_CHUNK = 1 << 16  # read/write in 64 KiB chunks (memory-safe for large files)


class ChecksumError(Exception):
    """Raised when a downloaded file's SHA256 does not match the expected value."""


def sha256_of(path) -> str:
    """Return the hex SHA256 of the file at `path`, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path, expected_sha256) -> bool:
    """True if the file's SHA256 matches `expected_sha256` (case-insensitive)."""
    actual = sha256_of(path)
    if actual.lower() == expected_sha256.lower():
        log_info("verify-checksum", f"Checksum verified for {path}")
        return True
    log_error("verify-checksum",
              f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual}")
    return False


def download(url, dest, *, tries: int = 3, wait: int = 5) -> None:
    """Download `url` to `dest`, retrying on failure (mirrors wget --tries)."""
    last_error: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            log_info("download", f"Downloading {url} -> {dest} (attempt {attempt}/{tries})")
            with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
                shutil.copyfileobj(response, out, _CHUNK)
            return
        except OSError as exc:  # URLError/HTTPError are subclasses of OSError
            last_error = exc
            log_error("download", f"Download failed: {exc}")
            if attempt < tries:
                time.sleep(wait)
    raise last_error  # type: ignore[misc]


def download_and_verify(url, sha256, dest_dir=".") -> Path:
    """Download `url` into `dest_dir` and verify its SHA256.

    Returns the path to the downloaded file. Raises ChecksumError if the
    checksum does not match.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(url).name

    download(url, dest)

    if not verify_checksum(dest, sha256):
        raise ChecksumError(f"SHA256 verification failed for {dest}")

    return dest


def fetch_verified(url, sha256, op: str, dest_dir=".") -> Path | None:
    """download_and_verify() for ports: the file's Path, or None on failure.

    Network errors, file errors and checksum mismatches are logged as an ERROR
    under the step's `op` label (with the URL), so the caller only checks for
    None and returns 3, instead of wrapping each download in try/except.
    """
    try:
        return download_and_verify(url, sha256, dest_dir=dest_dir)
    except (OSError, ValueError, ChecksumError) as exc:   # ValueError: malformed URL
        log_error(op, f"download of {url} failed: {exc}", url=url)
        return None


def fetch(url, op: str, dest_dir=".") -> Path | None:
    """Download `url` into `dest_dir` with no checksum: the Path, or None (logged).

    Only for files bash also downloads unverified (e.g. GitHub source
    tarballs, which have no published checksum). Prefer fetch_verified().
    """
    dest_dir = Path(dest_dir)
    dest = dest_dir / Path(url).name
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        download(url, dest)
    except (OSError, ValueError) as exc:
        log_error(op, f"download of {url} failed: {exc}", url=url)
        return None
    return dest