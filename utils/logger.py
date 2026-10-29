"""logger.py — shared logging helpers for azhpc-images (Python port of logger.sh).

Configuration (override via environment variables before import):
  LOG_FORMAT   text | json                 (default: text if stdout is a TTY, else json)
  LOG_LEVEL    debug | info | warn | error  (default: info)
  LOG_FILE     path                         (default: a timestamped file under LOG_DIR)
  RUN_ID       string         (default: ADO BUILD_BUILDID, else a UUID)

Every log helper also accepts optional keyword fields for telemetry, e.g.
    log_info("install-pmix", "Recorded version", component="PMIX", version="4.2.9")
JSON records carry them under "data"; text records append them as key=value.
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


LOG_FORMAT = os.environ.get("LOG_FORMAT", "")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info")
LOG_FILE = os.environ.get("LOG_FILE", "")
RUN_ID = os.environ.get("RUN_ID") or os.environ.get("BUILD_BUILDID") or str(uuid.uuid4())
IMPLEMENTATION = os.environ.get("IMPLEMENTATION", "python")
VERSION = os.environ.get("VERSION", "")
BUILD_COMMIT = os.environ.get("BUILD_COMMIT", "")
VENDOR = os.environ.get("VENDOR", "")
GPU = os.environ.get("GPU", "")
OS = os.environ.get("OS", "")
FIPS = os.environ.get("FIPS", "false")

# When no LOG_FILE was provided, give this run its own timestamped log file.
if not LOG_FILE:
    LOG_DIR = os.environ.get("LOG_DIR", "/var/log/azhpc")
    try:
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        LOG_FILE = f"{LOG_DIR}/azhpc_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    except OSError:
        LOG_FILE = ""  # couldn't create the dir

# text if stdout is a TTY, else json
if not LOG_FORMAT:
    LOG_FORMAT = "text" if sys.stdout.isatty() else "json"

# ANSI colors for the level, only when stdout is a TTY
_RESET = "\033[0m"
_COLORS = {
    "debug": "\033[2;37m",  # dim grey
    "info": "\033[36m",     # cyan
    "warn": "\033[33m",     # yellow
    "error": "\033[31m",    # red
}

# Severity ranking. A message is emitted only when its own level is at or above
# the configured LOG_LEVEL threshold, e.g. LOG_LEVEL=warn shows warn + error.
_LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}

def _enabled(level: str) -> bool:
    """True when `level` is at or above the configured LOG_LEVEL threshold.

    Unknown values (for either the message level or LOG_LEVEL) fall back to
    'info' so a typo never silences the logs entirely.
    """
    threshold = _LEVELS.get(LOG_LEVEL.lower(), _LEVELS["info"])
    return _LEVELS.get(level, _LEVELS["info"]) >= threshold

def configure(*, log_format=None, log_level=None, version="", commit="",
              vendor="", gpu="", os="", fips=False) -> None:
    """Set logger configuration explicitly (call once at startup)."""
    global LOG_FORMAT, LOG_LEVEL, VERSION, BUILD_COMMIT, VENDOR, GPU, OS, FIPS
    if log_format:
        LOG_FORMAT = log_format
    if log_level:
        LOG_LEVEL = log_level
    VERSION, BUILD_COMMIT = version, commit
    VENDOR, GPU, OS, FIPS = vendor, gpu, os, ("true" if fips else "false")

# redaction (port of the two sed expressions in _redact)
_BEARER = re.compile(r"(bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE)
_TOKENS = re.compile(r"((?:sig|sas|token|key)=)[A-Za-z0-9._%\-]+", re.IGNORECASE)

def _redact(s: str) -> str:
    s = _BEARER.sub(r"\1[REDACTED]", s)
    s = _TOKENS.sub(r"\1[REDACTED]", s)
    return s

# Each structured field is also shown in text logs, as key=value. Values that
# aren't plain JSON types (e.g. a Path) are converted with str().
_JSON_TYPES = (str, int, float, bool, type(None))


def _clean_fields(fields: dict) -> dict:
    """Make extra fields safe to log: JSON-friendly values, secrets redacted."""
    clean = {}
    for key, value in fields.items():
        if not isinstance(value, _JSON_TYPES):
            value = str(value)
        if isinstance(value, str):
            value = _redact(value)
        clean[key] = value
    return clean


def _write(line: str, line_raw: str) -> None:
    """Print `line` and append `line_raw` (no colors, full timestamp) to LOG_FILE."""
    print(line)
    if LOG_FILE:
        # If the log file isn't writable (e.g. /var/log/azhpc without root),
        # degrade gracefully — keep printing instead of crashing the caller.
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(line_raw + "\n")
        except OSError:
            pass


def _log(level: str, op: str, message: str, fields: dict | None = None,
         detail: str | None = None) -> None:
    """Emit a single log record.

    `fields` are optional structured values (e.g. component="pmix",
    version="4.2.9"). In JSON they go under "data" so a log query can filter on
    them; in text they're appended as key=value. `detail` is an optional
    multi-line block (command output, a stack trace), shown under the message.
    """
    # Skip anything below the configured LOG_LEVEL threshold.
    if not _enabled(level):
        return

    message = _redact(message)
    fields = _clean_fields(fields or {})
    if detail is not None:
        detail = _redact(detail)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if LOG_FORMAT == "json":
        record = {
            "ts": ts, "level": level, "op": op, "msg": message,
            "run_id": RUN_ID, "implementation": IMPLEMENTATION,
            "version": VERSION, "commit": BUILD_COMMIT,
            "vendor": VENDOR, "gpu": GPU, "os": OS,
            "fips": FIPS == "true",
        }
        if fields:
            record["data"] = fields
        if detail is not None:
            record["error_detail"] = detail
        line = json.dumps(record, separators=(",", ":"))
        _write(line, line)
        return

    upper = level.upper()
    ts_short = ts[11:19]  # HH:MM:SS

    lvl_pretty = upper
    if sys.stdout.isatty() and level in _COLORS:
        lvl_pretty = f"{_COLORS[level]}{upper}{_RESET}"

    suffix = ""
    if fields:
        suffix = "  [" + " ".join(f"{key}={value}" for key, value in fields.items()) + "]"

    _write(f"{ts_short}  {lvl_pretty:<14}  {op:<14} | {message}{suffix}",
           f"{ts}  {upper:<5}  {op:<14} | {message}{suffix}")
    for detail_line in (detail or "").splitlines():
        _write(f"    | {detail_line}", f"    | {detail_line}")


# Every helper takes optional keyword fields, e.g.
#     log_info("install-pmix", "Installing PMIx", version="4.2.9", distro="ubuntu24.04")
def log_info(op: str, message: str, **fields) -> None:
    """INFO: a high-level operation."""
    _log("info", op, message, fields)


def log_warn(op: str, message: str, **fields) -> None:
    """WARN: unexpected, but the build continues."""
    _log("warn", op, message, fields)


def log_debug(op: str, message: str, **fields) -> None:
    """DEBUG: only emitted when LOG_LEVEL=debug."""
    _log("debug", op, message, fields)


def log_error(op: str, message: str, **fields) -> None:
    """ERROR: caller decides whether to exit."""
    _log("error", op, message, fields)


def log_error_detail(op: str, message: str, detail: str, **fields) -> None:
    """ERROR plus a multi-line detail block (command output, stack trace).

    JSON records carry the same run/vendor/os fields as every other record, and
    both formats are written to LOG_FILE, detail included.
    """
    _log("error", op, message, fields, detail)