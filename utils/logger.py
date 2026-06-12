"""logger.py — shared logging helpers for azhpc-images (Python port of logger.sh).

Configuration (override via environment variables before import):
  LOG_FORMAT   text | json   (default: text if stdout is a TTY, else json)
  LOG_LEVEL    info | debug  (default: info)
  LOG_FILE     path          (default: a timestamped file under LOG_DIR)
  RUN_ID       string         (default: ADO BUILD_BUILDID, else a UUID)
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

# redaction (port of the two sed expressions in _redact)
_BEARER = re.compile(r"(bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE)
_TOKENS = re.compile(r"((?:sig|sas|token|key)=)[A-Za-z0-9._%\-]+", re.IGNORECASE)

def _redact(s: str) -> str:
    s = _BEARER.sub(r"\1[REDACTED]", s)
    s = _TOKENS.sub(r"\1[REDACTED]", s)
    return s

def _log(level: str, op: str, message: str) -> None:
    """Emit a single log record."""
    message = _redact(message)

    # debug messages only when LOG_LEVEL=debug
    if level == "debug" and LOG_LEVEL != "debug":
        return

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if LOG_FORMAT == "json":
        line = json.dumps({
            "ts": ts, "level": level, "op": op, "msg": message,
            "run_id": RUN_ID, "implementation": IMPLEMENTATION,
            "version": VERSION, "commit": BUILD_COMMIT,
            "vendor": VENDOR, "gpu": GPU, "os": OS,
            "fips": FIPS == "true",
        }, separators=(",", ":"))
        line_raw = line
    else:
        upper = level.upper()
        ts_short = ts[11:19]  # HH:MM:SS

        lvl_pretty = upper
        if sys.stdout.isatty() and level in _COLORS:
            lvl_pretty = f"{_COLORS[level]}{upper}{_RESET}"

        line = f"{ts_short}  {lvl_pretty:<14}  {op:<14} | {message}"
        line_raw = f"{ts}  {upper:<5}  {op:<14} | {message}"

    print(line)
    if LOG_FILE:
        # File gets the uncolored, full-timestamp version.
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(line_raw + "\n")

def log_info(op: str, message: str) -> None:
    """INFO: a high-level operation."""
    _log("info", op, message)


def log_warn(op: str, message: str) -> None:
    """WARN: unexpected, but the build continues."""
    _log("warn", op, message)


def log_debug(op: str, message: str) -> None:
    """DEBUG: only emitted when LOG_LEVEL=debug."""
    _log("debug", op, message)


def log_error(op: str, message: str) -> None:
    """ERROR: caller decides whether to exit."""
    _log("error", op, message)


def log_error_detail(op: str, message: str, detail: str) -> None:
    """ERROR plus a multi-line detail block (stderr / stack trace)."""
    if LOG_FORMAT == "json":
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(json.dumps({
            "ts": ts, "level": "error", "op": op,
            "msg": _redact(message), "error_detail": _redact(detail),
        }, separators=(",", ":")))
    else:
        log_error(op, message)
        for ln in detail.splitlines():
            print(f"    | {ln}")