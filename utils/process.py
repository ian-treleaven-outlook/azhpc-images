"""process.py — the single gateway for running external programs.

Every external command goes through exec_program so its output and return code
are captured and logged consistently. Output is streamed line-by-line as it
arrives (via subprocess.Popen) and logged at debug level, so it appears live
when --verbose is set without waiting for the command to finish.
"""

from __future__ import annotations

import subprocess

from utils.logger import log_info, log_debug, log_error

def exec_program(command: list[str], op: str, *, cwd: str | None = None, env = None) -> int:
    """Run an external program, streaming and logging its output.

    stdout and stderr are merged and read line-by-line so each line is logged
    (at debug/verbose level) as it is produced. Returns the program's exit code,
    or 127 if it could not be executed.
    """
    log_info(op, f"Running command: {' '.join(command)}")

    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,                      # pass the environment variables
            stdout=subprocess.PIPE,       # capture stdout...
            stderr=subprocess.STDOUT,     # ...and fold stderr into it
            text=True,                    # decode bytes -> str
            bufsize=1,                    # line-buffered for live streaming
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        # The program could not be started at all (no return code exists).
        log_error(op, f"Failed to execute {' '.join(command)}: {exc}")
        return 127

    # Stream output live: each line is logged as soon as it arrives.
    assert proc.stdout is not None
    for line in proc.stdout:
        # Tools like apt redraw progress with carriage returns instead of
        # newlines, so a single line can carry many '\r'-separated updates.
        # Keep only the final segment (the last redraw) and drop blank ones.
        text = line.split("\r")[-1].rstrip()
        if text:
            log_debug(op, text)

    returncode = proc.wait()
    log_debug(op, f"Command exited with code {returncode}")

    if returncode != 0:
        log_error(op, f"Command failed with exit code {returncode}")

    return returncode


def run_capture(command: list[str], op: str, *, cwd: str | None = None, env=None) -> tuple[int, str]:
    """Run an external program and return its exit code *and* captured stdout.

    Unlike exec_program (which streams output to the log and returns only the
    exit code), this captures stdout so callers can parse it — needed for the few
    commands whose output is data, e.g. `ofed_info` or `apt-cache show openmpi`.
    stderr is kept separate (logged, not returned) so it can't corrupt the parse.
    Returns (127, "") if the program could not be executed.
    """
    log_info(op, f"Running command: {' '.join(command)}")

    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,      # collect stdout and stderr separately
            text=True,                # decode bytes -> str
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        # The program could not be started at all (no return code exists).
        log_error(op, f"Failed to execute {' '.join(command)}: {exc}")
        return (127, "")

    # Log everything the command produced (stdout + stderr) at debug level, but
    # only stdout is returned for parsing.
    for stream in (proc.stdout, proc.stderr):
        for line in (stream or "").splitlines():
            if line.strip():
                log_debug(op, line.rstrip())

    log_debug(op, f"Command exited with code {proc.returncode}")
    if proc.returncode != 0:
        log_error(op, f"Command failed with exit code {proc.returncode}")

    return (proc.returncode, proc.stdout or "")