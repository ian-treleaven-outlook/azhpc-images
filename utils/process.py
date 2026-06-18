"""process.py — the single gateway for running external programs.

Every external command goes through exec_program so its stdout, stderr, and
return code are captured and logged consistently. Detailed output is logged at
debug level, so it only appears when --verbose is set.
"""

from __future__ import annotations

import subprocess

from utils.logger import log_info, log_debug, log_error

def exec_program(command: list[str], op: str, *, cwd: str | None = None, env = None) -> int:
    """Run an external program and log its output.

    Captures stdout and stderr (logged at debug/verbose level) and the return
    code. Returns the program's exit code, or 127 if it could not be executed.
    """
    log_info(op, f"Running command: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,   # capture both streams so we can log them
            text=True,             # decode bytes -> str
            env=env,               # pass the environment variables
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        # The program could not be started at all (no return code exists).
        log_error(op, f"Failed to execute {' '.join(command)}: {exc}")
        return 127

    # Only when --verbose is set 
    if result.stdout:
        log_debug(op, f"stdout:\n{result.stdout.rstrip()}")
    if result.stderr:
        log_debug(op, f"stderr:\n{result.stderr.rstrip()}")
    
    log_debug(op, f"Command exited with code {result.returncode}")

    if result.returncode != 0:
        log_error(op, f"Command failed with exit code {result.returncode}")

    return result.returncode