"""process.py — the single gateway for running external programs.

External commands go through exec_program so their output and return code are
captured and logged consistently. Output is streamed line-by-line as it arrives
(via subprocess.Popen) and logged at debug level, so it appears live when
--verbose is set without waiting for the command to finish.

Child lifecycle: a build step can run for tens of minutes (make, apt), so the
child is started in its own process group and torn down if we're interrupted or
terminated. Without that, killing azhpc.py leaves the compiler or package
manager running as an orphan, still holding the dpkg lock.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from collections import deque

from utils.logger import log_info, log_debug, log_warn, log_error, log_error_detail

# How long to wait for a child to exit after SIGTERM before escalating.
_GRACE_SECONDS = 5

# How many of a command's last output lines to keep. On failure they are logged
# at error level, so the cause is visible even when debug output is switched off.
_TAIL_LINES = 50


def _kill_process_group(proc: subprocess.Popen, op: str) -> None:
    """Tear down `proc` and everything it spawned. SIGTERM, then SIGKILL.

    Signals the process *group*, not just the pid: `make -j` and the install
    scripts fork freely, and killing only the direct child would leave the
    compilers behind. exec_program starts every child in its own group so this
    can never reach back into our own process.
    """
    if proc.poll() is not None:
        return                                  # already exited, nothing to do

    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        return

    log_warn(op, f"interrupted — terminating process group {pgid}")
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        # Ignored SIGTERM (or is wedged in uninterruptible IO). Stop asking.
        log_warn(op, f"process group {pgid} ignored SIGTERM; sending SIGKILL")
        try:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=_GRACE_SECONDS)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    except ProcessLookupError:
        pass                                    # exited between poll and killpg


def exec_program(command: list[str], op: str, *, cwd: str | None = None, env = None,
                 failure_is_error: bool = True) -> int:
    """Run an external program, streaming and logging its output.

    stdout and stderr are merged and read line-by-line so each line is logged
    (at debug/verbose level) as it is produced. Returns the program's exit code,
    or 127 if it could not be executed.

    Pass failure_is_error=False when a non-zero exit is a normal answer rather
    than a failure (see _log_result), so build logs don't show a false ERROR.

    If we're interrupted (Ctrl-C) or terminated while the child is running, the
    child's whole process group is torn down before we exit.
    """
    log_info(op, f"Running command: {' '.join(command)}")
    started = time.monotonic()

    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,                      # pass the environment variables
            stdin=subprocess.DEVNULL,     # no terminal: stops dpkg drawing progress
            stdout=subprocess.PIPE,       # capture stdout...
            stderr=subprocess.STDOUT,     # ...and fold stderr into it
            text=True,                    # decode bytes -> str
            bufsize=1,                    # line-buffered for live streaming
            start_new_session=True,       # own process group, so we can killpg it
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        # The program could not be started at all (no return code exists).
        log_error(op, f"Failed to execute {' '.join(command)}: {exc}",
                  command=command[0], exit_code=127)
        return 127

    def _on_signal(signum, _frame):
        # Take the child down first, then die the way we were asked to: restore
        # the default handler and re-raise, so our exit status is the usual
        # 128+signum rather than something invented.
        _kill_process_group(proc, op)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    # start_new_session detaches the child from the terminal's process group, so
    # Ctrl-C no longer reaches it on its own - we have to forward it ourselves.
    # signal.signal only works on the main thread; if we're not on it (a test
    # runner, say) skip the handlers and rely on the finally block below.
    previous = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, _on_signal)
    except ValueError:
        previous.clear()

    # The last few output lines, kept so a failure can show them (see below).
    # A deque with maxlen drops the oldest line once it is full.
    tail: deque[str] = deque(maxlen=_TAIL_LINES)

    try:
        # Stream output live: each line is logged as soon as it arrives.
        assert proc.stdout is not None
        for line in proc.stdout:
            # Tools like apt redraw progress with carriage returns instead of
            # newlines, so a single line can carry many '\r'-separated updates.
            # Keep only the final segment (the last redraw) and drop blank ones.
            text = line.split("\r")[-1].rstrip()
            if text:
                log_debug(op, text)
                tail.append(text)

        returncode = proc.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        # Covers the paths a signal handler can't: an exception mid-stream, or
        # the caller giving up on us. Never leave the child running.
        _kill_process_group(proc, op)
        proc.stdout.close()           # release the pipe (avoids a ResourceWarning)

    _log_result(op, command, returncode, time.monotonic() - started, tail,
                failure_is_error)
    return returncode


def _log_result(op: str, command: list[str], returncode: int, elapsed: float,
                tail, failure_is_error: bool = True) -> None:
    """Log how a finished command went, for build telemetry.

    Success is one INFO line with the exit code and duration. Failure is an
    ERROR that includes the command's last output lines, so the cause shows up
    even at the default INFO level, when the full output was only logged at
    debug level.

    With failure_is_error=False the caller is asking a yes/no question (e.g.
    "is this package installed?"), so a non-zero exit is an answer, not a
    failure: it is logged at INFO instead, and nothing shows up as an ERROR.
    """
    fields = {"command": command[0], "exit_code": returncode,
              "duration_s": round(elapsed, 1)}
    if returncode == 0:
        log_info(op, f"Command succeeded in {elapsed:.1f}s: {command[0]}", **fields)
        return
    if not failure_is_error:
        log_info(op, f"Command exited with code {returncode} after {elapsed:.1f}s "
                     f"(an expected answer for this check): {' '.join(command)}",
                 **fields)
        return
    log_error_detail(
        op,
        f"Command failed with exit code {returncode} after {elapsed:.1f}s: "
        f"{' '.join(command)}",
        "\n".join(tail) if tail else "(no output)",
        **fields)


def run_capture(command: list[str], op: str, *, cwd: str | None = None, env=None,
                failure_is_error: bool = True) -> tuple[int, str]:
    """Run an external program and return its exit code *and* captured stdout.

    Unlike exec_program (which streams output to the log and returns only the
    exit code), this captures stdout so callers can parse it — needed for the few
    commands whose output is data, e.g. `ofed_info` or `apt-cache show openmpi`.
    stderr is kept separate (logged, not returned) so it can't corrupt the parse.
    Returns (127, "") if the program could not be executed.

    Pass failure_is_error=False when a non-zero exit is a normal answer (a
    probe such as `dnf list installed <pkg>`), so it isn't logged as an ERROR.
    """
    log_info(op, f"Running command: {' '.join(command)}")
    started = time.monotonic()

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
        log_error(op, f"Failed to execute {' '.join(command)}: {exc}",
                  command=command[0], exit_code=127)
        return (127, "")

    # Log everything the command produced (stdout + stderr) at debug level, but
    # only stdout is returned for parsing.
    tail: deque[str] = deque(maxlen=_TAIL_LINES)
    for stream in (proc.stdout, proc.stderr):
        for line in (stream or "").splitlines():
            if line.strip():
                log_debug(op, line.rstrip())
                tail.append(line.rstrip())

    _log_result(op, command, proc.returncode, time.monotonic() - started, tail,
                failure_is_error)
    return (proc.returncode, proc.stdout or "")

# --- Convenience wrappers --------------------------------------------------
# Most ports only need "did it work?". These keep that one line long, so a
# port reads as what it does rather than how it checks exit codes.

def run(command: list[str], op: str, *, cwd: str | None = None, env=None) -> bool:
    """Run one command; True if it exited 0.

    exec_program logs the command, its exit code and, on failure, its last
    output lines, so callers only need to say what the failure means.
    """
    return exec_program(command, op, cwd=cwd, env=env) == 0


def run_all(commands: list[list[str]], op: str, *, cwd: str | None = None,
            env=None) -> bool:
    """Run commands in order, stopping at the first failure (bash's `set -e`)."""
    for command in commands:
        if not run(command, op, cwd=cwd, env=env):
            return False
    return True


def run_script(lines: list[str], op: str, *, modules: list[str] | tuple[str, ...] = (),
               cwd: str | None = None, env=None) -> bool:
    """Run shell lines as one `bash -c` script with `set -e`.

    For the steps that must share one shell, typically because they follow
    `module load` (an environment-modules shell function, so it only affects
    the shell it runs in). With `modules`, the script starts with
        source /etc/profile.d/modules.sh
        module load <module>...
    Bash's matching `module unload` lines are not needed: the shell exits.
    """
    header = ["set -e"]
    if modules:
        header.append("source /etc/profile.d/modules.sh")
        header += [f"module load {module}" for module in modules]
    script = "\n".join(header + list(lines)) + "\n"
    return run(["bash", "-c", script], op, cwd=cwd, env=env)


def command_exists(name: str, env=None) -> bool:
    """bash `command -v <name>`: True if `name` is on the build env's PATH."""
    return command_path(name, env) is not None

def command_path(name: str, env=None) -> str | None:
    """bash `$(command -v <name>)`: its full path on the build env's PATH, or None."""
    return shutil.which(name, path=(env or os.environ).get("PATH"))


def cpu_count() -> int:
    """bash `$(nproc)`: the CPUs this process may run on (its affinity set)."""
    return len(os.sched_getaffinity(0))