"""copy_test_file.py — Python port of components/copy_test_file.sh.

Copies the repo's test scripts into the image (/opt/azurehpc/test) so they
can be run on a finished VM: the top-level files of tests/ (`cp tests/*.*`),
then the whole tests/health_checks tree (`cp -r`). Where the CUDA compiler
/usr/local/cuda/bin/nvcc exists, it first compiles
health_checks/NDv4/gpu-copy.cu into the *repo's* tree, next to its source,
so the tree copy carries the binary along.

The source is AZHPC_IMAGES_TEST_DIR, the variable bash reads; component_env()
sets it to <repo>/tests, as utils/set_properties.sh does for bash.

Kept bash behaviour:
  * nvcc needs a host gcc newer than 6 (bash: `[ $(gcc -dumpversion | cut
    -d. -f1) -gt 6 ]`); otherwise it is pointed at /opt/gcc-9.2.0/bin.
  * `*.*` skips dotfiles and matches directories too; cp without -r fails on
    a directory, and so does this step. When nothing matches, bash's cp gets
    the literal "*.*" and fails; so does this step.

How the copies compare with cp:
  * `cp files dir` copies content and permission bits, following symlinks;
    shutil.copy does the same. Over an existing file, cp keeps that file's
    permissions where shutil.copy sets the source's.
  * `cp -r` is shutil.copytree(symlinks=True, dirs_exist_ok=True,
    copy_function=shutil.copy): links stay links, existing trees are merged,
    file timestamps are not copied. copytree also copies directory modes and
    timestamps, which nothing here depends on. On a rerun cp -r replaces an
    existing symlink where copytree fails; tests/health_checks has no
    symlinks (a unit test checks this), so that can't happen today.

Deliberate differences:
  * A missing AZHPC_IMAGES_TEST_DIR fails the step (bash would copy "/*.*").
  * Unusable `gcc -dumpversion` output fails the step. Bash's `[ ... -gt 6 ]`
    errors, the `if` treats that as false, and nvcc is silently pointed at
    /opt/gcc-9.2.0/bin, which may not exist.
"""

from __future__ import annotations

import os
import re
import shutil

from utils.logger import log_error, log_info
from utils.process import run, run_capture

_OP = "copy-test-file"

_DEST_TEST_DIR = "/opt/azurehpc/test"
_NVCC = "/usr/local/cuda/bin/nvcc"

# nvcc needs a host gcc newer than this; older ones use the gcc 9.2 in /opt.
_NEWEST_UNSUPPORTED_GCC = 6
_FALLBACK_GCC_BINDIR = "/opt/gcc-9.2.0/bin"

# Paths inside the source tests directory.
_HEALTH_CHECKS = "health_checks"
_GPU_COPY_SOURCE = os.path.join(_HEALTH_CHECKS, "NDv4", "gpu-copy.cu")
_GPU_COPY_BINARY = os.path.join(_HEALTH_CHECKS, "NDv4", "gpu-copy")


def top_level_test_files(names: list[str]) -> list[str]:
    """The names the shell glob `*.*` matches, sorted.

    A name matches if it contains a dot and doesn't start with one (bash skips
    dotfiles without `dotglob`), whether it is a file or a directory:

        ["run-tests.sh", "python", ".hidden.sh"]  ->  ["run-tests.sh"]
    """
    return sorted(name for name in names
                  if "." in name and not name.startswith("."))


def gcc_major_version(gcc_output: str) -> int | None:
    """`gcc -dumpversion | cut -d. -f1` as a number ("9.4.0\\n" -> 9).

    None unless the output is one line starting with digits.
    """
    major = gcc_output.strip().partition(".")[0]
    if not re.fullmatch(r"[0-9]+", major):
        return None
    return int(major)


def nvcc_command(nvcc: str, gcc_major: int, cufile: str, outfile: str) -> list[str]:
    """The nvcc command bash runs for this gcc version."""
    if gcc_major > _NEWEST_UNSUPPORTED_GCC:
        return [nvcc, "-lnuma", cufile, "-o", outfile]
    return [nvcc, "--compiler-bindir", _FALLBACK_GCC_BINDIR,
            "-lnuma", cufile, "-o", outfile]


def _make_dest_dir() -> bool:
    """bash `mkdir -p /opt/azurehpc/test`."""
    try:
        os.makedirs(_DEST_TEST_DIR, exist_ok=True)
    except OSError as exc:
        log_error(_OP, f"could not create {_DEST_TEST_DIR}: {exc}", path=_DEST_TEST_DIR)
        return False
    return True


def _copy_top_level_files(test_dir: str) -> bool:
    """bash `cp $AZHPC_IMAGES_TEST_DIR/*.* $DEST_TEST_DIR`."""
    try:
        names = top_level_test_files(os.listdir(test_dir))
        if not names:
            log_error(_OP, f"no files matching *.* in {test_dir}")
            return False
        for name in names:
            # Raises IsADirectoryError for a directory, as cp without -r fails.
            shutil.copy(os.path.join(test_dir, name), _DEST_TEST_DIR)
    except OSError as exc:
        log_error(_OP, f"could not copy test files from {test_dir} "
                       f"to {_DEST_TEST_DIR}: {exc}")
        return False
    log_info(_OP, f"Copied {', '.join(names)} to {_DEST_TEST_DIR}")
    return True


def _compile_gpu_copy(test_dir: str, env: dict[str, str]) -> bool:
    """Compile gpu-copy.cu with nvcc, choosing the host gcc as bash does."""
    rc, output = run_capture(["gcc", "-dumpversion"], _OP, env=env)
    if rc != 0:
        log_error(_OP, "could not determine the default gcc version, "
                       "needed to choose nvcc's host compiler")
        return False
    gcc_major = gcc_major_version(output)
    if gcc_major is None:
        log_error(_OP, f"unexpected gcc -dumpversion output: {output.strip()!r}")
        return False
    return run(nvcc_command(_NVCC, gcc_major,
                            os.path.join(test_dir, _GPU_COPY_SOURCE),
                            os.path.join(test_dir, _GPU_COPY_BINARY)),
               _OP, env=env)


def _copy_health_checks(test_dir: str) -> bool:
    """bash `cp -r $AZHPC_IMAGES_TEST_DIR/health_checks $DEST_TEST_DIR`."""
    source = os.path.join(test_dir, _HEALTH_CHECKS)
    destination = os.path.join(_DEST_TEST_DIR, _HEALTH_CHECKS)
    # These options make copytree behave like `cp -r` (module docstring).
    try:
        shutil.copytree(source, destination, symlinks=True,
                        copy_function=shutil.copy, dirs_exist_ok=True)
    except OSError as exc:  # includes shutil.Error
        log_error(_OP, f"could not copy {source} to {_DEST_TEST_DIR}: {exc}")
        return False
    log_info(_OP, f"Copied {source} to {_DEST_TEST_DIR}")
    return True


def install(env: dict[str, str]) -> int:
    """Copy the repo's tests into the image, building gpu-copy if nvcc exists."""
    test_dir = env.get("AZHPC_IMAGES_TEST_DIR", "")
    if not test_dir:
        log_error(_OP, "AZHPC_IMAGES_TEST_DIR is not set")
        return 3

    if not _make_dest_dir():
        return 3
    if not _copy_top_level_files(test_dir):
        return 3
    if os.path.isfile(_NVCC):                       # bash `test -f`
        if not _compile_gpu_copy(test_dir, env):
            return 3
    else:
        log_info(_OP, f"{_NVCC} not found; not building gpu-copy")
    if not _copy_health_checks(test_dir):
        return 3
    return 0
