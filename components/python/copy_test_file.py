"""copy_test_file.py — Python port of components/copy_test_file.sh.

What this step does
-------------------
Copies the image's test scripts from the repository into the image, so they
can be run on a finished VM:

  1. Creates /opt/azurehpc/test.
  2. Copies the top-level files of the repo's tests/ directory into it
     (run-tests.sh, test-definitions.sh, test-matrix_*.json).
  3. If the CUDA compiler (nvcc) is installed, compiles the gpu-copy benchmark,
     tests/health_checks/NDv4/gpu-copy.cu, into tests/health_checks/NDv4/gpu-copy.
     The binary is written into the *repo's* tests directory, next to its
     source, so step 4 copies it along with the scripts.
  4. Copies the whole tests/health_checks directory into /opt/azurehpc/test.

The original bash, for comparison:

    set -e

    DEST_TEST_DIR=/opt/azurehpc/test

    mkdir -p $DEST_TEST_DIR

    cp $AZHPC_IMAGES_TEST_DIR/*.* $DEST_TEST_DIR

    #Test if nvcc is installed and if so install gpu-copy test.
    if test -f "/usr/local/cuda/bin/nvcc"; then
        #Compile the gpu-copy benchmark.
        NVCC=/usr/local/cuda/bin/nvcc
        cufile="$AZHPC_IMAGES_TEST_DIR/health_checks/NDv4/gpu-copy.cu"
        outfile="$AZHPC_IMAGES_TEST_DIR/health_checks/NDv4/gpu-copy"

        #Test if the default gcc compiler is new enough to compile gpu-copy.
        #If it is not then use the 9.2 compiler, that should be installed in
        #/opt.
        if [ $(gcc -dumpversion | cut -d. -f1) -gt 6 ]; then
            $NVCC -lnuma $cufile -o $outfile
        else
            $NVCC --compiler-bindir /opt/gcc-9.2.0/bin \\
                -lnuma $cufile -o $outfile
        fi
    fi
    cp -r $AZHPC_IMAGES_TEST_DIR/health_checks $DEST_TEST_DIR

    exit 0

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"copy-test-file" Step. When a Step has an action, the build calls the Python
function instead of running the bash script. Deleting `action=` from that Step
makes the build run the bash script again. That is why the .sh file is kept.

The source directory comes from AZHPC_IMAGES_TEST_DIR, the variable bash
reads. In the bash build, utils/set_properties.sh sets it to <repo>/tests;
component_env() in build_config.py now sets it to the same value, for this
action and for the bash fallback.

Like every action, install() receives the build environment (a dict of
environment variables) and returns 0 for success or 3 for failure.

How the copies compare with cp
------------------------------
  * `cp a b dir` (no -r, no -p) copies each file's content and permission
    bits; it follows symbolic links and copies what they point to. shutil.copy
    does the same. One small difference: when the destination file already
    exists, cp keeps that file's permissions, while shutil.copy sets them to
    the source file's. Neither copies owner or timestamps.
  * `cp -r src dir` copies a directory tree. Symbolic links inside it are
    recreated as links (not followed), and if dir/src already exists the
    files are merged into it. shutil.copytree with symlinks=True and
    dirs_exist_ok=True does the same. We pass copy_function=shutil.copy so
    files get content and permissions only, like cp; copytree's default
    (shutil.copy2) would also copy timestamps. copytree also copies each
    directory's permissions and timestamps, which cp -r does not; nothing
    depends on directory timestamps here.
  * On a rerun, cp -r replaces a symbolic link that already exists in the
    destination; copytree fails on it ("File exists"), so the step returns 3.
    The repo's tests/health_checks contains no symbolic links (a unit test
    checks this), so this can't happen today.

Deliberate differences from bash
--------------------------------
  * A missing AZHPC_IMAGES_TEST_DIR fails the step. Bash would expand it to ""
    and try to copy "/*.*" from the root directory.
  * If `gcc -dumpversion` fails or prints something that isn't a version, the
    step fails. In bash, the `[ ... -gt 6 ]` test errors out, which an `if`
    treats as false, so bash silently compiles with /opt/gcc-9.2.0/bin, which
    may not exist.
"""

from __future__ import annotations

import os
import re
import shutil

from utils.logger import log_error, log_info
from utils.process import exec_program, run_capture

# The Step's `op` label in build_config.py. Every log line from this module uses
# it, so this step's output can be found in a build log.
_OP = "copy-test-file"

# Where the tests are copied to in the image. Like the other paths below, it is
# a module-level "constant" (by convention, names in capitals are not changed
# while the program runs) so the tests can point it at a temporary directory.
_DEST_TEST_DIR = "/opt/azurehpc/test"

# The CUDA compiler. The gpu-copy benchmark is only built when this file exists.
_NVCC = "/usr/local/cuda/bin/nvcc"

# The command that prints the default gcc's version, e.g. "13" or "9.4.0".
_GCC_VERSION_COMMAND = ("gcc", "-dumpversion")

# nvcc needs a host compiler newer than gcc 6 (bash: `-gt 6`). With an older
# default gcc, bash points nvcc at the gcc 9.2 installed under /opt.
_NEWEST_UNSUPPORTED_GCC = 6
_FALLBACK_GCC_BINDIR = "/opt/gcc-9.2.0/bin"

# Paths inside the source tests directory.
_HEALTH_CHECKS = "health_checks"
_GPU_COPY_SOURCE = os.path.join(_HEALTH_CHECKS, "NDv4", "gpu-copy.cu")
_GPU_COPY_BINARY = os.path.join(_HEALTH_CHECKS, "NDv4", "gpu-copy")


def top_level_test_files(names: list[str]) -> list[str]:
    """Pick the names the shell glob `*.*` would match, sorted.

    In bash, `*.*` matches names that contain a dot, but not names that start
    with one: hidden "dotfiles" such as ".gitignore" are skipped unless bash's
    `dotglob` option is on (it is not, in the original script). The glob
    doesn't care whether a name is a file or a directory; see install() for
    what happens with directories.

        ["run-tests.sh", "python", ".hidden.sh", "test-matrix_AMD.json"]
            ->  ["run-tests.sh", "test-matrix_AMD.json"]

    This is a "pure" function: it only transforms its input, so tests can call
    it directly with real names.
    """
    # A "list comprehension": builds a new list from the names that pass the
    # `if` test. sorted() returns them in a stable order, like bash's glob.
    return sorted(name for name in names
                  if "." in name and not name.startswith("."))


def gcc_major_version(gcc_output: str) -> int | None:
    """Turn `gcc -dumpversion` output into its major version number.

    Mirrors bash's `gcc -dumpversion | cut -d. -f1`, which keeps the text
    before the first dot:

        "13\\n"     ->  13
        "9.4.0\\n"  ->  9

    Returns None when the output isn't one line starting with a number (for
    example when gcc printed nothing). Bash's `[ ... -gt 6 ]` would fail on
    such a value; see the module docstring.
    """
    # .strip() removes the trailing newline, as bash's $(...) does.
    # .partition(".") splits at the first dot into (before, ".", after); [0]
    # takes the part before it, like `cut -d. -f1`.
    major = gcc_output.strip().partition(".")[0]
    # re.fullmatch succeeds only if the whole string matches: one or more
    # digits 0-9. That also rejects multi-line output.
    if not re.fullmatch(r"[0-9]+", major):
        return None
    return int(major)


def nvcc_command(nvcc: str, gcc_major: int, cufile: str, outfile: str) -> list[str]:
    """The nvcc command bash runs for this gcc version, as a list of arguments."""
    if gcc_major > _NEWEST_UNSUPPORTED_GCC:
        return [nvcc, "-lnuma", cufile, "-o", outfile]
    return [nvcc, "--compiler-bindir", _FALLBACK_GCC_BINDIR,
            "-lnuma", cufile, "-o", outfile]


def _copy_top_level_files(test_dir: str) -> bool:
    """`cp $AZHPC_IMAGES_TEST_DIR/*.* $DEST_TEST_DIR`.

    Returns True on success, or False after logging why the copy failed.
    """
    try:
        # os.listdir() returns the names (not full paths) in the directory.
        names = top_level_test_files(os.listdir(test_dir))
        if not names:
            # When `*.*` matches nothing, bash passes the literal text "*.*" to
            # cp, which fails because no such file exists.
            log_error(_OP, f"no files matching *.* in {test_dir}")
            return False
        for name in names:
            # Like cp without -r, this fails on a directory whose name contains
            # a dot (IsADirectoryError, a kind of OSError). No directory in
            # tests/ has one today.
            shutil.copy(os.path.join(test_dir, name), _DEST_TEST_DIR)
    except OSError as exc:
        log_error(_OP, f"could not copy test files from {test_dir} "
                       f"to {_DEST_TEST_DIR}: {exc}")
        return False
    log_info(_OP, f"Copied {', '.join(names)} to {_DEST_TEST_DIR}")
    return True


def _compile_gpu_copy(test_dir: str, env: dict[str, str]) -> bool:
    """Compile gpu-copy.cu with nvcc, choosing the host gcc as bash does.

    Returns True on success, or False after logging why it failed.
    """
    # run_capture() returns the exit code and what gcc printed to stdout.
    rc, output = run_capture(list(_GCC_VERSION_COMMAND), _OP, env=env)
    if rc != 0:
        log_error(_OP, "could not determine the default gcc version")
        return False
    gcc_major = gcc_major_version(output)
    if gcc_major is None:
        # !r adds quotes, which shows empty or odd output clearly in the log.
        log_error(_OP, f"unexpected gcc -dumpversion output: {output.strip()!r}")
        return False

    command = nvcc_command(_NVCC, gcc_major,
                           os.path.join(test_dir, _GPU_COPY_SOURCE),
                           os.path.join(test_dir, _GPU_COPY_BINARY))
    rc = exec_program(command, _OP, env=env)
    if rc != 0:
        log_error(_OP, f"compiling gpu-copy failed with exit code {rc}")
        return False
    return True


def _copy_health_checks(test_dir: str) -> bool:
    """`cp -r $AZHPC_IMAGES_TEST_DIR/health_checks $DEST_TEST_DIR`.

    Returns True on success, or False after logging why the copy failed.
    """
    source = os.path.join(test_dir, _HEALTH_CHECKS)
    destination = os.path.join(_DEST_TEST_DIR, _HEALTH_CHECKS)
    # See "How the copies compare with cp" in the module docstring for why
    # these three options make copytree behave like `cp -r`.
    try:
        shutil.copytree(source, destination, symlinks=True,
                        copy_function=shutil.copy, dirs_exist_ok=True)
    except OSError as exc:
        # shutil.Error, raised when some files in the tree failed to copy, is
        # a kind of OSError, so this catches it too.
        log_error(_OP, f"could not copy {source} to {_DEST_TEST_DIR}: {exc}")
        return False
    log_info(_OP, f"Copied {source} to {_DEST_TEST_DIR}")
    return True


def install(env: dict[str, str]) -> int:
    """Copy the repo's tests into the image, building gpu-copy if nvcc exists.

    `env` is the build environment made by component_env() in build_config.py.
    AZHPC_IMAGES_TEST_DIR (the repo's tests directory) is read from it.

    Returns 0 on success and 3 on failure.
    """
    test_dir = env.get("AZHPC_IMAGES_TEST_DIR", "")
    if not test_dir:
        log_error(_OP, "AZHPC_IMAGES_TEST_DIR is not set")
        return 3

    # ---- 1. mkdir -p /opt/azurehpc/test -----------------------------------
    # os.makedirs() also creates missing parent directories, and exist_ok=True
    # makes an existing directory not an error. Together they match `mkdir -p`.
    try:
        os.makedirs(_DEST_TEST_DIR, exist_ok=True)
    except OSError as exc:
        log_error(_OP, f"could not create {_DEST_TEST_DIR}: {exc}")
        return 3

    # ---- 2. Top-level test files -----------------------------------------
    if not _copy_top_level_files(test_dir):
        return 3

    # ---- 3. gpu-copy benchmark, only where nvcc is installed -------------
    # os.path.isfile() is like bash's `test -f`: True only for an existing
    # regular file (following symbolic links).
    if os.path.isfile(_NVCC):
        if not _compile_gpu_copy(test_dir, env):
            return 3
    else:
        log_info(_OP, f"{_NVCC} not found; not building gpu-copy")

    # ---- 4. The health_checks tree (now including gpu-copy, if built) ----
    if not _copy_health_checks(test_dir):
        return 3

    return 0
