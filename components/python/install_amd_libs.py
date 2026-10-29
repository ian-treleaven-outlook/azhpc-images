"""install_amd_libs.py — Python port of components/install_amd_libs.sh.

What this step does
-------------------
On x86_64 images we install two AMD packages under /opt/amd:

  * AOCL (AMD Optimizing CPU Libraries): the BLIS, FFTW and libFLAME math
    libraries. We run AMD's installer, copy its output into /opt/amd, point
    /opt/amd/lib and /opt/amd/include at the multi-threaded ("MT") build, and
    write an `amd/aocl` environment module for it.
  * AOCC (AMD Optimizing C/C++ and Fortran Compilers): we run AMD's installer
    and copy the whole compiler folder to /opt/amd/aocc-compiler-<version>.

Both versions are recorded in the component-versions file as AOCL and AOCC.

The original bash, for comparison (comments and blank lines trimmed):

    INSTALL_PREFIX=/opt/amd
    mkdir -p ${INSTALL_PREFIX}
    amd_metadata=$(get_component_config "amd")
    AOCL_VERSION=$(jq -r '.aocl.version' <<< $amd_metadata)
    AOCL_SHA256=$(jq -r '.aocl.sha256' <<< $amd_metadata)
    AOCL_TARBALL="aocl-linux-aocc-${AOCL_VERSION}.tar.gz"
    AOCL_FOLDER_VERSION=$(echo $AOCL_VERSION | cut -d'.' -f1-2 --output-delimiter='-')
    AOCL_DOWNLOAD_URL=https://download.amd.com/developer/eula/aocl/aocl-${AOCL_FOLDER_VERSION}/${AOCL_TARBALL}
    AOCL_FOLDER=$(basename $AOCL_TARBALL .tar.gz)
    download_and_verify $AOCL_DOWNLOAD_URL $AOCL_SHA256
    tar -xvf ${AOCL_TARBALL}
    pushd ${AOCL_FOLDER}
    ./install.sh -t amd -l blis fftw libflame -i lp64
    cp -r amd/${AOCL_VERSION}/aocc/* ${INSTALL_PREFIX}
    AOCL_MT_ROOT="${INSTALL_PREFIX}/MT"
    if [[ ! -d "${AOCL_MT_ROOT}/lib" || ! -d "${AOCL_MT_ROOT}/include" ]]; then
        echo "AOCL MT installation is incomplete: ..." >&2
        exit 1
    fi
    ln -sfn "${AOCL_MT_ROOT}/lib" "${INSTALL_PREFIX}/lib"
    ln -sfn "${AOCL_MT_ROOT}/include" "${INSTALL_PREFIX}/include"
    popd
    AMD_MODULE_FILES_DIRECTORY=${MODULE_FILES_DIRECTORY}/amd
    mkdir -p ${AMD_MODULE_FILES_DIRECTORY}
    cat << EOF >> ${AMD_MODULE_FILES_DIRECTORY}/aocl-${AOCL_VERSION}
    #%Module 1.0
    #
    #  AOCL
    #
    prepend-path    LD_LIBRARY_PATH   ${INSTALL_PREFIX}/lib
    setenv          AMD_FFTW_INCLUDE  ${INSTALL_PREFIX}/include
    EOF
    ln -s ${AMD_MODULE_FILES_DIRECTORY}/aocl-${AOCL_VERSION} ${AMD_MODULE_FILES_DIRECTORY}/aocl
    write_component_version "AOCL" ${AOCL_VERSION}

    AOCC_VERSION=$(jq -r '.aocc.version' <<< $amd_metadata)
    AOCC_SHA256=$(jq -r '.aocc.sha256' <<< $amd_metadata)
    AOCC_TARBALL="aocc-compiler-${AOCC_VERSION}.tar"
    AOCC_FOLDER_VERSION=$(echo $AOCC_VERSION | cut -d'.' -f1-2 --output-delimiter='-')
    AOCC_DOWNLOAD_URL=https://download.amd.com/developer/eula/aocc/aocc-${AOCC_FOLDER_VERSION}/${AOCC_TARBALL}
    AOCC_FOLDER=$(basename $AOCC_TARBALL .tar)
    download_and_verify $AOCC_DOWNLOAD_URL $AOCC_SHA256
    tar -xvf ${AOCC_TARBALL}
    pushd ${AOCC_FOLDER}
    ./install.sh
    popd
    cp -r ${AOCC_FOLDER} ${INSTALL_PREFIX}
    write_component_version "AOCC" ${AOCC_VERSION}

    rm -rf *.tar *.tar.gz
    rm -rf ${AOCL_FOLDER}
    rm -rf ${AOCC_FOLDER}

How it fits into the Python build
---------------------------------
utils/build_config.py sets this module's install() as the `action=` of the
"install-amd-libs" Step (x86_64 only). The build then calls install(env)
instead of running the bash script. Deleting `action=` runs the bash script
again, which is why the .sh file is kept.

install() returns 0 for success or 3 for failure. It logs problems and returns
3; it does not let exceptions escape to the build loop. Like bash's `set -e`,
it stops at the first failing step.

Which commands still run as programs, and why
---------------------------------------------
  * `tar -xvf`: the vendor archives may contain symlinks, which Python's
    tarfile module can refuse to extract in its safe mode. The tar command
    extracts them exactly as bash did.
  * The two vendor `./install.sh` installers.
  * `cp -r`: the AOCL and AOCC trees contain library symlinks (libfoo.so ->
    libfoo.so.5). `cp -r` copies them as symlinks and merges into existing
    folders. shutil.copytree does not behave the same way when files already
    exist, so we keep cp.

Everything else (checking folders, symlinks, the modulefile, cleanup) is plain
Python.

Deliberate differences from bash
--------------------------------
  * A missing AOCL/AOCC version or sha256, or a missing MODULE_FILES_DIRECTORY,
    fails the step before anything is downloaded. Bash would have built URLs
    containing "null", installed AOCL, and only failed later at the AOCC
    download (or written the modulefile under /amd).
  * Bash downloads and extracts into the distro build folder, its current
    directory. We use a dedicated work folder (_WORK_DIR) and delete it at the
    end. Bash's `rm -rf *.tar *.tar.gz` also deleted unrelated tarballs that
    earlier steps had left in the build folder; we only delete our own folder.
    Any extra files the vendor installers write beside their folders are now
    removed with the work folder instead of being left in the build folder.

Rerun behavior (same as bash)
-----------------------------
The modulefile is appended to (bash's `>>`), and the `amd/aocl` symlink is
created without replacing an existing one (bash's `ln -s` without `-f`). So a
second run appends the modulefile text again and then fails at the symlink,
exactly as the bash script does under `set -e`.
"""

from __future__ import annotations

import glob
import os
import shutil

from utils.component_config import config_for, write_component_version
from utils.download import ChecksumError, download_and_verify
from utils.logger import log_error, log_info
from utils.process import exec_program

# The Step's `op` label in build_config.py. Every log line uses it.
_OP = "install-amd-libs"

# Where both packages are installed (bash's INSTALL_PREFIX).
_INSTALL_PREFIX = "/opt/amd"

# Our private download/extract folder, deleted at the end (see the docstring).
_WORK_DIR = "/tmp/amd-libs"

_AOCL_URL = "https://download.amd.com/developer/eula/aocl/aocl-{folder_version}/{tarball}"
_AOCC_URL = "https://download.amd.com/developer/eula/aocc/aocc-{folder_version}/{tarball}"

# The libraries and integer size passed to AOCL's installer.
_AOCL_INSTALL_ARGS = ["-t", "amd", "-l", "blis", "fftw", "libflame", "-i", "lp64"]

# The modulefile body, byte for byte what bash's heredoc writes. Each line ends
# with "\n", including the last one, just like a heredoc.
_AOCL_MODULEFILE = (
    "#%Module 1.0\n"
    "#\n"
    "#  AOCL\n"
    "#\n"
    "prepend-path    LD_LIBRARY_PATH   {prefix}/lib\n"
    "setenv          AMD_FFTW_INCLUDE  {prefix}/include\n"
)


def folder_version(version: str) -> str:
    """Turn a version like "5.3.0" into the download folder name part "5-3".

    Mirrors bash's `echo $VERSION | cut -d'.' -f1-2 --output-delimiter='-'`:
    keep the first two dot-separated fields and join them with "-".

        "5.3.0" -> "5-3"      "5.3" -> "5-3"      "5" -> "5"

    Like cut, a version with no "." at all is returned unchanged.
    """
    # str.split(".") gives a list of fields; [:2] keeps at most the first two.
    return "-".join(version.split(".")[:2])


def aocl_modulefile(prefix: str) -> str:
    """The text bash appends to the aocl-<version> modulefile."""
    return _AOCL_MODULEFILE.format(prefix=prefix)


def make_symlink(target: str, link_name: str, *, replace: bool) -> None:
    """Create a symlink like `ln -sfn` (replace=True) or `ln -s` (replace=False).

    The rules `ln` follows, which this mirrors:
      * If link_name is an existing folder, ln creates the link *inside* it,
        named after target. With -n (used together with -f here), a symlink
        to a folder does not count; it is replaced instead.
      * With -f, an existing file or symlink at that name is removed first.
      * Without -f, an existing name is an error ("File exists").

    Raises OSError (for example FileExistsError) when ln would fail.
    """
    # os.path.isdir follows symlinks; os.path.islink checks the link itself.
    if os.path.isdir(link_name) and not (replace and os.path.islink(link_name)):
        link_name = os.path.join(link_name, os.path.basename(target))
    # lexists is True even for a broken symlink, which exists() would miss.
    if replace and os.path.lexists(link_name):
        os.remove(link_name)          # refuses folders, like `ln -f`
    os.symlink(target, link_name)


def _run(command: list[str], env: dict[str, str], cwd: str | None = None) -> bool:
    """Run one command; log and return False if it fails."""
    rc = exec_program(command, _OP, cwd=cwd, env=env)
    if rc != 0:
        log_error(_OP, f"{' '.join(command)} failed with exit code {rc}")
        return False
    return True


def _download_and_extract(url: str, sha256: str, env: dict[str, str]) -> bool:
    """Download + verify `url` into _WORK_DIR, then extract it there with tar."""
    try:
        tarball = download_and_verify(url, sha256, dest_dir=_WORK_DIR)
    # download() raises OSError (network/file errors); a bad hash raises ChecksumError.
    except (OSError, ChecksumError) as exc:
        log_error(_OP, f"download/verify of {url} failed: {exc}")
        return False
    # The tar command, not tarfile: see "Which commands still run" above.
    return _run(["tar", "-xvf", str(tarball)], env, cwd=_WORK_DIR)


def _install_aocl(version: str, sha256: str, env: dict[str, str]) -> bool:
    """Download, install and register AOCL. Returns True on success."""
    tarball = f"aocl-linux-aocc-{version}.tar.gz"
    url = _AOCL_URL.format(folder_version=folder_version(version), tarball=tarball)
    folder = os.path.join(_WORK_DIR, f"aocl-linux-aocc-{version}")

    if not _download_and_extract(url, sha256, env):
        return False

    # The installer puts its output under ./amd/<version>/aocc in its own folder.
    if not _run(["./install.sh", *_AOCL_INSTALL_ARGS], env, cwd=folder):
        return False

    # bash: cp -r amd/<version>/aocc/* /opt/amd
    # glob.glob, like the shell's `*`, skips names starting with "." and
    # returns nothing when there is no match. sorted() mimics the shell's order.
    entries = sorted(glob.glob(os.path.join(folder, "amd", version, "aocc", "*")))
    if not entries:
        log_error(_OP, f"AOCL installer produced nothing under {folder}/amd/{version}/aocc")
        return False
    if not _run(["cp", "-r", *entries, _INSTALL_PREFIX], env):
        return False

    mt_root = os.path.join(_INSTALL_PREFIX, "MT")
    mt_lib = os.path.join(mt_root, "lib")
    mt_include = os.path.join(mt_root, "include")
    if not (os.path.isdir(mt_lib) and os.path.isdir(mt_include)):
        log_error(_OP, "AOCL MT installation is incomplete: "
                       f"expected {mt_lib} and {mt_include}")
        return False

    module_dir = os.path.join(env["MODULE_FILES_DIRECTORY"], "amd")
    modulefile = os.path.join(module_dir, f"aocl-{version}")
    try:
        # Point the default lib/include folders at the multi-threaded build.
        make_symlink(mt_lib, os.path.join(_INSTALL_PREFIX, "lib"), replace=True)
        make_symlink(mt_include, os.path.join(_INSTALL_PREFIX, "include"), replace=True)

        os.makedirs(module_dir, exist_ok=True)
        # Mode "a" appends, like bash's `>>`.
        with open(modulefile, "a", encoding="utf-8") as handle:
            handle.write(aocl_modulefile(_INSTALL_PREFIX))

        # No replace: like bash's `ln -s`, this fails if amd/aocl already exists.
        make_symlink(modulefile, os.path.join(module_dir, "aocl"), replace=False)

        write_component_version("AOCL", version)
    except OSError as exc:
        log_error(_OP, f"could not finish the AOCL install: {exc}")
        return False
    return True


def _install_aocc(version: str, sha256: str, env: dict[str, str]) -> bool:
    """Download, install and register AOCC. Returns True on success."""
    tarball = f"aocc-compiler-{version}.tar"
    url = _AOCC_URL.format(folder_version=folder_version(version), tarball=tarball)
    folder = os.path.join(_WORK_DIR, f"aocc-compiler-{version}")

    if not _download_and_extract(url, sha256, env):
        return False
    if not _run(["./install.sh"], env, cwd=folder):
        return False
    # Copies the folder itself, giving /opt/amd/aocc-compiler-<version>.
    if not _run(["cp", "-r", folder, _INSTALL_PREFIX], env):
        return False

    try:
        write_component_version("AOCC", version)
    except OSError as exc:
        log_error(_OP, f"could not record AOCC={version}: {exc}")
        return False
    return True


def install(env: dict[str, str]) -> int:
    """Install AOCL then AOCC into /opt/amd. Returns 0 on success, 3 on failure."""
    try:
        os.makedirs(_INSTALL_PREFIX, exist_ok=True)
    except OSError as exc:
        log_error(_OP, f"could not create {_INSTALL_PREFIX}: {exc}")
        return 3

    # Both packages live in one "amd" entry of versions.json:
    #   "amd": {"common": {"aocl": {"version", "sha256"}, "aocc": {...}}}
    cfg = config_for("amd", env) or {}
    # `or {}` turns a missing entry (None) into an empty dict, so the lookups
    # below simply find nothing instead of raising.
    aocl = cfg.get("aocl") or {}
    aocc = cfg.get("aocc") or {}
    if not (aocl.get("version") and aocl.get("sha256")
            and aocc.get("version") and aocc.get("sha256")):
        log_error(_OP, "could not resolve amd aocl/aocc version and sha256 "
                       "from versions.json")
        return 3
    if not env.get("MODULE_FILES_DIRECTORY"):
        log_error(_OP, "MODULE_FILES_DIRECTORY is not set")
        return 3

    # Start from an empty work folder so leftovers from an earlier run can't mix in.
    shutil.rmtree(_WORK_DIR, ignore_errors=True)

    log_info(_OP, f"Installing AOCL {aocl['version']}")
    if not _install_aocl(aocl["version"], aocl["sha256"], env):
        return 3

    log_info(_OP, f"Installing AOCC {aocc['version']}")
    if not _install_aocc(aocc["version"], aocc["sha256"], env):
        return 3

    # Like bash's `rm -rf`, cleanup never fails the step.
    shutil.rmtree(_WORK_DIR, ignore_errors=True)
    log_info(_OP, f"AOCL {aocl['version']} and AOCC {aocc['version']} "
                  f"installed to {_INSTALL_PREFIX}")
    return 0
