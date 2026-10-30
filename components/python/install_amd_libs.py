"""install_amd_libs.py — Python port of components/install_amd_libs.sh.

On x86_64 images, installs two AMD packages under /opt/amd:

  * AOCL (AMD Optimizing CPU Libraries: BLIS, FFTW, libFLAME): run AMD's
    installer, copy its output into /opt/amd, point /opt/amd/lib and
    /opt/amd/include at the multi-threaded ("MT") build, and write an
    `amd/aocl` environment module.
  * AOCC (AMD's C/C++/Fortran compilers): run AMD's installer and copy the
    folder to /opt/amd/aocc-compiler-<version>.

Both versions come from the "amd" entry of versions.json and are recorded as
AOCL and AOCC.

The vendor `./install.sh` installers and `cp -r` still run as programs: the
AOCL/AOCC trees hold library symlinks that `cp -r` copies as symlinks while
merging into existing folders, which shutil.copytree doesn't do the same way.

Kept bash quirks
----------------
  * The modulefile is appended to (`>>`) and `amd/aocl` is linked with
    `ln -s` (no -f), so a rerun appends the modulefile text again and then
    fails at the symlink, as bash does under `set -e`; the error says so.
  * If /opt/amd/lib or /opt/amd/include is a real folder, `ln -sfn` creates
    the link inside it rather than replacing it. Kept, with a warning.

Deliberate differences
----------------------
  * A missing AOCL/AOCC version or sha256, or MODULE_FILES_DIRECTORY, fails
    the step before anything is downloaded. Bash would build URLs containing
    "null", install AOCL and fail later (or write the modulefile under /amd).
  * Downloads and extraction use a private work folder (_WORK_DIR), emptied
    first and deleted at the end. Bash used the build folder, and its
    `rm -rf *.tar *.tar.gz` also deleted unrelated tarballs left there by
    earlier steps; anything the vendor installers write beside their folders
    now goes with the work folder too.
"""

from __future__ import annotations

import glob
import os

from utils.component_config import require_config, write_component_version
from utils.download import fetch_verified
from utils.files import extract_tarball, remove_paths, write_file
from utils.logger import log_error, log_info, log_warn
from utils.process import run

_OP = "install-amd-libs"

_INSTALL_PREFIX = "/opt/amd"          # bash's INSTALL_PREFIX
_WORK_DIR = "/tmp/amd-libs"           # private download/extract folder

_AOCL_URL = "https://download.amd.com/developer/eula/aocl/aocl-{folder_version}/{tarball}"
_AOCC_URL = "https://download.amd.com/developer/eula/aocc/aocc-{folder_version}/{tarball}"

# The libraries and integer size passed to AOCL's installer.
_AOCL_INSTALL_ARGS = ["-t", "amd", "-l", "blis", "fftw", "libflame", "-i", "lp64"]

# Byte for byte what bash's heredoc appends.
_AOCL_MODULEFILE = (
    "#%Module 1.0\n"
    "#\n"
    "#  AOCL\n"
    "#\n"
    "prepend-path    LD_LIBRARY_PATH   {prefix}/lib\n"
    "setenv          AMD_FFTW_INCLUDE  {prefix}/include\n"
)


# ---- Pure helpers -------------------------------------------------------------
def folder_version(version: str) -> str:
    """bash `cut -d'.' -f1-2 --output-delimiter='-'`: "5.3.0" -> "5-3", "5" -> "5"."""
    return "-".join(version.split(".")[:2])


def aocl_url(version: str) -> str:
    return _AOCL_URL.format(folder_version=folder_version(version),
                            tarball=f"aocl-linux-aocc-{version}.tar.gz")


def aocc_url(version: str) -> str:
    return _AOCC_URL.format(folder_version=folder_version(version),
                            tarball=f"aocc-compiler-{version}.tar")


def aocl_modulefile(prefix: str) -> str:
    """The text bash appends to the aocl-<version> modulefile."""
    return _AOCL_MODULEFILE.format(prefix=prefix)


def make_symlink(target: str, link_name: str, *, replace: bool) -> None:
    """`ln -sfn` (replace=True) or `ln -s` (replace=False); OSError where ln fails.

    As ln does: an existing folder at link_name gets the link *inside* it
    (with -n, a symlink to a folder is replaced instead); -f removes an
    existing file or symlink first; without -f an existing name is
    "File exists" (FileExistsError).
    """
    if os.path.isdir(link_name) and not (replace and os.path.islink(link_name)):
        link_name = os.path.join(link_name, os.path.basename(target))
    if replace and os.path.lexists(link_name):    # lexists: broken links too
        os.remove(link_name)                      # refuses folders, like `ln -f`
    os.symlink(target, link_name)


# ---- Steps ----------------------------------------------------------------------
def _record_version(component: str, version: str) -> bool:
    try:
        write_component_version(component, version)
    except OSError as exc:
        log_error(_OP, f"could not record {component}={version}: {exc}")
        return False
    return True


def _download_and_extract(url: str, sha256: str, env: dict[str, str]) -> bool:
    """Download + verify `url` into _WORK_DIR and unpack it there."""
    tarball = fetch_verified(url, sha256, _OP, _WORK_DIR)
    return tarball is not None and extract_tarball(tarball, _OP, cwd=_WORK_DIR, env=env)


def _copy_aocl_output(folder: str, version: str, env: dict[str, str]) -> bool:
    """bash: `cp -r amd/<version>/aocc/* /opt/amd`."""
    # glob, like the shell's `*`, skips dotfiles; sorted() is the shell's order.
    entries = sorted(glob.glob(os.path.join(folder, "amd", version, "aocc", "*")))
    if not entries:
        log_error(_OP, f"AOCL installer produced nothing under {folder}/amd/{version}/aocc")
        return False
    return run(["cp", "-r", *entries, _INSTALL_PREFIX], _OP, env=env)


def _link_mt_build() -> bool:
    """Check the MT build is complete, then `ln -sfn` /opt/amd/{lib,include} to it."""
    mt_root = os.path.join(_INSTALL_PREFIX, "MT")
    links = [(os.path.join(mt_root, name), os.path.join(_INSTALL_PREFIX, name))
             for name in ("lib", "include")]
    if not all(os.path.isdir(target) for target, _ in links):
        log_error(_OP, "AOCL MT installation is incomplete: expected "
                       f"{links[0][0]} and {links[1][0]}")
        return False
    for target, link in links:
        if os.path.isdir(link) and not os.path.islink(link):
            log_warn(_OP, f"{link} is an existing folder, so `ln -sfn` creates the "
                          "link inside it instead of replacing it, as bash does",
                     path=link, target=target,
                     link=os.path.join(link, os.path.basename(target)))
        try:
            make_symlink(target, link, replace=True)
        except OSError as exc:
            log_error(_OP, f"could not link {link} -> {target}: {exc}", path=link)
            return False
    return True


def _write_aocl_module(version: str, module_root: str) -> bool:
    """Append the aocl-<version> modulefile and link amd/aocl to it (`ln -s`)."""
    module_dir = os.path.join(module_root, "amd")
    modulefile = os.path.join(module_dir, f"aocl-{version}")
    aocl_link = os.path.join(module_dir, "aocl")
    if not write_file(modulefile, aocl_modulefile(_INSTALL_PREFIX), _OP, append=True):
        return False
    try:
        make_symlink(modulefile, aocl_link, replace=False)
    except FileExistsError as exc:
        log_error(_OP, f"could not create {aocl_link}: {exc}. It already exists, "
                       "which happens when this step is run again: bash's `ln -s` "
                       "(no -f) fails the same way. The modulefile "
                       f"{modulefile} has also been appended to a second time.",
                  path=aocl_link, target=modulefile)
        return False
    except OSError as exc:
        log_error(_OP, f"could not create {aocl_link}: {exc}", path=aocl_link)
        return False
    return True


def _install_aocl(version: str, sha256: str, env: dict[str, str]) -> bool:
    folder = os.path.join(_WORK_DIR, f"aocl-linux-aocc-{version}")
    return (_download_and_extract(aocl_url(version), sha256, env)
            and run(["./install.sh", *_AOCL_INSTALL_ARGS], _OP, cwd=folder, env=env)
            and _copy_aocl_output(folder, version, env)
            and _link_mt_build()
            and _write_aocl_module(version, env["MODULE_FILES_DIRECTORY"])
            and _record_version("AOCL", version))


def _install_aocc(version: str, sha256: str, env: dict[str, str]) -> bool:
    folder = os.path.join(_WORK_DIR, f"aocc-compiler-{version}")
    return (_download_and_extract(aocc_url(version), sha256, env)
            and run(["./install.sh"], _OP, cwd=folder, env=env)
            # Copies the folder itself: /opt/amd/aocc-compiler-<version>.
            and run(["cp", "-r", folder, _INSTALL_PREFIX], _OP, env=env)
            and _record_version("AOCC", version))


def install(env: dict[str, str]) -> int:
    """Install AOCL then AOCC into /opt/amd. Returns 0 on success, 3 on failure."""
    try:
        os.makedirs(_INSTALL_PREFIX, exist_ok=True)
    except OSError as exc:
        log_error(_OP, f"could not create {_INSTALL_PREFIX}: {exc}")
        return 3
    cfg = require_config("amd", env, _OP, "aocl.version", "aocl.sha256",
                         "aocc.version", "aocc.sha256")
    if cfg is None:
        return 3
    if not env.get("MODULE_FILES_DIRECTORY"):
        log_error(_OP, "MODULE_FILES_DIRECTORY is not set")
        return 3
    aocl, aocc = cfg["aocl"], cfg["aocc"]

    remove_paths(_WORK_DIR, op=_OP)    # no leftovers from an earlier run
    log_info(_OP, f"Installing AOCL {aocl['version']}", version=aocl["version"])
    if not _install_aocl(aocl["version"], aocl["sha256"], env):
        return 3
    log_info(_OP, f"Installing AOCC {aocc['version']}", version=aocc["version"])
    if not _install_aocc(aocc["version"], aocc["sha256"], env):
        return 3
    remove_paths(_WORK_DIR, op=_OP)

    log_info(_OP, f"AOCL {aocl['version']} and AOCC {aocc['version']} "
                  f"installed to {_INSTALL_PREFIX}")
    return 0
