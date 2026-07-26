"""install_doca.py — Python port of components/install_doca.sh.

DOCA-OFED is NVIDIA's packaging of the InfiniBand / RDMA driver stack (the
successor to MLNX_OFED). This installs it, records the DOCA + OFED versions, and
sets up the openibd boot service.

The install is package-manager orchestration, so most steps shell out via
exec_program; the genuinely logic-bearing pieces are native, testable Python:
parse_openmpi_version, parse_ofed_version, newest_kernel_repo_rpm, and
strip_dnf_excludes. Distro differences are real (apt needs an equivs marker to
inhibit DOCA's bundled Open MPI; RHEL needs a kernel-repo rpm + dnf.conf edit),
so each distro family has its own named handler rather than one if/elif wall.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

from utils.component_config import config_for, write_component_version
from utils.download import download_and_verify
from utils.process import exec_program, run_capture
from utils.logger import log_info, log_error

_OP = "install-doca"

# Config/asset locations (mirror the paths in install_doca.sh).
_DKMS_PIN = "/etc/apt/preferences.d/doca-dkms-pin"
_EQUIVS_CTRL = "/tmp/hpcx-provides-openmpi"
_EQUIVS_DEB_GLOB = "/tmp/hpcx-provides-openmpi_*_all.deb"
_DOCA_KERNEL_SUPPORT = "/opt/mellanox/doca/tools/doca-kernel-support"
_DNF_CONF = "/etc/dnf/dnf.conf"
_OPENIBD_DROPIN_DIR = "/etc/systemd/system/openibd.service.d"
_OPENIB_CONF = "/etc/infiniband/openib.conf"

# We prefer the distro-shipped dkms over the one DOCA ships, so pin DOCA's out.
_DKMS_PIN_CONTENT = """Package: dkms
Pin: release l=DOCA-HOST*
Pin-Priority: -1
"""

# openibd should start only after udev settles and restart if it fails.
_OPENIBD_OVERRIDE = """[Unit]
After=systemd-udev-settle.service
Wants=systemd-udev-settle.service

[Service]
Restart=on-failure
RestartSec=5
"""


# --------------------------------------------------------------------------- #
# Pure, testable helpers (no side effects).
# --------------------------------------------------------------------------- #
def parse_openmpi_version(apt_cache_output: str) -> str | None:
    """Extract the Open MPI version from `apt-cache show openmpi` output.

    Mirrors: apt-cache show openmpi | awk '/^Version:/ {print $2; exit}'
    Returns the first ``Version:`` value found, or None if there isn't one.
    """
    for line in apt_cache_output.splitlines():
        if line.startswith("Version:"):
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()
    return None


def parse_ofed_version(ofed_info_output: str) -> str | None:
    """Extract the OFED version from `ofed_info` output.

    Mirrors: ofed_info | sed -n '1,1p' | awk -F'-' 'OFS="-" {print $3,$4}' | tr -d ':'
    i.e. take the first line, split on '-', join the 3rd and 4th fields with '-',
    and drop any ':'. Returns None if the first line has too few fields.
    """
    lines = ofed_info_output.splitlines()
    if not lines:
        return None
    fields = lines[0].split("-")
    if len(fields) < 4:
        return None
    result = f"{fields[2]}-{fields[3]}".replace(":", "")
    return result or None


def newest_kernel_repo_rpm(search_root: str) -> str | None:
    """Return the most recently modified doca-kernel-repo-*.rpm under search_root.

    Mirrors the shell:
        find /tmp/DOCA.*/ -name 'doca-kernel-repo-*.rpm' \
            -printf '%T@ %p\\n' | sort -n | tail -1 | cut -d' ' -f2-
    """
    pattern = os.path.join(search_root, "DOCA.*", "**", "doca-kernel-repo-*.rpm")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def strip_dnf_excludes(conf_text: str) -> str:
    """Drop any 'exclude=' lines from dnf.conf (mirrors sed '/^exclude=/d')."""
    kept = [ln for ln in conf_text.splitlines() if not ln.startswith("exclude=")]
    result = "\n".join(kept)
    if conf_text.endswith("\n"):
        result += "\n"
    return result


def _equivs_control(openmpi_version: str) -> str:
    """Build the equivs control file that Provides: openmpi (= version).

    This empty marker package satisfies doca-ofed's strict openmpi dependency so
    apt never pulls in DOCA's bundled Open MPI (HPC-X provides it at runtime, and
    the DOCA .deb collides with the separately-installed pmix package).
    """
    return (
        "Section: misc\n"
        "Priority: optional\n"
        "Homepage: https://github.com/Azure/azhpc-images\n"
        "Standards-Version: 3.9.2\n"
        "\n"
        "Package: hpcx-provides-openmpi\n"
        f"Provides: openmpi (= {openmpi_version})\n"
        f"Version: {openmpi_version}\n"
        "Maintainer: Azure HPC Platform team <hpcplat@microsoft.com>\n"
        "Description: marker package to inhibit the DOCA-bundled openmpi\n"
        " HPC-X (installed by install_mpis.sh into /opt) provides Open MPI at runtime,\n"
        " so the DOCA openmpi .deb is redundant and additionally collides with\n"
        " /etc/pmix-mca-params.conf from the separately-installed pmix package.\n"
    )


# --------------------------------------------------------------------------- #
# Per-distro install handlers. Each returns 0 on success, 3 on any failure.
# --------------------------------------------------------------------------- #
def _install_ubuntu(env: dict[str, str], deb_file: str) -> int:
    """Ubuntu/Debian: dpkg the repo package, inhibit DOCA's openmpi via an equivs
    marker, then install doca-ofed."""
    if exec_program(["dpkg", "-i", deb_file], _OP, env=env) != 0:
        return 3

    Path(_DKMS_PIN).write_text(_DKMS_PIN_CONTENT, encoding="utf-8")

    if exec_program(["apt-get", "update"], _OP, env=env) != 0:
        return 3
    if exec_program(["apt-get", "install", "-y", "equivs"], _OP, env=env) != 0:
        return 3

    # Read the openmpi version DOCA would install, so our marker can claim it.
    rc, out = run_capture(["apt-cache", "show", "openmpi"], _OP, env=env)
    openmpi_version = parse_openmpi_version(out)
    if not openmpi_version:
        log_error(_OP, "could not read openmpi version from DOCA repo")
        return 3

    # Build and install the empty marker package.
    Path(_EQUIVS_CTRL).write_text(_equivs_control(openmpi_version), encoding="utf-8")
    if exec_program(["equivs-build", _EQUIVS_CTRL], _OP, cwd="/tmp", env=env) != 0:
        return 3
    built = glob.glob(_EQUIVS_DEB_GLOB)
    if not built:
        log_error(_OP, "equivs-build did not produce a marker package")
        return 3
    if exec_program(["dpkg", "-i", built[0]], _OP, env=env) != 0:
        return 3

    # Clean up the marker artifacts.
    for path in built + [_EQUIVS_CTRL]:
        try:
            os.remove(path)
        except OSError:
            pass

    return 0 if exec_program(["apt-get", "-y", "install", "doca-ofed"], _OP, env=env) == 0 else 3


def _install_azurelinux(env: dict[str, str], deb_file: str) -> int:
    """Azure Linux: rpm the package, then dnf-install doca-ofed."""
    steps = [
        ["rpm", "-i", deb_file],
        ["dnf", "clean", "all"],
        ["dnf", "install", "-y", "doca-extra"],
        [_DOCA_KERNEL_SUPPORT],
        ["dnf", "install", "-y", "doca-ofed-userspace"],
        ["dnf", "-y", "install", "doca-ofed"],
    ]
    for cmd in steps:
        if exec_program(cmd, _OP, env=env) != 0:
            return 3
    return 0


def _install_rhel(env: dict[str, str], deb_file: str) -> int:
    """RHEL-family (Alma/Rocky/RHEL): like Azure Linux, plus install the newest
    kernel-repo rpm and temporarily lift dnf.conf's exclude= pins."""
    pre = [
        ["rpm", "-i", deb_file],
        ["dnf", "clean", "all"],
        ["dnf", "install", "-y", "doca-extra"],
        [_DOCA_KERNEL_SUPPORT],
    ]
    for cmd in pre:
        if exec_program(cmd, _OP, env=env) != 0:
            return 3

    repo = newest_kernel_repo_rpm("/tmp")
    if not repo:
        log_error(_OP, "could not find a doca-kernel-repo rpm under /tmp")
        return 3
    if exec_program(["rpm", "-i", repo], _OP, env=env) != 0:
        return 3

    # Lift any exclude= pins so doca-ofed's kernel packages can install, then
    # always restore the original dnf.conf.
    original = Path(_DNF_CONF).read_text(encoding="utf-8")
    try:
        Path(_DNF_CONF).write_text(strip_dnf_excludes(original), encoding="utf-8")
        for cmd in (["dnf", "-y", "install", "doca-ofed-userspace"],
                    ["dnf", "-y", "install", "doca-ofed"]):
            if exec_program(cmd, _OP, env=env) != 0:
                return 3
    finally:
        Path(_DNF_CONF).write_text(original, encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def install(env: dict[str, str]) -> int:
    """Install DOCA-OFED: resolve + download, dispatch by distro, record versions,
    and set up the openibd service. Returns 0 on success, 3 on any failure."""
    cfg = config_for("doca", env)
    if not cfg or not cfg.get("version"):
        log_error(_OP, "could not resolve doca version from versions.json")
        return 3

    version = cfg["version"]
    url = cfg.get("url", "")
    sha256 = cfg.get("sha256", "")

    log_info(_OP, f"Installing DOCA-OFED {version}")

    try:
        deb_file = str(download_and_verify(url, sha256, dest_dir="/tmp"))
    except Exception as exc:
        log_error(_OP, f"download/verify failed: {exc}")
        return 3

    distribution = env.get("DISTRIBUTION", "")
    if "ubuntu" in distribution:
        rc = _install_ubuntu(env, deb_file)
    elif distribution == "azurelinux3.0":
        rc = _install_azurelinux(env, deb_file)
    else:  # RHEL-family: AlmaLinux, Rocky Linux, RHEL, etc.
        rc = _install_rhel(env, deb_file)
    if rc != 0:
        return rc

    write_component_version("DOCA", version)

    # Best-effort OFED version: ofed_info may be absent on a build VM without IB.
    rc_ofed, ofed_out = run_capture(["ofed_info"], _OP, env=env)
    if rc_ofed == 0:
        ofed_version = parse_ofed_version(ofed_out)
        if ofed_version:
            write_component_version("OFED", ofed_version)

    # Drop-in so openibd starts after udev settles and restarts on failure.
    os.makedirs(_OPENIBD_DROPIN_DIR, exist_ok=True)
    Path(f"{_OPENIBD_DROPIN_DIR}/override.conf").write_text(
        _OPENIBD_OVERRIDE, encoding="utf-8")

    if env.get("NODE_TYPE", "azure-vm") == "baremetal":
        with open(_OPENIB_CONF, "a", encoding="utf-8") as handle:
            handle.write("\n# Load IPoIB\nIPOIB_LOAD=no\n")

    # Enable only; do NOT restart here — restarting probes the build VM's IB
    # hardware (which may be absent on general-purpose build SKUs).
    if exec_program(["systemctl", "daemon-reload"], _OP, env=env) != 0:
        return 3
    return 0 if exec_program(["systemctl", "enable", "openibd"], _OP, env=env) == 0 else 3
