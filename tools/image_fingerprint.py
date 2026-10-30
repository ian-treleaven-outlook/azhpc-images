#!/usr/bin/env python3
"""image_fingerprint.py — record what an image contains, to compare two builds.

Packer already saves "build evidence" for every build (packer/scripts/
collect_build_evidence.sh): installed packages, holds, repositories, service
states, dkms status and component versions. This adds what that leaves out,
so a Python-built image can be compared with a bash-built one:

  * config files the build writes (modprobe, udev, sysctl, limits, profile.d,
    systemd units and drop-ins, ld.so.conf.d, apt/dnf config, waagent, netplan
    ...): path, mode and SHA256 of every file;
  * the layout of /opt (three levels) and /usr/local (two), and the
    environment-modules trees (with file hashes);
  * enabled unit files, and Python packages installed with pip.

Run it on the build VM as root (Packer does, when CAPTURE_IMAGE_FINGERPRINT
is true):

    python3 tools/image_fingerprint.py --out /tmp/image-fingerprint.json

It uses only the standard library and runs on Python 3.6 or newer, so it
also works with the stock python3 of AlmaLinux/Rocky 8 on a bash build.
Compare two captures with tools/compare_images.py.
"""

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys

# Directories (and single files) whose every file is recorded with its hash.
CONFIG_PATHS = [
    "/etc/modprobe.d", "/etc/modules-load.d", "/etc/udev/rules.d",
    "/etc/sysctl.conf", "/etc/sysctl.d", "/etc/security/limits.conf",
    "/etc/security/limits.d", "/etc/profile.d", "/etc/environment",
    "/etc/systemd/system", "/etc/systemd/network", "/etc/ld.so.conf.d",
    "/etc/apt/sources.list", "/etc/apt/sources.list.d", "/etc/apt/preferences.d",
    "/etc/apt/apt.conf.d", "/etc/yum.repos.d", "/etc/dnf/dnf.conf",
    "/etc/dnf/plugins", "/etc/waagent.conf", "/etc/netplan", "/etc/cloud/cloud.cfg.d",
    "/etc/default/grub", "/etc/nvidia", "/etc/dkms", "/etc/rdma", "/etc/sudoers.d",
    "/usr/lib/udev/rules.d",
]
# Trees listed by path (and type) to a fixed depth: too big to hash.
LAYOUT_ROOTS = {"/opt": 3, "/usr/local": 2}
# Environment-modules trees: small, so every file is hashed.
MODULE_ROOTS = ["/usr/share/modules/modulefiles", "/usr/share/Modules/modulefiles"]


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path):
    """One entry for a path: its type, mode, and (files) hash or (links) target."""
    st = os.lstat(path)
    entry = {"mode": oct(stat.S_IMODE(st.st_mode))}
    if stat.S_ISLNK(st.st_mode):
        entry["type"] = "link"
        entry["target"] = os.readlink(path)
    elif stat.S_ISDIR(st.st_mode):
        entry["type"] = "dir"
    elif stat.S_ISREG(st.st_mode):
        entry["type"] = "file"
        try:
            entry["sha256"] = file_hash(path)
        except OSError as exc:
            entry["error"] = str(exc)
    else:
        entry["type"] = "other"
    return entry


def walk_paths(root):
    """Every path under `root` (or `root` itself if it isn't a directory), sorted."""
    if not os.path.lexists(root):
        return []
    if os.path.islink(root) or not os.path.isdir(root):
        return [root]
    found = []
    for directory, dirs, files in os.walk(root):
        dirs.sort()
        found += [os.path.join(directory, name) for name in sorted(dirs + files)]
    return found


def hashed_tree(roots):
    """{path: describe(path)} for every path under each of `roots`."""
    return {path: describe(path) for root in roots for path in walk_paths(root)}


def layout(root, depth):
    """{path: type} for paths under `root` down to `depth` levels (no hashes)."""
    entries = {}
    if not os.path.isdir(root):
        return entries
    base = root.rstrip("/").count("/")
    for directory, dirs, files in os.walk(root):
        if directory.rstrip("/").count("/") - base >= depth:
            dirs[:] = []
            continue
        dirs.sort()
        for name in sorted(dirs + files):
            path = os.path.join(directory, name)
            entries[path] = ("link" if os.path.islink(path)
                             else "dir" if os.path.isdir(path) else "file")
    return entries


def command_lines(command):
    """stdout lines of a command, or [] if it isn't available or fails."""
    try:
        out = subprocess.check_output(command, stderr=subprocess.DEVNULL,
                                      universal_newlines=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def enabled_units():
    lines = command_lines(["systemctl", "list-unit-files", "--state=enabled",
                           "--no-legend", "--no-pager"])
    return sorted(line.split()[0] for line in lines)


def pip_packages():
    return sorted(command_lines([sys.executable, "-m", "pip", "list",
                                 "--format=freeze", "--disable-pip-version-check"]))


def capture():
    return {
        "format": 1,
        "kernel": os.uname().release,
        "config_files": hashed_tree(CONFIG_PATHS),
        "layout": {root: layout(root, depth) for root, depth in LAYOUT_ROOTS.items()},
        "modulefiles": hashed_tree(MODULE_ROOTS),
        "enabled_units": enabled_units(),
        "pip_packages": pip_packages(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Record what an image contains.")
    parser.add_argument("--out", required=True, help="where to write the JSON")
    args = parser.parse_args(argv)
    data = capture()
    with open(args.out, "w") as out:
        json.dump(data, out, indent=1, sort_keys=True)
    print("image fingerprint: {} config files, {} modulefiles, {} enabled units -> {}".format(
        len(data["config_files"]), len(data["modulefiles"]),
        len(data["enabled_units"]), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
