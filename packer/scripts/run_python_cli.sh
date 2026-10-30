#!/bin/bash
# Packer's "Install HPC components" step when USE_PYTHON_CLI=true
# (docs/python-cli-builds.md): run the Python CLI, azhpc.py, from the repo root.
#
# azhpc.py needs Python 3.10 or newer. Ubuntu 22.04+ and Azure Linux 3.0 have
# it as python3. AlmaLinux, Rocky and RHEL have 3.6 (8.x) or 3.9 (9.x) as
# python3, and their install_utils.sh installs a newer Python later in the
# build without changing what python3 points at. On those, this script
# installs that same package now and runs azhpc.py with it, so the image gets
# the same packages as a bash build, and python3 stays the distro's own.
#
# Usage, from the repo root:  bash packer/scripts/run_python_cli.sh <azhpc.py arguments>

set -euo pipefail

OS_RELEASE=${OS_RELEASE:-/etc/os-release}   # overridable for the unit tests

is_new_enough() {
    "$1" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null
}

# The newer Python this distro's install_utils.sh installs anyway, or "".
distro_python() {
    local ID="" VERSION_ID=""
    source "$OS_RELEASE"
    case "$ID$VERSION_ID" in
        # distros/rocky8.10/install_utils.sh: python3.11-pyyaml (needs python3.11)
        rocky8*)                 echo python3.11 ;;
        # distros/{almalinux,rocky,rhel}*/install_utils.sh: dnf install -y python3.12
        almalinux*|rocky*|rhel*) echo python3.12 ;;
        *)                       echo "" ;;
    esac
}

python=python3
if ! is_new_enough "$python"; then
    python=$(distro_python)
    if [[ -z "$python" ]]; then
        echo "ERROR: azhpc.py needs Python 3.10 or newer; python3 is $(python3 --version 2>&1)," \
             "and no newer Python is known for this distro (packer/scripts/run_python_cli.sh)" >&2
        exit 1
    fi
    if ! command -v "$python" >/dev/null; then
        echo "python3 is $(python3 --version 2>&1); installing $python for azhpc.py"
        dnf install -y "$python"
    fi
fi

"$python" --version
exec "$python" azhpc.py "$@"
