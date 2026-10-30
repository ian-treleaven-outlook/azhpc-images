"""system_prep.py — the system changes utils/set_properties.sh makes.

Every distro's install.sh starts with `source ../../utils/set_properties.sh`.
Most of that script only exports variables (TOP_DIR, DISTRIBUTION, SKU_FAMILY,
...), which `component_env()` in utils/build_config.py provides for the Python
build. But it also changes the system, before install_utils.sh runs:

    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        if [[ "${TARGET_NODE_TYPE}" == "baremetal_3p" ]]; then
            echo "[set_properties.sh] Skipping apt update/upgrade on baremetal 3P node"
        else
            apt update
            apt upgrade -y
        fi
        apt install -y jq
    elif [[ $DISTRIBUTION == almalinux* ]]; then
        if [[ $DISTRIBUTION == "almalinux8.10" ]]; then
            rpm --import https://repo.almalinux.org/almalinux/RPM-GPG-KEY-AlmaLinux
        elif [[ $DISTRIBUTION == almalinux9* ]]; then
            rpm --import https://repo.almalinux.org/almalinux/RPM-GPG-KEY-AlmaLinux-9
        fi
        dnf install -y jq
    elif [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
        dnf install -y jq
    else
        # Rocky Linux, RHEL, and other RHEL-family distros
        dnf install -y jq
    fi

`prepare_system(env)` runs exactly those commands, in that order, and is
called by ImageBuilder._bootstrap() before install_utils.sh. jq is still
needed: install_utils.sh and the nested bash scripts read versions.json with
it. The `apt upgrade -y` matters for image equivalence: it brings every
package the marketplace image shipped up to date, as the bash build does.

The commands are kept exactly as bash writes them (`apt`, not `apt-get`) for
now; the package-manager abstraction work in ADO 991 will move them behind
utils/package_installer.py.
"""

from __future__ import annotations

from utils.component_config import DEFAULT_NODE_TYPE
from utils.logger import log_error, log_info
from utils.process import exec_program

_OP = "prepare-system"

ALMA8_GPG_KEY = "https://repo.almalinux.org/almalinux/RPM-GPG-KEY-AlmaLinux"
ALMA9_GPG_KEY = "https://repo.almalinux.org/almalinux/RPM-GPG-KEY-AlmaLinux-9"


def prepare_commands(distribution: str, node_type: str) -> list[list[str]]:
    """The commands set_properties.sh runs for this distro and node type, in order.

    A pure function (no side effects), so tests can check every branch.
    """
    if "ubuntu" in distribution:
        commands = []
        if (node_type or DEFAULT_NODE_TYPE) != "baremetal_3p":
            commands += [["apt", "update"], ["apt", "upgrade", "-y"]]
        return commands + [["apt", "install", "-y", "jq"]]

    if distribution.startswith("almalinux"):
        commands = []
        if distribution == "almalinux8.10":
            commands.append(["rpm", "--import", ALMA8_GPG_KEY])
        elif distribution.startswith("almalinux9"):
            commands.append(["rpm", "--import", ALMA9_GPG_KEY])
        return commands + [["dnf", "install", "-y", "jq"]]

    # Azure Linux 3.0, Rocky Linux, RHEL and other RHEL-family distros.
    return [["dnf", "install", "-y", "jq"]]


def prepare_system(env: dict[str, str]) -> int:
    """Run set_properties.sh's system changes. Returns 0, or 3 on failure.

    Like bash's `set -e`, the first failing command stops the rest.
    """
    distribution = env.get("DISTRIBUTION", "")
    node_type = env.get("TARGET_NODE_TYPE", "") or DEFAULT_NODE_TYPE
    commands = prepare_commands(distribution, node_type)

    upgrade = any(command[:2] == ["apt", "upgrade"] for command in commands)
    log_info(_OP, f"Preparing {distribution} as set_properties.sh does",
             distribution=distribution, node_type=node_type,
             commands=len(commands), apt_upgrade=upgrade)
    if "ubuntu" in distribution and not upgrade:
        # The same message bash echoes, so both builds' logs say why.
        log_info(_OP, "Skipping apt update/upgrade on baremetal 3P node",
                 node_type=node_type)

    for command in commands:
        rc = exec_program(command, _OP, env=env)
        if rc != 0:
            log_error(_OP, f"'{' '.join(command)}' failed with exit code {rc}; "
                           "stopping, as set_properties.sh's set -e would",
                      command=" ".join(command), exit_code=rc)
            return 3
    log_info(_OP, "System prepared", distribution=distribution)
    return 0
