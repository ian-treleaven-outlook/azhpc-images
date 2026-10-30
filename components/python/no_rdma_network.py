"""no_rdma_network.py — the inline no-RDMA steps of distros/ubuntu24.04/install.sh.

On SKUs without an RDMA fabric (sku_network_mode "no_rdma"; today only NCv6),
install.sh skips DOCA-OFED and does this instead, inline rather than in a
component script:

    else
        # Non-IB SKUs: install rdma-core for kernel-native IB module management (mana_ib support)
        apt-get install -y rdma-core libibverbs-dev ibverbs-utils librdmacm-dev pkg-config
        # Install libfabric — replaces UCX as the networking abstraction for MPI on non-IB SKUs
        $COMPONENT_DIR/install_libfabric.sh
        # Blacklist mana_ib — it exposes a non-functional verbs device (max_msg_size=0, no UD/SRQ,
        # guest RDMA not yet enabled) that causes UCX, libfabric verbs, and UCC to crash.
        # The mana ethernet driver (eth0/eth1) is unaffected.
        # Customers can re-enable: sudo rm /etc/modprobe.d/blacklist-mana-ib.conf && sudo modprobe mana_ib
        echo "blacklist mana_ib" | tee /etc/modprobe.d/blacklist-mana-ib.conf
    fi

build_plan() runs these as two steps around install-libfabric, in the same
order: `install-rdma-core` (install_rdma_core) and `blacklist-mana-ib`
(blacklist_mana_ib). They have no bash script of their own, like the
cleanup step.

Only Ubuntu 24.04's install.sh has this branch (NCv6 is built on Ubuntu 24.04
only). The package list is the port's; install_packages()
(utils/package_installer.py) picks apt-get from DISTRIBUTION and installs it
in one command, as bash does.

Deliberate differences
----------------------
  * install_packages() runs `apt-get install -y -o Dpkg::Use-Pty=0 ...`
    (the documented normalisation; same packages, same single command).
"""

from __future__ import annotations

from pathlib import Path

from utils.logger import log_error, log_info
from utils.package_installer import install_packages

RDMA_CORE_PACKAGES = ["rdma-core", "libibverbs-dev", "ibverbs-utils",
                      "librdmacm-dev", "pkg-config"]

# Where the blacklist goes, and its exact contents (echo adds the newline;
# tee overwrites, so a rerun leaves the same file).
BLACKLIST_FILE = "/etc/modprobe.d/blacklist-mana-ib.conf"
BLACKLIST_TEXT = "blacklist mana_ib\n"


def install_rdma_core(env: dict[str, str]) -> int:
    """Install the kernel-native RDMA userspace packages. 0, or 3 on failure."""
    op = "install-rdma-core"
    log_info(op, "Installing rdma-core for a no-RDMA SKU (kernel-native IB "
                 "modules, mana_ib support)",
             sku=env.get("SKU", ""), packages=" ".join(RDMA_CORE_PACKAGES))
    if not install_packages(RDMA_CORE_PACKAGES, env, op):
        return 3
    return 0


def blacklist_mana_ib(env: dict[str, str]) -> int:
    """Write the mana_ib modprobe blacklist. 0, or 3 on failure."""
    op = "blacklist-mana-ib"
    path = Path(BLACKLIST_FILE)
    action = "replaced" if path.exists() else "created"
    try:
        path.write_text(BLACKLIST_TEXT, encoding="utf-8")
    except OSError as exc:
        log_error(op, f"could not write {path}: {exc}", path=str(path))
        return 3
    log_info(op, f"Wrote {path} ({action}): mana_ib exposes a non-functional "
                 "verbs device on MANA-only hardware",
             path=str(path), action=action, sku=env.get("SKU", ""))
    return 0
