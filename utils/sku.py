"""sku.py — SKU capability helpers (Python port of the SKU checks in utilities.sh).

Mirrors _is_ncv6_sku / sku_has_infiniband / sku_uses_ucx from utils/utilities.sh.
NCv6 is the one non-InfiniBand SKU: it skips DOCA-OFED and uses libfabric instead
of UCX. Everything else has InfiniBand and uses UCX.

sku_network_mode / sku_uses_ipoib mirror the bash functions of the same names.
Unlike the helpers above, they also depend on the deployment target
(TARGET_NODE_TYPE in bash): baremetal_1p uses the "mrc" network, and
baremetal_3p nodes have InfiniBand but don't use IPoIB.
"""

from __future__ import annotations

from utils.component_config import DEFAULT_NODE_TYPE

# The values sku_network_mode() can return (the strings bash echoes).
NETWORK_MODE_NO_RDMA = "no_rdma"          # no RDMA-capable fabric (NCv6)
NETWORK_MODE_MRC = "mrc"                  # MRC network (currently baremetal_1p)
NETWORK_MODE_STANDARD_IB = "standard_ib"  # the regular InfiniBand path


def is_ncv6_sku(sku):
    """True if `sku` is NCv6 (the single non-InfiniBand SKU)."""
    return sku == "NCv6"


def sku_has_infiniband(sku):
    """Whether the SKU has InfiniBand hardware.

    Used to decide DOCA-OFED vs. libfabric (see distros/*/install.sh).
    """
    return not is_ncv6_sku(sku)


def sku_uses_ucx(sku):
    """Whether the SKU uses UCX as its MPI transport layer (else libfabric)."""
    return not is_ncv6_sku(sku)


def sku_network_mode(sku, node_type=DEFAULT_NODE_TYPE):
    """The network mode for this build target, as bash's sku_network_mode.

    Bash checks, in this order:
        if _is_ncv6_sku; then echo "no_rdma"
        elif [[ "${TARGET_NODE_TYPE:-azure_vm_regular}" == "baremetal_1p" ]]; then echo "mrc"
        else echo "standard_ib"

    `node_type` is TARGET_NODE_TYPE. An empty value means the default, like
    bash's ${TARGET_NODE_TYPE:-azure_vm_regular}.
    """
    if is_ncv6_sku(sku):
        return NETWORK_MODE_NO_RDMA
    if (node_type or DEFAULT_NODE_TYPE) == "baremetal_1p":
        return NETWORK_MODE_MRC
    return NETWORK_MODE_STANDARD_IB


def sku_uses_ipoib(sku, node_type=DEFAULT_NODE_TYPE):
    """Whether this target enables IPoIB (IP over InfiniBand), as bash's sku_uses_ipoib.

    True only on the standard InfiniBand path, and never on baremetal_3p: those
    nodes have InfiniBand cards, but their customer doesn't need IPoIB.
    """
    node_type = node_type or DEFAULT_NODE_TYPE
    return (node_type != "baremetal_3p"
            and sku_network_mode(sku, node_type) == NETWORK_MODE_STANDARD_IB)
