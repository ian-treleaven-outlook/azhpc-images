"""sku.py — SKU capability helpers (Python port of the SKU checks in utilities.sh).

Mirrors _is_ncv6_sku / sku_has_infiniband / sku_uses_ucx from utils/utilities.sh.
NCv6 is the one non-InfiniBand SKU: it skips DOCA-OFED and uses libfabric instead
of UCX. Everything else has InfiniBand and uses UCX.
"""

from __future__ import annotations


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
