"""Unit tests for utils/sku.py (SKU capability helpers).

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import unittest

from utils.sku import (is_ncv6_sku, sku_has_infiniband, sku_network_mode,
                       sku_uses_ipoib, sku_uses_ucx)


# The whole point of these helpers: NCv6 is the odd SKU out. It's the only one without
# InfiniBand, so it skips DOCA-OFED and uses libfabric instead of UCX. Everything else
# is "normal". These tests pin that down so a future SKU change can't silently flip it.
class SkuHelperTests(unittest.TestCase):
    def test_is_ncv6_sku(self):
        self.assertTrue(is_ncv6_sku("NCv6"))
        for sku in ("A100", "H100", "GB200", "MI300", "V100"):
            self.assertFalse(is_ncv6_sku(sku), sku)

    def test_sku_has_infiniband(self):
        self.assertFalse(sku_has_infiniband("NCv6"))
        for sku in ("A100", "H100", "GB200", "MI300"):
            self.assertTrue(sku_has_infiniband(sku), sku)

    def test_sku_uses_ucx(self):
        self.assertFalse(sku_uses_ucx("NCv6"))
        for sku in ("A100", "H100", "GB200", "MI300"):
            self.assertTrue(sku_uses_ucx(sku), sku)


# sku_network_mode / sku_uses_ipoib also depend on the node type (bash reads
# TARGET_NODE_TYPE). The expected values below come from running the bash
# functions in utils/utilities.sh with the same SKU and TARGET_NODE_TYPE.
class NetworkModeTests(unittest.TestCase):
    # (sku, node_type, expected network mode, expected uses_ipoib)
    CASES = [
        ("NCv6", "azure_vm_regular", "no_rdma", False),
        ("NCv6", "baremetal_1p", "no_rdma", False),      # NCv6 wins over MRC
        ("NCv6", "baremetal_3p", "no_rdma", False),
        ("H100", "azure_vm_regular", "standard_ib", True),
        ("H100", "azure_vm_akshost", "standard_ib", True),
        ("GB200", "baremetal_1p", "mrc", False),
        ("GB200", "baremetal_3p", "standard_ib", False),  # IB, but no IPoIB
        ("MI300", "azure_vm_regular", "standard_ib", True),
        ("A100", "", "standard_ib", True),                # empty -> default
    ]

    def test_network_mode_and_ipoib_match_bash(self):
        for sku, node_type, mode, ipoib in self.CASES:
            with self.subTest(sku=sku, node_type=node_type):
                self.assertEqual(sku_network_mode(sku, node_type), mode)
                self.assertIs(sku_uses_ipoib(sku, node_type), ipoib)

    def test_node_type_defaults_to_azure_vm_regular(self):
        self.assertEqual(sku_network_mode("H100"), "standard_ib")
        self.assertTrue(sku_uses_ipoib("H100"))
        self.assertFalse(sku_uses_ipoib("NCv6"))


if __name__ == "__main__":
    unittest.main()
