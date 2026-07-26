"""Unit tests for utils/sku.py (SKU capability helpers).

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import unittest

from utils.sku import is_ncv6_sku, sku_has_infiniband, sku_uses_ucx


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


if __name__ == "__main__":
    unittest.main()
