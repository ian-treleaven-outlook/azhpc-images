"""Unit tests for utils/version_validate.py (Step 2: CUDA-major consistency).

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import unittest

from utils.version_validate import (
    active_hpcx_component,
    cuda_major_from_config,
    cuda_major_in_url,
    check_cuda_consistency,
)


class HelperTests(unittest.TestCase):
    def test_cuda_major_from_config(self):
        self.assertEqual(cuda_major_from_config({"driver": {"version": "13.0"}}), 13)

    def test_cuda_major_from_config_missing(self):
        self.assertIsNone(cuda_major_from_config({}))
        self.assertIsNone(cuda_major_from_config({"driver": {}}))

    def test_cuda_major_in_url(self):
        self.assertEqual(
            cuda_major_in_url("https://x/hpcx-v2.25.1-...-cuda13-x86_64.tbz"), 13)

    def test_cuda_major_in_url_absent(self):
        self.assertIsNone(cuda_major_in_url("https://x/hpcx-nocuda.tbz"))
        self.assertIsNone(cuda_major_in_url(None))

    def test_active_hpcx_variant_selection(self):
        self.assertEqual(active_hpcx_component("AMD", "MI300"), "hpcx_amd")
        self.assertEqual(active_hpcx_component("NVIDIA", "NCv6"), "hpcx_inbox")
        self.assertEqual(active_hpcx_component("NVIDIA", "A100"), "hpcx")


class CudaConsistencyTests(unittest.TestCase):
    def _versions(self, hpcx_cuda="cuda13", driver="13.0"):
        return {
            "cuda": {"common": {"driver": {"version": driver}}},
            "hpcx": {"common": {
                "version": "2.25.1",
                "url": f"https://x/hpcx-v2.25.1-doca_ofed-ubuntu24.04-{hpcx_cuda}-x86_64.tbz",
            }},
            "hpcx_inbox": {"common": {"url": "https://x/hpcx-inbox-cuda13.tbz"}},
            "hpcx_amd": {"common": {"url": "https://x/hpcx-amd-cuda12.tbz"}},
        }

    def _check(self, versions, gpu="NVIDIA", sku="A100"):
        return check_cuda_consistency(
            versions, distribution="ubuntu24.04", architecture="x86_64",
            gpu=gpu, sku=sku)

    def test_consistent_returns_no_violations(self):
        self.assertEqual(self._check(self._versions("cuda13", "13.0")), [])

    def test_mismatch_is_flagged(self):
        issues = self._check(self._versions("cuda12", "13.0"))
        self.assertEqual(len(issues), 1)
        self.assertIn("hpcx", issues[0])
        self.assertIn("CUDA 12", issues[0])
        self.assertIn("CUDA 13", issues[0])

    def test_only_selected_variant_checked(self):
        # hpcx_amd's url says cuda12, but on an NVIDIA/A100 target only `hpcx`
        # is installed, so the mismatched sibling must not be flagged.
        self.assertEqual(self._check(self._versions("cuda13", "13.0")), [])

    def test_ncv6_checks_inbox_variant(self):
        v = self._versions("cuda13", "13.0")
        v["hpcx_inbox"]["common"]["url"] = "https://x/hpcx-inbox-cuda12.tbz"
        issues = self._check(v, gpu="NVIDIA", sku="NCv6")
        self.assertEqual(len(issues), 1)
        self.assertIn("hpcx_inbox", issues[0])

    def test_non_nvidia_target_skipped(self):
        # AMD images install no CUDA, so nothing is checked even though the
        # cuda entry resolves via fallback.
        self.assertEqual(self._check(self._versions(), gpu="AMD", sku="MI300"), [])


if __name__ == "__main__":
    unittest.main()
