"""Tests for the Ubuntu 26.04 build plan (utils/plans/ubuntu26_04.py).

The generic parity tests in test_distro_plans.py check the plan against
distros/ubuntu26.04/install.sh (order, coverage, scripts, cleanup). These
cover what they can't see: the SKUs install.sh refuses, and which SKUs get
which steps.
"""

from __future__ import annotations

import re
import unittest
from argparse import Namespace
from pathlib import Path

from utils.build_config import (BuildConfig, ConfigError, build_plan, pre_bootstrap_plan,
                                resolve_config)
from utils.plans import ubuntu26_04

_REPO = Path(__file__).resolve().parents[2]
_INSTALL_SH = _REPO / "distros" / "ubuntu26.04" / "install.sh"


def _cfg(vendor, gpu):
    return BuildConfig(vendor=vendor, gpu=gpu, os="Ubuntu26", fips=False, spec_path=None)


def _ops(vendor, gpu):
    cfg = _cfg(vendor, gpu)
    return [s.op for s in pre_bootstrap_plan(cfg) + build_plan(cfg) if s.when(cfg)]


class UnsupportedSkuTests(unittest.TestCase):
    """install.sh: `exit 1` for V100, GB200, VR200 and NCv6 before anything else."""

    def test_install_sh_rejects_exactly_these_skus(self):
        text = _INSTALL_SH.read_text(encoding="utf-8")
        in_bash = set(re.findall(r'"\$SKU" == "(\w+)"', text))
        self.assertEqual(in_bash, {"V100", "GB200", "VR200", "NCv6"})
        for sku in in_bash:
            with self.subTest(sku=sku):
                self.assertTrue(ubuntu26_04.reject_sku(sku))

    def test_messages_are_install_sh_text(self):
        text = _INSTALL_SH.read_text(encoding="utf-8")
        self.assertIn(f'echo "##[error]{ubuntu26_04.reject_sku("V100")}"', text)
        # The other three share one message, with $SKU in it.
        self.assertIn('echo "##[error]$SKU is not supported on Ubuntu 26.04 yet."', text)
        for sku in ("GB200", "VR200", "NCv6"):
            with self.subTest(sku=sku):
                self.assertEqual(ubuntu26_04.reject_sku(sku),
                                 f"{sku} is not supported on Ubuntu 26.04 yet.")
        for sku in ("A100", "H100", "H200", "GB300", "MI300X"):
            with self.subTest(sku=sku):
                self.assertEqual(ubuntu26_04.reject_sku(sku), "")

    def test_resolve_config_refuses_them_before_anything_runs(self):
        # Ubuntu 26.04 has no VR200 alias, so VR200 is named in its own message.
        for sku in ("V100", "GB200", "VR200", "NCv6"):
            with self.subTest(sku=sku):
                with self.assertRaises(ConfigError) as caught:
                    resolve_config(Namespace(vendor="NVidia", gpu=sku, os="Ubuntu26",
                                             spec=None, fips=False))
                self.assertEqual(str(caught.exception), ubuntu26_04.reject_sku(sku))
                self.assertEqual(caught.exception.code, 1)

    def test_supported_skus_resolve(self):
        for vendor, sku in (("NVidia", "A100"), ("NVidia", "GB300"), ("AMD", "MI300X")):
            with self.subTest(sku=sku):
                cfg = resolve_config(Namespace(vendor=vendor, gpu=sku, os="Ubuntu26",
                                               spec=None, fips=False))
                self.assertEqual(cfg.gpu, sku)

    def test_no_pre_bootstrap_steps(self):
        self.assertEqual(pre_bootstrap_plan(_cfg("NVidia", "A100")), [])


class GatingTests(unittest.TestCase):
    def test_nvidia_steps(self):
        for gpu in ("A100", "H100", "H200", "GB300"):
            with self.subTest(gpu=gpu):
                ops = _ops("NVidia", gpu)
                for op in ("install-nv-driver", "install-nccl", "install-dcgm"):
                    self.assertIn(op, ops)
                for op in ("install-rocm", "install-rccl", "install-nv-grid"):
                    self.assertNotIn(op, ops)

    def test_amd_steps(self):
        ops = _ops("AMD", "MI300X")
        self.assertIn("install-rocm", ops)
        self.assertIn("install-rccl", ops)
        for op in ("install-nv-driver", "install-nccl", "install-dcgm"):
            self.assertNotIn(op, ops)

    def test_no_pmix_and_no_no_rdma_path(self):
        # HPC-X brings PMIx 5; NCv6 (the no-RDMA SKU) is rejected, so DOCA always runs.
        for vendor, gpu in (("NVidia", "A100"), ("NVidia", "GB300"), ("AMD", "MI300X")):
            with self.subTest(gpu=gpu):
                ops = _ops(vendor, gpu)
                self.assertNotIn("install-pmix", ops)
                self.assertIn("install-doca", ops)
                for op in ("install-rdma-core", "install-libfabric", "blacklist-mana-ib"):
                    self.assertNotIn(op, ops)

    def test_order_differences_from_ubuntu_24_04(self):
        nvidia, amd = _ops("NVidia", "A100"), _ops("AMD", "MI300X")
        self.assertLess(nvidia.index("install-mpis"), nvidia.index("install-nv-driver"))
        self.assertLess(amd.index("install-rocm"), amd.index("install-mpis"))
        self.assertLess(nvidia.index("install-nccl"), nvidia.index("install-docker"))
        self.assertLess(amd.index("install-rccl"), amd.index("install-docker"))
        self.assertEqual(nvidia[-2:], ["disable-auto-upgrade",
                                       "disable-predictive-interface-renaming"])

    def test_every_sku_gets_the_monitoring_and_health_steps(self):
        # No GB200/NCv6 exceptions here: those SKUs are rejected.
        for vendor, gpu in (("NVidia", "A100"), ("NVidia", "GB300"), ("AMD", "MI300X")):
            with self.subTest(gpu=gpu):
                ops = _ops(vendor, gpu)
                for op in ("install-aznfs", "install-hpcdiag", "install-monitoring-tools",
                           "install-health-checks"):
                    self.assertIn(op, ops)

    def test_health_checks_get_the_gpu_argument(self):
        for vendor, gpu, arg in (("NVidia", "A100", "NVIDIA"), ("AMD", "MI300X", "AMD")):
            with self.subTest(vendor=vendor):
                steps = {s.op: s for s in build_plan(_cfg(vendor, gpu))}
                self.assertEqual(steps["install-health-checks"].args, (arg,))

    def test_x86_64_libraries_skip_aarch64(self):
        self.assertIn("install-amd-libs", _ops("NVidia", "A100"))
        self.assertIn("install-intel-libs", _ops("NVidia", "A100"))
        self.assertNotIn("install-amd-libs", _ops("NVidia", "GB300"))
        self.assertNotIn("install-intel-libs", _ops("NVidia", "GB300"))


if __name__ == "__main__":
    unittest.main()
