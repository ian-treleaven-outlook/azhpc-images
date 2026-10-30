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
from unittest import mock

from utils.build_config import (BuildConfig, ImageBuilder, build_plan, pre_bootstrap_plan,
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
        rejected = set(re.findall(r'"\$SKU" == "(\w+)"', text))
        self.assertEqual(rejected, set(ubuntu26_04.UNSUPPORTED_SKUS))

    def test_messages_are_install_sh_text(self):
        text = _INSTALL_SH.read_text(encoding="utf-8")
        self.assertIn(f'echo "##[error]{ubuntu26_04.unsupported_sku_error("V100")}"', text)
        # The other three share one message, with $SKU in it.
        self.assertIn('echo "##[error]$SKU is not supported on Ubuntu 26.04 yet."', text)
        for sku in ("GB200", "VR200", "NCv6"):
            with self.subTest(sku=sku):
                self.assertEqual(ubuntu26_04.unsupported_sku_error(sku),
                                 f"{sku} is not supported on Ubuntu 26.04 yet.")
        for sku in ("A100", "H100", "H200", "GB300", "MI300X"):
            with self.subTest(sku=sku):
                self.assertIsNone(ubuntu26_04.unsupported_sku_error(sku))

    def test_rejected_skus_get_only_the_rejection(self):
        # install.sh exits before install_utils.sh, so nothing else is planned.
        for gpu in ("V100", "GB200", "NCv6"):
            with self.subTest(gpu=gpu):
                self.assertEqual(_ops("NVidia", gpu), ["reject-unsupported-sku"])
        for vendor, gpu in (("NVidia", "A100"), ("NVidia", "GB300"), ("AMD", "MI300X")):
            with self.subTest(gpu=gpu):
                self.assertNotIn("reject-unsupported-sku", _ops(vendor, gpu))

    def test_vr200_is_rejected_too(self):
        # resolve_config builds VR200 as GB200 (SKU_ALIASES), which is rejected.
        with mock.patch("utils.build_config.log_warn"):
            cfg = resolve_config(Namespace(vendor="NVidia", gpu="VR200", os="Ubuntu26",
                                           spec=None, fips=False))
        step = pre_bootstrap_plan(cfg)[0]
        self.assertTrue(step.when(cfg))

    def test_the_step_logs_install_sh_message_and_fails(self):
        step = pre_bootstrap_plan(_cfg("NVidia", "V100"))[0]
        self.assertEqual(step.runner, "python")
        with mock.patch.object(ubuntu26_04, "log_error") as log_error:
            self.assertEqual(step.action({"SKU": "V100"}), 3)
        log_error.assert_called_once_with(
            "reject-unsupported-sku", ubuntu26_04.unsupported_sku_error("V100"), sku="V100")

    def test_a_rejected_sku_stops_before_install_utils(self):
        builder = ImageBuilder(_REPO, _cfg("NVidia", "NCv6"))
        with mock.patch("utils.build_config.prepare_system", return_value=0), \
             mock.patch("utils.build_config.exec_program") as exec_program, \
             mock.patch("utils.build_config.log_info"), \
             mock.patch("utils.build_config.log_error"), \
             mock.patch.object(ubuntu26_04, "log_error"):
            rc = builder._bootstrap(_REPO / ubuntu26_04.DISTRO_DIR, {"SKU": "NCv6"})
        self.assertEqual(rc, 3)
        exec_program.assert_not_called()


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
