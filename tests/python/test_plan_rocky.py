"""Tests for the Rocky Linux 8 and 9 build plan (utils/plans/rocky.py).

The generic parity tests in test_distro_plans.py check the plan against each
install.sh (order, coverage, scripts, cleanup). These cover what they can't
see: the per-vendor gating, the lack of SKU gating, the "$SKU" arguments,
the trailing clear_history.sh, and how the order differs from Ubuntu 24.04's.
"""

from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

from utils.build_config import BuildConfig, build_plan, pre_bootstrap_plan, resolve_config

_REPO = Path(__file__).resolve().parents[2]
_INSTALL_SH = {"Rocky8": _REPO / "distros" / "rocky8.10" / "install.sh",
               "Rocky9": _REPO / "distros" / "rocky9.x" / "install.sh"}
_NVIDIA_SKUS = ("A100", "V100", "H100", "H200", "NCv6", "GB200", "GB300", "VR200")


def _steps(vendor, gpu, os_name="Rocky9"):
    cfg = BuildConfig(vendor=vendor, gpu=gpu, os=os_name, fips=False, spec_path=None)
    return [s for s in pre_bootstrap_plan(cfg) + build_plan(cfg) if s.when(cfg)]


def _ops(vendor, gpu, os_name="Rocky9"):
    return [s.op for s in _steps(vendor, gpu, os_name)]


def _install_sh_text(os_name):
    return _INSTALL_SH[os_name].read_text(encoding="utf-8")


# Every step both vendors run after the GPU-specific ones, in install.sh's order.
_COMMON_TAIL = [
    "install-amd-libs", "install-intel-libs", "cleanup-downloads",
    "hpc-tuning", "install-waagent", "install-hpcdiag", "install-aznfs",
    "install-monitoring-tools", "install-persistent-rdma-naming", "copy-test-file",
    "install-health-checks", "write-kernel-os-version", "install-azsecpack-prereqs",
    "disable-cloudinit", "setup-sku-customizations", "trivy-scan",
    "network-config", "clear-history",
]


class VendorTests(unittest.TestCase):
    def test_both_vendors_resolve(self):
        for vendor, sku in (("NVidia", "A100"), ("AMD", "MI300X")):
            with self.subTest(vendor=vendor):
                cfg = resolve_config(Namespace(vendor=vendor, gpu=sku, os="Rocky8",
                                               spec=None, fips=False))
                self.assertEqual(cfg.gpu, sku)

    def test_nvidia(self):
        self.assertEqual(_ops("NVidia", "A100"), [
            "fix-setools-cyclecloud", "install-doca", "install-nv-driver",
            "install-pmix", "install-mpis", "install-lustre", "install-mpifileutils",
            "install-nccl", "install-docker", "install-dcgm", *_COMMON_TAIL])

    def test_amd(self):
        self.assertEqual(_ops("AMD", "MI300X"), [
            "fix-setools-cyclecloud", "install-doca", "install-rocm",
            "install-pmix", "install-mpis", "install-lustre", "install-mpifileutils",
            "install-docker", "install-rccl", *_COMMON_TAIL])

    def test_health_checks_get_the_gpu_argument(self):
        for vendor, gpu, arg in (("NVidia", "A100", "NVIDIA"), ("AMD", "MI300X", "AMD")):
            with self.subTest(vendor=vendor):
                step = {s.op: s for s in _steps(vendor, gpu)}["install-health-checks"]
                self.assertEqual(step.args, (arg,))


class SkuTests(unittest.TestCase):
    def test_install_sh_scripts_are_identical(self):
        self.assertEqual(_install_sh_text("Rocky8"), _install_sh_text("Rocky9"))

    def test_no_sku_gating_like_install_sh(self):
        # install.sh only passes $SKU on; it never tests it.
        self.assertNotIn('"$SKU" =', _install_sh_text("Rocky9"))   # = and ==
        for os_name in ("Rocky8", "Rocky9"):
            for sku in _NVIDIA_SKUS:
                with self.subTest(os=os_name, sku=sku):
                    self.assertEqual(_ops("NVidia", sku, os_name), _ops("NVidia", "A100"))

    def test_driver_and_rocm_get_the_sku_like_install_sh(self):
        text = _install_sh_text("Rocky9")
        self.assertIn('$COMPONENT_DIR/install_nvidiagpudriver.sh "$SKU"', text)
        self.assertIn('$COMPONENT_DIR/install_rocm.sh "$SKU"', text)
        for vendor, gpu, op in (("NVidia", "H100", "install-nv-driver"),
                                ("NVidia", "NCv6", "install-nv-driver"),
                                ("AMD", "MI300X", "install-rocm")):
            with self.subTest(gpu=gpu):
                step = {s.op: s for s in _steps(vendor, gpu)}[op]
                self.assertEqual(step.args, (gpu,))

    def test_each_os_cleans_its_own_folder(self):
        for os_name, folder in (("Rocky8", "distros/rocky8.10"), ("Rocky9", "distros/rocky9.x")):
            with self.subTest(os=os_name):
                step = {s.op: s for s in _steps("NVidia", "A100", os_name)}["cleanup-downloads"]
                self.assertEqual(step.action.keywords["distro_dir"], folder)


class EndOfBuildTests(unittest.TestCase):
    def test_clear_history_runs_last_as_bash_from_utils(self):
        self.assertTrue(_install_sh_text("Rocky9").rstrip().endswith(
            "$UTILS_DIR/clear_history.sh"))
        last = _steps("NVidia", "A100")[-1]
        self.assertEqual((last.op, last.base, last.runner), ("clear-history", "utils", "bash"))

    def test_every_other_step_runs_python(self):
        for vendor, gpu in (("NVidia", "A100"), ("AMD", "MI300X")):
            with self.subTest(vendor=vendor):
                self.assertTrue(all(s.runner == "python" for s in _steps(vendor, gpu)[:-1]))

    def test_order_differences_from_ubuntu_24_04(self):
        ops = _ops("NVidia", "A100")
        self.assertLess(ops.index("install-mpifileutils"), ops.index("install-nccl"))
        self.assertLess(ops.index("copy-test-file"), ops.index("install-health-checks"))
        for op in ("add-udev-rules", "disable-predictive-interface-renaming",
                   "install-nv-grid", "install-rdma-core", "install-nvshmem",
                   "upgrade-all-packages"):
            self.assertNotIn(op, ops)


if __name__ == "__main__":
    unittest.main()
