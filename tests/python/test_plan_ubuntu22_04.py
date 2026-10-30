"""Tests for utils/plans/ubuntu22_04.py: what the generic parity tests in
test_distro_plans.py can't see.

Those check the plan against distros/ubuntu22.04/install.sh (order, coverage,
scripts, cleanup). These check the gating: install.sh only branches on the
GPU vendor, never on the SKU, so unlike Ubuntu 24.04 every SKU gets DOCA,
the shared NVIDIA driver and the monitoring/health-check steps.
"""

from __future__ import annotations

import unittest

from components.python import remove_unused_packages
from utils.build_config import BuildConfig, build_plan, pre_bootstrap_plan

_NVIDIA_SKUS = ["A100", "H100", "H200", "V100", "NCv6", "GB200"]

# distros/ubuntu22.04/install.sh for an NVIDIA SKU, top to bottom.
_NVIDIA_STEPS = [
    "install-doca", "install-nv-driver", "install-pmix", "install-mpis",
    "install-nccl", "install-docker", "install-dcgm",
    "install-lustre", "install-mpifileutils", "install-amd-libs", "install-intel-libs",
    "cleanup-downloads", "hpc-tuning", "install-waagent", "install-aznfs",
    "install-hpcdiag", "install-monitoring-tools", "install-persistent-rdma-naming",
    "add-udev-rules", "copy-test-file", "install-health-checks",
    "write-kernel-os-version", "install-azsecpack-prereqs", "disable-cloudinit",
    "setup-sku-customizations", "trivy-scan", "disable-predictive-interface-renaming",
]


def _cfg(vendor, gpu):
    return BuildConfig(vendor=vendor, gpu=gpu, os="Ubuntu22", fips=False, spec_path=None)


def _ops(cfg):
    return [s.op for s in build_plan(cfg) if s.when(cfg)]


class PreBootstrapTests(unittest.TestCase):
    def test_remove_unused_packages_runs_before_install_utils(self):
        for vendor, gpu in [("NVidia", "A100"), ("AMD", "MI300X")]:
            with self.subTest(gpu=gpu):
                pre = pre_bootstrap_plan(_cfg(vendor, gpu))
                self.assertEqual([s.op for s in pre], ["remove-unused-packages"])
                self.assertEqual(pre[0].base, "distro")
                self.assertIs(pre[0].action, remove_unused_packages.install)


class GatingTests(unittest.TestCase):
    def test_every_nvidia_sku_gets_the_same_steps(self):
        # No no-RDMA path, GRID driver or GB200 NVLink tools on 22.04.
        for gpu in _NVIDIA_SKUS:
            with self.subTest(gpu=gpu):
                self.assertEqual(_ops(_cfg("NVidia", gpu)), _NVIDIA_STEPS)

    def test_amd_gets_rocm_before_mpi_and_rccl_instead_of_nvidia_steps(self):
        ops = _ops(_cfg("AMD", "MI300X"))
        self.assertEqual(ops[:4], ["install-doca", "install-rocm", "install-pmix",
                                   "install-mpis"])
        self.assertEqual(ops[4:7], ["install-docker", "install-rccl", "install-lustre"])
        for nvidia_only in ("install-nv-driver", "install-nccl", "install-dcgm"):
            self.assertNotIn(nvidia_only, ops)

    def test_health_checks_get_the_install_sh_gpu_argument(self):
        for vendor, gpu, arg in [("NVidia", "A100", "NVIDIA"), ("AMD", "MI300X", "AMD")]:
            with self.subTest(gpu=gpu):
                cfg = _cfg(vendor, gpu)
                step = {s.op: s for s in build_plan(cfg)}["install-health-checks"]
                self.assertEqual(step.args, (arg,))


if __name__ == "__main__":
    unittest.main()
