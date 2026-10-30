"""Tests for the Azure Linux 3.0 build plan (utils/plans/azurelinux3_0.py).

test_distro_plans.py already checks the plan against install.sh (order,
coverage, scripts, cleanup). These pin the per-SKU gating it can't see: the
exact steps each SKU runs.
"""

from __future__ import annotations

import unittest

from utils.build_config import BuildConfig, build_plan


def _steps(vendor, gpu):
    cfg = BuildConfig(vendor=vendor, gpu=gpu, os="Azure3", fips=False, spec_path=None)
    return [s for s in build_plan(cfg) if s.when(cfg)]


def _ops(vendor, gpu):
    return [s.op for s in _steps(vendor, gpu)]


class AzureLinuxPlanTests(unittest.TestCase):
    def test_a100(self):
        self.assertEqual(_ops("NVidia", "A100"), [
            "install-mofed", "install-nv-driver",
            "install-pmix", "install-mpis", "install-mpifileutils",
            "install-nccl", "install-docker", "install-dcgm",
            "install-amd-libs", "install-intel-libs", "cleanup-downloads",
            "hpc-tuning", "install-waagent", "install-aznfs", "install-health-checks",
            "write-kernel-os-version", "install-azsecpack-prereqs", "install-hpcdiag",
            "install-monitoring-tools", "install-persistent-rdma-naming",
            "add-udev-rules", "copy-test-file", "setup-sku-customizations",
            "trivy-scan", "network-config",
        ])

    def test_gb200(self):
        # NVLink tools after NCCL; no AZNFS/NHC; no AMD/Intel libraries (aarch64).
        self.assertEqual(_ops("NVidia", "GB200"), [
            "install-mofed", "install-nv-driver",
            "install-pmix", "install-mpis", "install-mpifileutils",
            "install-nccl", "install-nvshmem", "install-nvloom", "install-nvbandwidth",
            "install-docker", "install-dcgm", "cleanup-downloads",
            "hpc-tuning", "install-waagent",
            "write-kernel-os-version", "install-azsecpack-prereqs", "install-hpcdiag",
            "install-monitoring-tools", "install-persistent-rdma-naming",
            "add-udev-rules", "copy-test-file", "setup-sku-customizations",
            "trivy-scan", "network-config",
        ])

    def test_mi300x(self):
        self.assertEqual(_ops("AMD", "MI300X"), [
            "install-mofed", "install-rocm",
            "install-pmix", "install-mpis", "install-mpifileutils",
            "install-docker", "install-rccl",
            "install-amd-libs", "install-intel-libs", "cleanup-downloads",
            "hpc-tuning", "install-waagent", "install-aznfs", "install-health-checks",
            "write-kernel-os-version", "install-azsecpack-prereqs", "install-hpcdiag",
            "install-monitoring-tools", "install-persistent-rdma-naming",
            "add-udev-rules", "copy-test-file", "setup-sku-customizations",
            "trivy-scan", "network-config",
        ])

    def test_every_nvidia_sku_gets_mofed_and_the_one_driver_script(self):
        # Unlike Ubuntu: no DOCA/rdma-core split, and no GRID driver for NCv6.
        for gpu in ("A100", "H100", "H200", "V100", "NCv6", "GB200", "GB300"):
            with self.subTest(gpu=gpu):
                ops = _ops("NVidia", gpu)
                self.assertEqual(ops[:2], ["install-mofed", "install-nv-driver"])
                self.assertNotIn("install-nv-grid", ops)
                self.assertNotIn("install-doca", ops)

    def test_only_gb200_skips_aznfs_and_health_checks(self):
        # install.sh tests SKU == "GB200" only, so GB300 and NCv6 keep them.
        for gpu in ("A100", "V100", "NCv6", "GB300"):
            with self.subTest(gpu=gpu):
                self.assertIn("install-aznfs", _ops("NVidia", gpu))
                self.assertIn("install-health-checks", _ops("NVidia", gpu))

    def test_gb300_gets_no_nvlink_tools_like_install_sh(self):
        ops = _ops("NVidia", "GB300")
        for op in ("install-nvshmem", "install-nvloom", "install-nvbandwidth"):
            self.assertNotIn(op, ops)

    def test_health_checks_get_the_gpu_vendor_argument(self):
        for vendor, gpu, arg in (("NVidia", "A100", "NVIDIA"), ("AMD", "MI300X", "AMD")):
            with self.subTest(vendor=vendor):
                step = {s.op: s for s in _steps(vendor, gpu)}["install-health-checks"]
                self.assertEqual(step.args, (arg,))

    def test_network_config_is_last_and_from_the_distro_folder(self):
        step = _steps("NVidia", "A100")[-1]
        self.assertEqual((step.op, step.script, step.base),
                         ("network-config", "network-config.sh", "distro"))

    def test_no_step_runs_bash_for_lack_of_a_port(self):
        for vendor, gpu in (("NVidia", "A100"), ("NVidia", "GB200"), ("AMD", "MI300X")):
            with self.subTest(gpu=gpu):
                self.assertTrue(all(s.runner == "python" for s in _steps(vendor, gpu)))


if __name__ == "__main__":
    unittest.main()
