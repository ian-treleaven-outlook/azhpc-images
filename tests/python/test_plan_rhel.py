"""Tests for the RHEL 8 and 9 build plan (utils/plans/rhel.py).

The generic parity tests in test_distro_plans.py check the plan against each
install.sh (order, coverage, scripts, cleanup). These cover what they can't
see: the refusal of other vendors, the lack of SKU gating, the "$SKU"
argument, the trailing clear_history.sh, and how it relates to Rocky's plan.
"""

from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

from utils.build_config import (BuildConfig, ConfigError, build_plan, pre_bootstrap_plan,
                                resolve_config)
from utils.plans import rhel

_REPO = Path(__file__).resolve().parents[2]
_INSTALL_SH = {"RHEL8": _REPO / "distros" / "rhel8.10" / "install.sh",
               "RHEL9": _REPO / "distros" / "rhel9.x" / "install.sh"}
_NVIDIA_SKUS = ("A100", "V100", "H100", "H200", "NCv6", "GB200", "GB300", "VR200")


def _steps(gpu="A100", os_name="RHEL9", vendor="NVidia"):
    cfg = BuildConfig(vendor=vendor, gpu=gpu, os=os_name, fips=False, spec_path=None)
    return [s for s in pre_bootstrap_plan(cfg) + build_plan(cfg) if s.when(cfg)]


def _ops(gpu="A100", os_name="RHEL9", vendor="NVidia"):
    return [s.op for s in _steps(gpu, os_name, vendor)]


def _install_sh_text(os_name):
    return _INSTALL_SH[os_name].read_text(encoding="utf-8")


class VendorTests(unittest.TestCase):
    def test_amd_is_refused_with_install_sh_message(self):
        for os_name in ("RHEL8", "RHEL9"):
            with self.subTest(os=os_name):
                self.assertIn(f'echo "Error: {rhel.VENDOR_ERROR}"', _install_sh_text(os_name))
                with self.assertRaises(ConfigError) as caught:
                    resolve_config(Namespace(vendor="AMD", gpu="MI300X", os=os_name,
                                             spec=None, fips=False))
                self.assertEqual(str(caught.exception), rhel.VENDOR_ERROR)
                self.assertEqual(caught.exception.code, 1)

    def test_install_sh_scripts_are_identical(self):
        self.assertEqual(_install_sh_text("RHEL8"), _install_sh_text("RHEL9"))


class PlanTests(unittest.TestCase):
    def test_a100(self):
        self.assertEqual(_ops(), [
            "fix-setools-cyclecloud", "install-doca", "install-nv-driver",
            "install-pmix", "install-mpis", "install-lustre", "install-mpifileutils",
            "install-nccl", "install-docker", "install-dcgm",
            "install-amd-libs", "install-intel-libs", "cleanup-downloads",
            "hpc-tuning", "install-waagent", "install-hpcdiag", "install-aznfs",
            "install-monitoring-tools", "install-persistent-rdma-naming", "copy-test-file",
            "install-health-checks", "write-kernel-os-version", "install-azsecpack-prereqs",
            "disable-cloudinit", "setup-sku-customizations", "trivy-scan",
            "network-config", "clear-history",
        ])

    def test_same_steps_as_rocky_on_nvidia(self):
        # RHEL's install.sh is Rocky's without the AMD branches.
        for os_name, rocky in (("RHEL8", "Rocky8"), ("RHEL9", "Rocky9")):
            with self.subTest(os=os_name):
                self.assertEqual(_ops(os_name=os_name), _ops(os_name=rocky))

    def test_no_sku_gating_like_install_sh(self):
        self.assertNotIn('"$SKU" =', _install_sh_text("RHEL9"))   # = and ==
        for os_name in ("RHEL8", "RHEL9"):
            for sku in _NVIDIA_SKUS:
                with self.subTest(os=os_name, sku=sku):
                    self.assertEqual(_ops(sku, os_name), _ops())

    def test_step_arguments_are_install_sh_ones(self):
        self.assertIn('$COMPONENT_DIR/install_nvidiagpudriver.sh "$SKU"', _install_sh_text("RHEL9"))
        self.assertIn('$COMPONENT_DIR/install_health_checks.sh "$GPU"', _install_sh_text("RHEL9"))
        for sku in ("V100", "NCv6"):
            with self.subTest(sku=sku):
                steps = {s.op: s for s in _steps(sku)}
                self.assertEqual(steps["install-nv-driver"].args, (sku,))
                self.assertEqual(steps["install-health-checks"].args, ("NVIDIA",))

    def test_clear_history_runs_last_as_bash_from_utils(self):
        self.assertTrue(_install_sh_text("RHEL9").rstrip().endswith(
            "$UTILS_DIR/clear_history.sh"))
        last = _steps()[-1]
        self.assertEqual((last.op, last.base, last.runner), ("clear-history", "utils", "bash"))
        self.assertTrue(all(s.runner == "python" for s in _steps()[:-1]))

    def test_each_os_cleans_its_own_folder(self):
        for os_name, folder in (("RHEL8", "distros/rhel8.10"), ("RHEL9", "distros/rhel9.x")):
            with self.subTest(os=os_name):
                step = {s.op: s for s in _steps(os_name=os_name)}["cleanup-downloads"]
                self.assertEqual(step.action.keywords["distro_dir"], folder)


if __name__ == "__main__":
    unittest.main()
