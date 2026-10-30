"""Tests for the AlmaLinux 8 and 9 build plan (utils/plans/almalinux.py).

The generic parity tests in test_distro_plans.py check the plan against each
install.sh (order, coverage, scripts, cleanup). These cover what they can't
see: the AMD refusal, the lack of SKU gating, the inline `dnf update -y`,
and how the order differs from Ubuntu 24.04's.
"""

from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

from utils.build_config import (BuildConfig, ConfigError, build_plan, pre_bootstrap_plan,
                                resolve_config)
from utils.plans import almalinux

_REPO = Path(__file__).resolve().parents[2]
_INSTALL_SH = {"Alma8": _REPO / "distros" / "almalinux8.10" / "install.sh",
               "Alma9": _REPO / "distros" / "almalinux9.x" / "install.sh"}
_NVIDIA_SKUS = ("A100", "V100", "H100", "H200", "NCv6", "GB200", "GB300", "VR200")


def _cfg(os_name, gpu="A100"):
    return BuildConfig(vendor="NVidia", gpu=gpu, os=os_name, fips=False, spec_path=None)


def _steps(os_name, gpu="A100"):
    cfg = _cfg(os_name, gpu)
    return [s for s in pre_bootstrap_plan(cfg) + build_plan(cfg) if s.when(cfg)]


def _ops(os_name, gpu="A100"):
    return [s.op for s in _steps(os_name, gpu)]


def _commands(os_name):
    """install.sh's lines that aren't comments or blank."""
    lines = _INSTALL_SH[os_name].read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


class VendorTests(unittest.TestCase):
    def test_amd_is_refused_with_install_sh_message(self):
        for os_name in ("Alma8", "Alma9"):
            with self.subTest(os=os_name):
                self.assertIn(f'echo "Error: {almalinux.VENDOR_ERROR}"', _commands(os_name))
                with self.assertRaises(ConfigError) as caught:
                    resolve_config(Namespace(vendor="AMD", gpu="MI300X", os=os_name,
                                             spec=None, fips=False))
                self.assertEqual(str(caught.exception), almalinux.VENDOR_ERROR)
                self.assertEqual(caught.exception.code, 1)

    def test_every_nvidia_sku_resolves_unchanged(self):
        # install.sh has no SKU aliases or refusals.
        for sku in _NVIDIA_SKUS:
            with self.subTest(sku=sku):
                cfg = resolve_config(Namespace(vendor="NVidia", gpu=sku, os="Alma9",
                                               spec=None, fips=False))
                self.assertEqual(cfg.gpu, sku)


class PlanTests(unittest.TestCase):
    def test_alma9_a100(self):
        self.assertEqual(_ops("Alma9"), [
            "fix-setools-cyclecloud", "install-doca", "install-nv-driver",
            "install-pmix", "install-mpis", "install-lustre", "install-mpifileutils",
            "install-nccl", "install-docker", "install-dcgm",
            "install-amd-libs", "install-intel-libs", "cleanup-downloads",
            "hpc-tuning", "install-waagent", "install-hpcdiag", "install-aznfs",
            "install-monitoring-tools", "install-persistent-rdma-naming", "copy-test-file",
            "install-health-checks", "write-kernel-os-version", "install-azsecpack-prereqs",
            "disable-cloudinit", "setup-sku-customizations", "trivy-scan",
            "network-config", "upgrade-all-packages",
        ])

    def test_install_sh_scripts_differ_only_by_the_upgrade(self):
        alma8 = _INSTALL_SH["Alma8"].read_text(encoding="utf-8")
        alma9 = _INSTALL_SH["Alma9"].read_text(encoding="utf-8")
        self.assertEqual(alma9.replace("dnf update -y\n\n", "", 1), alma8)

    def test_only_alma9_upgrades_every_package_after_network_config(self):
        self.assertIn("dnf update -y", _commands("Alma9"))
        self.assertNotIn("dnf update -y", _commands("Alma8"))
        self.assertEqual(_ops("Alma9")[-2:], ["network-config", "upgrade-all-packages"])
        self.assertEqual(_ops("Alma8")[-1], "network-config")
        self.assertEqual(_ops("Alma8"), _ops("Alma9")[:-1])

    def test_no_sku_gating_like_install_sh(self):
        # install.sh never tests $SKU: GB200 and NCv6 run the same steps as A100.
        for os_name in ("Alma8", "Alma9"):
            for sku in _NVIDIA_SKUS:
                with self.subTest(os=os_name, sku=sku):
                    self.assertEqual(_ops(os_name, sku), _ops(os_name))
        commands = _commands("Alma9")
        after_export = commands[commands.index("export SKU=$2") + 1:]
        self.assertFalse([line for line in after_export if "$SKU" in line])

    def test_clear_history_is_commented_out_so_not_run(self):
        self.assertNotIn("$UTILS_DIR/clear_history.sh", _commands("Alma9"))
        self.assertNotIn("clear-history", _ops("Alma9"))

    def test_step_arguments_are_install_sh_ones(self):
        steps = {s.op: s for s in _steps("Alma9")}
        self.assertEqual(steps["install-health-checks"].args, ("NVIDIA",))
        # AlmaLinux passes no SKU to the driver script (Rocky and RHEL do).
        self.assertIn("$COMPONENT_DIR/install_nvidiagpudriver.sh", _commands("Alma9"))
        self.assertEqual(steps["install-nv-driver"].args, ())

    def test_order_differences_from_ubuntu_24_04(self):
        ops = _ops("Alma9")
        self.assertEqual(ops[0], "fix-setools-cyclecloud")
        self.assertLess(ops.index("install-mpifileutils"), ops.index("install-nccl"))
        self.assertLess(ops.index("install-monitoring-tools"),
                        ops.index("install-persistent-rdma-naming"))
        self.assertLess(ops.index("copy-test-file"), ops.index("install-health-checks"))
        for op in ("add-udev-rules", "disable-predictive-interface-renaming",
                   "install-nv-grid", "install-rdma-core", "install-nvshmem"):
            self.assertNotIn(op, ops)

    def test_each_os_cleans_its_own_folder(self):
        for os_name, folder in (("Alma8", "distros/almalinux8.10"),
                                ("Alma9", "distros/almalinux9.x")):
            with self.subTest(os=os_name):
                step = {s.op: s for s in _steps(os_name)}["cleanup-downloads"]
                self.assertEqual(step.action.keywords["distro_dir"], folder)

    def test_every_step_runs_python(self):
        self.assertTrue(all(s.runner == "python" for s in _steps("Alma9")))


if __name__ == "__main__":
    unittest.main()
