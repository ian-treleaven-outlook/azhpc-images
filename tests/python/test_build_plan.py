"""Parity tests for utils/build_config.build_plan against distros/*/install.sh.

These assert the Python build plan installs the same components, in the same
order, and with the same SKU/vendor/arch gating as the authoritative bash
orchestrator distros/ubuntu24.04/install.sh. They parse install.sh statically
(running it would do real installs), mirroring the differential approach used
for component-config parity.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest import mock

from utils import build_config
from utils.build_config import BuildConfig, ImageBuilder, Step, build_plan

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INSTALL_SH = _REPO_ROOT / "distros" / "ubuntu24.04" / "install.sh"

# Matches `$COMPONENT_DIR/foo.sh` and `./foo.sh` invocations.
_SCRIPT_RE = re.compile(r"(?:\$COMPONENT_DIR/|\./)([\w-]+\.sh)")


def _install_sh_scripts():
    """Ordered list of script basenames invoked by install.sh.

    Excludes install_utils.sh, which is the bootstrap run separately (via
    ImageBuilder._bootstrap), not part of build_plan.
    """
    text = _INSTALL_SH.read_text(encoding="utf-8")
    scripts = _SCRIPT_RE.findall(text)
    return [s for s in scripts if s != "install_utils.sh"]


def _is_subsequence(sub, full):
    """True if `sub` appears in `full` in the same relative order."""
    it = iter(full)
    return all(item in it for item in sub)


def _plan_scripts(cfg):
    # the bash script names the Python plan would run for this target (gates applied),
    # in order — this is what we line up against install.sh
    return [s.script for s in build_plan(cfg) if s.when(cfg) and s.script]


def _cfg(vendor, gpu, os="Ubuntu24"):
    return BuildConfig(vendor=vendor, gpu=gpu, os=os, fips=False, spec_path=None)


# Core idea: don't hardcode the expected flow — read install.sh itself and check the
# Python plan lines up with it (same scripts, same order). If someone reorders
# install.sh, these fail until build_plan is updated to match.
class InstallShParityTests(unittest.TestCase):
    def test_install_sh_is_parseable(self):
        scripts = _install_sh_scripts()
        self.assertIn("install_cmake.sh", scripts)
        self.assertIn("install_dcgm.sh", scripts)

    def test_a100_plan_is_ordered_subsequence_of_install_sh(self):
        # Every script the A100 plan runs must appear in install.sh, in order.
        plan = _plan_scripts(_cfg("NVidia", "A100"))
        install_sh = _install_sh_scripts()
        self.assertTrue(
            _is_subsequence(plan, install_sh),
            f"A100 plan is not an ordered subsequence of install.sh\n"
            f"plan={plan}\ninstall.sh={install_sh}")

    def test_a100_plan_exact_order(self):
        # The full expected A100 flow, in order, per install.sh.
        expected = [
            "install_cmake.sh",
            "install_lustre_client.sh",
            "install_doca.sh",
            "install_pmix.sh",
            "install_mpis.sh",
            "install_mpifileutils.sh",
            "install_nvidiagpudriver.sh",
            "install_nccl.sh",
            "install_docker.sh",
            "install_dcgm.sh",
            "install_amd_libs.sh",
            "install_intel_libs.sh",
            "install_dynolog_drl.sh",
            "hpc-tuning.sh",
            "install_azure_persistent_rdma_naming.sh",
            "install_aznfs.sh",
            "install_hpcdiag.sh",
            "install_monitoring_tools.sh",
            "install_health_checks.sh",
            "add-udev-rules.sh",
            "copy_test_file.sh",
            "disable_cloudinit.sh",
            "setup_sku_customizations.sh",
            "trivy_scan.sh",
            "disable_auto_upgrade.sh",
            "disable_predictive_interface_renaming.sh",
        ]
        self.assertEqual(_plan_scripts(_cfg("NVidia", "A100")), expected)


# Same plan, different targets: check the SKU/vendor/arch gates pull in the right
# components and leave out the ones meant for other paths (IB vs non-IB, NVIDIA vs AMD,
# GB200's aarch64 branch, etc.).
class GatingTests(unittest.TestCase):
    def test_a100_includes_ib_and_excludes_other_paths(self):
        plan = _plan_scripts(_cfg("NVidia", "A100"))
        self.assertIn("install_doca.sh", plan)          # IB SKU -> DOCA
        self.assertIn("install_intel_libs.sh", plan)    # x86_64
        self.assertNotIn("install_libfabric.sh", plan)  # non-IB only
        self.assertNotIn("install_nvidiagriddriver.sh", plan)  # NCv6 only
        self.assertNotIn("install_rocm.sh", plan)       # AMD only
        self.assertNotIn("install_nvshmem.sh", plan)    # GB200 only

    def test_ncv6_uses_libfabric_and_grid_driver(self):
        plan = _plan_scripts(_cfg("NVidia", "NCv6"))
        self.assertIn("install_libfabric.sh", plan)
        self.assertIn("install_nvidiagriddriver.sh", plan)
        self.assertNotIn("install_doca.sh", plan)
        self.assertNotIn("install_nvidiagpudriver.sh", plan)
        self.assertNotIn("install_health_checks.sh", plan)  # NHC not on NCv6

    def test_amd_branch(self):
        plan = _plan_scripts(_cfg("AMD", "MI300"))
        self.assertIn("install_rocm.sh", plan)
        self.assertIn("install_rccl.sh", plan)
        self.assertIn("install_intel_libs.sh", plan)     # still x86_64
        self.assertNotIn("install_nccl.sh", plan)
        self.assertNotIn("install_nvidiagpudriver.sh", plan)

    def test_gb200_branch(self):
        plan = _plan_scripts(_cfg("NVidia", "GB200"))
        self.assertIn("install_nvidiagpudriver_gb200.sh", plan)
        self.assertIn("install_nvshmem.sh", plan)
        self.assertIn("install_nvloom.sh", plan)
        self.assertIn("install_nvbandwidth_tool.sh", plan)
        self.assertNotIn("install_cmake.sh", plan)       # skipped on GB200
        self.assertNotIn("install_intel_libs.sh", plan)  # aarch64
        self.assertNotIn("install_amd_libs.sh", plan)    # aarch64
        self.assertNotIn("install_aznfs.sh", plan)       # not-GB200 group
        self.assertNotIn("install_health_checks.sh", plan)


# A component action is just a python function, so it can raise as well as return
# a bad code (corrupt tarball, missing file, changed upstream layout...). build()
# has to turn either one into exit code 3 — a traceback would escape as exit 1 and
# break the exit-status contract azhpc.py --help publishes.
class BuildFailureHandlingTests(unittest.TestCase):
    def _run_build_with(self, action):
        """Run build() against a single-step plan whose only step is `action`.

        Everything around the step is stubbed: bootstrap (would run
        install_utils.sh) and component_env (would shell out to dpkg/rpm).
        """
        cfg = _cfg("NVidia", "A100")
        builder = ImageBuilder(_REPO_ROOT, cfg)
        with mock.patch.object(ImageBuilder, "_bootstrap", return_value=0), \
             mock.patch.object(build_config, "component_env", return_value={}), \
             mock.patch.object(build_config, "build_plan",
                               return_value=[Step("test-step", action=action)]), \
             mock.patch.object(build_config, "log_error") as logged:
            rc = builder.build()
        return rc, logged

    def test_step_returning_failure_gives_exit_3(self):
        rc, logged = self._run_build_with(lambda env: 3)
        self.assertEqual(rc, 3)
        logged.assert_called_once()

    def test_raising_step_gives_exit_3_not_a_traceback(self):
        def boom(env):
            raise FileNotFoundError("/tmp/cmake-4.3.1/bin/ccmake")

        rc, logged = self._run_build_with(boom)   # must not propagate
        self.assertEqual(rc, 3)
        # the log names the exception type so the cause isn't lost
        self.assertIn("FileNotFoundError", logged.call_args[0][1])

    def test_successful_step_still_returns_0(self):
        rc, _ = self._run_build_with(lambda env: 0)
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
