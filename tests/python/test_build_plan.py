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
from argparse import Namespace
from pathlib import Path
from unittest import mock

from utils import build_config
from utils.build_config import BuildConfig, ImageBuilder, Step, build_plan, resolve_config

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
        self.assertIn("install_doca.sh", scripts)
        self.assertIn("install_dcgm.sh", scripts)

    def test_a100_plan_is_ordered_subsequence_of_install_sh(self):
        # Every script the A100 plan runs must appear in install.sh, in order.
        plan = _plan_scripts(_cfg("NVidia", "A100"))
        install_sh = _install_sh_scripts()
        self.assertTrue(
            _is_subsequence(plan, install_sh),
            f"A100 plan is not an ordered subsequence of install.sh\n"
            f"plan={plan}\ninstall.sh={install_sh}")

    def test_other_targets_are_ordered_subsequences_of_install_sh(self):
        # Same check for the AMD, non-InfiniBand and GB200 paths, which the A100
        # test can't see (this is how the ROCm-after-MPI ordering slipped through).
        install_sh = _install_sh_scripts()
        for vendor, gpu in (("AMD", "MI300"), ("NVidia", "NCv6"), ("NVidia", "H100"),
                            ("NVidia", "GB200")):
            with self.subTest(gpu=gpu):
                plan = _plan_scripts(_cfg(vendor, gpu))
                self.assertTrue(
                    _is_subsequence(plan, install_sh),
                    f"{gpu} plan is not an ordered subsequence of install.sh\n"
                    f"plan={plan}\ninstall.sh={install_sh}")

    def test_amd_installs_rocm_before_pmix_and_mpis(self):
        plan = _plan_scripts(_cfg("AMD", "MI300"))
        self.assertLess(plan.index("install_rocm.sh"), plan.index("install_pmix.sh"))
        self.assertLess(plan.index("install_rocm.sh"), plan.index("install_mpis.sh"))

    def test_nvidia_installs_the_driver_before_pmix_and_mpis(self):
        # Upstream install.sh: "Install CUDA before MPI so HPC-X can rebuild
        # Open MPI with CUDA support." NCv6 gets the GRID driver instead.
        for gpu, driver in (("A100", "install_nvidiagpudriver.sh"),
                            ("GB200", "install_nvidiagpudriver.sh"),
                            ("NCv6", "install_nvidiagriddriver.sh")):
            with self.subTest(gpu=gpu):
                plan = _plan_scripts(_cfg("NVidia", gpu))
                self.assertLess(plan.index(driver), plan.index("install_pmix.sh"))
                self.assertLess(plan.index(driver), plan.index("install_mpis.sh"))

    def test_a100_plan_exact_order(self):
        # The full expected A100 flow, in order, per install.sh.
        expected = [
            "install_doca.sh",
            "install_nvidiagpudriver.sh",
            "install_pmix.sh",
            "install_mpis.sh",
            "install_nccl.sh",
            "install_docker.sh",
            "install_dcgm.sh",
            "install_lustre_client.sh",
            "install_mpifileutils.sh",
            "install_amd_libs.sh",
            "install_intel_libs.sh",
            "hpc-tuning.sh",
            "install_waagent.sh",
            "install_azure_persistent_rdma_naming.sh",
            "install_aznfs.sh",
            "install_hpcdiag.sh",
            "install_monitoring_tools.sh",
            "install_health_checks.sh",
            "write_kernel_os_version.sh",
            "install_azsecpack_prereqs.sh",
            "add-udev-rules.sh",
            "copy_test_file.sh",
            "disable_cloudinit.sh",
            "setup_sku_customizations.sh",
            "trivy_scan.sh",
            "disable_predictive_interface_renaming.sh",
        ]
        self.assertEqual(_plan_scripts(_cfg("NVidia", "A100")), expected)


# Since the upstream sync (ADO 983), some ports are behind their bash script and
# run bash until a resync ticket updates them (Step.resync). This list is pinned
# on purpose: re-enabling a port, or marking another one stale, should be a
# deliberate change that also updates this test.
class ResyncTests(unittest.TestCase):
    EXPECTED = {
        "install-rocm": "ADO 986",
        "install-rccl": "ADO 986",
        "install-doca": "ADO 987",
        "install-nccl": "ADO 987",
        "install-dcgm": "ADO 987",
        "install-lustre": "ADO 988",
        "hpc-tuning": "ADO 988",
        "install-persistent-rdma-naming": "ADO 988",
        "install-aznfs": "ADO 988",
    }

    def _steps(self):
        return build_plan(_cfg("NVidia", "A100"))

    def test_stale_ports_are_marked_with_their_ticket(self):
        marked = {s.op: s.resync for s in self._steps() if s.resync}
        self.assertEqual(marked, self.EXPECTED)

    def test_marked_steps_have_a_bash_script_to_fall_back_to(self):
        for step in self._steps():
            if step.resync:
                with self.subTest(op=step.op):
                    self.assertTrue(step.script)
                    self.assertIsNotNone(step.action)
                    self.assertTrue((_REPO_ROOT / "components" / step.script).is_file())

    def test_new_upstream_step_runs_as_bash(self):
        steps = {s.op: s for s in self._steps()}
        self.assertIsNone(steps["install-azsecpack-prereqs"].action)   # ADO 990


# Same plan, different targets: check the SKU/vendor/arch gates pull in the right
# components and leave out the ones meant for other paths (IB vs non-IB, NVIDIA vs AMD,
# GB200's aarch64 branch, etc.).
class GatingTests(unittest.TestCase):
    def test_vr200_is_built_as_gb200_with_a_warning(self):
        # distros/ubuntu24.04/install.sh maps VR200 -> GB200; resolve_config does too.
        args = Namespace(vendor="NVidia", gpu="VR200", os="Ubuntu24", spec=None, fips=False)
        with mock.patch.object(build_config, "log_warn") as warned:
            cfg = resolve_config(args)
        self.assertEqual(cfg.gpu, "GB200")
        self.assertEqual(cfg.architecture, "aarch64")
        warned.assert_called_once()
        self.assertEqual(warned.call_args.kwargs["requested_gpu"], "VR200")
        self.assertEqual(_plan_scripts(cfg), _plan_scripts(_cfg("NVidia", "GB200")))

    def test_other_gpus_are_not_remapped(self):
        args = Namespace(vendor="NVidia", gpu="H100", os="Ubuntu24", spec=None, fips=False)
        with mock.patch.object(build_config, "log_warn") as warned:
            cfg = resolve_config(args)
        self.assertEqual(cfg.gpu, "H100")
        warned.assert_not_called()
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

    def test_docker_runs_for_every_vendor(self):
        # install.sh runs install_docker.sh unconditionally, on every distro.
        for vendor, gpu in (("NVidia", "A100"), ("NVidia", "NCv6"), ("AMD", "MI300")):
            with self.subTest(gpu=gpu):
                self.assertIn("install_docker.sh", _plan_scripts(_cfg(vendor, gpu)))

    def test_steps_a_distros_install_sh_never_runs_are_skipped(self):
        # Azure Linux's install.sh has no disable_cloudinit.sh; only the Ubuntu
        # install.sh scripts run the distro-local disable script.
        ubuntu_only = {"disable_predictive_interface_renaming.sh"}
        cases = {
            "Ubuntu24": (True, True),   # (runs disable_cloudinit, runs ubuntu-only)
            "Ubuntu22": (True, True),
            "Azure3":   (False, False),
            "Alma9":    (True, False),
            "Rocky8":   (True, False),
        }
        for os_name, (cloudinit, ubuntu) in cases.items():
            with self.subTest(os=os_name):
                plan = set(_plan_scripts(_cfg("NVidia", "A100", os=os_name)))
                self.assertEqual("disable_cloudinit.sh" in plan, cloudinit)
                self.assertEqual(ubuntu_only <= plan, ubuntu)
                self.assertFalse(ubuntu_only & plan and not ubuntu)

    def test_those_skips_carry_a_warning_reason(self):
        steps = {s.op: s for s in build_plan(_cfg("NVidia", "A100"))}
        for op in ("disable-cloudinit", "disable-predictive-interface-renaming"):
            with self.subTest(op=op):
                self.assertTrue(steps[op].skip_warning)
        self.assertEqual(steps["install-rocm"].skip_warning, "")   # normal gating

    def test_gb200_branch(self):
        plan = _plan_scripts(_cfg("NVidia", "GB200"))
        # GB200 uses the shared driver script, as install.sh does; the old
        # dedicated install_nvidiagpudriver_gb200.sh was removed upstream.
        self.assertIn("install_nvidiagpudriver.sh", plan)
        self.assertNotIn("install_nvidiagpudriver_gb200.sh", plan)
        self.assertIn("install_nvshmem.sh", plan)
        self.assertIn("install_nvloom.sh", plan)
        self.assertIn("install_nvbandwidth_tool.sh", plan)
        # The driver must be installed before the NVLink tools that need it.
        for tool in ("install_nvshmem.sh", "install_nvloom.sh", "install_nvbandwidth_tool.sh"):
            self.assertLess(plan.index("install_nvidiagpudriver.sh"), plan.index(tool))
        self.assertNotIn("install_intel_libs.sh", plan)  # aarch64
        self.assertNotIn("install_amd_libs.sh", plan)    # aarch64
        self.assertNotIn("install_aznfs.sh", plan)       # not-GB200 group
        self.assertNotIn("install_health_checks.sh", plan)

    def test_every_plan_script_exists(self):
        # A Step whose bash script is missing fails the build (exit 127) when it
        # runs as bash, and has no parity reference. This is how the deleted
        # install_nvidiagpudriver_gb200.sh went unnoticed.
        for vendor, gpu in (("NVidia", "A100"), ("NVidia", "GB200"),
                            ("NVidia", "NCv6"), ("AMD", "MI300")):
            cfg = _cfg(vendor, gpu)
            for step in build_plan(cfg):
                if not step.script or not step.when(cfg):
                    continue
                base = (_REPO_ROOT / cfg.distro_dir if step.base == "distro"
                        else _REPO_ROOT / "components")
                with self.subTest(gpu=gpu, script=step.script):
                    self.assertTrue((base / step.script).is_file(),
                                    f"{base / step.script} does not exist")


# A component action is just a python function, so it can raise as well as return
# a bad code (corrupt tarball, missing file, changed upstream layout...). build()
# has to turn either one into exit code 3 — a traceback would escape as exit 1 and
# break the exit-status contract azhpc.py --help publishes.
class BuildFailureHandlingTests(unittest.TestCase):
    def _run_build_with(self, action):
        """Run build() against a single-step plan whose only step is `action`.

        Everything around the step is stubbed: bootstrap (would run
        install_utils.sh) and component_env (would shell out to dpkg/rpm).
        Returns the exit code plus the two error loggers: a failed step uses
        log_error; an exception uses log_error_detail (message + traceback).
        """
        cfg = _cfg("NVidia", "A100")
        builder = ImageBuilder(_REPO_ROOT, cfg)
        with mock.patch.object(ImageBuilder, "_bootstrap", return_value=0), \
             mock.patch.object(build_config, "component_env", return_value={}), \
             mock.patch.object(build_config, "build_plan",
                               return_value=[Step("test-step", action=action)]), \
             mock.patch.object(build_config, "log_error") as logged, \
             mock.patch.object(build_config, "log_error_detail") as logged_detail:
            rc = builder.build()
        return rc, logged, logged_detail

    def test_step_returning_failure_gives_exit_3(self):
        rc, logged, _ = self._run_build_with(lambda env: 3)
        self.assertEqual(rc, 3)
        logged.assert_called_once()
        # Telemetry: the failure records the runner and exit code as fields.
        self.assertEqual(logged.call_args.kwargs["runner"], "python")
        self.assertEqual(logged.call_args.kwargs["exit_code"], 3)

    def test_raising_step_gives_exit_3_not_a_traceback(self):
        def boom(env):
            raise FileNotFoundError("/tmp/mpifileutils-build/bin/dcp")

        rc, logged, logged_detail = self._run_build_with(boom)   # must not propagate
        self.assertEqual(rc, 3)
        logged.assert_not_called()
        logged_detail.assert_called_once()
        message, detail = logged_detail.call_args[0][1], logged_detail.call_args[0][2]
        # the log names the exception type so the cause isn't lost...
        self.assertIn("FileNotFoundError", message)
        # ...and the traceback is kept as the detail block, not printed raw.
        self.assertIn("Traceback", detail)

    def test_successful_step_still_returns_0(self):
        rc, _, _ = self._run_build_with(lambda env: 0)
        self.assertEqual(rc, 0)

    def test_success_logs_step_duration_and_build_summary(self):
        cfg = _cfg("NVidia", "A100")
        builder = ImageBuilder(_REPO_ROOT, cfg)
        with mock.patch.object(ImageBuilder, "_bootstrap", return_value=0), \
             mock.patch.object(build_config, "component_env", return_value={}), \
             mock.patch.object(build_config, "build_plan",
                               return_value=[Step("test-step", action=lambda env: 0)]), \
             mock.patch.object(build_config, "log_info") as info:
            self.assertEqual(builder.build(), 0)

        done = [c for c in info.call_args_list if c.args[0] == "test-step"
                and c.args[1].startswith("done")]
        self.assertEqual(len(done), 1)
        self.assertIn("duration_s", done[0].kwargs)
        summary = info.call_args_list[-1]
        self.assertEqual(summary.args[0], "build-image")
        self.assertEqual(summary.kwargs["python_steps"], 1)
        self.assertEqual(summary.kwargs["bash_steps"], 0)

    def _run_skipped(self, step):
        """Run build() with one step whose `when` is false; return the loggers."""
        builder = ImageBuilder(_REPO_ROOT, _cfg("NVidia", "A100"))
        with mock.patch.object(ImageBuilder, "_bootstrap", return_value=0), \
             mock.patch.object(build_config, "component_env", return_value={}), \
             mock.patch.object(build_config, "build_plan", return_value=[step]), \
             mock.patch.object(build_config, "log_info") as info, \
             mock.patch.object(build_config, "log_warn") as warn:
            self.assertEqual(builder.build(), 0)
        return info, warn

    def test_skip_with_a_reason_is_a_warning(self):
        step = Step("test-step", "x.sh", action=lambda env: 0,
                    when=lambda c: False, skip_warning="install.sh never runs x.sh")
        _, warn = self._run_skipped(step)
        skipped = [c for c in warn.call_args_list if c.args[0] == "test-step"]
        self.assertEqual(len(skipped), 1)
        self.assertIn("install.sh never runs x.sh", skipped[0].args[1])

    def test_ordinary_skip_is_info_only(self):
        step = Step("test-step", "x.sh", action=lambda env: 0, when=lambda c: False)
        info, warn = self._run_skipped(step)
        self.assertFalse([c for c in warn.call_args_list if c.args[0] == "test-step"])
        self.assertTrue([c for c in info.call_args_list
                         if c.args[0] == "test-step" and "skipped" in c.args[1]])

    def _run_one_step(self, step):
        """Run build() with a one-step plan; bash is stubbed to succeed.

        Returns (exit code, exec_program mock, log_info mock, log_warn mock).
        """
        builder = ImageBuilder(_REPO_ROOT, _cfg("NVidia", "A100"))
        with mock.patch.object(ImageBuilder, "_bootstrap", return_value=0), \
             mock.patch.object(build_config, "component_env", return_value={}), \
             mock.patch.object(build_config, "build_plan", return_value=[step]), \
             mock.patch.object(build_config, "exec_program", return_value=0) as bash, \
             mock.patch.object(build_config, "log_info") as info, \
             mock.patch.object(build_config, "log_warn") as warn:
            rc = builder.build()
        return rc, bash, info, warn

    def test_step_marked_resync_runs_bash_with_a_warning(self):
        action = mock.Mock(return_value=0)
        step = Step("test-step", "x.sh", action=action, resync="ADO 987")
        rc, bash, info, warn = self._run_one_step(step)

        self.assertEqual(rc, 0)
        action.assert_not_called()                    # the stale port never runs
        bash.assert_called_once()
        self.assertEqual(Path(bash.call_args.args[0][0]).name, "x.sh")
        warned = [c for c in warn.call_args_list if c.args[0] == "test-step"]
        self.assertEqual(len(warned), 1)
        self.assertIn("ADO 987", warned[0].args[1])
        self.assertEqual(warned[0].kwargs["resync"], "ADO 987")
        summary = info.call_args_list[-1]
        self.assertEqual(summary.kwargs["bash_steps"], 1)
        self.assertEqual(summary.kwargs["python_steps"], 0)

    def test_step_without_resync_runs_python_and_does_not_warn(self):
        action = mock.Mock(return_value=0)
        rc, bash, _, warn = self._run_one_step(Step("test-step", "x.sh", action=action))
        self.assertEqual(rc, 0)
        action.assert_called_once()
        bash.assert_not_called()
        self.assertFalse([c for c in warn.call_args_list if c.args[0] == "test-step"])

    def test_runner_property(self):
        port = lambda env: 0
        self.assertEqual(Step("a", "a.sh", action=port).runner, "python")
        self.assertEqual(Step("a", "a.sh").runner, "bash")
        self.assertEqual(Step("a", "a.sh", action=port, resync="ADO 1").runner, "bash")

    def test_dry_run_shows_the_runner_a_build_would_use(self):
        import contextlib
        import io
        import azhpc

        plan = [Step("ported", "p.sh", action=lambda env: 0),
                Step("stale", "s.sh", action=lambda env: 0, resync="ADO 987"),
                Step("bash-only", "b.sh")]
        out = io.StringIO()
        with mock.patch.object(azhpc, "build_plan", return_value=plan), \
             contextlib.redirect_stdout(out):
            azhpc.print_build_plan(_cfg("NVidia", "A100"))
        lines = {line.split()[1]: line for line in out.getvalue().splitlines()[1:]}
        self.assertTrue(lines["ported"].endswith("(python)"))
        self.assertTrue(lines["stale"].endswith("(bash, resync ADO 987)"))
        self.assertTrue(lines["bash-only"].endswith("(bash)"))

    def _rackscale_warnings(self, gpu, env):
        """Run build() for `gpu` with an empty plan; return the rackscale warnings."""
        builder = ImageBuilder(_REPO_ROOT, _cfg("NVidia", gpu))
        with mock.patch.object(ImageBuilder, "_bootstrap", return_value=0), \
             mock.patch.object(build_config, "component_env", return_value=env), \
             mock.patch.object(build_config, "build_plan", return_value=[]), \
             mock.patch.object(build_config, "log_warn") as warn:
            self.assertEqual(builder.build(), 0)
        return [c for c in warn.call_args_list if "NVLINK_RACKSCALE" in c.args[1]]

    def test_gb200_without_nvlink_rackscale_warns(self):
        self.assertEqual(len(self._rackscale_warnings("GB200", {})), 1)

    def test_gb200_with_nvlink_rackscale_is_quiet(self):
        self.assertEqual(self._rackscale_warnings("GB200", {"NVLINK_RACKSCALE": "TRUE"}), [])

    def test_non_rackscale_sku_is_quiet(self):
        self.assertEqual(self._rackscale_warnings("A100", {}), [])


if __name__ == "__main__":
    unittest.main()
