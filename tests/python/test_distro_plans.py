"""Tests for the per-distro plan framework (utils/plans/).

The parity tests here run for EVERY registered plan, against that distro's own
install.sh, so a new distro plan (ADO 992-997) is checked as soon as it is
added to PLANS:

* order: for representative SKUs, the scripts the plan runs appear in
  install.sh in the same order;
* coverage: across those SKUs, the plan runs every script install.sh runs;
* every planned bash script exists;
* the inline cleanup is install.sh's text word for word.

install.sh is parsed statically (running it would do real installs).
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import azhpc
from utils import build_config
from utils.build_config import (DISTRIBUTIONS, DISTRO_DIRS, BuildConfig, ConfigError,
                                ImageBuilder, build_plan, pre_bootstrap_plan,
                                resolve_config)
from utils.plans import OS_TARGETS, PENDING, PLANS, DistroPlan, steps
from utils.plans.step import Step

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT_RE = re.compile(r"(?:\$COMPONENT_DIR/|\$UTILS_DIR/|\./)([\w.-]+\.sh)")
# Run by the bootstrap, not by a plan step.
_BOOTSTRAP_SCRIPTS = {"install_utils.sh"}
# Representative SKUs per vendor; together they reach every branch of the
# install.sh scripts (IB vs no-RDMA, x86_64 vs aarch64, NVIDIA vs AMD).
_SKUS = {"NVidia": ["A100", "V100", "H100", "GB200", "NCv6"], "AMD": ["MI300X"]}


def install_sh_scripts(os_name: str) -> list[str]:
    """The scripts os_name's install.sh runs, in order (comment lines skipped)."""
    text = (_REPO / DISTRO_DIRS[os_name] / "install.sh").read_text(encoding="utf-8")
    scripts = []
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        scripts += _SCRIPT_RE.findall(line)
    return [s for s in scripts if s not in _BOOTSTRAP_SCRIPTS]


def _cfg(os_name, vendor, gpu):
    return BuildConfig(vendor=vendor, gpu=gpu, os=os_name, fips=False, spec_path=None)


def _targets(os_name):
    plan = PLANS[os_name]
    # Some install.sh scripts refuse SKUs before doing anything (Ubuntu 26.04:
    # V100, GB200, VR200, NCv6); resolve_config fails those, so only the SKUs
    # a distro builds are checked against its install.sh.
    return [(vendor, gpu) for vendor in sorted(plan.vendors) for gpu in _SKUS[vendor]
            if not plan.reject_sku(gpu)]


def _planned(cfg):
    """Every step this target runs, pre-bootstrap first, gates applied."""
    return [s for s in pre_bootstrap_plan(cfg) + build_plan(cfg) if s.when(cfg)]


def _is_subsequence(sub, full):
    it = iter(full)
    return all(item in it for item in sub)


class OsTableTests(unittest.TestCase):
    def test_every_os_has_its_distro_folder_and_install_sh(self):
        for os_name, target in OS_TARGETS.items():
            with self.subTest(os=os_name):
                self.assertTrue((_REPO / "distros" / target.folder / "install.sh").is_file())

    def test_every_distribution_is_a_versions_json_key(self):
        versions = json.loads((_REPO / "versions.json").read_text(encoding="utf-8"))
        keys = {k for comp in versions.values() if isinstance(comp, dict) for k in comp}
        for os_name, distribution in DISTRIBUTIONS.items():
            with self.subTest(os=os_name):
                self.assertIn(distribution, keys)

    def test_every_os_is_planned_or_pending(self):
        self.assertEqual(set(PLANS) | set(PENDING), set(OS_TARGETS))
        self.assertFalse(set(PLANS) & set(PENDING))


class ResolveConfigTests(unittest.TestCase):
    def _resolve(self, os_name, vendor="NVidia", gpu="A100"):
        return resolve_config(Namespace(vendor=vendor, gpu=gpu, os=os_name,
                                        spec=None, fips=False))

    def _without_a_plan(self, os_name, ticket):
        """Patches that make os_name an OS still waiting for its plan (`ticket`).

        Every --os has a plan today, so the refusal path is tested this way.
        patch.dict puts both dicts back as they were afterwards; clear=True
        empties PLANS first, then fills it with every plan but os_name's.
        """
        others = {name: plan for name, plan in PLANS.items() if name != os_name}
        return (mock.patch.dict(build_config.PLANS, others, clear=True),
                mock.patch.dict(build_config.PENDING, {os_name: ticket}))

    def test_an_os_without_a_plan_is_rejected_with_its_ticket(self):
        plans, pending = self._without_a_plan("RHEL9", "ADO 999")
        with plans, pending:
            with self.assertRaises(ConfigError) as caught:
                self._resolve("RHEL9")
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("ADO 999", str(caught.exception))
        self.assertIn("distros/rhel9.x/install.sh", str(caught.exception))
        self.assertIn("RHEL9", PLANS)                      # restored

    def test_a_vendor_the_install_sh_rejects_is_rejected(self):
        # e.g. AlmaLinux's install.sh: "AMD GPU support is not implemented yet".
        plan = DistroPlan(build=lambda cfg: [], vendors=frozenset({"NVidia"}),
                          vendor_error="AMD GPU support is not implemented yet for X.")
        with mock.patch.dict(build_config.PLANS, {"Ubuntu24": plan}):
            with self.assertRaises(ConfigError) as caught:
                self._resolve("Ubuntu24", vendor="AMD", gpu="MI300X")
            self.assertEqual(str(caught.exception),
                             "AMD GPU support is not implemented yet for X.")
            self.assertEqual(self._resolve("Ubuntu24").vendor, "NVidia")

    def test_build_plan_refuses_an_os_without_a_plan(self):
        # resolve_config() refuses it first; this guards direct callers.
        plans, pending = self._without_a_plan("Alma8", "ADO 999")
        with plans, pending:
            with self.assertRaises(ConfigError) as caught:
                build_plan(_cfg("Alma8", "NVidia", "A100"))
        self.assertIn("ADO 999", str(caught.exception))


class InstallShParityTests(unittest.TestCase):
    """Run for every registered plan against its own install.sh."""

    def test_planned_scripts_follow_install_sh_order(self):
        for os_name in PLANS:
            install_sh = install_sh_scripts(os_name)
            for vendor, gpu in _targets(os_name):
                with self.subTest(os=os_name, gpu=gpu):
                    planned = [s.script for s in _planned(_cfg(os_name, vendor, gpu)) if s.script]
                    self.assertTrue(_is_subsequence(planned, install_sh),
                                    f"{os_name}/{gpu} plan is not in install.sh order\n"
                                    f"plan={planned}\ninstall.sh={install_sh}")

    def test_every_install_sh_script_is_planned_for_some_sku(self):
        for os_name in PLANS:
            with self.subTest(os=os_name):
                covered = {s.script for vendor, gpu in _targets(os_name)
                           for s in _planned(_cfg(os_name, vendor, gpu))}
                missing = set(install_sh_scripts(os_name)) - covered
                self.assertFalse(missing, f"{os_name}: install.sh runs {sorted(missing)}, "
                                          "which no plan step covers")

    def test_every_planned_script_exists(self):
        for os_name in PLANS:
            for vendor, gpu in _targets(os_name):
                cfg = _cfg(os_name, vendor, gpu)
                for step in _planned(cfg):
                    if not step.script:
                        continue
                    with self.subTest(os=os_name, gpu=gpu, script=step.script):
                        self.assertTrue(step.script_path(_REPO, cfg.distro_dir).is_file())

    def test_cleanup_is_install_sh_text_word_for_word_in_the_distro_folder(self):
        for os_name in PLANS:
            with self.subTest(os=os_name):
                text = (_REPO / DISTRO_DIRS[os_name] / "install.sh").read_text(encoding="utf-8")
                self.assertIn(steps.CLEANUP_SCRIPT, text)
                cfg = _cfg(os_name, "NVidia", "A100")
                step = {s.op: s for s in build_plan(cfg)}["cleanup-downloads"]
                self.assertEqual(step.action.keywords["distro_dir"], DISTRO_DIRS[os_name])


class PreBootstrapTests(unittest.TestCase):
    """A plan's pre-bootstrap steps run after set_properties.sh's changes and
    before install_utils.sh, like Ubuntu 22.04's remove_unused_packages.sh."""

    def _plan(self, pre_action):
        return DistroPlan(build=lambda cfg: [],
                          pre_bootstrap=lambda cfg: [Step("pre-step", "remove_unused_packages.sh",
                                                          action=pre_action, base="distro")])

    def _bootstrap(self, pre_rc):
        order = []
        plan = self._plan(lambda env: order.append("pre") or pre_rc)
        builder = ImageBuilder(_REPO, _cfg("Ubuntu24", "NVidia", "A100"))
        with mock.patch.dict(build_config.PLANS, {"Ubuntu24": plan}), \
             mock.patch.object(build_config, "prepare_system",
                               side_effect=lambda env: order.append("prep") or 0), \
             mock.patch.object(build_config, "exec_program",
                               side_effect=lambda *a, **k: order.append("install_utils") or 0), \
             mock.patch.object(build_config, "log_info"), \
             mock.patch.object(build_config, "log_error"):
            rc = builder._bootstrap(_REPO / "distros" / "ubuntu24.04", {})
        return rc, order

    def test_order(self):
        rc, order = self._bootstrap(0)
        self.assertEqual(rc, 0)
        self.assertEqual(order, ["prep", "pre", "install_utils"])

    def test_a_failing_pre_bootstrap_step_stops_the_bootstrap(self):
        rc, order = self._bootstrap(5)
        self.assertEqual(rc, 3)
        self.assertEqual(order, ["prep", "pre"])

    def test_dry_run_lists_them_first_and_marks_them(self):
        plan = DistroPlan(build=lambda cfg: [Step("main-step", "x.sh", action=lambda e: 0)],
                          pre_bootstrap=lambda cfg: [Step("pre-step", "p.sh", action=lambda e: 0)])
        out = io.StringIO()
        with mock.patch.dict(build_config.PLANS, {"Ubuntu24": plan}), \
             contextlib.redirect_stdout(out):
            azhpc.print_build_plan(_cfg("Ubuntu24", "NVidia", "A100"))
        lines = out.getvalue().splitlines()
        self.assertIn("2 steps", lines[0])
        self.assertIn("pre-step", lines[1])
        self.assertTrue(lines[1].endswith("(python, before install_utils.sh)"))
        self.assertTrue(lines[2].endswith("(python)"))


class EndOfBuildStepTests(unittest.TestCase):
    """The shared steps some install.sh scripts end with (AlmaLinux, Rocky, RHEL)."""

    def test_clear_history_runs_the_utils_script_as_bash(self):
        step = steps.clear_history()
        self.assertEqual((step.op, step.script, step.base),
                         ("clear-history", "clear_history.sh", "utils"))
        self.assertEqual(step.runner, "bash")
        self.assertTrue(step.script_path(_REPO, "distros/rocky9.x").is_file())

    def test_system_upgrade_is_inline_and_returns_the_contract_codes(self):
        step = steps.system_upgrade()
        self.assertEqual((step.op, step.script, step.runner), ("upgrade-all-packages", "", "python"))
        for upgraded, rc in ((True, 0), (False, 3)):
            with self.subTest(upgraded=upgraded):
                with mock.patch.object(steps, "upgrade_all_packages",
                                       return_value=upgraded) as upgrade:
                    self.assertEqual(step.action({"DISTRIBUTION": "almalinux9.8"}), rc)
                upgrade.assert_called_once_with({"DISTRIBUTION": "almalinux9.8"},
                                                "upgrade-all-packages")


if __name__ == "__main__":
    unittest.main()
