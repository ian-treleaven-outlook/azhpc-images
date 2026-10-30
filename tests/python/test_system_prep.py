"""Unit tests for utils/system_prep.py (set_properties.sh's system changes)."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from utils import build_config, system_prep
from utils.build_config import BuildConfig, ImageBuilder

_REPO = Path(__file__).resolve().parents[2]
_SET_PROPERTIES = (_REPO / "utils" / "set_properties.sh").read_text(encoding="utf-8")

_UPGRADE = [["apt", "update"], ["apt", "upgrade", "-y"]]
_APT_JQ = [["apt", "install", "-y", "jq"]]
_DNF_JQ = [["dnf", "install", "-y", "jq"]]


class PrepareCommandsTests(unittest.TestCase):
    def test_every_branch(self):
        cases = {
            ("ubuntu24.04", "azure_vm_regular"): _UPGRADE + _APT_JQ,
            ("ubuntu22.04", ""): _UPGRADE + _APT_JQ,            # empty -> default
            ("ubuntu26.04", "baremetal_1p"): _UPGRADE + _APT_JQ,
            ("ubuntu24.04", "baremetal_3p"): _APT_JQ,           # no upgrade
            ("almalinux8.10", "azure_vm_regular"):
                [["rpm", "--import", system_prep.ALMA8_GPG_KEY]] + _DNF_JQ,
            ("almalinux9.7", "azure_vm_regular"):
                [["rpm", "--import", system_prep.ALMA9_GPG_KEY]] + _DNF_JQ,
            ("almalinux10.0", "azure_vm_regular"): _DNF_JQ,      # no key branch
            ("azurelinux3.0", "azure_vm_regular"): _DNF_JQ,
            ("rocky9.7", "azure_vm_regular"): _DNF_JQ,
            ("rhel9.8", "azure_vm_regular"): _DNF_JQ,
        }
        for (distribution, node_type), expected in cases.items():
            with self.subTest(distribution=distribution, node_type=node_type):
                self.assertEqual(system_prep.prepare_commands(distribution, node_type),
                                 expected)

    def test_commands_match_the_bash_script(self):
        for line in ["apt update", "apt upgrade -y", "apt install -y jq",
                     "dnf install -y jq", '"baremetal_3p"',
                     f"rpm --import {system_prep.ALMA8_GPG_KEY}\n",
                     f"rpm --import {system_prep.ALMA9_GPG_KEY}\n"]:
            with self.subTest(line=line):
                self.assertIn(line, _SET_PROPERTIES)


class PrepareSystemTests(unittest.TestCase):
    def _run(self, env, results):
        with mock.patch.object(system_prep, "exec_program", side_effect=results) as run, \
             mock.patch.object(system_prep, "log_info") as info, \
             mock.patch.object(system_prep, "log_error") as error:
            rc = system_prep.prepare_system(env)
        return rc, run, info, error

    def test_success_runs_every_command_in_order(self):
        env = {"DISTRIBUTION": "ubuntu24.04", "TARGET_NODE_TYPE": "azure_vm_regular"}
        rc, run, info, error = self._run(env, [0, 0, 0])
        self.assertEqual(rc, 0)
        self.assertEqual([c.args[0] for c in run.call_args_list], _UPGRADE + _APT_JQ)
        error.assert_not_called()
        start = info.call_args_list[0].kwargs
        self.assertEqual((start["distribution"], start["apt_upgrade"]), ("ubuntu24.04", True))

    def test_first_failure_stops_the_rest(self):
        env = {"DISTRIBUTION": "ubuntu24.04"}
        rc, run, _, error = self._run(env, [0, 100])
        self.assertEqual(rc, 3)
        self.assertEqual(run.call_count, 2)                 # jq never attempted
        self.assertEqual(error.call_args.kwargs["command"], "apt upgrade -y")

    def test_baremetal_3p_logs_why_it_skips_the_upgrade(self):
        env = {"DISTRIBUTION": "ubuntu24.04", "TARGET_NODE_TYPE": "baremetal_3p"}
        rc, _, info, _ = self._run(env, [0])
        self.assertEqual(rc, 0)
        self.assertIn("Skipping apt update/upgrade on baremetal 3P node",
                      [c.args[1] for c in info.call_args_list])


class BootstrapTests(unittest.TestCase):
    def _bootstrap(self, prep_rc):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24", fips=False,
                          spec_path=None)
        builder = ImageBuilder(_REPO, cfg)
        order = []
        with mock.patch.object(build_config, "prepare_system",
                               side_effect=lambda env: order.append("prep") or prep_rc), \
             mock.patch.object(build_config, "exec_program",
                               side_effect=lambda *a, **k: order.append("install_utils") or 0), \
             mock.patch.object(build_config, "log_info"):
            rc = builder._bootstrap(_REPO / cfg.distro_dir, {})
        return rc, order

    def test_system_prep_runs_before_install_utils(self):
        rc, order = self._bootstrap(0)
        self.assertEqual(rc, 0)
        self.assertEqual(order, ["prep", "install_utils"])

    def test_failed_system_prep_stops_the_bootstrap(self):
        rc, order = self._bootstrap(3)
        self.assertEqual(rc, 3)
        self.assertEqual(order, ["prep"])


if __name__ == "__main__":
    unittest.main()
