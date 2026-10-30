"""Unit tests for utils/system_prep.py (set_properties.sh's system changes)."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from utils import build_config, package_installer, system_prep
from utils.build_config import BuildConfig, ImageBuilder

_REPO = Path(__file__).resolve().parents[2]
_SET_PROPERTIES = (_REPO / "utils" / "set_properties.sh").read_text(encoding="utf-8")

_UPGRADE = [["apt", "update"], ["apt", "upgrade", "-y"]]
# bash's `apt install -y jq` / `dnf install -y jq`, as install_packages() runs them.
_APT_JQ = [["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0", "jq"]]
_DNF_JQ = [["dnf", "install", "-y", "jq"]]


class PrepareCommandsTests(unittest.TestCase):
    def test_every_branch(self):
        # The commands before the jq install, which every branch ends with.
        cases = {
            ("ubuntu24.04", "azure_vm_regular"): _UPGRADE,
            ("ubuntu22.04", ""): _UPGRADE,                      # empty -> default
            ("ubuntu26.04", "baremetal_1p"): _UPGRADE,
            ("ubuntu24.04", "baremetal_3p"): [],                # no upgrade
            ("almalinux8.10", "azure_vm_regular"):
                [["rpm", "--import", system_prep.ALMA8_GPG_KEY]],
            ("almalinux9.7", "azure_vm_regular"):
                [["rpm", "--import", system_prep.ALMA9_GPG_KEY]],
            ("almalinux10.0", "azure_vm_regular"): [],          # no key branch
            ("azurelinux3.0", "azure_vm_regular"): [],
            ("rocky9.7", "azure_vm_regular"): [],
            ("rhel9.8", "azure_vm_regular"): [],
        }
        for (distribution, node_type), expected in cases.items():
            with self.subTest(distribution=distribution, node_type=node_type):
                self.assertEqual(system_prep.prepare_commands(distribution, node_type),
                                 expected)

    def test_commands_match_the_bash_script(self):
        jq = " ".join(system_prep.PREREQUISITE_PACKAGES)
        for line in ["apt update", "apt upgrade -y", f"apt install -y {jq}",
                     f"dnf install -y {jq}", '"baremetal_3p"',
                     f"rpm --import {system_prep.ALMA8_GPG_KEY}\n",
                     f"rpm --import {system_prep.ALMA9_GPG_KEY}\n"]:
            with self.subTest(line=line):
                self.assertIn(line, _SET_PROPERTIES)


class PrepareSystemTests(unittest.TestCase):
    def _run(self, env, results):
        """Run prepare_system(env). One mock stands in for exec_program here and
        in install_packages, so `run` records every command in order."""
        run = mock.Mock(side_effect=results)
        with mock.patch.object(system_prep, "exec_program", run), \
             mock.patch.object(package_installer, "exec_program", run), \
             mock.patch.object(package_installer, "log_info"), \
             mock.patch.object(package_installer, "log_error"), \
             mock.patch.object(system_prep, "log_info") as info, \
             mock.patch.object(system_prep, "log_error") as error:
            rc = system_prep.prepare_system(env)
        for call in run.call_args_list:
            self.assertEqual(call.args[1], "prepare-system")
            self.assertIs(call.kwargs["env"], env)
        return rc, run, info, error

    def test_success_runs_every_command_in_order(self):
        cases = {
            ("ubuntu24.04", "azure_vm_regular"): _UPGRADE + _APT_JQ,
            ("ubuntu24.04", "baremetal_3p"): _APT_JQ,
            ("almalinux8.10", "azure_vm_regular"):
                [["rpm", "--import", system_prep.ALMA8_GPG_KEY]] + _DNF_JQ,
            ("almalinux9.7", "azure_vm_regular"):
                [["rpm", "--import", system_prep.ALMA9_GPG_KEY]] + _DNF_JQ,
            ("azurelinux3.0", "azure_vm_regular"): _DNF_JQ,
            ("rocky9.7", "azure_vm_regular"): _DNF_JQ,
            ("rhel9.8", "azure_vm_regular"): _DNF_JQ,
        }
        for (distribution, node_type), expected in cases.items():
            with self.subTest(distribution=distribution, node_type=node_type):
                env = {"DISTRIBUTION": distribution, "TARGET_NODE_TYPE": node_type}
                rc, run, _, error = self._run(env, [0] * len(expected))
                self.assertEqual(rc, 0)
                self.assertEqual([c.args[0] for c in run.call_args_list], expected)
                error.assert_not_called()

    def test_start_log(self):
        env = {"DISTRIBUTION": "ubuntu24.04", "TARGET_NODE_TYPE": "azure_vm_regular"}
        _, _, info, _ = self._run(env, [0, 0, 0])
        start = info.call_args_list[0].kwargs
        self.assertEqual((start["distribution"], start["apt_upgrade"], start["packages"]),
                         ("ubuntu24.04", True, "jq"))

    def test_first_failure_stops_the_rest(self):
        env = {"DISTRIBUTION": "ubuntu24.04"}
        rc, run, _, error = self._run(env, [0, 100])
        self.assertEqual(rc, 3)
        self.assertEqual(run.call_count, 2)                 # jq never attempted
        self.assertEqual(error.call_args.kwargs["command"], "apt upgrade -y")

    def test_jq_failure_returns_3(self):
        env = {"DISTRIBUTION": "rocky9.7"}
        rc, run, info, error = self._run(env, [1])
        self.assertEqual(rc, 3)
        self.assertEqual([c.args[0] for c in run.call_args_list], _DNF_JQ)
        self.assertIn("installing jq failed", error.call_args.args[1])
        self.assertNotIn("System prepared", [c.args[1] for c in info.call_args_list])

    def test_unknown_distribution_fails_at_jq(self):
        # Bash's else branch would run `dnf install -y jq`; install_packages()
        # knows no manager for it, so nothing runs.
        rc, run, _, error = self._run({"DISTRIBUTION": "sles15"}, [])
        self.assertEqual(rc, 3)
        run.assert_not_called()
        error.assert_called_once()

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
