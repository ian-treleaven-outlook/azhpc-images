"""Unit tests for components/python/azurelinux_network_config.py.

  * networkd.conf's text and the iptables commands are compared with
    distros/azurelinux3.0/network-config.sh, so they can't drift from the bash.
  * install() runs with systemctl, service_is_active and run_all mocked and
    networkd.conf written to a temporary directory, so no service, firewall
    or /etc is touched.

Run from the repo root:
    python3 -m unittest tests.python.test_azurelinux_network_config -v
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import azurelinux_network_config as m
from utils.plans import steps

_OP = "network-config"
_BASH_SCRIPT = (Path(__file__).resolve().parents[2]
                / "distros" / "azurelinux3.0" / "network-config.sh")
_ENV = {"DISTRIBUTION": "azurelinux3.0"}


def _bash_text() -> str:
    return _BASH_SCRIPT.read_text(encoding="utf-8")


class BashParityTests(unittest.TestCase):
    def test_networkd_conf_is_the_bash_heredoc(self):
        match = re.search(r"cat << EOF > /etc/systemd/networkd.conf\n(.*?)^EOF$",
                          _bash_text(), re.S | re.M)
        self.assertEqual(m._NETWORKD_SETTINGS, match.group(1))

    def test_iptables_commands_are_the_bash_ones_in_order(self):
        bash = [line.split() for line in _bash_text().splitlines()
                if line.startswith("iptables ")]
        self.assertEqual(m.accept_policy_commands(), bash)

    def test_systemctl_calls_are_the_bash_ones_in_order(self):
        bash = [line.split()[1:] for line in _bash_text().splitlines()
                if line.startswith("systemctl ")]
        self.assertEqual(bash, [["restart", "systemd-networkd"],
                                ["is-active", "systemd-networkd"],
                                ["stop", "iptables"], ["disable", "iptables"]])


class InstallTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.conf = os.path.join(tmp.name, "networkd.conf")
        for name in ("log_info", "log_error"):
            patcher = mock.patch.object(m, name)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

    def _install(self, *, systemctl=True, active=True, iptables=True):
        # systemctl: one result for every call, or a function giving each one.
        fake_systemctl = systemctl if callable(systemctl) else (lambda *a, **k: systemctl)
        with mock.patch.object(m, "_NETWORKD_CONF", self.conf), \
             mock.patch("utils.files.log_info"), \
             mock.patch.object(m, "systemctl", side_effect=fake_systemctl) as systemctl_mock, \
             mock.patch.object(m, "service_is_active", return_value=active) as is_active, \
             mock.patch.object(m, "run_all", return_value=iptables) as run_all:
            rc = m.install(_ENV)
        self.systemctl, self.is_active, self.run_all = systemctl_mock, is_active, run_all
        return rc

    def _systemctl_calls(self):
        return [c.args for c in self.systemctl.call_args_list]

    def test_happy_path(self):
        self.assertEqual(self._install(), 0)

        with open(self.conf, encoding="utf-8") as f:
            self.assertEqual(f.read(), m._NETWORKD_SETTINGS)
        self.assertEqual(self._systemctl_calls(), [
            ("restart", "systemd-networkd"), ("stop", "iptables"), ("disable", "iptables")])
        self.is_active.assert_called_once_with("systemd-networkd", op=_OP, env=_ENV)
        self.run_all.assert_called_once_with(m.accept_policy_commands(), _OP, env=_ENV)
        self.log_error.assert_not_called()

    def test_networkd_not_active_returns_3_before_the_firewall(self):
        self.assertEqual(self._install(active=False), 3)
        self.log_error.assert_called_once()
        self.assertEqual(self._systemctl_calls(), [("restart", "systemd-networkd")])
        self.run_all.assert_not_called()

    def test_failing_restart_returns_3(self):
        self.assertEqual(self._install(systemctl=False), 3)
        self.is_active.assert_not_called()
        self.run_all.assert_not_called()

    def test_failing_iptables_service_stop_returns_3(self):
        results = iter([True, False])
        self.assertEqual(self._install(systemctl=lambda *a, **k: next(results)), 3)
        self.assertEqual(self._systemctl_calls(), [
            ("restart", "systemd-networkd"), ("stop", "iptables")])
        self.run_all.assert_not_called()

    def test_failing_iptables_policy_returns_3(self):
        self.assertEqual(self._install(iptables=False), 3)

    def test_unwritable_conf_returns_3_without_restarting(self):
        with mock.patch.object(m, "write_file", return_value=False):
            self.assertEqual(self._install(), 3)
        self.systemctl.assert_not_called()


class StepWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_from_the_distro_folder(self):
        step = steps.azurelinux_network_config()
        self.assertEqual(step.op, "network-config")
        self.assertEqual(step.script, "network-config.sh")
        self.assertEqual(step.base, "distro")
        self.assertIs(step.action, m.install)


if __name__ == "__main__":
    unittest.main()
