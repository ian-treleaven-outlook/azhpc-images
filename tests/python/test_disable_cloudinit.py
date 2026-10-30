"""Unit tests for components/python/disable_cloudinit.py.

Two kinds of test:

  * The pure helpers (branch checks, the sed-like line deletion, and the two
    generated EL8 files) are tested directly with real inputs.
  * install() runs against a temporary directory: every path constant in the
    module is pointed into it with mock.patch.object, exec_program (under
    run(), systemctl() and service_is_active()) is replaced by a fake, and
    utils.logger._log records every log line. The tests check the files
    written, their modes, the exact commands, and the key log lines. They
    never touch /etc or run real commands.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from collections import namedtuple
from unittest import mock

from components.python import disable_cloudinit
from utils.build_config import BuildConfig, build_plan
from utils.distro import Distro

# The real paths, relative to "/". Tests re-root them in a temporary directory.
_PATHS = {
    "_CUSTOM_NETWORKING_CFG": "etc/cloud/cloud.cfg.d/99-custom-networking.cfg",
    "_IFCFG_FILE": "etc/sysconfig/network-scripts/ifcfg-eth0",
    "_SCRIPT_PATH": "usr/sbin/disable_cloudinit.sh",
    "_SERVICE_PATH": "etc/systemd/system/disable_cloudinit.service",
    "_DISABLE_NETWORK_CFG": "etc/cloud/cloud.cfg.d/99-disable-network-config.cfg",
    "_NETPLAN_BAREMETAL": "etc/netplan/net.yaml",
    "_NETPLAN_CLOUD_INIT": "etc/netplan/50-cloud-init.yaml",
    "_CLOUD_INIT_DISABLED": "etc/cloud/cloud-init.disabled",
}

_IFCFG = ("DEVICE=eth0\n"
          "HWADDR=00:0d:3a:aa:bb:cc\n"
          "ONBOOT=yes\n"
          "DHCP_HOSTNAME=buildvm\n"
          "BOOTPROTO=dhcp\n"
          "IPV6INIT=yes\n")
_IFCFG_CLEANED = "DEVICE=eth0\nONBOOT=yes\nBOOTPROTO=dhcp\n"

_SERVICE_COMMANDS = [
    ["systemctl", "daemon-reload"],
    ["systemctl", "enable", "disable_cloudinit.service"],
    ["systemctl", "start", "disable_cloudinit.service"],
    ["systemctl", "is-active", "--quiet", "disable_cloudinit.service"],
]


# --------------------------------------------------------------------------- #
# Pure helpers: real inputs, no mocks.
# --------------------------------------------------------------------------- #
class BranchHelperTests(unittest.TestCase):
    def test_rhel_branch_is_alma_rocky_and_rhel(self):
        # install() takes the RHEL branch on Distro.is_rhel_family.
        for distro in ("almalinux8.10", "almalinux9.7", "rocky8.10", "rocky9.7",
                       "rhel8.10", "rhel9.6"):
            self.assertTrue(Distro(distro).is_rhel_family, distro)
        # Azure Linux is none of them, so it takes the netplan branch.
        for distro in ("ubuntu24.04", "ubuntu22.04", "azurelinux3.0", ""):
            self.assertFalse(Distro(distro).is_rhel_family, distro)

    def test_el8_service_only_for_alma810_rocky810_and_rhel8(self):
        for distro in ("almalinux8.10", "rocky8.10", "rhel8.10", "rhel8.8"):
            self.assertTrue(disable_cloudinit.needs_ifcfg_cleanup_service(Distro(distro)))
        for distro in ("almalinux8.9", "rocky8.9", "almalinux9.7", "rocky9.7",
                       "rhel9.6", "ubuntu24.04"):
            self.assertFalse(disable_cloudinit.needs_ifcfg_cleanup_service(Distro(distro)))

    def test_node_type_defaults_when_unset_or_empty(self):
        # Like bash ${TARGET_NODE_TYPE:-azure_vm_regular}.
        self.assertEqual(disable_cloudinit.node_type({}), "azure_vm_regular")
        self.assertEqual(disable_cloudinit.node_type({"TARGET_NODE_TYPE": ""}),
                         "azure_vm_regular")
        self.assertEqual(
            disable_cloudinit.node_type({"TARGET_NODE_TYPE": "baremetal_1p"}),
            "baremetal_1p")


class DeleteLinesTests(unittest.TestCase):
    def test_deletes_only_lines_starting_with_the_prefix(self):
        text = "A=1\nHWADDR=x\n  HWADDR=indented\nXHWADDR=y\nHWADDR=z\n"
        self.assertEqual(
            disable_cloudinit.delete_lines_starting_with(text, "HWADDR="),
            ("A=1\n  HWADDR=indented\nXHWADDR=y\n", 2))

    def test_last_line_without_newline(self):
        # sed keeps the file's lack of a final newline, and deleting the last
        # line leaves the previous line's newline in place.
        self.assertEqual(
            disable_cloudinit.delete_lines_starting_with("A=1\nB=2", "HWADDR="),
            ("A=1\nB=2", 0))
        self.assertEqual(
            disable_cloudinit.delete_lines_starting_with("A=1\nHWADDR=x", "HWADDR="),
            ("A=1\n", 1))

    def test_carriage_returns_are_kept(self):
        self.assertEqual(
            disable_cloudinit.delete_lines_starting_with("A=1\r\nIPV6INIT=no\r\n",
                                                         "IPV6INIT="),
            ("A=1\r\n", 1))

    def test_empty_text(self):
        self.assertEqual(disable_cloudinit.delete_lines_starting_with("", "X="),
                         ("", 0))


class GeneratedFileTests(unittest.TestCase):
    # Both expected texts were produced by bash's unquoted heredocs in WSL.
    def test_cleanup_script_expands_the_ifcfg_path(self):
        self.assertEqual(
            disable_cloudinit.cleanup_script_text(
                "/etc/sysconfig/network-scripts/ifcfg-eth0"),
            "#!/bin/bash\n"
            "\n"
            "# Remove Hardware Mac Address and DHCP Name\n"
            "sed -i '/^HWADDR=.*$/d' \"/etc/sysconfig/network-scripts/ifcfg-eth0\"\n"
            "sed -i '/^DHCP_HOSTNAME=.*$/d' \"/etc/sysconfig/network-scripts/ifcfg-eth0\"\n"
            "sed -i '/^IPV6INIT=.*$/d' \"/etc/sysconfig/network-scripts/ifcfg-eth0\"\n"
            "\n"
            "# Restart NetworkManager to apply changes\n"
            "systemctl restart NetworkManager\n")

    def test_cleanup_service_expands_the_script_path(self):
        self.assertEqual(
            disable_cloudinit.cleanup_service_text("/usr/sbin/disable_cloudinit.sh"),
            "[Unit]\n"
            "Description=Clean network config after boot\n"
            "After=network.target NetworkManager.service\n"
            "Wants=NetworkManager.service\n"
            "\n"
            "[Service]\n"
            "Type=oneshot\n"
            "ExecStart=/usr/sbin/disable_cloudinit.sh\n"
            "RemainAfterExit=yes\n"
            "\n"
            "[Install]\n"
            "WantedBy=multi-user.target\n")


# --------------------------------------------------------------------------- #
# install(): real file I/O in a temporary directory, commands mocked.
# --------------------------------------------------------------------------- #
_Record = namedtuple("_Record", "level message fields")


class InstallTestBase(unittest.TestCase):
    def setUp(self):
        # TemporaryDirectory() is deleted again by cleanup at the end of the test.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.paths = {name: os.path.join(self.root, rel)
                      for name, rel in _PATHS.items()}
        for name, path in self.paths.items():
            patcher = mock.patch.object(disable_cloudinit, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def mkdirs(self, *rel_dirs):
        for rel in rel_dirs:
            os.makedirs(os.path.join(self.root, rel), exist_ok=True)

    def write(self, name, text):
        with open(self.paths[name], "w", encoding="utf-8") as f:
            f.write(text)

    def read(self, name):
        with open(self.paths[name], encoding="utf-8", newline="") as f:
            return f.read()

    def mode(self, name):
        return stat.S_IMODE(os.stat(self.paths[name]).st_mode)

    def install(self, env, *, rcs=None):
        """Run install(env). `rcs` maps a command (tuple) to a fake exit code.

        Returns (exit code, commands run, and the info, warn and error log
        records).
        """
        rcs = rcs or {}
        commands = []
        records = []

        def fake_exec(command, op, **kwargs):
            commands.append(command)
            return rcs.get(tuple(command), 0)

        def fake_log(level, op, message, fields=None, detail=None):
            self.assertEqual(op, "disable-cloudinit")
            records.append(_Record(level, message, fields or {}))

        with mock.patch("utils.process.exec_program", side_effect=fake_exec), \
             mock.patch("utils.services.exec_program", side_effect=fake_exec), \
             mock.patch("utils.logger._log", side_effect=fake_log):
            rc = disable_cloudinit.install(env)
        by_level = {level: [r for r in records if r.level == level]
                    for level in ("info", "warn", "error")}
        return rc, commands, by_level["info"], by_level["warn"], by_level["error"]

    @staticmethod
    def messages(records):
        return [r.message for r in records]


class RhelInstallTests(InstallTestBase):
    def setUp(self):
        super().setUp()
        self.mkdirs("etc/cloud/cloud.cfg.d", "etc/sysconfig/network-scripts",
                    "usr/sbin", "etc/systemd/system")
        self.write("_CUSTOM_NETWORKING_CFG", "existing: 1\n")
        self.write("_IFCFG_FILE", _IFCFG)

    def test_el8_writes_files_and_starts_the_service(self):
        for distro in ("almalinux8.10", "rocky8.10", "rhel8.10"):
            with self.subTest(distro=distro):
                self.write("_CUSTOM_NETWORKING_CFG", "existing: 1\n")
                self.write("_IFCFG_FILE", _IFCFG)
                rc, commands, info, _, error = self.install({"DISTRIBUTION": distro})

                self.assertEqual(rc, 0)
                self.assertEqual(error, [])
                # Appended, not replaced.
                self.assertEqual(self.read("_CUSTOM_NETWORKING_CFG"),
                                 "existing: 1\nnetwork: {config: disabled}\n")
                self.assertEqual(self.read("_IFCFG_FILE"), _IFCFG_CLEANED)
                self.assertEqual(self.read("_SCRIPT_PATH"),
                                 disable_cloudinit.cleanup_script_text(
                                     self.paths["_IFCFG_FILE"]))
                self.assertEqual(self.mode("_SCRIPT_PATH"), 0o755)
                self.assertEqual(self.read("_SERVICE_PATH"),
                                 disable_cloudinit.cleanup_service_text(
                                     self.paths["_SCRIPT_PATH"]))
                self.assertEqual(commands, _SERVICE_COMMANDS)
                # Start line with structured fields, and the closing summary.
                start = info[0]
                self.assertEqual(start.message,
                                 "Disabling cloud-init network configuration")
                self.assertEqual(start.fields["distro"], distro)
                self.assertEqual(start.fields["branch"], "rhel-ifcfg")
                self.assertIn("Cloud-init network configuration disabled",
                              self.messages(info))

    def test_el9_skips_the_service(self):
        rc, commands, info, _, _ = self.install({"DISTRIBUTION": "almalinux9.7"})

        self.assertEqual(rc, 0)
        self.assertEqual(self.read("_IFCFG_FILE"), _IFCFG_CLEANED)
        self.assertFalse(os.path.exists(self.paths["_SCRIPT_PATH"]))
        self.assertFalse(os.path.exists(self.paths["_SERVICE_PATH"]))
        self.assertEqual(commands, [])
        decision = [r for r in info if r.fields.get("el8_service") is False]
        self.assertEqual(len(decision), 1)

    def test_logs_each_file_change(self):
        _, _, info, _, _ = self.install({"DISTRIBUTION": "rocky8.10"})

        by_path = {r.fields.get("path"): r.fields for r in info if "action" in r.fields}
        self.assertEqual(by_path[self.paths["_CUSTOM_NETWORKING_CFG"]]["action"],
                         "appended")
        edited = [r.fields for r in info
                  if r.fields.get("path") == self.paths["_IFCFG_FILE"]]
        self.assertEqual(edited, [{"path": self.paths["_IFCFG_FILE"], "changed": True}])
        self.assertEqual(by_path[self.paths["_SCRIPT_PATH"]]["action"], "created")
        self.assertEqual(by_path[self.paths["_SERVICE_PATH"]]["action"], "created")

    def test_rerun_appends_again_and_warns(self):
        self.install({"DISTRIBUTION": "almalinux9.7"})
        rc, _, _, warn, _ = self.install({"DISTRIBUTION": "almalinux9.7"})

        self.assertEqual(rc, 0)
        self.assertEqual(self.read("_CUSTOM_NETWORKING_CFG"),
                         "existing: 1\n" + "network: {config: disabled}\n" * 2)
        self.assertEqual(len(warn), 1)

    def test_missing_ifcfg_returns_3_before_any_service_work(self):
        os.remove(self.paths["_IFCFG_FILE"])
        rc, commands, _, _, error = self.install({"DISTRIBUTION": "almalinux8.10"})

        self.assertEqual(rc, 3)
        self.assertEqual(commands, [])
        self.assertFalse(os.path.exists(self.paths["_SCRIPT_PATH"]))
        self.assertIn("ifcfg-eth0", error[-1].message)

    def test_each_failing_service_command_returns_3_and_stops(self):
        for i, failing in enumerate(_SERVICE_COMMANDS):
            with self.subTest(command=failing):
                self.write("_IFCFG_FILE", _IFCFG)
                rc, commands, info, _, _ = self.install(
                    {"DISTRIBUTION": "almalinux8.10"}, rcs={tuple(failing): 1})

                self.assertEqual(rc, 3)
                self.assertEqual(commands, _SERVICE_COMMANDS[:i + 1])
                self.assertNotIn("Cloud-init network configuration disabled",
                                 self.messages(info))

    def test_inactive_service_is_reported(self):
        rc, _, _, _, error = self.install(
            {"DISTRIBUTION": "almalinux8.10"},
            rcs={tuple(_SERVICE_COMMANDS[-1]): 3})

        self.assertEqual(rc, 3)
        self.assertIn("not active", error[-1].message)

    def test_unwritable_script_returns_3_without_raising(self):
        os.makedirs(self.paths["_SCRIPT_PATH"])   # a folder can't be written as a file
        rc, commands, _, _, error = self.install({"DISTRIBUTION": "almalinux8.10"})

        self.assertEqual(rc, 3)
        self.assertEqual(commands, [])
        self.assertIn(self.paths["_SCRIPT_PATH"], error[-1].message)


class NetplanInstallTests(InstallTestBase):
    def setUp(self):
        super().setUp()
        self.mkdirs("etc/cloud/cloud.cfg.d", "etc/netplan")
        self.write("_DISABLE_NETWORK_CFG", "old\ncontent\n")
        self.write("_NETPLAN_CLOUD_INIT", "old: yaml\n")

    def test_regular_node_replaces_cloud_init_yaml_and_applies(self):
        for env in ({"DISTRIBUTION": "ubuntu24.04"},
                    {"DISTRIBUTION": "ubuntu22.04", "TARGET_NODE_TYPE": ""},
                    {"DISTRIBUTION": "ubuntu24.04",
                     "TARGET_NODE_TYPE": "baremetal_3p"}):
            with self.subTest(env=env):
                rc, commands, info, _, error = self.install(env)

                self.assertEqual(rc, 0)
                self.assertEqual(error, [])
                # tee (no -a) replaces the file.
                self.assertEqual(self.read("_DISABLE_NETWORK_CFG"),
                                 "network: {config: disabled}\n")
                self.assertEqual(self.read("_NETPLAN_CLOUD_INIT"),
                                 "network:\n"
                                 "    ethernets:\n"
                                 "        eth0:\n"
                                 "            dhcp4: true\n"
                                 "    version: 2\n")
                self.assertFalse(os.path.exists(self.paths["_NETPLAN_BAREMETAL"]))
                self.assertEqual(commands, [["netplan", "apply"]])
                self.assertEqual(info[0].fields["branch"], "netplan")

    def test_baremetal_1p_writes_net_yaml_and_disables_cloud_init(self):
        rc, commands, info, _, _ = self.install(
            {"DISTRIBUTION": "ubuntu24.04", "TARGET_NODE_TYPE": "baremetal_1p"})

        self.assertEqual(rc, 0)
        self.assertEqual(self.read("_NETPLAN_BAREMETAL"),
                         "network:\n"
                         "  ethernets:\n"
                         "    mana-nics:\n"
                         "      dhcp4: true\n"
                         "      match:\n"
                         "        driver: mana\n"
                         "        name: e*\n"
                         "    eth0:\n"
                         "      dhcp4: true\n")
        self.assertEqual(self.mode("_NETPLAN_BAREMETAL"), 0o600)
        self.assertFalse(os.path.exists(self.paths["_NETPLAN_CLOUD_INIT"]))
        self.assertEqual(self.read("_CLOUD_INIT_DISABLED"), "")
        self.assertEqual(commands, [])          # no netplan apply on bare metal
        removed = [r for r in info if r.fields.get("action") == "removed"]
        self.assertEqual(removed[0].fields["path"], self.paths["_NETPLAN_CLOUD_INIT"])

    def test_baremetal_1p_without_cloud_init_yaml(self):
        os.remove(self.paths["_NETPLAN_CLOUD_INIT"])
        rc, _, _, _, _ = self.install(
            {"DISTRIBUTION": "ubuntu24.04", "TARGET_NODE_TYPE": "baremetal_1p"})

        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(self.paths["_CLOUD_INIT_DISABLED"]))

    def test_netplan_apply_failure_returns_3(self):
        rc, commands, info, _, _ = self.install({"DISTRIBUTION": "ubuntu24.04"},
                                                rcs={("netplan", "apply"): 1})

        self.assertEqual(rc, 3)
        self.assertEqual(commands, [["netplan", "apply"]])
        self.assertNotIn("Cloud-init network configuration disabled", self.messages(info))

    def test_azure_linux_takes_netplan_branch_and_fails_without_netplan(self):
        os.remove(self.paths["_NETPLAN_CLOUD_INIT"])
        os.rmdir(os.path.join(self.root, "etc/netplan"))
        rc, commands, _, warn, _ = self.install({"DISTRIBUTION": "azurelinux3.0"},
                                                rcs={("netplan", "apply"): 127})

        # As in bash the cloud-init file is written; write_file creates the
        # missing /etc/netplan, so the step fails at `netplan apply` (not found).
        self.assertEqual(rc, 3)
        self.assertEqual(self.read("_DISABLE_NETWORK_CFG"),
                         "network: {config: disabled}\n")
        self.assertEqual(commands, [["netplan", "apply"]])
        self.assertIn("Azure Linux", warn[-1].message)


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "disable-cloudinit"]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, disable_cloudinit.install)
        self.assertEqual(steps[0].script, "disable_cloudinit.sh")


if __name__ == "__main__":
    unittest.main()
