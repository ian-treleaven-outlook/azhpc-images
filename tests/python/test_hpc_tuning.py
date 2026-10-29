"""Unit tests for components/python/hpc_tuning.py.

Two kinds of test:

  * The pure helpers (distro branch, firewalld rule, NVLINK_RACKSCALE parsing)
    and the exact generated text are tested directly.
  * install() runs against a temporary directory: every path constant in the
    module is pointed into it with mock.patch.object, and exec_program /
    run_capture (which would run apt, pip3, dnf, systemctl and udevadm) are
    replaced by mocks. The tests check files, modes, exact command lists and
    key log lines. They never touch /etc or run real commands.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from unittest import mock

from components.python import hpc_tuning
from utils.build_config import BuildConfig, build_plan

# The real paths, relative to "/". Tests re-root them in a temporary directory.
_PATHS = {
    "_SYSCTL_CONF": "etc/sysctl.conf",
    "_SYSTEMD_SYSTEM_CONF": "etc/systemd/system.conf",
    "_SYSTEMD_USER_CONF": "etc/systemd/user.conf",
    "_LIMITS_CONF": "etc/security/limits.conf",
    "_SUNRPC_SCRIPT": "usr/sbin/sunrpc_tcp_settings.sh",
    "_SUNRPC_UNIT_FILE": "etc/systemd/system/sunrpc_tcp_settings.service",
    "_NFS_READAHEAD_RULES": "etc/udev/rules.d/90-nfs-readahead.rules",
}

# Exact bytes bash writes (checked against bash in WSL).
_LIMITS = (
    "*               hard    memlock         unlimited\n"
    "*               soft    memlock         unlimited\n"
    "*               hard    nofile          1048576\n"
    "*               soft    nofile          1048576\n"
    "*               hard    stack           unlimited\n"
    "*               soft    stack           unlimited\n")
_SYSCTL = (
    "vm.zone_reclaim_mode = 1\n"
    "net.ipv4.neigh.default.gc_thresh1 = 4096\n"
    "net.ipv4.neigh.default.gc_thresh2 = 8192\n"
    "net.ipv4.neigh.default.gc_thresh3 = 16384\n"
    "sunrpc.tcp_max_slot_table_entries = 128\n")
_RACKSCALE_SYSCTL = (
    "net.core.rmem_max = 2147483647\n"
    "net.core.wmem_max = 2147483647\n"
    "net.ipv4.tcp_rmem = 4096 67108864 1073741824\n"
    "net.ipv4.tcp_wmem = 4096 67108864 1073741824\n")
_UDEV_RULE = (
    'SUBSYSTEM=="bdi", ACTION=="add", PROGRAM="/usr/bin/awk -v bdi=$kernel '
    "'BEGIN{ret=1} {if ($4 == bdi) {ret=0}} END{exit ret}' "
    '/proc/fs/nfsfs/volumes", ATTR{read_ahead_kb}="15380"\n')

_AZSEC_PROBE = ["dnf", "list", "installed", "azsec-monitor"]
_AZSEC_REMOVE = ["dnf", "remove", "-y", "azsec-monitor"]

# The commands every distro runs after the distro block.
_COMMON_TAIL = [
    ["systemctl", "enable", "sunrpc_tcp_settings"],
    ["systemctl", "start", "sunrpc_tcp_settings"],
    ["systemctl", "is-active", "--quiet", "sunrpc_tcp_settings"],
    ["udevadm", "control", "--reload"],
]


# --------------------------------------------------------------------------- #
# Pure helpers: real inputs, no mocks.
# --------------------------------------------------------------------------- #
class HelperTests(unittest.TestCase):
    def test_distro_branch(self):
        cases = {
            "ubuntu22.04": "ubuntu22.04",
            "ubuntu24.04": "ubuntu24.04",
            "almalinux8.10": "rhel-family",
            "almalinux9.7": "rhel-family",
            "rocky8.10": "rhel-family",
            "rhel9.6": "rhel-family",
            "azurelinux3.0": "azurelinux3.0",
            "ubuntu20.04": None,
            "sles15": None,
            "": None,
        }
        for distro, expected in cases.items():
            self.assertEqual(hpc_tuning.distro_branch(distro), expected, distro)

    def test_rhel_family_is_a_prefix_match(self):
        self.assertTrue(hpc_tuning.is_rhel_family("rocky9.7"))
        # Unlike disable_cloudinit's substring check, "xrhel" doesn't match.
        self.assertFalse(hpc_tuning.is_rhel_family("xrhel9"))
        self.assertFalse(hpc_tuning.is_rhel_family("azurelinux3.0"))

    def test_firewalld_only_alma810_and_rhel8(self):
        for distro in ("almalinux8.10", "rhel8.10", "rhel8.8"):
            self.assertTrue(hpc_tuning.disables_firewalld(distro), distro)
        for distro in ("rocky8.10", "almalinux9.7", "almalinux8.9", "rhel9.6"):
            self.assertFalse(hpc_tuning.disables_firewalld(distro), distro)

    def test_rackscale_is_case_insensitive_true(self):
        for value in ("true", "True", "TRUE"):
            self.assertTrue(hpc_tuning.is_rackscale({"NVLINK_RACKSCALE": value}))
        for env in ({}, {"NVLINK_RACKSCALE": ""}, {"NVLINK_RACKSCALE": "false"},
                    {"NVLINK_RACKSCALE": "1"}, {"NVLINK_RACKSCALE": " true"}):
            self.assertFalse(hpc_tuning.is_rackscale(env), env)

    def test_udev_rule_is_one_line_with_literal_dollars(self):
        # Bash's unquoted heredoc joins the backslash-continued lines and turns
        # \$kernel / \$4 into $kernel / $4.
        self.assertEqual(hpc_tuning._NFS_READAHEAD_RULE, _UDEV_RULE)

    def test_limits_block_uses_spaces(self):
        self.assertEqual(hpc_tuning._LIMITS_BLOCK, _LIMITS)
        self.assertNotIn("\t", hpc_tuning._LIMITS_BLOCK)


# --------------------------------------------------------------------------- #
# install(): real file I/O in a temporary directory, commands mocked.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.paths = {name: os.path.join(self.root, rel)
                      for name, rel in _PATHS.items()}
        for name, path in self.paths.items():
            patcher = mock.patch.object(hpc_tuning, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        for rel in ("etc/systemd/system", "etc/security", "usr/sbin",
                    "etc/udev/rules.d"):
            os.makedirs(os.path.join(self.root, rel))
        self.write("_SYSCTL_CONF", "kernel.x = 1\n")
        self.write("_SYSTEMD_SYSTEM_CONF", "[Manager]\n")

    def write(self, name, text):
        with open(self.paths[name], "w", encoding="utf-8") as f:
            f.write(text)

    def read(self, name):
        with open(self.paths[name], encoding="utf-8") as f:
            return f.read()

    def install(self, env, *, rcs=None, azsec_installed=True):
        """Run install(env) with commands mocked.

        `rcs` maps a command (tuple) to a fake exit code. Returns (exit code,
        all commands in order, info/warn/error log mocks). The keyword
        arguments of each run_capture call are kept in self.capture_kwargs.
        """
        rcs = rcs or {}
        commands = []
        self.capture_kwargs = []

        def fake_exec(command, op, **kwargs):
            commands.append(command)
            return rcs.get(tuple(command), 0)

        def fake_capture(command, op, **kwargs):
            commands.append(command)
            self.capture_kwargs.append(kwargs)
            return (0 if azsec_installed else 1), ""

        with mock.patch.object(hpc_tuning, "exec_program", side_effect=fake_exec), \
             mock.patch.object(hpc_tuning, "run_capture", side_effect=fake_capture), \
             mock.patch.object(hpc_tuning, "log_info") as info, \
             mock.patch.object(hpc_tuning, "log_warn") as warn, \
             mock.patch.object(hpc_tuning, "log_error") as error:
            rc = hpc_tuning.install(env)
        return rc, commands, info, warn, error

    @staticmethod
    def messages(log_mock):
        return [c.args[1] for c in log_mock.call_args_list]

    # ---- success paths, one per distro branch ------------------------------
    def test_ubuntu2204(self):
        rc, commands, info, _, error = self.install({"DISTRIBUTION": "ubuntu22.04"})

        self.assertEqual(rc, 0)
        error.assert_not_called()
        self.assertEqual(commands, [
            ["pip3", "install", "-U", "netifaces"],
            ["pip3", "install", "-U", "PyYAML"],
            ["systemctl", "disable", "ufw"],
        ] + _COMMON_TAIL)
        self.assertEqual(self.read("_LIMITS_CONF"), _LIMITS)
        self.assertEqual(self.read("_SYSCTL_CONF"), "kernel.x = 1\n" + _SYSCTL)
        # No systemd memlock lines outside the RHEL family.
        self.assertEqual(self.read("_SYSTEMD_SYSTEM_CONF"), "[Manager]\n")
        self.assertFalse(os.path.exists(self.paths["_SYSTEMD_USER_CONF"]))
        self.assertEqual(self.read("_SUNRPC_SCRIPT"),
                         "#!/bin/bash\n\nmodprobe sunrpc\nsysctl -p\n")
        self.assertEqual(stat.S_IMODE(os.stat(self.paths["_SUNRPC_SCRIPT"]).st_mode),
                         0o755)
        self.assertEqual(self.read("_SUNRPC_UNIT_FILE"),
                         "[Unit]\n"
                         "Description=Set sunrpc tcp settings\n"
                         "\n"
                         "[Service]\n"
                         "User=root\n"
                         "Type=oneshot\n"
                         "ExecStart=/usr/sbin/sunrpc_tcp_settings.sh\n"
                         "RemainAfterExit=true\n"
                         "\n"
                         "[Install]\n"
                         "WantedBy=multi-user.target\n")
        self.assertEqual(self.read("_NFS_READAHEAD_RULES"), _UDEV_RULE)
        # Start line with fields, and the closing summary.
        start = info.call_args_list[0]
        self.assertEqual(start.args[1], "Applying HPC tuning")
        self.assertEqual(start.kwargs["branch"], "ubuntu22.04")
        self.assertIs(start.kwargs["nvlink_rackscale"], False)
        self.assertEqual(info.call_args_list[-1].args[1], "HPC tuning applied")

    def test_ubuntu2404(self):
        rc, commands, _, _, _ = self.install({"DISTRIBUTION": "ubuntu24.04"})

        self.assertEqual(rc, 0)
        self.assertEqual(commands, [
            ["apt", "install", "-y", "python3-netifaces"],
            ["apt", "install", "-y", "python3-yaml"],
            ["systemctl", "disable", "ufw"],
        ] + _COMMON_TAIL)

    def test_rhel_family_firewalld_and_azsec(self):
        cases = {
            "almalinux8.10": True, "rhel8.10": True,
            "rocky8.10": False, "almalinux9.7": False, "rocky9.7": False,
        }
        for distro, firewalld in cases.items():
            with self.subTest(distro=distro):
                self.write("_SYSTEMD_SYSTEM_CONF", "[Manager]\n")
                if os.path.exists(self.paths["_SYSTEMD_USER_CONF"]):
                    os.remove(self.paths["_SYSTEMD_USER_CONF"])
                rc, commands, _, _, _ = self.install({"DISTRIBUTION": distro})

                self.assertEqual(rc, 0)
                expected = ([["systemctl", "disable", "firewalld"]] if firewalld
                            else [])
                expected += [_AZSEC_PROBE, _AZSEC_REMOVE,
                             ["systemctl", "daemon-reexec"]] + _COMMON_TAIL
                self.assertEqual(commands, expected)
                self.assertEqual(self.read("_SYSTEMD_SYSTEM_CONF"),
                                 "[Manager]\nDefaultLimitMEMLOCK=infinity\n")
                self.assertEqual(self.read("_SYSTEMD_USER_CONF"),
                                 "DefaultLimitMEMLOCK=infinity\n")

    def test_azsec_monitor_not_installed_is_left_alone(self):
        rc, commands, info, _, error = self.install({"DISTRIBUTION": "rocky9.7"},
                                                    azsec_installed=False)

        self.assertEqual(rc, 0)
        error.assert_not_called()
        self.assertIn(_AZSEC_PROBE, commands)
        self.assertNotIn(_AZSEC_REMOVE, commands)
        self.assertTrue(any(c.kwargs.get("installed") is False
                            for c in info.call_args_list))
        # The probe's non-zero exit is an answer, not a failure: the command
        # runner is told not to log it as an ERROR.
        self.assertEqual(self.capture_kwargs[0].get("failure_is_error"), False)

    def test_azure_linux_only_removes_azsec(self):
        rc, commands, _, _, _ = self.install({"DISTRIBUTION": "azurelinux3.0"})

        self.assertEqual(rc, 0)
        self.assertEqual(commands, [_AZSEC_PROBE, _AZSEC_REMOVE] + _COMMON_TAIL)
        # Azure Linux isn't in the RHEL family: no systemd memlock lines.
        self.assertEqual(self.read("_SYSTEMD_SYSTEM_CONF"), "[Manager]\n")

    def test_other_distro_skips_the_distro_block(self):
        rc, commands, info, _, _ = self.install({"DISTRIBUTION": "sles15"})

        self.assertEqual(rc, 0)
        self.assertEqual(commands, _COMMON_TAIL)
        self.assertIn("No distro-specific tuning for this distribution",
                      self.messages(info))

    # ---- NVLINK_RACKSCALE ----------------------------------------------------
    def test_rackscale_appends_network_and_numa_settings(self):
        rc, _, info, _, _ = self.install({"DISTRIBUTION": "almalinux9.7",
                                          "NVLINK_RACKSCALE": "True"})

        self.assertEqual(rc, 0)
        # Rack-scale lines come before the always-added sysctl lines.
        self.assertEqual(self.read("_SYSCTL_CONF"),
                         "kernel.x = 1\n" + _RACKSCALE_SYSCTL + _SYSCTL)
        self.assertEqual(self.read("_SYSTEMD_SYSTEM_CONF"),
                         "[Manager]\nNUMAPolicy=bind\nNUMAMask=0-1\n"
                         "DefaultLimitMEMLOCK=infinity\n")
        self.assertIs(info.call_args_list[0].kwargs["nvlink_rackscale"], True)
        appended = [c.kwargs for c in info.call_args_list
                    if c.kwargs.get("action") == "appended"
                    and c.kwargs["path"] == self.paths["_SYSCTL_CONF"]]
        self.assertEqual([a["lines_added"] for a in appended], [4, 5])

    def test_rackscale_false_adds_nothing_extra(self):
        rc, _, info, _, _ = self.install({"DISTRIBUTION": "ubuntu24.04",
                                          "NVLINK_RACKSCALE": "false"})

        self.assertEqual(rc, 0)
        self.assertEqual(self.read("_SYSCTL_CONF"), "kernel.x = 1\n" + _SYSCTL)
        self.assertEqual(self.read("_SYSTEMD_SYSTEM_CONF"), "[Manager]\n")
        skipped = [c for c in info.call_args_list
                   if c.kwargs.get("nvlink_rackscale_value") == "false"]
        self.assertEqual(len(skipped), 1)

    # ---- reruns --------------------------------------------------------------
    def test_rerun_appends_again_and_warns(self):
        self.install({"DISTRIBUTION": "ubuntu24.04"})
        rc, _, _, warn, _ = self.install({"DISTRIBUTION": "ubuntu24.04"})

        self.assertEqual(rc, 0)
        self.assertEqual(self.read("_LIMITS_CONF"), _LIMITS * 2)
        self.assertEqual(self.read("_SYSCTL_CONF"), "kernel.x = 1\n" + _SYSCTL * 2)
        # One warning per file appended to (limits.conf and sysctl.conf).
        self.assertEqual(warn.call_count, 2)
        # Generated files are replaced, not duplicated.
        self.assertEqual(self.read("_NFS_READAHEAD_RULES"), _UDEV_RULE)

    # ---- failures --------------------------------------------------------------
    def test_each_failing_command_returns_3_and_stops(self):
        full = ([["systemctl", "disable", "firewalld"], _AZSEC_PROBE, _AZSEC_REMOVE,
                 ["systemctl", "daemon-reexec"]] + _COMMON_TAIL)
        # The probe's non-zero exit isn't a failure, so it's not in this loop.
        for i, failing in enumerate(full):
            if failing == _AZSEC_PROBE:
                continue
            with self.subTest(command=failing):
                rc, commands, _, _, error = self.install(
                    {"DISTRIBUTION": "almalinux8.10"}, rcs={tuple(failing): 1})

                self.assertEqual(rc, 3)
                self.assertEqual(commands, full[:i + 1])
                error.assert_called_once()

    def test_ubuntu_package_failure_stops_before_any_file_change(self):
        rc, commands, _, _, _ = self.install(
            {"DISTRIBUTION": "ubuntu24.04"},
            rcs={("apt", "install", "-y", "python3-netifaces"): 100})

        self.assertEqual(rc, 3)
        self.assertEqual(len(commands), 1)
        self.assertFalse(os.path.exists(self.paths["_LIMITS_CONF"]))

    def test_inactive_sunrpc_service_returns_3(self):
        rc, commands, _, _, error = self.install(
            {"DISTRIBUTION": "ubuntu24.04"},
            rcs={("systemctl", "is-active", "--quiet", "sunrpc_tcp_settings"): 3})

        self.assertEqual(rc, 3)
        self.assertNotIn(["udevadm", "control", "--reload"], commands)
        self.assertIn("not active", error.call_args.args[1])
        self.assertFalse(os.path.exists(self.paths["_NFS_READAHEAD_RULES"]))

    def test_unwritable_file_returns_3_without_raising(self):
        os.rmdir(os.path.join(self.root, "etc/udev/rules.d"))
        rc, commands, _, _, error = self.install({"DISTRIBUTION": "ubuntu24.04"})

        self.assertEqual(rc, 3)
        self.assertNotIn(["udevadm", "control", "--reload"], commands)
        self.assertIn(self.paths["_NFS_READAHEAD_RULES"], error.call_args.args[1])

    def test_unwritable_limits_conf_stops_early(self):
        os.rmdir(os.path.join(self.root, "etc/security"))
        rc, commands, _, _, error = self.install({"DISTRIBUTION": "sles15"})

        self.assertEqual(rc, 3)
        self.assertEqual(commands, [])
        self.assertEqual(error.call_args.kwargs["path"], self.paths["_LIMITS_CONF"])


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "hpc-tuning"]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, hpc_tuning.install)
        self.assertEqual(steps[0].script, "hpc-tuning.sh")


if __name__ == "__main__":
    unittest.main()
