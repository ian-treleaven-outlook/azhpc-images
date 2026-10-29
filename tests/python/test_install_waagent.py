"""Unit tests for components/python/install_waagent.py.

Two kinds of test:

  * The pure helpers (distro tests, command lists, the waagent.conf and
    waagent.service text edits, the `waagent --version` parser) are tested
    directly with realistic inputs. Their grep/sed/awk equivalents were
    compared against real bash in WSL with the same inputs.
  * install() is tested with exec_program, run_capture, download_and_verify,
    write_component_version and time.sleep replaced by fakes. The conf file,
    the service file and the work directory point into a temporary directory,
    so the real file edits run without touching /etc, /usr or /tmp.

Run from the repo root:
    python3 -m unittest tests.python.test_install_waagent -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_waagent
from utils.build_config import BuildConfig, build_plan
from utils.download import ChecksumError

_OP = "install-waagent"
_VERSION = "2.15.0.1"
_SHA = "f405cb90179dbe582a1c71a36af8016072c0defeb969aeee7861d5aab10499d8"
_URL = f"https://github.com/Azure/WALinuxAgent/archive/refs/tags/v{_VERSION}.tar.gz"
_TIMESTAMP = "2026-10-29_15:07:42"

# What `waagent --version` prints.
_VERSION_OUTPUT = ("WALinuxAgent-2.15.0.1 running on ubuntu 24.04\n"
                   "Python: 3.12.3\n"
                   "Goal state agent: 2.15.1.3\n")

# A cut-down waagent.conf as shipped by the agent.
_CONF_BEFORE = ("# Enable RDMA management and set up\n"
                "OS.EnableRDMA=n\n"
                "Extensions.GoalStatePeriod=6\n"
                "Provisioning.MonitorHostName=n\n")

# After the non-Azure-Linux settings on a target that uses IPoIB.
_CONF_AFTER_WITH_RDMA = ("# Enable RDMA management and set up\n"
                         "OS.EnableRDMA=y\n"
                         "Extensions.GoalStatePeriod=300\n"
                         "Provisioning.MonitorHostName=y\n"
                         "Extensions.InitialGoalStatePeriod=6\n"
                         "OS.EnableFirewallPeriod=300\n"
                         "OS.RemovePersistentNetRulesPeriod=300\n"
                         "OS.RootDeviceScsiTimeoutPeriod=300\n"
                         "OS.MonitorDhcpClientRestartPeriod=60\n"
                         "Provisioning.MonitorHostNamePeriod=60\n")

# The same without OS.EnableRDMA=y: the shipped "n" stays.
_CONF_AFTER_WITHOUT_RDMA = _CONF_AFTER_WITH_RDMA.replace("OS.EnableRDMA=y", "OS.EnableRDMA=n")

# After the Azure Linux settings (no Provisioning.MonitorHostName).
_CONF_AFTER_AZURE_LINUX = ("# Enable RDMA management and set up\n"
                           "OS.EnableRDMA=y\n"
                           "Extensions.GoalStatePeriod=300\n"
                           "Provisioning.MonitorHostName=n\n"
                           "Extensions.InitialGoalStatePeriod=6\n"
                           "OS.EnableFirewallPeriod=300\n"
                           "OS.RemovePersistentNetRulesPeriod=300\n"
                           "OS.RootDeviceScsiTimeoutPeriod=300\n"
                           "OS.MonitorDhcpClientRestartPeriod=60\n"
                           "Provisioning.MonitorHostNamePeriod=60\n")

_SERVICE_BEFORE = ("[Unit]\nDescription=Azure Linux Agent\n\n[Service]\nType=simple\n"
                   "ExecStart=/usr/bin/python3 -u /usr/sbin/waagent -daemon\n"
                   "Restart=always\n")
_SERVICE_AFTER = _SERVICE_BEFORE.replace("python3 -u", "python3.12 -u")

_SETUP = ["setup.py", "install", "--register-service"]


def _pip(python):
    return [[python, "-m", "ensurepip", "--upgrade"],
            [python, "-m", "pip", "install", "--upgrade", "pip", "setuptools"],
            [python, "-m", "pip", "install", "distro"]]


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
class DistroHelperTests(unittest.TestCase):
    def test_install_handler_follows_bash_patterns(self):
        cases = {
            "almalinux8.10": "el8", "rocky8.10": "el8", "rhel8.10": "el8",
            "almalinux8.9": "default",          # bash's EL8 test is an exact match
            "almalinux9.6": "el9", "rocky9.7": "el9", "rhel9.4": "el9",
            "ubuntu24.04": "default", "ubuntu22.04": "default",
        }
        for distribution, handler in cases.items():
            with self.subTest(distribution=distribution):
                self.assertEqual(install_waagent.install_handler(distribution), handler)

    def test_rhel_family_is_a_contains_test(self):
        for distribution in ("almalinux8.10", "rocky9.7", "rhel9.4"):
            self.assertTrue(install_waagent.is_rhel_family(distribution))
        for distribution in ("ubuntu24.04", "azurelinux3.0"):
            self.assertFalse(install_waagent.is_rhel_family(distribution))

    def test_setup_commands(self):
        self.assertEqual(install_waagent.setup_commands("el8"),
                         _pip("python3") + [["python3", *_SETUP]])
        self.assertEqual(install_waagent.setup_commands("el9"),
                         _pip("python3.12") + [["python3.12", *_SETUP]])
        self.assertEqual(install_waagent.setup_commands("default"), [["python3", *_SETUP]])

    def test_version_command_and_restart_service(self):
        self.assertEqual(install_waagent.version_command("rocky9.7"),
                         ["python3.12", "-u", "/usr/sbin/waagent", "--version"])
        self.assertEqual(install_waagent.version_command("rocky8.10"), ["waagent", "--version"])
        self.assertEqual(install_waagent.restart_service("ubuntu22.04"), "walinuxagent")
        for distribution in ("almalinux9.6", "rocky8.10", "rhel9.4", "azurelinux3.0"):
            self.assertEqual(install_waagent.restart_service(distribution), "waagent")
        self.assertIsNone(install_waagent.restart_service("debian12"))

    def test_backup_timestamp_matches_date_f_t(self):
        # `date +%F_%T` -> 2026-10-29_15:07:42
        self.assertRegex(install_waagent.backup_timestamp(),
                         r"^\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}$")


class ConfTextTests(unittest.TestCase):
    def test_replaces_an_existing_key(self):
        text, n = install_waagent.update_waagent_conf_text(
            "A=1\nOS.EnableRDMA=n\nB=2\n", "OS.EnableRDMA", "y")
        self.assertEqual((text, n), ("A=1\nOS.EnableRDMA=y\nB=2\n", 1))

    def test_appends_a_missing_key(self):
        text, n = install_waagent.update_waagent_conf_text("A=1\n", "OS.EnableRDMA", "y")
        self.assertEqual((text, n), ("A=1\nOS.EnableRDMA=y\n", 0))

    def test_creates_content_for_an_empty_file(self):
        self.assertEqual(install_waagent.update_waagent_conf_text("", "K.X", "1"),
                         ("K.X=1\n", 0))

    def test_dot_is_a_wildcard_and_every_matching_line_is_replaced(self):
        # Like grep/sed: "." matches any character, so OSXEnableRDMA matches too.
        text, n = install_waagent.update_waagent_conf_text(
            "OS.EnableRDMA=n\nOSXEnableRDMA=n\n", "OS.EnableRDMA", "y")
        self.assertEqual((text, n), ("OS.EnableRDMA=y\nOS.EnableRDMA=y\n", 2))

    def test_comments_indented_lines_and_longer_keys_are_not_matched(self):
        original = "# OS.EnableRDMA=y\n  OS.EnableRDMA=n\nOS.EnableRDMAX=n\n"
        text, n = install_waagent.update_waagent_conf_text(original, "OS.EnableRDMA", "y")
        self.assertEqual((text, n), (original + "OS.EnableRDMA=y\n", 0))

    def test_no_final_newline_is_kept_like_sed_and_echo(self):
        # sed keeps a missing final newline; `echo >>` glues onto the last line.
        self.assertEqual(install_waagent.update_waagent_conf_text("K=1", "K", "2"), ("K=2", 1))
        self.assertEqual(install_waagent.update_waagent_conf_text("A=1", "K", "2"),
                         ("A=1K=2\n", 0))

    def test_retarget_exec_start(self):
        text, n = install_waagent.retarget_exec_start(_SERVICE_BEFORE)
        self.assertEqual((text, n), (_SERVICE_AFTER, 1))
        # A rerun on the edited file changes nothing.
        self.assertEqual(install_waagent.retarget_exec_start(_SERVICE_AFTER), (_SERVICE_AFTER, 0))


class ParseVersionTests(unittest.TestCase):
    def test_real_output(self):
        self.assertEqual(install_waagent.parse_waagent_version(_VERSION_OUTPUT),
                         ("2.15.0.1", "2.15.1.3"))

    def test_extra_dash_and_leading_spaces(self):
        # awk -F' ' skips leading blanks; awk -F- '{print $2}' takes the 2nd part.
        output = "  WALinuxAgent-2.11.8.8-extra running\nPython: 3.12\nGoal state agent: 2.11.1.4\n"
        self.assertEqual(install_waagent.parse_waagent_version(output),
                         ("2.11.8.8", "2.11.1.4"))

    def test_missing_pieces_give_empty_strings(self):
        self.assertEqual(install_waagent.parse_waagent_version(""), ("", ""))
        self.assertEqual(install_waagent.parse_waagent_version("WALinuxAgent\nPython: 3\n"),
                         ("", ""))


# --------------------------------------------------------------------------- #
# install(): external effects faked.
# --------------------------------------------------------------------------- #
class _FakeExec:
    """Stands in for exec_program: records (command, cwd) and imitates `tar`.

    fail_at, when given, is the index of the call that should fail (exit 1).
    """

    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def __call__(self, command, op, *, cwd=None, env=None):
        assert op == _OP
        self.calls.append((command, cwd))
        if len(self.calls) - 1 == self.fail_at:
            return 1
        if command[0] == "tar":
            os.makedirs(os.path.join(cwd, f"WALinuxAgent-{_VERSION}"))
        return 0

    @property
    def commands(self):
        return [command for command, _ in self.calls]


def _fake_download(url, sha256, dest_dir="."):
    """Stands in for download_and_verify: writes an empty 'tarball'."""
    os.makedirs(dest_dir, exist_ok=True)
    path = Path(dest_dir) / Path(url).name
    path.write_bytes(b"")
    return path


def _env(distribution, *, sku="H100", node_type="azure_vm_regular", config=True):
    versions = {"waagent": {"common": {"version": _VERSION, "sha256": _SHA}}} if config else {}
    return {
        "DISTRIBUTION": distribution,
        "ARCHITECTURE": "x86_64",
        "GPU": "NVIDIA",
        "SKU": sku,
        "TARGET_NODE_TYPE": node_type,
        "COMPONENT_VERSIONS": json.dumps(versions),
    }


class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.conf = self.root / "etc" / "waagent.conf"
        self.service = self.root / "usr" / "waagent.service"
        self.work = self.root / "work"
        self.conf.parent.mkdir(parents=True)
        self.service.parent.mkdir(parents=True)
        self.conf.write_text(_CONF_BEFORE, encoding="utf-8")
        self.service.write_text(_SERVICE_BEFORE, encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _install(self, env, *, fake=None, capture=(0, _VERSION_OUTPUT),
                 download_error=None, write_error=None):
        """Run install(env) with every external effect faked.

        Returns (rc, mocks); mocks is a dict of the fakes so tests can check
        what install() did and logged.
        """
        fake = fake or _FakeExec()
        download = mock.Mock(side_effect=download_error or _fake_download)
        with mock.patch.object(install_waagent, "_WAAGENT_CONF", str(self.conf)), \
             mock.patch.object(install_waagent, "_WAAGENT_SERVICE_FILE", str(self.service)), \
             mock.patch.object(install_waagent, "_WORK_DIR", str(self.work)), \
             mock.patch.object(install_waagent, "backup_timestamp", return_value=_TIMESTAMP), \
             mock.patch.object(install_waagent, "exec_program", fake), \
             mock.patch.object(install_waagent, "download_and_verify", download), \
             mock.patch.object(install_waagent, "run_capture",
                               return_value=capture) as run_capture, \
             mock.patch.object(install_waagent, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(install_waagent.time, "sleep") as sleep, \
             mock.patch.object(install_waagent, "log_info") as log_info, \
             mock.patch.object(install_waagent, "log_warn") as log_warn, \
             mock.patch.object(install_waagent, "log_error") as log_error:
            rc = install_waagent.install(env)
        return rc, {"exec": fake, "download": download, "run_capture": run_capture,
                    "record": record, "sleep": sleep, "log_info": log_info,
                    "log_warn": log_warn, "log_error": log_error}

    def _source_dir(self):
        return str(self.work / f"WALinuxAgent-{_VERSION}")

    # ---- success paths -----------------------------------------------------
    def test_ubuntu_installs_from_source_and_enables_rdma(self):
        env = _env("ubuntu24.04")
        rc, m = self._install(env)

        self.assertEqual(rc, 0)
        m["download"].assert_called_once_with(_URL, _SHA, dest_dir=str(self.work))
        self.assertEqual(m["exec"].calls, [
            (["tar", "-xvf", f"v{_VERSION}.tar.gz"], str(self.work)),
            (["python3", *_SETUP], self._source_dir()),
            (["systemctl", "daemon-reload"], None),
            (["systemctl", "restart", "walinuxagent"], None),
        ])
        self.assertEqual(self.conf.read_text(encoding="utf-8"), _CONF_AFTER_WITH_RDMA)
        m["run_capture"].assert_called_once_with(["waagent", "--version"], _OP, env=env)
        self.assertEqual(m["record"].call_args_list, [
            mock.call("WAAGENT", "2.15.0.1"), mock.call("WAAGENT_EXTENSIONS", "2.15.1.3")])
        m["sleep"].assert_called_once_with(60)
        # Like bash: the unpacked folder is removed, the tarball is kept.
        self.assertFalse(os.path.exists(self._source_dir()))
        self.assertTrue((self.work / f"v{_VERSION}.tar.gz").exists())
        m["log_error"].assert_not_called()

    def test_targets_without_ipoib_leave_enable_rdma_alone(self):
        for sku, node_type in (("GB200", "baremetal_3p"), ("GB200", "baremetal_1p"),
                               ("NCv6", "azure_vm_regular")):
            with self.subTest(sku=sku, node_type=node_type):
                self.conf.write_text(_CONF_BEFORE, encoding="utf-8")
                rc, _ = self._install(_env("ubuntu24.04", sku=sku, node_type=node_type))
                self.assertEqual(rc, 0)
                self.assertEqual(self.conf.read_text(encoding="utf-8"),
                                 _CONF_AFTER_WITHOUT_RDMA)

    def test_azure_linux_updates_the_package(self):
        env = _env("azurelinux3.0", config=False)   # the config isn't needed
        rc, m = self._install(env)

        self.assertEqual(rc, 0)
        m["download"].assert_not_called()
        self.assertEqual(m["exec"].commands, [
            ["dnf", "update", "-y", "WALinuxAgent"],
            ["systemctl", "daemon-reload"],
            ["systemctl", "restart", "waagent"],
        ])
        self.assertEqual(self.conf.read_text(encoding="utf-8"), _CONF_AFTER_AZURE_LINUX)
        m["run_capture"].assert_called_once_with(["waagent", "--version"], _OP, env=env)

    def test_azure_linux_sets_rdma_even_on_baremetal_3p(self):
        rc, _ = self._install(_env("azurelinux3.0", sku="GB200", node_type="baremetal_3p"))
        self.assertEqual(rc, 0)
        self.assertIn("OS.EnableRDMA=y\n", self.conf.read_text(encoding="utf-8"))

    def test_el8_runs_the_pip_setup_twice_with_python3(self):
        rc, m = self._install(_env("almalinux8.10"))

        self.assertEqual(rc, 0)
        src = self._source_dir()
        self.assertEqual(m["exec"].calls,
                         [(c, None) for c in _pip("python3")]
                         + [(["tar", "-xvf", f"v{_VERSION}.tar.gz"], str(self.work))]
                         + [(c, src) for c in _pip("python3") + [["python3", *_SETUP]]]
                         + [(["systemctl", "daemon-reload"], None),
                            (["systemctl", "restart", "waagent"], None)])
        self.assertEqual(self.service.read_text(encoding="utf-8"), _SERVICE_BEFORE)

    def test_el9_uses_python312_and_rewrites_the_service(self):
        env = _env("rocky9.7")
        rc, m = self._install(env)

        self.assertEqual(rc, 0)
        src = self._source_dir()
        self.assertEqual(m["exec"].calls,
                         [(c, None) for c in _pip("python3")]
                         + [(["tar", "-xvf", f"v{_VERSION}.tar.gz"], str(self.work))]
                         + [(c, src) for c in _pip("python3.12") + [["python3.12", *_SETUP]]]
                         + [(["systemctl", "stop", "waagent"], src),
                            (["systemctl", "disable", "waagent"], src),
                            (["systemctl", "daemon-reexec"], src),
                            (["systemctl", "enable", "waagent"], src),
                            (["systemctl", "daemon-reload"], None),
                            (["systemctl", "restart", "waagent"], None)])
        self.assertEqual(self.service.read_text(encoding="utf-8"), _SERVICE_AFTER)
        backup = Path(f"{self.service}.bak.{_TIMESTAMP}")
        self.assertEqual(backup.read_text(encoding="utf-8"), _SERVICE_BEFORE)
        m["run_capture"].assert_called_once_with(
            ["python3.12", "-u", "/usr/sbin/waagent", "--version"], _OP, env=env)

    def test_missing_conf_file_is_created(self):
        self.conf.unlink()
        rc, m = self._install(_env("ubuntu24.04"))
        self.assertEqual(rc, 0)
        self.assertTrue(self.conf.read_text(encoding="utf-8").startswith(
            "Extensions.GoalStatePeriod=300\n"))
        actions = [c.kwargs.get("action") for c in m["log_info"].call_args_list
                   if c.kwargs.get("key")]
        self.assertEqual(actions[0], "created")

    # ---- failure paths -----------------------------------------------------
    def test_missing_config_returns_3_before_doing_anything(self):
        rc, m = self._install(_env("ubuntu24.04", config=False))
        self.assertEqual(rc, 3)
        self.assertEqual(m["exec"].calls, [])
        m["download"].assert_not_called()
        m["log_error"].assert_called_once()

    def test_each_failing_command_returns_3_and_stops(self):
        for distribution in ("ubuntu24.04", "almalinux9.6", "azurelinux3.0"):
            total = len(self._install(_env(distribution))[1]["exec"].calls)
            for index in range(total):
                with self.subTest(distribution=distribution, fail_at=index):
                    self.conf.write_text(_CONF_BEFORE, encoding="utf-8")
                    self.service.write_text(_SERVICE_BEFORE, encoding="utf-8")
                    fake = _FakeExec(fail_at=index)
                    rc, m = self._install(_env(distribution), fake=fake)
                    self.assertEqual(rc, 3)
                    self.assertEqual(len(fake.calls), index + 1)
                    m["sleep"].assert_not_called()
                    m["log_error"].assert_called()

    def test_download_failure_returns_3(self):
        rc, m = self._install(_env("ubuntu24.04"),
                              download_error=ChecksumError("bad sha"))
        self.assertEqual(rc, 3)
        self.assertEqual(m["exec"].calls, [])
        self.assertIn("bad sha", m["log_error"].call_args[0][1])

    def test_version_command_failure_or_bad_output_returns_3(self):
        for capture in ((1, ""), (0, "WALinuxAgent\n")):
            with self.subTest(capture=capture):
                rc, m = self._install(_env("ubuntu24.04"), capture=capture)
                self.assertEqual(rc, 3)
                m["record"].assert_not_called()
                self.assertNotIn(["systemctl", "daemon-reload"], m["exec"].commands)

    def test_unwritable_conf_returns_3(self):
        self.conf = self.root / "missing-dir" / "waagent.conf"
        rc, m = self._install(_env("ubuntu24.04"))
        self.assertEqual(rc, 3)
        self.assertIn("Extensions.GoalStatePeriod", m["log_error"].call_args[0][1])
        m["run_capture"].assert_not_called()

    def test_unwritable_versions_file_returns_3(self):
        rc, m = self._install(_env("ubuntu24.04"),
                              write_error=PermissionError("/opt/azurehpc: denied"))
        self.assertEqual(rc, 3)
        self.assertIn("denied", m["log_error"].call_args[0][1])
        m["sleep"].assert_not_called()

    def test_missing_service_file_on_el9_returns_3(self):
        self.service.unlink()
        rc, m = self._install(_env("almalinux9.6"))
        self.assertEqual(rc, 3)
        self.assertNotIn(["systemctl", "daemon-reexec"], m["exec"].commands)

    # ---- logging -----------------------------------------------------------
    def test_key_log_lines(self):
        rc, m = self._install(_env("rocky9.7"))
        self.assertEqual(rc, 0)
        calls = m["log_info"].call_args_list
        by_message = {c.args[1]: c.kwargs for c in calls}

        start = by_message[f"Installing WALinuxAgent {_VERSION} from source"]
        self.assertEqual((start["version"], start["source"], start["handler"]),
                         (_VERSION, _URL, "el9"))
        ipoib = by_message["Target uses IPoIB: OS.EnableRDMA=y will be set"]
        self.assertEqual((ipoib["network_mode"], ipoib["uses_ipoib"]), ("standard_ib", True))
        conf = [c.kwargs for c in calls if c.kwargs.get("key") == "OS.EnableRDMA"]
        self.assertEqual(conf[0]["action"], "replaced")
        self.assertIn(f"{self.service}.bak.{_TIMESTAMP}",
                      [c.kwargs.get("backup") for c in calls])
        self.assertIn({"component": "WAAGENT", "version": "2.15.0.1"},
                      [c.kwargs for c in calls])
        summary = calls[-1]
        self.assertEqual(summary.args[1], "WALinuxAgent 2.15.0.1 installed and configured")
        self.assertEqual(summary.kwargs["waagent_extensions"], "2.15.1.3")

    def test_no_ipoib_decision_is_logged(self):
        rc, m = self._install(_env("ubuntu24.04", sku="GB200", node_type="baremetal_1p"))
        self.assertEqual(rc, 0)
        decision = [c.kwargs for c in m["log_info"].call_args_list
                    if "uses_ipoib" in c.kwargs]
        self.assertEqual(decision, [{"sku": "GB200", "node_type": "baremetal_1p",
                                     "network_mode": "mrc", "uses_ipoib": False}])

    def test_unchanged_service_file_is_a_warning(self):
        self.service.write_text(_SERVICE_AFTER, encoding="utf-8")
        rc, m = self._install(_env("almalinux9.6"))
        self.assertEqual(rc, 0)
        m["log_warn"].assert_called_once()


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "install-waagent"]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, install_waagent.install)
        self.assertEqual(steps[0].script, "install_waagent.sh")


if __name__ == "__main__":
    unittest.main()
