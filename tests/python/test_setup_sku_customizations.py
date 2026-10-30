"""Unit tests for components/python/setup_sku_customizations.py.

Three kinds of test:

  * The generated files are compared with fixtures in
    tests/python/fixtures/setup_sku_customizations/{nvidia,amd}/. Those
    fixtures were written by running the real bash heredocs in bash, once per
    vendor, so they are byte for byte what the bash step writes.
  * The pure sed-like helpers are tested with inputs whose expected results
    were produced by the real `sed` commands from the bash script.
  * install() runs with utils.process.exec_program (under systemctl()) and
    utils.logger._log faked and every path pointed at a temporary folder, so
    the tests never touch /opt, /usr, /etc, /lib or /boot.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

from components.python import setup_sku_customizations as sku
from utils.build_config import BuildConfig, build_plan

_FIXTURES = Path(__file__).parent / "fixtures" / "setup_sku_customizations"
_OP = "setup-sku-customizations"


def _fixture(vendor: str, name: str) -> str:
    # newline="" keeps the text exactly as stored (no newline translation).
    with open(_FIXTURES / vendor / name, encoding="utf-8", newline="") as handle:
        return handle.read()


def _read(path: str) -> str:
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


def _mode(path: str) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


_Record = namedtuple("_Record", "level message fields")


class _Logs:
    """Stands in for utils.logger._log, so it sees every module's log lines."""

    def __init__(self):
        self.records = []

    def __call__(self, level, op, message, fields=None, detail=None):
        assert op == _OP
        self.records.append(_Record(level, message, fields or {}))

    def at(self, level):
        return [record for record in self.records if record.level == level]


# --------------------------------------------------------------------------- #
# Generated text, compared with what real bash writes.
# --------------------------------------------------------------------------- #
class GeneratedTextTests(unittest.TestCase):
    def test_scripts_match_bash_output_for_both_vendors(self):
        for gpu, vendor in (("NVIDIA", "nvidia"), ("AMD", "amd")):
            with self.subTest(gpu=gpu):
                setup, remove = sku._SCRIPTS[gpu]
                self.assertEqual(setup, _fixture(vendor, "setup_sku_customizations.sh"))
                self.assertEqual(remove, _fixture(vendor, "remove_sku_customizations.sh"))

    def test_service_unit_matches_bash_output_for_both_vendors(self):
        self.assertEqual(sku.service_unit("NVIDIA"),
                         _fixture("nvidia", "sku-customizations.service"))
        self.assertEqual(sku.service_unit("AMD"),
                         _fixture("amd", "sku-customizations.service"))

    def test_peermem_dropin_matches_bash_output(self):
        self.assertEqual(sku._PEERMEM_DROPIN, _fixture("nvidia", "10-nvidia-peermem.conf"))


# --------------------------------------------------------------------------- #
# Pure sed-like helpers. Expected values come from running the real sed.
# --------------------------------------------------------------------------- #
class SedHelperTests(unittest.TestCase):
    def test_fabric_manager_after_lines_are_deleted_like_sed(self):
        unit = ("[Unit]\nDescription=FM\nAfter=network-online.target\n"
                "After=multi-user.target\n"
                "After=multi-user.target.wants\n"      # also starts with the pattern
                " After=multi-user.target\n"           # not at line start: kept
                "AfterXmulti-user-target\n"            # "=" must match: kept
                "[Service]\nAfter=multi-user.target")  # no final newline
        text, removed = sku.delete_matching_lines(unit, sku._FM_AFTER_PATTERN)

        self.assertEqual(text, "[Unit]\nDescription=FM\nAfter=network-online.target\n"
                               " After=multi-user.target\nAfterXmulti-user-target\n"
                               "[Service]\n")
        self.assertEqual(removed, 3)

    def test_gb200_options_added_after_first_lockdown_on_each_line(self):
        opts = sku._GB200_CMDLINE
        grub = ('GRUB_CMDLINE_LINUX="console=ttyAMA0 lockdown=integrity rd.x lockdown=integrity"\n'
                "other line\n"
                "\tlinux /vmlinuz root=/dev/sda lockdown=integrity\n"
                "no newline lockdown=integrity")
        text, changed = sku.replace_first_per_line(grub, sku._LOCKDOWN, opts)

        self.assertEqual(text,
                         f'GRUB_CMDLINE_LINUX="console=ttyAMA0 {opts} rd.x lockdown=integrity"\n'
                         "other line\n"
                         f"\tlinux /vmlinuz root=/dev/sda {opts}\n"
                         f"no newline {opts}")
        self.assertEqual(changed, 3)
        self.assertTrue(opts.startswith("lockdown=integrity efi_mmap_nr_slack_slots=256 "))
        self.assertTrue(opts.endswith(" nvme_core.io_timeout=240 panic=-1"))


# --------------------------------------------------------------------------- #
# install(): systemctl faked, file system in a temporary folder.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = self.root = self._tmp.name
        # The repo side: customization scripts and topology files to copy.
        self.top = os.path.join(root, "repo")
        self.components = os.path.join(self.top, "components")
        self._make_files(os.path.join(self.components, "customizations"),
                         {"ndv5.sh": "#!/bin/bash\necho ndv5\n", "setup_nvme.sh": "nvme\n",
                          ".hidden": "skipped by the glob\n"}, mode=0o755)
        self._make_files(os.path.join(self.top, "topology"),
                         {"ndv5-topo.xml": "<system/>\n", "ncv4-graph.xml": "<graphs/>\n"},
                         mode=0o644)
        # The image side.
        self.paths = {
            "_CUSTOMIZATIONS_DIR": os.path.join(root, "opt", "azurehpc", "customizations"),
            "_TOPOLOGY_DIR": os.path.join(root, "opt", "microsoft"),
            "_SETUP_SCRIPT": os.path.join(root, "usr", "sbin", "setup_sku_customizations.sh"),
            "_REMOVE_SCRIPT": os.path.join(root, "usr", "sbin", "remove_sku_customizations.sh"),
            "_SERVICE_FILE": os.path.join(root, "etc", "systemd", "system",
                                          "sku-customizations.service"),
            "_OPENIBD_DROPIN_DIR": os.path.join(root, "etc", "systemd", "system",
                                                "openibd.service.d"),
            "_FABRICMANAGER_UNIT": os.path.join(root, "lib", "systemd", "system",
                                                "nvidia-fabricmanager.service"),
            "_GRUB_DEFAULT": os.path.join(root, "etc", "default", "grub"),
            "_GRUB_CFG": os.path.join(root, "boot", "grub2", "grub.cfg"),
        }
        for folder in ("usr/sbin", "etc/systemd/system", "lib/systemd/system",
                       "etc/default", "boot/grub2"):
            os.makedirs(os.path.join(root, folder))
        self.dropin = os.path.join(self.paths["_OPENIBD_DROPIN_DIR"], "10-nvidia-peermem.conf")
        self.env = {"GPU": "NVIDIA", "SKU": "H100", "DISTRIBUTION": "ubuntu24.04",
                    "COMPONENT_DIR": self.components, "TOP_DIR": self.top,
                    "TARGET_NODE_TYPE": "azure_vm_regular"}

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _make_files(folder, files, mode):
        os.makedirs(folder, exist_ok=True)
        for name, text in files.items():
            path = os.path.join(folder, name)
            Path(path).write_text(text, encoding="utf-8")
            os.chmod(path, mode)

    def _install(self, env=None, fail_at=None, paths=None):
        """Run install() with systemctl faked and paths redirected.

        Returns (rc, commands, logs): the systemctl commands that were run, and
        every log record. fail_at=N makes the Nth command (counting from 0)
        return exit code 1.
        """
        commands = []

        def run(command, op, *, cwd=None, env=None):
            commands.append(command)
            return 1 if len(commands) - 1 == fail_at else 0

        logs = _Logs()
        with mock.patch.multiple(sku, **dict(self.paths, **(paths or {}))), \
             mock.patch("utils.process.exec_program", side_effect=run), \
             mock.patch("utils.logger._log", logs):
            rc = sku.install(env or self.env)
        return rc, commands, logs

    def _assert_copied(self):
        customizations = self.paths["_CUSTOMIZATIONS_DIR"]
        self.assertEqual(sorted(os.listdir(customizations)), ["ndv5.sh", "setup_nvme.sh"])
        self.assertEqual(_read(os.path.join(customizations, "ndv5.sh")),
                         "#!/bin/bash\necho ndv5\n")
        self.assertEqual(_mode(os.path.join(customizations, "ndv5.sh")), 0o755)
        self.assertEqual(sorted(os.listdir(self.paths["_TOPOLOGY_DIR"])),
                         ["ncv4-graph.xml", "ndv5-topo.xml"])

    def _assert_generated(self, vendor):
        setup, remove = self.paths["_SETUP_SCRIPT"], self.paths["_REMOVE_SCRIPT"]
        self.assertEqual(_read(setup), _fixture(vendor, "setup_sku_customizations.sh"))
        self.assertEqual(_read(remove), _fixture(vendor, "remove_sku_customizations.sh"))
        self.assertEqual(_mode(setup), 0o755)
        self.assertEqual(_mode(remove), 0o755)
        self.assertEqual(_read(self.paths["_SERVICE_FILE"]),
                         _fixture(vendor, "sku-customizations.service"))

    # ---- success, per vendor ----------------------------------------------
    def test_nvidia_writes_scripts_service_and_peermem_dropin(self):
        rc, commands, logs = self._install()

        self.assertEqual(rc, 0)
        self._assert_copied()
        self._assert_generated("nvidia")
        self.assertEqual(_read(self.dropin), _fixture("nvidia", "10-nvidia-peermem.conf"))
        self.assertFalse(os.path.exists(self.paths["_FABRICMANAGER_UNIT"]))
        # Enabled, never started.
        self.assertEqual(commands, [["systemctl", "enable", "sku-customizations"]])
        self.assertEqual(logs.at("error"), [])

    def test_amd_writes_amd_scripts_and_no_nvidia_extras(self):
        Path(self.paths["_FABRICMANAGER_UNIT"]).write_text("After=multi-user.target\n")
        rc, commands, _ = self._install(env=dict(self.env, GPU="AMD", SKU="MI300X"))

        self.assertEqual(rc, 0)
        self._assert_copied()
        self._assert_generated("amd")
        self.assertFalse(os.path.exists(self.paths["_OPENIBD_DROPIN_DIR"]))
        # The fabric manager fix is NVIDIA-only.
        self.assertEqual(_read(self.paths["_FABRICMANAGER_UNIT"]), "After=multi-user.target\n")
        self.assertEqual(commands, [["systemctl", "enable", "sku-customizations"]])

    def test_nvidia_removes_fabric_manager_after_multi_user_lines(self):
        Path(self.paths["_FABRICMANAGER_UNIT"]).write_text(
            "[Unit]\nAfter=network-online.target\nAfter=multi-user.target\n[Service]\n")
        rc, _, logs = self._install()

        self.assertEqual(rc, 0)
        self.assertEqual(_read(self.paths["_FABRICMANAGER_UNIT"]),
                         "[Unit]\nAfter=network-online.target\n[Service]\n")
        edits = [r for r in logs.at("info")
                 if r.fields.get("path") == self.paths["_FABRICMANAGER_UNIT"]]
        self.assertEqual(edits[0].fields["lines_removed"], 1)

    def test_existing_generated_files_are_overwritten(self):
        for key in ("_SETUP_SCRIPT", "_REMOVE_SCRIPT", "_SERVICE_FILE"):
            Path(self.paths[key]).write_text("old\n" * 500)
        rc, _, _ = self._install()

        self.assertEqual(rc, 0)
        self._assert_generated("nvidia")

    # ---- GB200 on Azure Linux 3.0 -----------------------------------------
    def _write_grub(self):
        Path(self.paths["_GRUB_DEFAULT"]).write_text(
            'GRUB_CMDLINE_LINUX="console=ttyAMA0 lockdown=integrity"\n')
        Path(self.paths["_GRUB_CFG"]).write_text(
            "menuentry 'x' {\n\tlinux /vmlinuz lockdown=integrity ro\n}\n")

    def test_gb200_on_azure_linux_edits_grub_and_disables_irqbalance(self):
        self._write_grub()
        env = dict(self.env, SKU="GB200", DISTRIBUTION="azurelinux3.0")
        rc, commands, logs = self._install(env=env)

        self.assertEqual(rc, 0)
        opts = sku._GB200_CMDLINE
        self.assertEqual(_read(self.paths["_GRUB_DEFAULT"]),
                         f'GRUB_CMDLINE_LINUX="console=ttyAMA0 {opts}"\n')
        self.assertEqual(_read(self.paths["_GRUB_CFG"]),
                         f"menuentry 'x' {{\n\tlinux /vmlinuz {opts} ro\n}}\n")
        self.assertEqual(commands, [["systemctl", "stop", "irqbalance"],
                                    ["systemctl", "disable", "irqbalance"],
                                    ["systemctl", "enable", "sku-customizations"]])
        messages = [r.message for r in logs.at("info")]
        self.assertIn("Stopped and disabled irqbalance (GB200)", messages)
        self.assertEqual(logs.at("warn"), [])

    def test_gb200_changes_need_both_azure_linux_and_gb200(self):
        self._write_grub()
        for distro, sku_name in (("ubuntu24.04", "GB200"), ("azurelinux3.0", "H100"),
                                 ("azurelinux3.0", "VR200")):
            with self.subTest(distro=distro, sku=sku_name):
                env = dict(self.env, SKU=sku_name, DISTRIBUTION=distro)
                rc, commands, _ = self._install(env=env)
                self.assertEqual(rc, 0)
                self.assertNotIn("efi_mmap", _read(self.paths["_GRUB_DEFAULT"]))
                self.assertEqual(commands, [["systemctl", "enable", "sku-customizations"]])

    def test_gb200_without_lockdown_option_warns_and_continues(self):
        Path(self.paths["_GRUB_DEFAULT"]).write_text("GRUB_TIMEOUT=0\n")
        Path(self.paths["_GRUB_CFG"]).write_text("lockdown=integrity\n")
        env = dict(self.env, SKU="GB200", DISTRIBUTION="azurelinux3.0")
        rc, _, logs = self._install(env=env)

        self.assertEqual(rc, 0)
        self.assertEqual(_read(self.paths["_GRUB_DEFAULT"]), "GRUB_TIMEOUT=0\n")
        self.assertEqual(len(logs.at("warn")), 1)

    def test_gb200_rerun_warns_that_options_are_already_there(self):
        self._write_grub()
        env = dict(self.env, SKU="GB200", DISTRIBUTION="azurelinux3.0")
        self.assertEqual(self._install(env=env)[0], 0)
        rc, _, logs = self._install(env=env)

        self.assertEqual(rc, 0)
        # Added again, like bash's sed.
        self.assertEqual(_read(self.paths["_GRUB_DEFAULT"]).count("efi_mmap"), 2)
        warned = [r.fields["path"] for r in logs.at("warn")]
        self.assertEqual(warned, [self.paths["_GRUB_DEFAULT"], self.paths["_GRUB_CFG"]])

    def test_gb200_missing_grub_file_returns_3_before_systemctl(self):
        # sed -i fails on a missing file, which stops bash.
        env = dict(self.env, SKU="GB200", DISTRIBUTION="azurelinux3.0")
        rc, commands, logs = self._install(env=env)

        self.assertEqual(rc, 3)
        self.assertEqual(commands, [])
        self.assertIn("could not edit", logs.at("error")[-1].message)

    # ---- logging -----------------------------------------------------------
    def test_logs_start_decision_files_enable_and_summary(self):
        _, _, logs = self._install()
        info = logs.at("info")

        start = info[0]
        self.assertEqual(start.message, "Setting up SKU customizations")
        self.assertEqual(start.fields["gpu"], "NVIDIA")
        self.assertEqual(start.fields["sku"], "H100")
        messages = [r.message for r in info]
        self.assertIn("Using the NVIDIA setup/remove scripts", messages)
        copied = [r for r in info if r.message.startswith("Copied ")]
        self.assertEqual([r.fields["count"] for r in copied], [2, 2])
        wrote = [r for r in info if r.fields.get("path") == self.paths["_SETUP_SCRIPT"]]
        self.assertEqual(wrote[0].fields["action"], "created")
        self.assertIn("Enabled sku-customizations (not started during the build; "
                      "it runs at boot)", messages)
        summary = info[-1]
        self.assertEqual(summary.message, "SKU customizations set up")
        self.assertFalse(summary.fields["gb200_changes"])

    # ---- failures ----------------------------------------------------------
    def test_unknown_gpu_returns_3_before_changing_anything(self):
        for env in (dict(self.env, GPU="INTEL"),
                    {k: v for k, v in self.env.items() if k != "GPU"}):
            with self.subTest(gpu=env.get("GPU")):
                rc, commands, logs = self._install(env=env)
                self.assertEqual(rc, 3)
                self.assertEqual(commands, [])
                self.assertFalse(os.path.exists(self.paths["_CUSTOMIZATIONS_DIR"]))
                self.assertIn("GPU is not NVIDIA or AMD", logs.at("error")[-1].message)

    def test_missing_repo_dirs_returns_3(self):
        for key in ("COMPONENT_DIR", "TOP_DIR"):
            with self.subTest(missing=key):
                env = {k: v for k, v in self.env.items() if k != key}
                rc, commands, _ = self._install(env=env)
                self.assertEqual(rc, 3)
                self.assertEqual(commands, [])

    def test_folder_in_customizations_fails_like_cp_after_copying_files(self):
        os.makedirs(os.path.join(self.components, "customizations", "subdir"))
        rc, commands, logs = self._install()

        self.assertEqual(rc, 3)
        # cp copies the files it can, then fails; nothing after it runs.
        self.assertEqual(sorted(os.listdir(self.paths["_CUSTOMIZATIONS_DIR"])),
                         ["ndv5.sh", "setup_nvme.sh"])
        self.assertFalse(os.path.exists(self.paths["_TOPOLOGY_DIR"]))
        self.assertEqual(commands, [])
        self.assertIn("contains folders", logs.at("error")[-1].message)

    def test_empty_topology_folder_returns_3(self):
        for name in os.listdir(os.path.join(self.top, "topology")):
            os.remove(os.path.join(self.top, "topology", name))
        rc, _, logs = self._install()

        self.assertEqual(rc, 3)
        self.assertFalse(os.path.exists(self.paths["_SETUP_SCRIPT"]))
        self.assertIn("no files to copy", logs.at("error")[-1].message)

    def test_unwritable_setup_script_returns_3(self):
        a_folder = os.path.join(self.root, "a-folder")    # can't be opened for writing
        os.makedirs(a_folder)
        rc, commands, logs = self._install(paths={"_SETUP_SCRIPT": a_folder})

        self.assertEqual(rc, 3)
        self.assertFalse(os.path.exists(self.paths["_SERVICE_FILE"]))
        self.assertEqual(commands, [])
        self.assertIn("could not write", logs.at("error")[-1].message)

    def test_each_failing_command_returns_3_and_stops(self):
        expected = [["systemctl", "stop", "irqbalance"],
                    ["systemctl", "disable", "irqbalance"],
                    ["systemctl", "enable", "sku-customizations"]]
        env = dict(self.env, SKU="GB200", DISTRIBUTION="azurelinux3.0")
        for index in range(len(expected)):
            with self.subTest(failing=expected[index]):
                self._write_grub()
                rc, commands, logs = self._install(env=env, fail_at=index)
                self.assertEqual(rc, 3)
                self.assertEqual(commands, expected[:index + 1])
                self.assertNotEqual(logs.at("info")[-1].message, "SKU customizations set up")


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == _OP]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, sku.install)
        self.assertEqual(steps[0].script, "setup_sku_customizations.sh")


if __name__ == "__main__":
    unittest.main()
