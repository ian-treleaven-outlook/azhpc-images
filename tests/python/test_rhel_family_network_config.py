"""Unit tests for components/python/rhel_family_network_config.py.

  * The six AlmaLinux/Rocky/RHEL network-config.sh copies are checked to be
    identical, and the port's paths, drop-in text and sed edit against them
    (the sed one against real GNU sed, when it is there).
  * install() writes to a temporary directory, so /etc and /lib are never
    touched.

Run from the repo root:
    python3 -m unittest tests.python.test_rhel_family_network_config -v
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import rhel_family_network_config as m
from utils.plans import steps

_OP = "network-config"
_REPO = Path(__file__).resolve().parents[2]
_FOLDERS = ("almalinux8.10", "almalinux9.x", "rocky8.10", "rocky9.x", "rhel8.10", "rhel9.x")
_BASH_SCRIPT = _REPO / "distros" / "rocky9.x" / "network-config.sh"
_SED_EXPRESSION = "/\\[main\\]/a no-auto-default=*"
_CONF = "[main]\nplugins = ifcfg-rh\n\n[logging]\n"


def _bash_text() -> str:
    return _BASH_SCRIPT.read_text(encoding="utf-8")


class BashParityTests(unittest.TestCase):
    def test_all_six_scripts_are_the_same(self):
        for folder in _FOLDERS:
            with self.subTest(folder=folder):
                text = (_REPO / "distros" / folder / "network-config.sh").read_text(
                    encoding="utf-8")
                self.assertEqual(text, _bash_text())

    def test_sed_command_and_conf_path_are_the_bash_ones(self):
        self.assertIn(f"sed -i '{_SED_EXPRESSION}' {m._NETWORKMANAGER_CONF}\n", _bash_text())

    def test_drop_in_is_the_bash_heredoc(self):
        match = re.search(r"^cat <<EOF > (\S+)\n(.*?)^EOF$", _bash_text(), re.S | re.M)
        self.assertEqual(match.group(1), m._CLOUD_INIT_DROP_IN)
        self.assertEqual(match.group(2), m._CLOUD_INIT_SETTINGS)
        drop_in_dir = m._CLOUD_INIT_DROP_IN.rsplit("/", 1)[0]
        self.assertIn(f"mkdir -p {drop_in_dir}/\n", _bash_text())
        # The heredoc's EOF is unquoted: nothing in it may be expanded by bash.
        self.assertNotRegex(match.group(2), r"[$`\\]")

    @unittest.skipUnless(shutil.which("sed") and os.name == "posix", "needs GNU sed")
    def test_add_no_auto_default_matches_the_real_sed(self):
        texts = [_CONF, "[main]", "a\n[main]\r\nb\n", "# see [main]\n[main]\nx=1\n",
                 "[main]\nno-auto-default=*\n", "[logging]\n", ""]
        with tempfile.TemporaryDirectory() as tmp:
            conf = Path(tmp) / "NetworkManager.conf"
            for text in texts:
                with self.subTest(text=text):
                    conf.write_bytes(text.encode())
                    subprocess.run(["sed", "-i", _SED_EXPRESSION, str(conf)], check=True)
                    self.assertEqual(conf.read_bytes().decode(), m.add_no_auto_default(text))


class AddNoAutoDefaultTests(unittest.TestCase):
    def test_adds_the_line_after_main(self):
        self.assertEqual(m.add_no_auto_default(_CONF),
                         "[main]\nno-auto-default=*\nplugins = ifcfg-rh\n\n[logging]\n")

    def test_last_line_without_newline_gets_one_first(self):
        self.assertEqual(m.add_no_auto_default("[main]"), "[main]\nno-auto-default=*\n")

    def test_every_line_containing_main_matches_like_sed(self):
        self.assertEqual(m.add_no_auto_default("# [main]\n[main]\n"),
                         "# [main]\nno-auto-default=*\n[main]\nno-auto-default=*\n")

    def test_no_main_line_changes_nothing(self):
        self.assertEqual(m.add_no_auto_default("[logging]\n"), "[logging]\n")


class InstallTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.conf = Path(tmp.name) / "etc" / "NetworkManager" / "NetworkManager.conf"
        self.conf.parent.mkdir(parents=True)
        self.drop_in = (Path(tmp.name) / "lib" / "cloud-init-local.service.d"
                        / "50-azure-clear-persistent-obj-pkl.conf")
        patchers = [mock.patch.object(m, "_NETWORKMANAGER_CONF", str(self.conf)),
                    mock.patch.object(m, "_CLOUD_INIT_DROP_IN", str(self.drop_in)),
                    mock.patch("utils.files.log_info"), mock.patch("utils.files.log_error")]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ("log_info", "log_warn"):
            patcher = mock.patch.object(m, name)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

    def test_happy_path(self):
        self.conf.write_text(_CONF, encoding="utf-8")
        self.assertEqual(m.install({}), 0)
        self.assertEqual(self.conf.read_text(encoding="utf-8"),
                         m.add_no_auto_default(_CONF))
        self.assertEqual(self.drop_in.read_text(encoding="utf-8"), m._CLOUD_INIT_SETTINGS)
        self.log_warn.assert_not_called()

    def test_rerun_adds_a_duplicate_line_with_a_warning(self):
        self.conf.write_text(_CONF, encoding="utf-8")
        self.assertEqual(m.install({}), 0)
        self.assertEqual(m.install({}), 0)
        self.assertEqual(self.conf.read_text(encoding="utf-8").count("no-auto-default=*\n"), 2)
        self.log_warn.assert_called_once()
        self.assertIn("already has", self.log_warn.call_args.args[1])

    def test_several_main_lines_are_warned_about(self):
        self.conf.write_text("# [main]\n" + _CONF, encoding="utf-8")
        self.assertEqual(m.install({}), 0)
        self.assertEqual(self.log_warn.call_args.kwargs["main_lines"], 2)

    def test_no_main_line_succeeds_with_a_warning(self):
        self.conf.write_text("[logging]\n", encoding="utf-8")
        self.assertEqual(m.install({}), 0)
        self.assertEqual(self.conf.read_text(encoding="utf-8"), "[logging]\n")
        self.assertIn("no [main] line", self.log_warn.call_args.args[1])

    def test_missing_networkmanager_conf_fails_before_the_drop_in(self):
        self.assertEqual(m.install({}), 3)
        self.assertFalse(self.drop_in.exists())

    def test_unwritable_drop_in_fails(self):
        self.conf.write_text(_CONF, encoding="utf-8")
        with mock.patch.object(m, "write_file", return_value=False):
            self.assertEqual(m.install({}), 3)


class StepWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_from_the_distro_folder(self):
        step = steps.rhel_family_network_config()
        self.assertEqual((step.op, step.script, step.base),
                         (_OP, "network-config.sh", "distro"))
        self.assertIs(step.action, m.install)


if __name__ == "__main__":
    unittest.main()
