"""Tests for packer/scripts/run_python_cli.sh (USE_PYTHON_CLI builds).

The script picks a Python 3.10+ for azhpc.py: python3 if it is new enough,
otherwise the newer Python the distro's own install_utils.sh installs (so the
image gets the same packages as a bash build), installing it with dnf first
if needed. These tests run the real script with bash, against a fake
os-release and fake python3 / python3.X / dnf commands on PATH, so nothing is
installed. They skip themselves where bash is missing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "packer" / "scripts" / "run_python_cli.sh"
_ARGS = ["--vendor", "NVidia", "--gpu", "A100", "--os", "Rocky9"]

# os-release ID + VERSION_ID for each RPM distro folder.
_RPM_FOLDERS = {
    "almalinux8.10": ("almalinux", "8.10"), "almalinux9.x": ("almalinux", "9.6"),
    "rocky8.10": ("rocky", "8.10"), "rocky9.x": ("rocky", "9.6"),
    "rhel8.10": ("rhel", "8.10"), "rhel9.x": ("rhel", "9.6"),
}


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class RunPythonCliTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.bin = self.dir / "bin"
        self.bin.mkdir()
        self.log = self.dir / "calls.log"

    def _fake(self, name, body):
        path = self.bin / name
        path.write_text(f"#!/bin/bash\n{body}\n")
        path.chmod(0o755)

    def _fake_python(self, name, new_enough):
        # `-c <version check>` exits 0 for a new-enough Python; anything else is
        # azhpc.py (or --version) and is logged.
        self._fake(name, f'if [[ $1 == -c ]]; then exit {0 if new_enough else 1}; fi\n'
                         f'echo "{name} $*" >> "{self.log}"')

    def _run(self, os_id, version_id):
        os_release = self.dir / "os-release"
        os_release.write_text(f'NAME="Test"\nID="{os_id}"\nVERSION_ID="{version_id}"\n')
        env = {"PATH": f"{self.bin}:/usr/bin:/bin", "OS_RELEASE": str(os_release)}
        result = subprocess.run(["bash", str(_SCRIPT), *_ARGS], cwd=self.dir, env=env,
                                capture_output=True, text=True)
        calls = self.log.read_text().splitlines() if self.log.exists() else []
        return result, calls

    def test_a_new_enough_python3_is_used_as_is(self):
        self._fake_python("python3", new_enough=True)
        result, calls = self._run("ubuntu", "24.04")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ["python3 --version", "python3 azhpc.py " + " ".join(_ARGS)])

    def test_an_old_python3_uses_the_installed_newer_python(self):
        self._fake_python("python3", new_enough=False)
        self._fake_python("python3.12", new_enough=True)
        self._fake("dnf", f'echo "dnf $*" >> "{self.log}"')
        result, calls = self._run("almalinux", "9.6")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ["python3.12 --version",
                                 "python3.12 azhpc.py " + " ".join(_ARGS)])

    def test_a_missing_newer_python_is_installed_with_dnf_first(self):
        self._fake_python("python3", new_enough=False)
        # dnf "installs" python3.11 by creating the fake.
        self._fake("dnf", f'echo "dnf $*" >> "{self.log}"\n'
                          f'printf \'#!/bin/bash\\necho "python3.11 $*" >> "{self.log}"\\n\' '
                          f'> "{self.bin}/python3.11"; chmod +x "{self.bin}/python3.11"')
        result, calls = self._run("rocky", "8.10")
        self.assertEqual(result.returncode, 0, result.stderr)
        # (the first call is the "python3 is ..." message)
        self.assertEqual(calls, ["python3 --version", "dnf install -y python3.11",
                                 "python3.11 --version",
                                 "python3.11 azhpc.py " + " ".join(_ARGS)])

    def test_an_old_python3_on_an_unknown_distro_fails_clearly(self):
        self._fake_python("python3", new_enough=False)
        result, calls = self._run("debian", "11")
        self.assertEqual(result.returncode, 1)
        self.assertIn("needs Python 3.10 or newer", result.stderr)
        self.assertEqual(calls, ["python3 --version"])      # only the error message's

    def test_each_rpm_distro_gets_the_python_its_install_utils_installs(self):
        # The image must not gain a package a bash build wouldn't have.
        for folder, (os_id, version_id) in _RPM_FOLDERS.items():
            with self.subTest(folder=folder):
                for name in os.listdir(self.bin):
                    os.remove(self.bin / name)
                self._fake_python("python3", new_enough=False)
                self._fake("dnf", "exit 1")         # never needed: the fakes exist
                for candidate in ("python3.11", "python3.12"):
                    self._fake_python(candidate, new_enough=True)
                self.log.unlink(missing_ok=True)
                result, calls = self._run(os_id, version_id)
                self.assertEqual(result.returncode, 0, result.stderr)
                chosen = calls[0].split()[0]
                install_utils = (_REPO / "distros" / folder / "install_utils.sh").read_text(
                    encoding="utf-8")
                self.assertRegex(install_utils, rf"(?m)^[^#]*\b{re.escape(chosen)}\b")


if __name__ == "__main__":
    unittest.main()
