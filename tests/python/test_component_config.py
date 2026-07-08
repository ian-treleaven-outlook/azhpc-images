"""Differential tests for the Python get_component_config.

The Python port must agree with the real Bash get_component_config for every
component in versions.json. Rather than hard-coding expected values, each run
regenerates the baseline by invoking the actual Bash function over the *current*
versions.json — so these tests automatically track versions.json as it evolves,
and fail if the Python and Bash implementations ever diverge.

Requires bash + jq (used throughout the repo); skipped if either is missing.

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from utils.component_config import get_component_config

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSIONS_JSON = REPO_ROOT / "versions.json"
UTILITIES_SH = REPO_ROOT / "utils" / "utilities.sh"

# The lookup scenario under test: Ubuntu 24.04 + A100 + x64 (the design slice).
SCENARIO = {
    "DISTRIBUTION": "ubuntu24.04",
    "ARCHITECTURE": "x86_64",
    "GPU": "nvidia_a100",
    "SKU": "A100",
    "NODE_TYPE": "azure-vm",
}


def _tools_available() -> bool:
    return bool(shutil.which("bash") and shutil.which("jq"))


def _bash_get_component_config(component: str, versions_text: str):
    """Invoke the real Bash get_component_config for one component."""
    env = os.environ.copy()
    env.update(SCENARIO)
    env["COMPONENT_VERSIONS"] = versions_text
    script = f'source "{UTILITIES_SH}"; get_component_config "{component}"'
    proc = subprocess.run(["bash", "-c", script], env=env,
                          capture_output=True, text=True, check=True)
    out = proc.stdout.strip()
    if out in ("", "null"):
        return None
    return json.loads(out)


@unittest.skipUnless(_tools_available(), "bash and jq are required for this test")
class ComponentConfigParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.versions_text = VERSIONS_JSON.read_text(encoding="utf-8")
        cls.versions = json.loads(cls.versions_text)

    def _python_config(self, component):
        return get_component_config(
            component, self.versions,
            distribution=SCENARIO["DISTRIBUTION"],
            architecture=SCENARIO["ARCHITECTURE"],
            gpu=SCENARIO["GPU"], sku=SCENARIO["SKU"],
            node_type=SCENARIO["NODE_TYPE"],
        )

    def test_python_matches_bash_for_every_component(self):
        mismatches = []
        for component in self.versions:
            expected = _bash_get_component_config(component, self.versions_text)
            actual = self._python_config(component)
            if expected != actual:
                mismatches.append((component, expected, actual))

        if mismatches:
            report = "\n".join(
                f"  {c}:\n    bash   = {e}\n    python = {a}"
                for c, e, a in mismatches
            )
            self.fail(f"Python and Bash disagree for {len(mismatches)} component(s):\n{report}")

    def test_at_least_one_component_resolved(self):
        # Sanity check: the scenario should resolve real config for common cases.
        cfg = self._python_config("mpifileutils")
        self.assertIsInstance(cfg, dict)
        self.assertIn("version", cfg)


class NormalizeKeyTests(unittest.TestCase):
    def test_lowercases_and_collapses_separators(self):
        from utils.component_config import normalize_key
        self.assertEqual(normalize_key("NVIDIA_A100"), "nvidia_a100")
        self.assertEqual(normalize_key("azure-vm"), "azure_vm")
        self.assertEqual(normalize_key("GB200--x"), "gb200_x")


if __name__ == "__main__":
    unittest.main()
