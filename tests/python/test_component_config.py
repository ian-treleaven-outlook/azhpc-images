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
import tempfile
import unittest
from pathlib import Path

from utils.component_config import get_component_config, write_component_version

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSIONS_JSON = REPO_ROOT / "versions.json"
UTILITIES_SH = REPO_ROOT / "utils" / "utilities.sh"

# Lookup scenarios under test. The differential check runs for every scenario,
# so the harder paths (baremetal, aarch64, RPM distros) get exercised too — not
# just the A100/x64 case, which mostly falls straight through to `common`.
SCENARIOS = {
    "ubuntu24 / A100 / x64 / vm": {
        "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "x86_64",
        "GPU": "NVIDIA", "SKU": "A100", "NODE_TYPE": "azure-vm",
    },
    "ubuntu24 / GB200 / aarch64 / baremetal": {
        "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "aarch64",
        "GPU": "NVIDIA", "SKU": "GB200", "NODE_TYPE": "baremetal",
    },
    "ubuntu24 / GB200 / aarch64 / vm": {
        "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "aarch64",
        "GPU": "NVIDIA", "SKU": "GB200", "NODE_TYPE": "azure-vm",
    },
    "azurelinux3 / MI300 / x64 / vm": {
        "DISTRIBUTION": "azurelinux3.0", "ARCHITECTURE": "x86_64",
        "GPU": "AMD", "SKU": "MI300", "NODE_TYPE": "azure-vm",
    },
}


def _tools_available() -> bool:
    return bool(shutil.which("bash") and shutil.which("jq"))


def _bash_get_component_config(component: str, versions_text: str, scenario: dict):
    """Invoke the real Bash get_component_config for one component + scenario.

    This is the heart of the differential test: instead of hardcoding what we THINK
    bash returns, we actually source utilities.sh and run the real function, then diff
    Python against it. If either side ever drifts, the diff catches it.
    """
    env = os.environ.copy()
    env.update(scenario)
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

    def _python_config(self, component, scenario):
        return get_component_config(
            component, self.versions,
            distribution=scenario["DISTRIBUTION"],
            architecture=scenario["ARCHITECTURE"],
            gpu=scenario["GPU"], sku=scenario["SKU"],
            node_type=scenario["NODE_TYPE"],
        )

    def test_python_matches_bash_across_scenarios(self):
        mismatches = []
        for name, scenario in SCENARIOS.items():
            for component in self.versions:
                expected = _bash_get_component_config(component, self.versions_text, scenario)
                actual = self._python_config(component, scenario)
                if expected != actual:
                    mismatches.append((name, component, expected, actual))

        if mismatches:
            report = "\n".join(
                f"  [{s}] {c}:\n    bash   = {e}\n    python = {a}"
                for s, c, e, a in mismatches
            )
            self.fail(f"Python and Bash disagree in {len(mismatches)} case(s):\n{report}")

    def test_baremetal_scenario_exercises_a_distinct_path(self):
        # Confirm the baremetal scenario actually reaches a baremetal-specific
        # config (a different result than the A100/common case), so the parity
        # check above is meaningfully covering the baremetal / nested logic.
        baremetal = self._python_config(
            "cmake", SCENARIOS["ubuntu24 / GB200 / aarch64 / baremetal"])
        common = self._python_config(
            "cmake", SCENARIOS["ubuntu24 / A100 / x64 / vm"])
        self.assertIsInstance(baremetal, dict)
        self.assertNotEqual(baremetal, common)

    def test_at_least_one_component_resolved(self):
        cfg = self._python_config(
            "mpifileutils", SCENARIOS["ubuntu24 / A100 / x64 / vm"])
        self.assertIsInstance(cfg, dict)
        self.assertIn("version", cfg)


class NormalizeKeyTests(unittest.TestCase):
    def test_lowercases_and_collapses_separators(self):
        from utils.component_config import normalize_key
        self.assertEqual(normalize_key("NVIDIA_A100"), "nvidia_a100")
        self.assertEqual(normalize_key("azure-vm"), "azure_vm")
        self.assertEqual(normalize_key("GB200--x"), "gb200_x")


class WriteComponentVersionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "component_versions.txt"

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_file_with_component(self):
        write_component_version("MPIFILEUTILS", "0.12", path=self.path)
        self.assertEqual(json.loads(self.path.read_text()), {"MPIFILEUTILS": "0.12"})

    def test_appends_and_updates(self):
        write_component_version("A", "1", path=self.path)
        write_component_version("B", "2", path=self.path)
        write_component_version("A", "1.1", path=self.path)  # update existing key
        self.assertEqual(json.loads(self.path.read_text()), {"A": "1.1", "B": "2"})

    def test_recovers_from_corrupt_file(self):
        self.path.write_text("not json at all")
        write_component_version("A", "1", path=self.path)
        self.assertEqual(json.loads(self.path.read_text()), {"A": "1"})


if __name__ == "__main__":
    unittest.main()
