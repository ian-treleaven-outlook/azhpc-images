"""Tests for the Python get_component_config.

Two layers:

  * Differential tests (ComponentConfigParityTests): the Python port must agree
    with the real Bash get_component_config for every component in
    versions.json. Rather than hard-coding expected values, each run regenerates
    the baseline by invoking the actual Bash function over the *current*
    versions.json — so these tests automatically track versions.json as it
    evolves, and fail if the Python and Bash implementations ever diverge.
    Requires bash + jq (used throughout the repo); skipped if either is missing.
  * Lookup-order tests (LookupOrderTests): a small made-up versions.json shows
    each of the five lookup steps on its own. These always run.

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

from utils.component_config import config_for, get_component_config, write_component_version

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSIONS_JSON = REPO_ROOT / "versions.json"
UTILITIES_SH = REPO_ROOT / "utils" / "utilities.sh"

# Lookup scenarios under test. The differential check runs every component in
# versions.json through every scenario, comparing Python with the real bash.
#
# Each scenario is chosen to reach a different step of the lookup (see the
# numbered steps at the top of utils/component_config.py). The comments name
# overrides that exist in versions.json today. TARGET_NODE_TYPE is the variable
# bash reads, so it is what each scenario sets.
SCENARIOS = {
    # Steps 4/5: no override applies to an A100 on a regular VM.
    "ubuntu24 / A100 / x64 / regular": {
        "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "x86_64",
        "GPU": "NVIDIA", "SKU": "A100", "TARGET_NODE_TYPE": "azure_vm_regular",
    },
    # Step 1: cuda and hpcx have azure_vm_regular.nvidia_v100 overrides.
    "ubuntu24 / V100 / x64 / regular": {
        "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "x86_64",
        "GPU": "NVIDIA", "SKU": "V100", "TARGET_NODE_TYPE": "azure_vm_regular",
    },
    # Step 1: cuda has an azure_vm_regular.nvidia_ncv6 override.
    "ubuntu24 / NCv6 / x64 / regular": {
        "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "x86_64",
        "GPU": "NVIDIA", "SKU": "NCv6", "TARGET_NODE_TYPE": "azure_vm_regular",
    },
    # Step 1 under baremetal_3p: nvidia and dcgm have nvidia_h100 overrides.
    "ubuntu24 / H100 / x64 / baremetal_3p": {
        "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "x86_64",
        "GPU": "NVIDIA", "SKU": "H100", "TARGET_NODE_TYPE": "baremetal_3p",
    },
    # Step 1 (nvidia.nvidia_gb200) and step 3 (direct baremetal_3p configs,
    # e.g. hpcx, ompi, nccl, dcgm, waagent).
    "ubuntu24 / GB200 / aarch64 / baremetal_3p": {
        "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "aarch64",
        "GPU": "NVIDIA", "SKU": "GB200", "TARGET_NODE_TYPE": "baremetal_3p",
    },
    # Step 1: nvidia.nvidia_gb300.
    "ubuntu24 / GB300 / aarch64 / baremetal_3p": {
        "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "aarch64",
        "GPU": "NVIDIA", "SKU": "GB300", "TARGET_NODE_TYPE": "baremetal_3p",
    },
    # Step 3: doca and gdrcopy have direct azure_vm_akshost configs.
    "ubuntu24 / GB200 / aarch64 / akshost": {
        "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "aarch64",
        "GPU": "NVIDIA", "SKU": "GB200", "TARGET_NODE_TYPE": "azure_vm_akshost",
    },
    # Steps 4/5 on aarch64.
    "ubuntu24 / GB200 / aarch64 / regular": {
        "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "aarch64",
        "GPU": "NVIDIA", "SKU": "GB200", "TARGET_NODE_TYPE": "azure_vm_regular",
    },
    # An RPM distro with V100 overrides (cuda, hpcx).
    "almalinux9.8 / V100 / x64 / regular": {
        "DISTRIBUTION": "almalinux9.8", "ARCHITECTURE": "x86_64",
        "GPU": "NVIDIA", "SKU": "V100", "TARGET_NODE_TYPE": "azure_vm_regular",
    },
    # AMD on Azure Linux.
    "azurelinux3 / MI300 / x64 / regular": {
        "DISTRIBUTION": "azurelinux3.0", "ARCHITECTURE": "x86_64",
        "GPU": "AMD", "SKU": "MI300", "TARGET_NODE_TYPE": "azure_vm_regular",
    },
    # No GPU or SKU: bash skips steps 1-3 entirely, even for a V100-style target.
    "ubuntu24 / no GPU or SKU / x64": {
        "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "x86_64",
        "GPU": "", "SKU": "", "TARGET_NODE_TYPE": "azure_vm_regular",
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
    env.update(scenario)      # sets GPU, SKU, TARGET_NODE_TYPE, ... for bash
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
            node_type=scenario["TARGET_NODE_TYPE"],
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

    # The differential test proves Python == bash, but it would also pass if both
    # ignored every override. These guards check that the scenarios really reach
    # overrides: each compares two targets that differ in ONE selector only, and
    # expects different results. If versions.json drops one of these overrides,
    # point the guard at another override that still exists.
    def test_sku_override_is_reached(self):
        v100 = self._python_config("cuda", SCENARIOS["ubuntu24 / V100 / x64 / regular"])
        a100 = self._python_config("cuda", SCENARIOS["ubuntu24 / A100 / x64 / regular"])
        self.assertIsInstance(v100, dict)
        self.assertNotEqual(v100, a100)

    def test_node_type_override_is_reached(self):
        baremetal = self._python_config(
            "nvidia", SCENARIOS["ubuntu24 / GB200 / aarch64 / baremetal_3p"])
        regular = self._python_config(
            "nvidia", SCENARIOS["ubuntu24 / GB200 / aarch64 / regular"])
        self.assertIsInstance(baremetal, dict)
        self.assertNotEqual(baremetal, regular)

    def test_at_least_one_component_resolved(self):
        cfg = self._python_config(
            "mpifileutils", SCENARIOS["ubuntu24 / A100 / x64 / regular"])
        self.assertIsInstance(cfg, dict)
        self.assertIn("version", cfg)


# A small, made-up versions.json with one entry per lookup step, so each step can
# be shown on its own. These need no bash or jq, and don't change when the real
# versions.json does. The version strings say which step should win.
_SYNTHETIC_VERSIONS = {
    "tool": {
        "common": {"version": "common"},                          # step 5
        "ubuntu24.04": {
            "x86_64": {
                "version": "architecture",                        # step 4
                "azure_vm_regular": {                             # an override table
                    "nvidia_v100": {"version": "regular-v100"},   # step 1
                },
                "azure_vm_akshost": {"version": "akshost"},       # step 3
                "baremetal_3p": {
                    "default": {"version": "baremetal-default"},  # step 2
                    "nvidia_h100": {"version": "baremetal-h100"}, # step 1
                },
            },
        },
    },
    # A config that is present but empty still counts as found, as in bash.
    "empty_arch": {
        "common": {"version": "common"},
        "ubuntu24.04": {"x86_64": {}},
    },
}


class LookupOrderTests(unittest.TestCase):
    def _resolve(self, sku="A100", node_type="azure_vm_regular", *,
                 component="tool", distribution="ubuntu24.04", gpu="NVIDIA"):
        return get_component_config(
            component, _SYNTHETIC_VERSIONS, distribution=distribution,
            architecture="x86_64", gpu=gpu, sku=sku, node_type=node_type)

    def _version(self, **kwargs):
        return self._resolve(**kwargs)["version"]

    def test_step1_sku_override_for_the_node_type(self):
        self.assertEqual(self._version(sku="V100"), "regular-v100")

    def test_step1_wins_over_step2(self):
        self.assertEqual(self._version(sku="H100", node_type="baremetal_3p"),
                         "baremetal-h100")

    def test_step2_node_type_default(self):
        self.assertEqual(self._version(sku="A100", node_type="baremetal_3p"),
                         "baremetal-default")

    def test_step3_node_type_direct_config(self):
        self.assertEqual(self._resolve(node_type="azure_vm_akshost"),
                         {"version": "akshost"})

    def test_override_table_is_skipped_for_other_skus(self):
        # azure_vm_regular only holds a V100 override, so an A100 must not get
        # that table back; it falls through to the architecture level (step 4).
        self.assertEqual(self._version(sku="A100"), "architecture")

    def test_step5_common_when_distribution_has_no_entry(self):
        self.assertEqual(self._version(distribution="rocky9.8"), "common")

    def test_no_gpu_skips_the_overrides(self):
        # Without a vendor, bash skips steps 1-3 even for an overridden SKU.
        self.assertEqual(self._version(sku="V100", gpu=None), "architecture")

    def test_empty_node_type_uses_the_default(self):
        self.assertEqual(self._version(sku="V100", node_type=""), "regular-v100")

    def test_empty_config_counts_as_found(self):
        self.assertEqual(self._resolve(component="empty_arch"), {})

    def test_missing_component_returns_none(self):
        self.assertIsNone(self._resolve(component="not_in_versions"))

    def test_config_for_reads_target_node_type(self):
        env = {
            "COMPONENT_VERSIONS": json.dumps(_SYNTHETIC_VERSIONS),
            "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "x86_64",
            "GPU": "NVIDIA", "SKU": "H100", "TARGET_NODE_TYPE": "baremetal_3p",
        }
        self.assertEqual(config_for("tool", env)["version"], "baremetal-h100")

    def test_config_for_defaults_to_a_regular_vm(self):
        env = {
            "COMPONENT_VERSIONS": json.dumps(_SYNTHETIC_VERSIONS),
            "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "x86_64",
            "GPU": "NVIDIA", "SKU": "V100",
        }
        self.assertEqual(config_for("tool", env)["version"], "regular-v100")

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
