"""Unit tests for utils/version_report.py (Step 1: resolve all versions).

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import unittest

from utils.version_report import resolve_report, format_report, _extract_versions


class ExtractVersionsTests(unittest.TestCase):
    def test_flat_version(self):
        self.assertEqual(_extract_versions({"version": "1.2", "url": "u"}), [("", "1.2")])

    def test_nested_versions(self):
        cfg = {"driver": {"version": "13.0"},
               "samples": {"version": "13.0", "sha256": "s"}}
        self.assertEqual(sorted(_extract_versions(cfg)),
                         [("driver", "13.0"), ("samples", "13.0")])

    def test_flat_version_wins_over_nested_override(self):
        # arch-level config with a direct version plus a SKU-specific override
        cfg = {"version": "2.25", "nvidia_v100": {"version": "2.24"}}
        self.assertEqual(_extract_versions(cfg), [("", "2.25")])

    def test_no_version_returns_empty(self):
        self.assertEqual(_extract_versions({"url": "u"}), [])


class ResolveReportTests(unittest.TestCase):
    def _versions(self):
        return {
            "note": {"azurelinux3.0": {"_comment": "metadata, not a component"}},
            "cmake": {"common": {"version": "4.3.1"}},
            "cuda": {"common": {"driver": {"version": "13.0"},
                                "samples": {"version": "13.0", "sha256": "s"}}},
            "aznfs": {"ubuntu22.04": {"x86_64": {"version": "1.0"}}},  # no ubuntu24 match
        }

    def _report(self):
        return resolve_report(self._versions(),
                              distribution="ubuntu24.04", architecture="x86_64")

    def test_skips_note_metadata(self):
        self.assertNotIn("note", {r["component"] for r in self._report()})

    def test_flat_component_resolves(self):
        cmake = next(r for r in self._report() if r["component"] == "cmake")
        self.assertEqual(cmake["version"], "4.3.1")
        self.assertTrue(cmake["resolved"])

    def test_nested_component_expands_to_sublabels(self):
        labels = {r["component"] for r in self._report()}
        self.assertIn("cuda.driver", labels)
        self.assertIn("cuda.samples", labels)

    def test_unmatched_component_marked_unresolved(self):
        aznfs = next(r for r in self._report() if r["component"] == "aznfs")
        self.assertFalse(aznfs["resolved"])
        self.assertIsNone(aznfs["version"])


class FormatReportTests(unittest.TestCase):
    def test_aligns_and_labels_states(self):
        report = [
            {"component": "cmake", "version": "4.3.1", "resolved": True},
            {"component": "aznfs", "version": None, "resolved": False},
            {"component": "moneo", "version": None, "resolved": True},
        ]
        out = format_report(report)
        self.assertIn("cmake  4.3.1", out)
        self.assertIn("(no match)", out)      # unresolved
        self.assertIn("(no version)", out)    # resolved but versionless


if __name__ == "__main__":
    unittest.main()
