"""Unit tests for the full DOCA port (components/python/install_doca.py).

Two layers:
  * Real-logic tests for the pure helpers (parse_ofed_version,
    newest_kernel_repo_rpm, strip_dnf_excludes) — these run actual code.
  * Mocked orchestration tests for install() across the three distro families.
    The package operations (dpkg/rpm/apt/dnf) can't run in CI, so exec_program /
    run_capture / downloads / file writes are mocked. The Ubuntu path is the one
    additionally proven end-to-end by a real A100 build; RHEL and Azure Linux are
    mock-tested only (no RHEL/Azure Linux hardware to run them on yet).

Run from the repo root:
    python3 -m unittest discover -s tests/python -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from components.python import install_doca


# Builds the runtime env dict for a given distro. doca=False drops the doca entry so we
# can hit the "version not found" path. DISTRIBUTION is what selects the branch
# (ubuntu / azurelinux / else RHEL).
def _env(distribution="ubuntu24.04", node_type="azure-vm", doca=True):
    versions = {}
    if doca:
        versions["doca"] = {"common": {
            "version": "2.9.1",
            "url": "https://example/doca-host_2.9.1_amd64.deb",
            "sha256": "s",
        }}
    return {
        "COMPONENT_VERSIONS": json.dumps(versions),
        "DISTRIBUTION": distribution,
        "ARCHITECTURE": "x86_64",
        "GPU": "NVIDIA",
        "SKU": "A100",
        "NODE_TYPE": node_type,
    }


# --------------------------------------------------------------------------- #
# Pure helpers — real logic, no mocks.
# --------------------------------------------------------------------------- #
# Pure parsing (no mocks). parse_ofed_version mirrors the old
# `ofed_info | sed -n 1,1p | awk -F- ...` one-liner exactly — quirks and all.
class ParseOfedVersionTests(unittest.TestCase):
    def test_joins_third_and_fourth_dash_fields(self):
        # Mirrors awk -F'-' '{print $3,$4}' with OFS='-' over the first line.
        self.assertEqual(install_doca.parse_ofed_version("a-b-c-d-e\nsecond\n"), "c-d")

    def test_strips_colons(self):
        self.assertEqual(install_doca.parse_ofed_version("x-y-1.2:-3.4:"), "1.2-3.4")

    def test_empty_output_returns_none(self):
        self.assertIsNone(install_doca.parse_ofed_version(""))

    def test_too_few_fields_returns_none(self):
        self.assertIsNone(install_doca.parse_ofed_version("only-two"))


class NewestKernelRepoRpmTests(unittest.TestCase):
    def test_picks_the_most_recently_modified_rpm(self):
        with tempfile.TemporaryDirectory() as root:
            doca_dir = os.path.join(root, "DOCA.2.9")
            os.makedirs(doca_dir)
            older = os.path.join(doca_dir, "doca-kernel-repo-1.rpm")
            newer = os.path.join(doca_dir, "doca-kernel-repo-2.rpm")
            for path in (older, newer):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("rpm")
            os.utime(older, (1000, 1000))
            os.utime(newer, (2000, 2000))
            self.assertEqual(install_doca.newest_kernel_repo_rpm(root), newer)

    def test_returns_none_when_no_match(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(install_doca.newest_kernel_repo_rpm(root))


class StripDnfExcludesTests(unittest.TestCase):
    def test_removes_exclude_lines_keeps_the_rest(self):
        conf = "[main]\ngpgcheck=1\nexclude=kernel* kernel-devel*\ninstallonly=3\n"
        self.assertEqual(
            install_doca.strip_dnf_excludes(conf),
            "[main]\ngpgcheck=1\ninstallonly=3\n",
        )

    def test_no_exclude_lines_is_unchanged(self):
        conf = "[main]\ngpgcheck=1\n"
        self.assertEqual(install_doca.strip_dnf_excludes(conf), conf)


# --------------------------------------------------------------------------- #
# install() orchestration — mocked package operations.
# --------------------------------------------------------------------------- #
# Ubuntu is the branch we can actually run end-to-end on the A100 VM, so it gets the
# most attention. Everything external (dpkg/apt, run_capture, download, file writes) is
# mocked — we're checking the orchestration, not the real package installs.
class InstallUbuntuTests(unittest.TestCase):
    def test_install_runs_full_ubuntu_flow(self):
        """Ubuntu path: resolves, downloads, runs the equivs dance, records versions."""
        env = _env(distribution="ubuntu24.04")
        with mock.patch.object(install_doca, "exec_program", return_value=0) as ex, \
             mock.patch.object(install_doca, "run_capture",
                               side_effect=[(0, "Version: 4.1.5-1"), (0, "a-b-c-d")]), \
             mock.patch.object(install_doca, "download_and_verify",
                               return_value="/tmp/doca.deb") as dl, \
             mock.patch.object(install_doca, "write_component_version") as wcv, \
             mock.patch.object(install_doca.glob, "glob",
                               return_value=["/tmp/hpcx-provides-openmpi_1_all.deb"]), \
             mock.patch("pathlib.Path.write_text"), \
             mock.patch("os.makedirs"), \
             mock.patch("os.remove"):
            rc = install_doca.install(env)
        self.assertEqual(rc, 0)
        dl.assert_called_once()
        wcv.assert_any_call("DOCA", "2.9.1")
        wcv.assert_any_call("OFED", "c-d")   # parsed from the ofed_info output
        self.assertGreater(ex.call_count, 0)

    def test_install_fails_when_openmpi_version_unreadable(self):
        """Negative: no openmpi version from the DOCA repo → error is logged, rc=3."""
        env = _env(distribution="ubuntu24.04")
        with mock.patch.object(install_doca, "exec_program", return_value=0), \
             mock.patch.object(install_doca, "run_capture", return_value=(0, "")), \
             mock.patch.object(install_doca, "download_and_verify",
                               return_value="/tmp/doca.deb"), \
             mock.patch.object(install_doca, "log_error") as logged, \
             mock.patch("pathlib.Path.write_text"):
            rc = install_doca.install(env)
        self.assertEqual(rc, 3)
        logged.assert_called()

    def test_install_stops_on_first_failed_command(self):
        """Negative: the very first package op failing aborts with rc=3."""
        env = _env(distribution="ubuntu24.04")
        with mock.patch.object(install_doca, "exec_program", return_value=1), \
             mock.patch.object(install_doca, "download_and_verify",
                               return_value="/tmp/doca.deb"), \
             mock.patch("pathlib.Path.write_text"):
            rc = install_doca.install(env)
        self.assertEqual(rc, 3)


class InstallAzureLinuxTests(unittest.TestCase):
    def test_install_runs_azurelinux_flow(self):
        env = _env(distribution="azurelinux3.0")
        with mock.patch.object(install_doca, "exec_program", return_value=0), \
             mock.patch.object(install_doca, "run_capture", return_value=(0, "a-b-c-d")), \
             mock.patch.object(install_doca, "download_and_verify",
                               return_value="/tmp/doca.rpm"), \
             mock.patch.object(install_doca, "write_component_version") as wcv, \
             mock.patch("pathlib.Path.write_text"), \
             mock.patch("os.makedirs"):
            rc = install_doca.install(env)
        self.assertEqual(rc, 0)
        wcv.assert_any_call("DOCA", "2.9.1")


# RHEL/AzLinux are mock-only for now (no such hardware to run them on). The bit worth
# really checking is that dnf.conf gets restored even mid-flow — that's the try/finally
# the bash version could skip if an install failed.
class InstallRhelTests(unittest.TestCase):
    def test_install_restores_dnf_conf_after_install(self):
        """RHEL path lifts dnf.conf excludes then restores the original."""
        env = _env(distribution="almalinux9.7")
        original = "[main]\ngpgcheck=1\nexclude=kernel*\n"
        with mock.patch.object(install_doca, "exec_program", return_value=0), \
             mock.patch.object(install_doca, "run_capture", return_value=(0, "a-b-c-d")), \
             mock.patch.object(install_doca, "download_and_verify",
                               return_value="/tmp/doca.rpm"), \
             mock.patch.object(install_doca, "newest_kernel_repo_rpm",
                               return_value="/tmp/DOCA.2.9/doca-kernel-repo-1.rpm"), \
             mock.patch.object(install_doca, "write_component_version"), \
             mock.patch("pathlib.Path.read_text", return_value=original), \
             mock.patch("pathlib.Path.write_text") as wt, \
             mock.patch("os.makedirs"):
            rc = install_doca.install(env)
        self.assertEqual(rc, 0)
        # The original dnf.conf must be written back (restore in finally).
        self.assertIn(mock.call(original, encoding="utf-8"), wt.call_args_list)

    def test_install_fails_when_no_kernel_repo_found(self):
        """Negative: no doca-kernel-repo rpm → error logged, rc=3."""
        env = _env(distribution="rocky9.7")
        with mock.patch.object(install_doca, "exec_program", return_value=0), \
             mock.patch.object(install_doca, "download_and_verify",
                               return_value="/tmp/doca.rpm"), \
             mock.patch.object(install_doca, "newest_kernel_repo_rpm", return_value=None), \
             mock.patch.object(install_doca, "log_error") as logged:
            rc = install_doca.install(env)
        self.assertEqual(rc, 3)
        logged.assert_called()


class InstallCommonTests(unittest.TestCase):
    def test_install_fails_when_no_version(self):
        """Negative: doca missing from versions.json → error logged, rc=3."""
        with mock.patch.object(install_doca, "log_error") as logged:
            rc = install_doca.install(_env(doca=False))
        self.assertEqual(rc, 3)
        logged.assert_called_once()

    def test_install_fails_when_download_raises(self):
        """Negative: a failed download/verify → error logged, rc=3."""
        env = _env(distribution="ubuntu24.04")
        with mock.patch.object(install_doca, "download_and_verify",
                               side_effect=RuntimeError("boom")), \
             mock.patch.object(install_doca, "log_error") as logged:
            rc = install_doca.install(env)
        self.assertEqual(rc, 3)
        logged.assert_called()


if __name__ == "__main__":
    unittest.main()
