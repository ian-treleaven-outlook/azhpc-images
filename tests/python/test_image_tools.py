"""Tests for tools/image_fingerprint.py and tools/compare_images.py (ADO 1002)."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import compare_images as cmp
from tools import image_fingerprint as fp

_REPO = Path(__file__).resolve().parents[2]


class FingerprintTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_hashed_tree_records_files_links_and_modes(self):
        conf = self.root / "modprobe.d"
        conf.mkdir()
        (conf / "blacklist.conf").write_text("blacklist mana_ib\n")
        (conf / "blacklist.conf").chmod(0o644)
        (conf / "link.conf").symlink_to("blacklist.conf")
        tree = fp.hashed_tree([str(conf), str(self.root / "missing")])
        entry = tree[str(conf / "blacklist.conf")]
        self.assertEqual(entry["type"], "file")
        self.assertEqual(entry["mode"], "0o644")
        self.assertEqual(len(entry["sha256"]), 64)
        self.assertEqual(tree[str(conf / "link.conf")],
                         {"type": "link", "target": "blacklist.conf", "mode": "0o777"})
        self.assertNotIn(str(self.root / "missing"), tree)

    def test_layout_stops_at_the_depth(self):
        deep = self.root / "opt" / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "file").write_text("x")
        entries = fp.layout(str(self.root / "opt"), 2)
        self.assertEqual(entries, {str(self.root / "opt" / "a"): "dir",
                                   str(self.root / "opt" / "a" / "b"): "dir"})

    def test_capture_writes_json(self):
        with mock.patch.object(fp, "CONFIG_PATHS", [str(self.root)]), \
             mock.patch.object(fp, "MODULE_ROOTS", []), \
             mock.patch.object(fp, "LAYOUT_ROOTS", {str(self.root): 1}), \
             mock.patch.object(fp, "command_lines", return_value=[]), \
             contextlib_redirect() as out:
            (self.root / "x.conf").write_text("x")
            self.assertEqual(fp.main(["--out", str(self.root / "fp.json")]), 0)
        data = json.loads((self.root / "fp.json").read_text())
        self.assertEqual(data["format"], 1)
        self.assertIn(str(self.root / "x.conf"), data["config_files"])
        self.assertIn("image fingerprint:", out.getvalue())

    @unittest.skipUnless(shutil.which("python3"), "needs python3")
    def test_script_runs_standalone(self):
        # Packer runs it as a plain script, with whatever python3 the VM has.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fp.json"
            subprocess.run([sys.executable, str(_REPO / "tools" / "image_fingerprint.py"),
                            "--out", str(out)], check=True, capture_output=True)
            self.assertIn("enabled_units", json.loads(out.read_text()))


def contextlib_redirect():
    import contextlib
    buffer = io.StringIO()
    manager = contextlib.redirect_stdout(buffer)

    class _Both:
        def __enter__(self):
            manager.__enter__()
            return buffer

        def __exit__(self, *exc):
            return manager.__exit__(*exc)
    return _Both()


class ParserTests(unittest.TestCase):
    def test_packages_join_repeated_names(self):
        tsv = "bash\t5.2-1\nkernel\t0:6.6.1\nkernel\t0:6.6.2\n"
        self.assertEqual(cmp.parse_packages(tsv),
                         {"bash": "5.2-1", "kernel": "0:6.6.1, 0:6.6.2"})

    def test_services(self):
        text = ("  UNIT LOAD ACTIVE SUB DESCRIPTION\n"
                "  ssh.service       loaded active running OpenBSD Secure Shell\n"
                "● nvidia-dcgm.service loaded failed failed NVIDIA DCGM\n"
                "  foo.mount loaded active mounted Foo\n")
        self.assertEqual(cmp.parse_services(text),
                         {"ssh.service": "loaded active running",
                          "nvidia-dcgm.service": "loaded failed failed"})

    def test_trivy_findings(self):
        report = {"Results": [{"Vulnerabilities": [
            {"VulnerabilityID": "CVE-1", "PkgName": "openssl", "Severity": "HIGH"}]},
            {"Vulnerabilities": None}]}
        self.assertEqual(cmp.trivy_findings(report), {"CVE-1 openssl": "HIGH"})

    def test_diff_and_ignore(self):
        import re
        a = {"x": "1", "y": "1", "only-a": ""}
        b = {"x": "1", "y": "2", "only-b.log": ""}
        self.assertEqual(cmp.diff(a, b), (["only-a"], ["only-b.log"], {"y": ("1", "2")}))
        self.assertEqual(cmp.diff(a, b, re.compile(r"\.log$"))[1], [])


class CompareTests(unittest.TestCase):
    """Two fake manifest folders, as Packer leaves them."""

    def _build(self, folder: Path, packages: str, components: dict, units: list[str]):
        folder.mkdir()
        run = "2026.1030.010203"
        evidence = {f"{run}/build/packages.tsv": packages,
                    f"{run}/build/service-state.txt": "  ssh.service loaded active running SSH\n",
                    f"{run}/build/dkms.txt": "nvidia/580, 6.8.0: installed\n"}
        with tarfile.open(folder / "build-evidence.tgz", "w:gz") as archive:
            for name, text in evidence.items():
                data = text.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        (folder / "component-versions.json").write_text(json.dumps(components))
        (folder / "image-fingerprint.json").write_text(json.dumps(
            {"config_files": {}, "modulefiles": {}, "enabled_units": units,
             "pip_packages": [], "layout": {"/opt": {"/opt/hpcx": "dir"}}}))
        return folder

    def test_same_builds_have_no_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self._build(Path(tmp) / "a", "bash\t5.2\n", {"NCCL": "2.30", "ImageVersion": "1"}, ["ssh.service"])
            b = self._build(Path(tmp) / "b", "bash\t5.2\n", {"NCCL": "2.30", "ImageVersion": "2"}, ["ssh.service"])
            out = io.StringIO()
            total = cmp.report(cmp.load_build(a), cmp.load_build(b), ("bash", "python"), out=out)
        self.assertEqual(total, 0)                         # ImageVersion is ignored
        self.assertIn("== packages: same (1 items)", out.getvalue())

    def test_differences_are_listed_and_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self._build(Path(tmp) / "a", "bash\t5.2\njq\t1.7\n", {"NCCL": "2.30"}, ["ssh.service"])
            b = self._build(Path(tmp) / "b", "bash\t5.3\n", {"NCCL": "2.30"}, ["ssh.service", "x.service"])
            out = io.StringIO()
            total = cmp.report(cmp.load_build(a), cmp.load_build(b), ("bash", "python"), out=out)
            with mock.patch("sys.stdout", io.StringIO()):
                self.assertEqual(cmp.main([str(a), str(b)]), 1)
        text = out.getvalue()
        self.assertEqual(total, 3)
        self.assertIn("only in bash: jq 1.7", text)
        self.assertIn("differs: bash", text)
        self.assertIn("only in python: x.service", text)

    def test_unusable_inputs(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(cmp.main([str(Path(tmp) / "nope"), tmp]), 2)
            self.assertEqual(cmp.main([tmp, tmp]), 2)       # empty folder


if __name__ == "__main__":
    unittest.main()
