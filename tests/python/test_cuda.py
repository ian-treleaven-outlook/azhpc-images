"""Unit tests for utils/cuda.py, the CUDA toolkit pieces the driver ports share.

The nvcc parser was compared against the real grep/awk/cut in WSL on the same
inputs; CUDA_PROFILE_TEXT against what both bash scripts' `echo | tee` lines
write.

Run from the repo root:
    python3 -m unittest tests.python.test_cuda -v
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils import cuda

_REPO = Path(__file__).resolve().parents[2]

_NVCC_13 = ("nvcc: NVIDIA (R) Cuda compiler driver\n"
            "Copyright (c) 2005-2025 NVIDIA Corporation\n"
            "Built on Wed_Aug_20_01:58:59_PM_PDT_2025\n"
            "Cuda compilation tools, release 13.0, V13.0.88\n"
            "Build cuda_13.0.r13.0/compiler.36424714_0\n")


class ToolkitPackageTests(unittest.TestCase):
    def test_every_dot_becomes_a_dash(self):
        self.assertEqual(cuda.cuda_toolkit_package("13.0"), "cuda-toolkit-13-0")
        self.assertEqual(cuda.cuda_toolkit_package("12.9"), "cuda-toolkit-12-9")
        self.assertEqual(cuda.cuda_toolkit_package("12.9.1"), "cuda-toolkit-12-9-1")


class NvccVersionTests(unittest.TestCase):
    def test_release_line(self):
        self.assertEqual(cuda.nvcc_cuda_version(_NVCC_13), "13.0.88")
        self.assertEqual(cuda.nvcc_cuda_version(
            "Cuda compilation tools, release 12.9, V12.9.86\n"), "12.9.86")

    def test_first_word_wins_like_bash(self):
        self.assertEqual(cuda.nvcc_cuda_version("x release 1 2 3 Vy\n" + _NVCC_13), "y")

    def test_short_matching_line_is_dropped_like_word_splitting(self):
        self.assertEqual(cuda.nvcc_cuda_version("short release\n" + _NVCC_13), "13.0.88")

    def test_no_match_is_none(self):
        self.assertIsNone(cuda.nvcc_cuda_version(""))
        self.assertIsNone(cuda.nvcc_cuda_version("bash: nvcc: command not found\n"))
        self.assertIsNone(cuda.nvcc_cuda_version("release a b\n"))

    def test_command_sources_profile_in_its_own_bash(self):
        self.assertEqual(list(cuda.NVCC_VERSION_COMMAND),
                         ["bash", "-c", "source /etc/profile; nvcc --version"])


class CudaProfileTests(unittest.TestCase):
    def test_matches_the_bash_render(self):
        fixture = (_REPO / "tests" / "python" / "fixtures" / "install_nvidiagpudriver"
                   / "cuda.sh")
        with open(fixture, encoding="utf-8", newline="") as handle:
            self.assertEqual(cuda.CUDA_PROFILE_TEXT, handle.read())
        self.assertEqual(cuda.CUDA_PROFILE, "/etc/profile.d/cuda.sh")

    def test_matches_the_echo_lines_in_both_bash_scripts(self):
        # Each `echo '<text>' | tee [-a] /etc/profile.d/cuda.sh` writes <text>
        # plus a newline; read the texts from the .sh so the two can't drift.
        for script in ("install_nvidiagpudriver.sh", "install_nvidiagriddriver.sh"):
            with self.subTest(script=script):
                text = (_REPO / "components" / script).read_text(encoding="utf-8")
                lines = re.findall(r"^\s*echo '([^']*)' \| tee (?:-a )?"
                                   r"/etc/profile\.d/cuda\.sh", text, flags=re.M)
                self.assertEqual(len(lines), 2)
                self.assertEqual(cuda.CUDA_PROFILE_TEXT, "".join(t + "\n" for t in lines))


class CudaAptSourceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.sources = self._tmp.name

    def _check(self):
        with mock.patch.object(cuda, "log_info") as info, \
             mock.patch.object(cuda, "log_warn") as warn:
            found = cuda.check_cuda_apt_source("op", self.sources)
        return found, info, warn

    def test_present(self):
        for name in ("cuda-ubuntu2404-x86_64.list", "cuda-extra.list", "other.list"):
            open(os.path.join(self.sources, name), "w").close()
        found, info, warn = self._check()
        self.assertTrue(found)
        warn.assert_not_called()
        info.assert_called_once()
        self.assertIn("NVIDIA CUDA apt source present", info.call_args.args[1])
        self.assertEqual(info.call_args.kwargs["cuda_apt_sources"],
                         ", ".join(os.path.join(self.sources, name) for name in
                                   ("cuda-extra.list", "cuda-ubuntu2404-x86_64.list")))

    def test_missing_warns_and_returns_false(self):
        open(os.path.join(self.sources, "other.list"), "w").close()
        found, info, warn = self._check()
        self.assertFalse(found)
        info.assert_not_called()
        message = warn.call_args.args[1]
        self.assertIn("no CUDA apt source", message)
        self.assertIn("install_cuda_repo_and_dkms.sh", message)
        self.assertEqual(warn.call_args.kwargs["path"], self.sources)

    def test_missing_directory_is_just_missing(self):
        self.sources = os.path.join(self.sources, "absent")
        found, _, warn = self._check()
        self.assertFalse(found)
        warn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
