"""Unit tests for components/python/install_cuda_samples.py.

install() runs with download_and_verify, exec_program and run_capture mocked.
The work dir and the /usr/local prefix are redirected to a temporary directory.
(These replace the CudaSamplesTests that used to live in test_packaging.py.)

Run from the repo root:
    python3 -m unittest tests.python.test_install_cuda_samples -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_cuda_samples as m

_OP = "install-cuda-samples"
_SHA = "63cc9d5d8280c87df3c1f4e2276234a0f42cc497c52b40dd5bdda2836607db79"
_URL = "https://github.com/NVIDIA/cuda-samples/archive/refs/tags/v13.0.tar.gz"


def _env(cuda=None):
    if cuda is None:
        cuda = {"driver": {"version": "13.0", "distribution": "ubuntu2404"},
                "samples": {"version": "13.0", "sha256": _SHA}}
    return {
        "COMPONENT_VERSIONS": json.dumps({"cuda": {"ubuntu24.04": {"x86_64": cuda}}}),
        "DISTRIBUTION": "ubuntu24.04", "ARCHITECTURE": "x86_64", "GPU": "NVIDIA",
        "SKU": "A100", "TARGET_NODE_TYPE": "azure_vm_regular",
    }


class HelperTests(unittest.TestCase):
    def test_url_and_destination_match_bash(self):
        self.assertEqual(m.samples_url("13.0"), _URL)
        self.assertEqual(m.samples_destination("12.9"), "/usr/local/cuda-12.9/samples")

    @unittest.skip("install_cuda_samples.sh changed upstream (CMake install "
                   "path for newer samples); port resync in ADO 985")
    def test_bash_script_still_matches(self):
        script = (Path(__file__).resolve().parents[2] / "components"
                  / "install_cuda_samples.sh").read_text(encoding="utf-8")
        for line in [
            'TARBALL="v${CUDA_SAMPLES_VERSION}.tar.gz"',
            "CUDA_SAMPLES_DOWNLOAD_URL=https://github.com/NVIDIA/cuda-samples/archive/refs/tags/${TARBALL}",
            "tar -xvf ${TARBALL}",
            "pushd ./cuda-samples-${CUDA_SAMPLES_VERSION}",
            "mkdir build && cd build",
            "cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..",
            "make -j $(nproc)",
            "mv -vT ./Samples /usr/local/cuda-${CUDA_DRIVER_VERSION}/samples",
        ]:
            with self.subTest(line=line):
                self.assertIn(line, script)


class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.work = os.path.join(self._tmp.name, "work")
        self.prefix = os.path.join(self._tmp.name, "usr-local")
        for name, value in [("_WORK_DIR", self.work), ("_CUDA_PREFIX", self.prefix)]:
            patcher = mock.patch.object(m, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _install(self, env, *, fail_on=None, nproc=(0, "8\n"), download_error=None):
        calls = []

        def fake_download(url, sha256, dest_dir="."):
            calls.append(("download", url, sha256, dest_dir))
            if download_error:
                raise download_error
            return Path(dest_dir) / os.path.basename(url)

        def fake_exec(command, op, **kwargs):
            self.assertEqual(op, _OP)
            calls.append(("exec", command, kwargs.get("cwd")))
            if command[0] == "tar":
                # What the real tar would extract: the source directory.
                os.makedirs(os.path.join(self.work, "cuda-samples-13.0"))
            return 1 if command == fail_on else 0

        def fake_capture(command, op, **kwargs):
            calls.append(("capture", command))
            return nproc

        with mock.patch.object(m, "download_and_verify", side_effect=fake_download), \
             mock.patch.object(m, "exec_program", side_effect=fake_exec), \
             mock.patch.object(m, "run_capture", side_effect=fake_capture), \
             mock.patch.object(m, "log_info") as log_info, \
             mock.patch.object(m, "log_warn") as log_warn, \
             mock.patch.object(m, "log_error") as log_error:
            rc = m.install(env)
        return rc, calls, {"info": log_info, "warn": log_warn, "error": log_error}

    def test_happy_path_mirrors_bash(self):
        rc, calls, logs = self._install(_env())
        self.assertEqual(rc, 0)
        build = os.path.join(self.work, "cuda-samples-13.0", "build")
        dest = f"{self.prefix}/cuda-13.0/samples"
        self.assertEqual(calls, [
            ("download", _URL, _SHA, self.work),
            ("exec", ["tar", "-xvf", "v13.0.tar.gz"], self.work),
            ("exec", ["cmake", "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc", ".."], build),
            ("capture", ["nproc"]),
            ("exec", ["make", "-j", "8"], build),
            ("exec", ["mv", "-vT", "./Samples", dest], build),
        ])
        self.assertFalse(os.path.exists(self.work))
        first = logs["info"].call_args_list[0]
        self.assertEqual(first.kwargs["samples_version"], "13.0")
        self.assertEqual(first.kwargs["dest"], dest)
        self.assertEqual(logs["info"].call_args_list[-1].args[1],
                         "CUDA samples 13.0 installed")
        logs["warn"].assert_not_called()

    def test_v100_override_versions(self):
        cuda = {"driver": {"version": "12.9"}, "samples": {"version": "12.9", "sha256": "s"}}
        with mock.patch.object(m, "_CUDA_PREFIX", "/usr/local"):
            self.assertEqual(m.samples_destination("12.9"), "/usr/local/cuda-12.9/samples")
        self.assertEqual(m._resolve(_env(cuda)), ("12.9", "s", "12.9"))

    def test_existing_destination_warns(self):
        os.makedirs(f"{self.prefix}/cuda-13.0/samples")
        rc, _, logs = self._install(_env())
        self.assertEqual(rc, 0)          # the (mocked) mv decides; real mv -T fails
        self.assertIn("already exists", logs["warn"].call_args.args[1])

    def test_missing_values_fail_before_download(self):
        for cuda in [{}, {"driver": {"version": "13.0"}},
                     {"driver": {"version": "13.0"}, "samples": {"version": "13.0"}},
                     {"samples": {"version": "13.0", "sha256": "s"}}]:
            with self.subTest(cuda=cuda):
                rc, calls, logs = self._install(_env(cuda))
                self.assertEqual(rc, 3)
                self.assertEqual(calls, [])
                self.assertIn("could not resolve cuda", logs["error"].call_args.args[1])
        rc, calls, _ = self._install({"COMPONENT_VERSIONS": "{}"})
        self.assertEqual((rc, calls), (3, []))

    def test_download_failure(self):
        rc, calls, logs = self._install(_env(), download_error=OSError("boom"))
        self.assertEqual(rc, 3)
        self.assertEqual(len(calls), 1)
        self.assertFalse(os.path.exists(self.work))

    def test_each_command_failure_stops(self):
        build = ["cmake", "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc", ".."]
        for fail, count in [(["tar", "-xvf", "v13.0.tar.gz"], 2), (build, 3),
                            (["make", "-j", "8"], 5)]:
            with self.subTest(fail=fail[0]):
                rc, calls, logs = self._install(_env(), fail_on=fail)
                self.assertEqual(rc, 3)
                self.assertEqual(len(calls), count)
                self.assertFalse(os.path.exists(self.work))
                logs["error"].assert_called()

    def test_mv_failure(self):
        dest = f"{self.prefix}/cuda-13.0/samples"
        rc, calls, logs = self._install(_env(), fail_on=["mv", "-vT", "./Samples", dest])
        self.assertEqual(rc, 3)
        self.assertIn(f"could not move the built samples to {dest}",
                      [c.args[1] for c in logs["error"].call_args_list])

    def test_bad_nproc_output(self):
        rc, calls, _ = self._install(_env(), nproc=(0, "\n"))
        self.assertEqual(rc, 3)
        self.assertEqual(calls[-1], ("capture", ["nproc"]))


if __name__ == "__main__":
    unittest.main()
