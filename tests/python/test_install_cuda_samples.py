"""Unit tests for components/python/install_cuda_samples.py.

install() runs with utils.download.download_and_verify, utils.process.exec_program
(where run() and extract_tarball() end up) and run_capture mocked. The work dir
and the /usr/local prefix are redirected to a temporary directory.
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
from utils import component_config, download, process

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

    def test_cmake_options(self):
        dest = "/usr/local/cuda-13.0/samples"
        self.assertEqual(m.cmake_options(dest, False),
                         ["-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"])
        self.assertEqual(m.cmake_options(dest, True),
                         ["-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
                          f"-DCUDA_SAMPLES_INSTALL_DIR={dest}"])

    def test_bash_script_still_matches(self):
        script = (Path(__file__).resolve().parents[2] / "components"
                  / "install_cuda_samples.sh").read_text(encoding="utf-8")
        for line in [
            'TARBALL="v${CUDA_SAMPLES_VERSION}.tar.gz"',
            "CUDA_SAMPLES_DOWNLOAD_URL=https://github.com/NVIDIA/cuda-samples/archive/refs/tags/${TARBALL}",
            "tar -xvf ${TARBALL}",
            "pushd ./cuda-samples-${CUDA_SAMPLES_VERSION}",
            "mkdir build && cd build",
            "CMAKE_OPTIONS=(-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc)",
            "if [[ -f ../cmake/InstallSamples.cmake ]]; then",
            "\tCMAKE_OPTIONS+=(-DCUDA_SAMPLES_INSTALL_DIR=/usr/local/cuda-${CUDA_DRIVER_VERSION}/samples)",
            'cmake "${CMAKE_OPTIONS[@]}" ..',
            "make -j $(nproc)",
            "\tcmake --install .",
            "\tmv -vT ./Samples /usr/local/cuda-${CUDA_DRIVER_VERSION}/samples",
        ]:
            with self.subTest(line=line):
                self.assertIn(line, script)
        # The InstallSamples.cmake test appears twice: before cmake and after make.
        self.assertEqual(script.count("if [[ -f ../cmake/InstallSamples.cmake ]]; then"), 2)


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

    def _install(self, env, *, fail_on=None, nproc=(0, "8\n"), download_error=None,
                 cmake_install=False):
        """Run install(env) with the externals mocked.

        cmake_install: the extracted source ships cmake/InstallSamples.cmake.
        """
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
                source = os.path.join(self.work, "cuda-samples-13.0")
                os.makedirs(os.path.join(source, "cmake"))
                if cmake_install:
                    Path(source, "cmake", "InstallSamples.cmake").write_text("")
            return 1 if command == fail_on else 0

        def fake_capture(command, op, **kwargs):
            calls.append(("capture", command))
            return nproc

        with mock.patch.object(download, "download_and_verify", side_effect=fake_download), \
             mock.patch.object(download, "log_error") as fetch_error, \
             mock.patch.object(process, "exec_program", side_effect=fake_exec), \
             mock.patch.object(m, "run_capture", side_effect=fake_capture), \
             mock.patch.object(component_config, "log_error") as config_error, \
             mock.patch.object(m, "log_info") as log_info, \
             mock.patch.object(m, "log_warn") as log_warn, \
             mock.patch.object(m, "log_error") as log_error:
            rc = m.install(env)
        return rc, calls, {"info": log_info, "warn": log_warn, "error": log_error,
                           "config_error": config_error,
                           "download_error": fetch_error}

    def test_happy_path_mirrors_bash(self):
        rc, calls, logs = self._install(_env())
        self.assertEqual(rc, 0)
        build = os.path.join(self.work, "cuda-samples-13.0", "build")
        dest = f"{self.prefix}/cuda-13.0/samples"
        self.assertEqual(calls, [
            ("download", _URL, _SHA, self.work),
            ("exec", ["tar", "-xf", "v13.0.tar.gz"], self.work),
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

    def test_install_samples_cmake_uses_cmake_install(self):
        rc, calls, logs = self._install(_env(), cmake_install=True)
        self.assertEqual(rc, 0)
        build = os.path.join(self.work, "cuda-samples-13.0", "build")
        dest = f"{self.prefix}/cuda-13.0/samples"
        self.assertEqual(calls, [
            ("download", _URL, _SHA, self.work),
            ("exec", ["tar", "-xf", "v13.0.tar.gz"], self.work),
            ("exec", ["cmake", "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
                      f"-DCUDA_SAMPLES_INSTALL_DIR={dest}", ".."], build),
            ("capture", ["nproc"]),
            ("exec", ["make", "-j", "8"], build),
            ("exec", ["cmake", "--install", "."], build),
        ])
        self.assertFalse(os.path.exists(self.work))
        decision = [c for c in logs["info"].call_args_list
                    if c.kwargs.get("install_method")]
        self.assertEqual(len(decision), 1)
        self.assertEqual(decision[0].kwargs["install_method"], "cmake --install")
        self.assertEqual(decision[0].kwargs["dest"], dest)
        self.assertEqual(logs["info"].call_args.args[1], "CUDA samples 13.0 installed")
        logs["warn"].assert_not_called()

    def test_without_install_samples_cmake_logs_mv_method(self):
        _, _, logs = self._install(_env())
        decision = [c for c in logs["info"].call_args_list
                    if c.kwargs.get("install_method")]
        self.assertEqual([c.kwargs["install_method"] for c in decision], ["mv -vT"])

    def test_cmake_install_rerun_does_not_warn(self):
        # cmake --install overwrites in place; only the mv -T path can't rerun.
        os.makedirs(f"{self.prefix}/cuda-13.0/samples")
        rc, _, logs = self._install(_env(), cmake_install=True)
        self.assertEqual(rc, 0)
        logs["warn"].assert_not_called()

    def test_cmake_install_failure(self):
        rc, calls, logs = self._install(_env(), cmake_install=True,
                                        fail_on=["cmake", "--install", "."])
        self.assertEqual(rc, 3)
        self.assertEqual(calls[-1][1], ["cmake", "--install", "."])
        self.assertEqual(logs["error"].call_args.args[1], "CUDA samples installation failed")
        self.assertFalse(os.path.exists(self.work))

    def test_v100_override_versions(self):
        # The samples go under the cuda driver.version, not the samples version.
        cuda = {"driver": {"version": "12.9"}, "samples": {"version": "13.0", "sha256": "s"}}
        rc, calls, _ = self._install(_env(cuda))
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0], ("download", _URL, "s", self.work))
        self.assertEqual(calls[-1][1], ["mv", "-vT", "./Samples",
                                        f"{self.prefix}/cuda-12.9/samples"])

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
                self.assertIn("no usable 'cuda' config",
                              logs["config_error"].call_args.args[1])
        rc, calls, _ = self._install({"COMPONENT_VERSIONS": "{}"})
        self.assertEqual((rc, calls), (3, []))

    def test_download_failure(self):
        rc, calls, logs = self._install(_env(), download_error=OSError("boom"))
        self.assertEqual(rc, 3)
        self.assertEqual(len(calls), 1)
        self.assertIn("boom", logs["download_error"].call_args.args[1])
        self.assertFalse(os.path.exists(self.work))

    def test_each_command_failure_stops(self):
        build = ["cmake", "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc", ".."]
        for fail, count in [(["tar", "-xf", "v13.0.tar.gz"], 2), (build, 3),
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
        self.assertEqual(calls[-1][1], ["mv", "-vT", "./Samples", dest])
        self.assertEqual(logs["error"].call_args.args[1], "CUDA samples installation failed")

    def test_bad_nproc_output(self):
        rc, calls, _ = self._install(_env(), nproc=(0, "\n"))
        self.assertEqual(rc, 3)
        self.assertEqual(calls[-1], ("capture", ["nproc"]))


if __name__ == "__main__":
    unittest.main()
