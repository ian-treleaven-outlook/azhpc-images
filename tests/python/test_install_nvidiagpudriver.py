"""Unit tests for components/python/install_nvidiagpudriver.py.

  * Pure helpers are tested directly. The dnf/nvcc/awk parsers were also
    compared against the real grep/sed/awk/cut in WSL on the same inputs.
  * Generated files are compared with fixtures in
    tests/python/fixtures/install_nvidiagpudriver/, rendered by running the
    bash script's own echo/tee/heredoc lines in real bash. (00-nvidia-prefer
    is stored as 00-nvidia-prefer.txt: .gitattributes forces LF only on names
    with a dot, and a CRLF checkout would break the byte comparison.)
  * install() runs with every external effect mocked: exec_program,
    run_capture, download_and_verify, dnf_pin_packages,
    write_component_version and install_cuda_samples.install. System paths are
    redirected under a temporary directory (_ROOT), as is the work dir.

Run from the repo root:
    python3 -m unittest tests.python.test_install_nvidiagpudriver -v
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_nvidiagpudriver as m
from utils.build_config import BuildConfig, build_plan

_OP = "install-nv-driver"
_FIXTURES = Path(__file__).parent / "fixtures" / "install_nvidiagpudriver"
_COMPONENT_DIR = "/repo/components"
_TOP_DIR = "/repo"
_KERNEL = "5.14.0-test.el9.x86_64"

_REPO_FILE = "nvidia-driver-local-repo-ubuntu2404-580.178.04_1.0-1_amd64.deb"
_REPO_TEXT = "[cuda-azl3-x86_64]\nname=cuda\nbaseurl=https://example\n"
_NVCC_13 = ("nvcc: NVIDIA (R) Cuda compiler driver\n"
            "Cuda compilation tools, release 13.0, V13.0.88\n"
            "Build cuda_13.0.r13.0/compiler.36424714_0\n")
_NVCC_12 = "Cuda compilation tools, release 12.9, V12.9.86\n"
_DNF_LIST = ("Installed Packages\n"
             "cuda-open.x86_64      580.126.20-1_6.6.104.2.azl3   @azurelinux-nvidia\n"
             "cuda-open-hwe.aarch64 580.159.04-1_6.6.104.2.azl3   @azurelinux-nvidia\n"
             "cuda.x86_64           575.57.08-1_6.6.104.2.azl3    @azurelinux-nvidia\n"
             "cuda-toolkit-13-0.x86_64 13.0.2-1                   @cuda-azl3\n")

_CUDA_13 = {"driver": {"version": "13.0", "distribution": "{d}"}}
_V100_CUDA = {"driver": {"version": "12.9", "distribution": "{d}"}}


def _cuda(distribution):
    """A cuda entry with a V100 override, like versions.json."""
    base = json.loads(json.dumps(_CUDA_13).replace("{d}", distribution))
    v100 = json.loads(json.dumps(_V100_CUDA).replace("{d}", distribution))
    base["azure_vm_regular"] = {"nvidia_v100": v100}
    return base


_VERSIONS = {
    "nvidia": {
        "ubuntu24.04": {
            "x86_64": {
                "baremetal_1p": {"default": {"driver": {
                    "version": "580.178.04", "repo_file": _REPO_FILE}}},
                "driver": {"version": "580.178.04"},
            },
            "aarch64": {
                "baremetal_3p": {"nvidia_gb200": {
                    "driver": {"version": "580.167.08", "major_version": "580"},
                    "imex": {"version": "580.167.08-1"}}},
                "driver": {"version": "580.126.20", "major_version": "580"},
                "imex": {"version": "580.126.20-1"},
            },
        },
        "almalinux9.8": {"x86_64": {"driver": {"version": "580.126.20",
                                               "sha256": "abc123"}}},
        "sles15": {"x86_64": {"driver": {"version": "580.126.20", "sha256": "abc123"}}},
        "azurelinux3.0": {"aarch64": {"imex": {"version": "580.159.04"}}},
    },
    "cuda": {
        "ubuntu24.04": {"x86_64": _cuda("ubuntu2404"), "aarch64": _cuda("ubuntu2404")},
        "almalinux9.8": {"x86_64": _cuda("rhel9")},
        "sles15": {"x86_64": _cuda("sles15")},
        "azurelinux3.0": {"x86_64": _cuda("azl3"), "aarch64": _cuda("azl3")},
    },
}


def _env(distribution="ubuntu24.04", sku="A100", arch="x86_64",
         node_type="azure_vm_regular", versions=None, **extra):
    env = {
        "DISTRIBUTION": distribution, "ARCHITECTURE": arch, "GPU": "NVIDIA",
        "SKU": sku, "TARGET_NODE_TYPE": node_type, "TOP_DIR": _TOP_DIR,
        "COMPONENT_DIR": _COMPONENT_DIR,
        "COMPONENT_VERSIONS": json.dumps(versions or _VERSIONS),
    }
    env.update(extra)
    return env


def _fixture(name):
    with open(_FIXTURES / name, encoding="utf-8", newline="") as handle:
        return handle.read()


def _nested(script):
    return ("exec", [f"{_COMPONENT_DIR}/{script}"])


_GDRCOPY = _nested("install_gdrcopy.sh")
_FABRIC_MANAGER = _nested("install_nvidia_fabric_manager.sh")
_PERSISTENCE = _nested("configure_nvidia_persistence.sh")
_NVCC = ("capture", ["bash", "-c", "source /etc/profile; nvcc --version"])
_SAMPLES = ("samples",)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
class HelperTests(unittest.TestCase):
    def test_first_word(self):
        self.assertEqual(m.first_word("580.1-1\n580.2-1"), "580.1-1")
        self.assertEqual(m.first_word("  \t13.0.88 \n"), "13.0.88")
        self.assertIsNone(m.first_word(""))
        self.assertIsNone(m.first_word(" \n\t"))

    def test_is_rackscale_is_case_insensitive(self):
        for value, expected in [("true", True), ("TRUE", True), ("True", True),
                                ("false", False), ("", False), ("1", False),
                                (" true", False)]:
            with self.subTest(value=value):
                self.assertEqual(m.is_rackscale({"NVLINK_RACKSCALE": value}), expected)
        self.assertFalse(m.is_rackscale({}))

    def test_azurelinux_driver_package(self):
        self.assertEqual(m.azurelinux_driver_package("V100", "x86_64"), "cuda")
        self.assertEqual(m.azurelinux_driver_package("V100", "aarch64"), "cuda")
        self.assertEqual(m.azurelinux_driver_package("GB200", "aarch64"), "cuda-open-hwe")
        self.assertEqual(m.azurelinux_driver_package("H100", "x86_64"), "cuda-open")

    def test_azurelinux_driver_version(self):
        self.assertEqual(m.azurelinux_driver_version(_DNF_LIST, "cuda-open"), "580.126.20-1")
        self.assertEqual(m.azurelinux_driver_version(_DNF_LIST, "cuda-open-hwe"),
                         "580.159.04-1")
        # "^cuda\." must not match cuda-open.* or cuda-toolkit-*.
        self.assertEqual(m.azurelinux_driver_version(_DNF_LIST, "cuda"), "575.57.08-1")
        self.assertIsNone(m.azurelinux_driver_version("", "cuda-open"))
        self.assertIsNone(m.azurelinux_driver_version("other.x86_64 1-1_a\n", "cuda-open"))
        # sed keeps a non-matching line whole: bash would record its first word.
        self.assertEqual(m.azurelinux_driver_version(
            "cuda-open.x86_64  580.126.20-1.azl3  @nv\n", "cuda-open"), "cuda-open.x86_64")
        # The last version followed by "_" on the line wins (greedy .*).
        self.assertEqual(m.azurelinux_driver_version(
            "cuda-open.x86_64  1-2_a 3.4-5_b @x\n", "cuda-open"), "3.4-5")

    def test_looks_like_version(self):
        self.assertTrue(m.looks_like_version("580.126.20-1", r"[0-9.]+-[0-9]+"))
        self.assertFalse(m.looks_like_version("cuda-open.x86_64", r"[0-9.]+-[0-9]+"))
        self.assertFalse(m.looks_like_version(None, r"[0-9.]+-[0-9]+"))

    def test_nvcc_cuda_version(self):
        self.assertEqual(m.nvcc_cuda_version(_NVCC_13), "13.0.88")
        self.assertEqual(m.nvcc_cuda_version(_NVCC_12), "12.9.86")
        self.assertIsNone(m.nvcc_cuda_version(""))
        self.assertIsNone(m.nvcc_cuda_version("bash: nvcc: command not found\n"))
        self.assertIsNone(m.nvcc_cuda_version("release a b\n"))

    def test_mrc_repo_dir(self):
        self.assertEqual(m.mrc_repo_dir(_REPO_FILE),
                         "nvidia-driver-local-repo-ubuntu2404-580.178.04")
        self.assertEqual(m.mrc_repo_dir("plain.deb"), "plain.deb")

    def test_cuda_toolkit_package(self):
        self.assertEqual(m.cuda_toolkit_package("13.0"), "cuda-toolkit-13-0")
        self.assertEqual(m.cuda_toolkit_package("12.9"), "cuda-toolkit-12-9")

    def test_rhel_cuda_excludes(self):
        self.assertEqual(m.rhel_cuda_excludes("13.0"), "mft* kernel-mft*")
        self.assertEqual(m.rhel_cuda_excludes("12.9"), "mft* kernel-mft* cccl-*")
        self.assertEqual(m.rhel_cuda_excludes("120.1"), "mft* kernel-mft*")


class FixtureTests(unittest.TestCase):
    """The module's file texts are byte-identical to what bash writes."""

    def test_texts_match_bash_renders(self):
        self.assertEqual(m._CUDA_PROFILE_TEXT, _fixture("cuda.sh"))
        self.assertEqual(m._APT_PREFERENCES_TEXT, _fixture("00-nvidia-prefer.txt"))
        self.assertEqual(m._NVPROFILING_TEXT, _fixture("nvprofiling.conf"))
        self.assertEqual(m._OPENRM_TEXT, _fixture("nvidia-openrm.conf"))
        self.assertEqual(m._PEERMEM_LINE + "\n", _fixture("nvidia-peermem.conf"))
        self.assertEqual(m._GRDMA_LINE + "\n" + m._IMEX_CHANNEL_LINE + "\n",
                         _fixture("nvidia.conf.baremetal-rackscale"))

    def test_bash_script_still_has_these_lines(self):
        """Guard against upstream drift in the lines the fixtures came from."""
        script = (Path(__file__).resolve().parents[2] / "components"
                  / "install_nvidiagpudriver.sh").read_text(encoding="utf-8")
        for line in [
            "echo 'export PATH=\"${PATH:+$PATH:}/usr/local/cuda/bin\"' | tee /etc/profile.d/cuda.sh > /dev/null",
            "echo 'export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/cuda/lib64\"' | tee -a /etc/profile.d/cuda.sh > /dev/null",
            "echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | tee /etc/modprobe.d/nvprofiling.conf",
            "echo 'options nvidia NVreg_CoherentGPUMemoryMode=driver' | tee /etc/modprobe.d/nvidia-openrm.conf",
            'echo "nvidia_peermem" >> /etc/modules-load.d/nvidia-peermem.conf',
            'echo "options nvidia NVreg_GrdmaPciTopoCheckOverride=1" >> /etc/modprobe.d/nvidia.conf',
            "options nvidia NVreg_CreateImexChannel0=1",
            "Pin-Priority: 1001",
            "dnf install -y --disablerepo='cuda-azl3*' $AL3_GPU_DRIVER_PACKAGES",
        ]:
            with self.subTest(line=line):
                self.assertIn(line, script)


# --------------------------------------------------------------------------- #
# install()
# --------------------------------------------------------------------------- #
class _InstallBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.work = os.path.join(self.root, "work")
        for d in ["etc/yum.repos.d", "etc/apt/preferences.d", "etc/modprobe.d",
                  "etc/modules-load.d", "etc/profile.d", "usr/share/keyrings",
                  "var/nvidia-driver-local-repo-ubuntu2404-580.178.04"]:
            os.makedirs(os.path.join(self.root, d))
        for name, value in [("_ROOT", self.root), ("_WORK_DIR", self.work)]:
            patcher = mock.patch.object(m, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def path(self, p):
        return self.root + p

    def read(self, p):
        with open(self.path(p), encoding="utf-8", newline="") as handle:
            return handle.read()

    def _install(self, env, *, fail_on=None, capture=None, samples_rc=0,
                 pin_ok=True, download_error=None, write_error=None):
        """Run install(env) with everything external mocked.

        fail_on: a command (list) for which exec_program returns 1.
        capture: {first-two-words tuple: (rc, stdout)} overrides for run_capture.
        Returns (rc, calls, mocks); calls is the ordered list of external calls.
        """
        calls = []
        seen_work = []
        outputs = {("curl",): (0, _REPO_TEXT),
                   ("dnf", "list"): (0, _DNF_LIST),
                   ("bash", "-c"): (0, _NVCC_12 if "V100" == env.get("SKU") else _NVCC_13)}
        outputs.update(capture or {})

        def fake_exec(command, op, **kwargs):
            self.assertEqual(op, _OP)
            calls.append(("exec", command))
            if kwargs.get("cwd") is not None:
                seen_work.append((command[0], kwargs["cwd"]))
            return 1 if command == fail_on else 0

        def fake_capture(command, op, **kwargs):
            self.assertEqual(op, _OP)
            calls.append(("capture", command))
            key = tuple(command[:2]) if tuple(command[:2]) in outputs else (command[0],)
            return outputs[key]

        def fake_samples(env_arg):
            self.assertIs(env_arg, env)
            calls.append(_SAMPLES)
            return samples_rc

        def fake_pin(packages, op, env_arg):
            calls.append(("pin", packages))
            return pin_ok

        def fake_download(url, sha256, dest_dir="."):
            calls.append(("download", url, sha256, dest_dir))
            if download_error:
                raise download_error
            return Path(dest_dir) / os.path.basename(url)

        with mock.patch.object(m, "exec_program", side_effect=fake_exec), \
             mock.patch.object(m, "run_capture", side_effect=fake_capture), \
             mock.patch.object(m.install_cuda_samples, "install",
                               side_effect=fake_samples), \
             mock.patch.object(m, "dnf_pin_packages", side_effect=fake_pin), \
             mock.patch.object(m, "download_and_verify", side_effect=fake_download), \
             mock.patch.object(m, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(m.platform, "release", return_value=_KERNEL), \
             mock.patch.object(m, "log_info") as log_info, \
             mock.patch.object(m, "log_warn") as log_warn, \
             mock.patch.object(m, "log_error") as log_error:
            rc = m.install(env)
        mocks = {"record": record, "info": log_info, "warn": log_warn,
                 "error": log_error, "cwd": seen_work}
        return rc, calls, mocks

    @staticmethod
    def recorded(mocks):
        return [c.args for c in mocks["record"].call_args_list]

    @staticmethod
    def messages(log_mock):
        return [c.args[1] for c in log_mock.call_args_list]

    def assert_logged(self, log_mock, fragment):
        self.assertTrue(any(fragment in msg for msg in self.messages(log_mock)),
                        f"{fragment!r} not in {self.messages(log_mock)}")

    def assert_not_logged(self, log_mock, fragment):
        self.assertFalse(any(fragment in msg for msg in self.messages(log_mock)),
                         f"{fragment!r} unexpectedly logged")


_TAIL_FM = [_GDRCOPY, _FABRIC_MANAGER, _PERSISTENCE]


class AzureLinuxTests(_InstallBase):
    def _repo_calls(self, cuda_arch, pmc_arch):
        return [
            ("capture", ["curl", "https://packages.microsoft.com/azurelinux/3.0/prod/"
                                 f"nvidia/{pmc_arch}/config.repo"]),
            ("capture", ["curl", "https://developer.download.nvidia.com/compute/cuda/"
                                 f"repos/azl3/{cuda_arch}/cuda-azl3.repo"]),
        ]

    def test_x86_64_open_driver(self):
        rc, calls, mocks = self._install(_env("azurelinux3.0"))
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [
            *self._repo_calls("x86_64", "x86_64"),
            ("exec", ["dnf", "install", "-y", "--disablerepo=cuda-azl3*", "cuda-open"]),
            ("capture", ["dnf", "list", "installed"]),
            ("pin", ["cuda-open"]),
            ("exec", ["dnf", "install", "-y", "cuda-toolkit-13-0"]),
            _NVCC, _SAMPLES, *_TAIL_FM,
        ])
        self.assertEqual(self.recorded(mocks), [("NVIDIA", "580.126.20-1"),
                                                ("CUDA", "13.0.88")])
        self.assertEqual(self.read("/etc/yum.repos.d/azurelinux-nvidia-prod.repo"), _REPO_TEXT)
        self.assertEqual(self.read("/etc/yum.repos.d/cuda-azl3.repo"), _REPO_TEXT)
        self.assertEqual(self.read("/etc/profile.d/cuda.sh"), _fixture("cuda.sh"))
        self.assertEqual(stat.S_IMODE(os.stat(self.path("/etc/profile.d/cuda.sh")).st_mode),
                         0o644)
        self.assertEqual(self.read("/etc/modules-load.d/nvidia-peermem.conf"),
                         _fixture("nvidia-peermem.conf"))
        self.assertFalse(os.path.exists(self.path("/etc/modprobe.d/nvidia.conf")))
        self.assert_logged(mocks["info"], "installing driver package cuda-open from PMC")

    def test_aarch64_rackscale_uses_hwe_package_and_imex(self):
        env = _env("azurelinux3.0", sku="GB200", arch="aarch64", NVLINK_RACKSCALE="true")
        rc, calls, mocks = self._install(env)
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [
            *self._repo_calls("sbsa", "aarch64"),
            ("exec", ["dnf", "install", "-y", "--disablerepo=cuda-azl3*", "cuda-open-hwe"]),
            ("capture", ["dnf", "list", "installed"]),
            ("pin", ["cuda-open-hwe"]),
            ("exec", ["dnf", "install", "-y", "cuda-toolkit-13-0"]),
            _NVCC, _SAMPLES, _GDRCOPY,
            ("exec", ["dnf", "install", "-y", "nvidia-imex-580.159.04"]),
            ("exec", ["dracut", "--force"]),
            ("exec", ["systemctl", "enable", "nvidia-imex.service"]),
            _PERSISTENCE,
        ])
        self.assertEqual(self.recorded(mocks), [("NVIDIA", "580.159.04-1"),
                                                ("CUDA", "13.0.88"),
                                                ("IMEX", "580.159.04")])
        self.assertEqual(self.read("/etc/modprobe.d/nvprofiling.conf"),
                         _fixture("nvprofiling.conf"))
        self.assertEqual(self.read("/etc/modprobe.d/nvidia-openrm.conf"),
                         _fixture("nvidia-openrm.conf"))
        self.assertEqual(self.read("/etc/modprobe.d/nvidia.conf"),
                         m._IMEX_CHANNEL_LINE + "\n")
        self.assert_logged(mocks["info"], "instead of fabric manager")
        self.assert_not_logged(mocks["warn"], "nvidia-imex without a version")

    def test_v100_uses_proprietary_cuda_package(self):
        rc, calls, mocks = self._install(_env("azurelinux3.0", sku="V100"))
        self.assertEqual(rc, 0)
        self.assertIn(("exec", ["dnf", "install", "-y", "--disablerepo=cuda-azl3*", "cuda"]),
                      calls)
        self.assertIn(("pin", ["cuda"]), calls)
        self.assertIn(("exec", ["dnf", "install", "-y", "cuda-toolkit-12-9"]), calls)
        self.assertEqual(self.recorded(mocks)[0], ("NVIDIA", "575.57.08-1"))

    def test_repo_file_without_section_warns(self):
        rc, _, mocks = self._install(_env("azurelinux3.0"),
                                     capture={("curl",): (0, "<html>404</html>\n")})
        self.assertEqual(rc, 0)
        self.assert_logged(mocks["warn"], "curl runs without -f")

    def test_good_repo_file_does_not_warn(self):
        _, _, mocks = self._install(_env("azurelinux3.0"))
        self.assert_not_logged(mocks["warn"], "curl runs without -f")

    def test_curl_failure_returns_3(self):
        rc, calls, mocks = self._install(_env("azurelinux3.0"),
                                         capture={("curl",): (6, "")})
        self.assertEqual(rc, 3)
        self.assertEqual(len(calls), 1)
        self.assertFalse(os.path.exists(self.path("/etc/yum.repos.d/cuda-azl3.repo")))

    def test_unparsable_driver_version_returns_3(self):
        bad = "cuda-open.x86_64  580.126.20-1.azl3  @nv\n"
        rc, calls, mocks = self._install(_env("azurelinux3.0"),
                                         capture={("dnf", "list"): (0, bad)})
        self.assertEqual(rc, 3)
        self.assertNotIn(("pin", ["cuda-open"]), calls)
        self.assertEqual(self.recorded(mocks), [])
        self.assert_logged(mocks["error"], "could not read the installed cuda-open version")

    def test_failing_dnf_list_returns_3(self):
        rc, _, mocks = self._install(_env("azurelinux3.0"),
                                     capture={("dnf", "list"): (1, _DNF_LIST)})
        self.assertEqual(rc, 3)
        self.assertEqual(self.recorded(mocks), [])

    def test_pin_failure_returns_3(self):
        rc, calls, mocks = self._install(_env("azurelinux3.0"), pin_ok=False)
        self.assertEqual(rc, 3)
        self.assertEqual(calls[-1], ("pin", ["cuda-open"]))
        self.assertEqual(self.recorded(mocks), [])

    def test_rackscale_x86_64_without_imex_version_fails_up_front(self):
        rc, calls, mocks = self._install(_env("azurelinux3.0", NVLINK_RACKSCALE="true"))
        self.assertEqual(rc, 3)
        self.assertEqual(calls, [])
        self.assert_logged(mocks["error"], "nvidia imex.version")


class UbuntuTests(_InstallBase):
    def _driver_calls(self, arch="x86_64", distribution="ubuntu2404"):
        return [
            ("exec", ["wget", "https://developer.download.nvidia.com/compute/cuda/repos/"
                              f"{distribution}/{arch}/cuda-keyring_1.1-1_all.deb"]),
            ("exec", ["dpkg", "-i", "./cuda-keyring_1.1-1_all.deb"]),
            ("exec", ["apt-get", "update"]),
        ]

    def test_open_driver(self):
        open(self.path("/etc/modprobe.d/nvidia-graphics-drivers-kms.conf"), "w").close()
        rc, calls, mocks = self._install(_env())
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [
            *self._driver_calls(),
            ("exec", ["apt", "install", "nvidia-driver-pinning-580.178.04", "-y"]),
            ("exec", ["apt", "install", "nvidia-open", "-y"]),
            ("exec", ["apt", "install", "-y", "cuda-toolkit-13-0"]),
            _NVCC, _SAMPLES, *_TAIL_FM,
        ])
        self.assertEqual(self.recorded(mocks), [("NVIDIA", "580.178.04"),
                                                ("CUDA", "13.0.88")])
        self.assertFalse(os.path.exists(
            self.path("/etc/modprobe.d/nvidia-graphics-drivers-kms.conf")))
        self.assertEqual(self.read("/etc/modprobe.d/nvprofiling.conf"),
                         _fixture("nvprofiling.conf"))
        self.assertEqual(self.read("/etc/profile.d/cuda.sh"), _fixture("cuda.sh"))
        self.assertFalse(os.path.exists(self.path("/etc/apt/preferences.d/00-nvidia-prefer")))
        self.assertFalse(os.path.exists(self.path("/etc/modprobe.d/nvidia-openrm.conf")))
        # The keyring download and the nested scripts run from the work dir.
        self.assertEqual(mocks["cwd"], [("wget", self.work), ("dpkg", self.work),
                                        ("apt-get", self.work),
                                        (f"{_COMPONENT_DIR}/install_gdrcopy.sh", self.work),
                                        (f"{_COMPONENT_DIR}/install_nvidia_fabric_manager.sh",
                                         self.work),
                                        (f"{_COMPONENT_DIR}/configure_nvidia_persistence.sh",
                                         self.work)])
        self.assert_logged(mocks["info"], "Not the MRC network")
        self.assert_logged(mocks["info"], "Running nested script install_gdrcopy.sh")
        self.assert_logged(mocks["info"], "fabric manager (NVSwitch) instead of IMEX")
        self.assert_logged(mocks["info"], "Recorded NVIDIA version 580.178.04")

    def test_v100_installs_cuda_drivers(self):
        rc, calls, mocks = self._install(_env(sku="V100"))
        self.assertEqual(rc, 0)
        self.assertEqual(calls[3:6], [
            ("exec", ["apt", "install", "nvidia-driver-pinning-580.178.04", "-y"]),
            ("exec", ["apt", "install", "cuda-drivers", "-y"]),
            ("exec", ["apt", "install", "-y", "cuda-toolkit-12-9"]),
        ])
        self.assertEqual(self.recorded(mocks)[1], ("CUDA", "12.9.86"))

    def test_rackscale_aarch64(self):
        env = _env(sku="GB200", arch="aarch64", NVLINK_RACKSCALE="TRUE")
        rc, calls, mocks = self._install(env)
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [
            *self._driver_calls(arch="sbsa"),
            ("exec", ["apt", "install", "nvidia-driver-pinning-580.126.20", "-y"]),
            ("exec", ["apt", "install", "nvidia-dkms-580-open", "nvidia-driver-580-open",
                      "nvidia-modprobe", "-y"]),
            ("exec", ["apt", "install", "-y", "cuda-toolkit-13-0"]),
            _NVCC, _SAMPLES, _GDRCOPY,
            ("exec", ["apt-get", "install", "nvidia-imex", "-y"]),
            ("exec", ["update-initramfs", "-u", "-k", "all"]),
            ("exec", ["systemctl", "enable", "nvidia-imex.service"]),
            _PERSISTENCE,
        ])
        self.assertEqual(self.recorded(mocks), [("NVIDIA", "580.126.20"),
                                                ("CUDA", "13.0.88"),
                                                ("IMEX", "580.126.20-1")])
        self.assertEqual(self.read("/etc/modprobe.d/nvidia-openrm.conf"),
                         _fixture("nvidia-openrm.conf"))
        self.assertEqual(self.read("/etc/modprobe.d/nvidia.conf"),
                         m._IMEX_CHANNEL_LINE + "\n")
        self.assert_logged(mocks["warn"], "nvidia-imex without a version")
        self.assert_logged(mocks["info"], "Enabled nvidia-imex.service")

    def test_rackscale_without_major_version_fails_up_front(self):
        versions = json.loads(json.dumps(_VERSIONS))
        del versions["nvidia"]["ubuntu24.04"]["aarch64"]["driver"]["major_version"]
        env = _env(sku="GB200", arch="aarch64", NVLINK_RACKSCALE="true", versions=versions)
        rc, calls, mocks = self._install(env)
        self.assertEqual(rc, 3)
        self.assertEqual(calls, [])
        self.assert_logged(mocks["error"], "nvidia driver.major_version")

    def test_baremetal_3p_rackscale_nvidia_conf(self):
        env = _env(sku="GB200", arch="aarch64", node_type="baremetal_3p",
                   NVLINK_RACKSCALE="true")
        rc, calls, mocks = self._install(env)
        self.assertEqual(rc, 0)
        self.assertEqual(self.read("/etc/modprobe.d/nvidia.conf"),
                         _fixture("nvidia.conf.baremetal-rackscale"))
        self.assertIn(("exec", ["apt", "install", "nvidia-driver-pinning-580.167.08", "-y"]),
                      calls)
        self.assertEqual(self.recorded(mocks)[-1], ("IMEX", "580.167.08-1"))
        self.assert_not_logged(mocks["info"], "MRC network (baremetal_1p)")

    def test_mrc_local_repo(self):
        repo_dir = "/var/nvidia-driver-local-repo-ubuntu2404-580.178.04"
        for name in ["nvidia-driver-local-A1B2-keyring.gpg", "unrelated.gpg"]:
            with open(self.path(f"{repo_dir}/{name}"), "w") as handle:
                handle.write(name)
        rc, calls, mocks = self._install(_env(node_type="baremetal_1p"))
        self.assertEqual(rc, 0)
        self.assertEqual(calls[:8], [
            *self._driver_calls(),
            ("exec", ["dpkg", "-i", f"{_TOP_DIR}/internal_bits/{_REPO_FILE}"]),
            ("exec", ["apt", "update"]),
            ("exec", ["apt", "install", "nvidia-driver-pinning-580.178.04", "-y"]),
            ("exec", ["apt", "install", "nvidia-open", "-y"]),
            ("exec", ["apt", "install", "-y", "cuda-toolkit-13-0"]),
        ])
        self.assertEqual(sorted(os.listdir(self.path("/usr/share/keyrings"))),
                         ["nvidia-driver-local-A1B2-keyring.gpg"])
        self.assertEqual(self.read("/etc/apt/preferences.d/00-nvidia-prefer"),
                         _fixture("00-nvidia-prefer.txt"))
        self.assertEqual(self.read("/etc/modprobe.d/nvidia.conf"), m._GRDMA_LINE + "\n")
        self.assert_logged(mocks["info"], "MRC network (baremetal_1p)")

    def test_mrc_without_keyring_returns_3(self):
        rc, calls, mocks = self._install(_env(node_type="baremetal_1p"))
        self.assertEqual(rc, 3)
        self.assertEqual(calls[-1],
                         ("exec", ["dpkg", "-i", f"{_TOP_DIR}/internal_bits/{_REPO_FILE}"]))
        self.assert_logged(mocks["error"], "no keyring matches")

    def test_mrc_without_repo_file_fails_up_front(self):
        versions = json.loads(json.dumps(_VERSIONS))
        del versions["nvidia"]["ubuntu24.04"]["x86_64"]["baremetal_1p"]
        rc, calls, mocks = self._install(_env(node_type="baremetal_1p", versions=versions))
        self.assertEqual(rc, 3)
        self.assertEqual(calls, [])
        self.assert_logged(mocks["error"], "nvidia driver.repo_file")

    def test_aks_host_skips_toolkit_and_samples(self):
        rc, calls, mocks = self._install(_env(node_type="azure_vm_akshost"))
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [
            *self._driver_calls(),
            ("exec", ["apt", "install", "nvidia-driver-pinning-580.178.04", "-y"]),
            ("exec", ["apt", "install", "nvidia-open", "-y"]),
            *_TAIL_FM,
        ])
        self.assertEqual(self.recorded(mocks), [("NVIDIA", "580.178.04")])
        self.assertFalse(os.path.exists(self.path("/etc/profile.d/cuda.sh")))
        self.assert_logged(mocks["info"], "AKS host (azure_vm_akshost): skipping the CUDA")

    def test_missing_driver_version_fails_up_front(self):
        versions = json.loads(json.dumps(_VERSIONS))
        del versions["nvidia"]["ubuntu24.04"]["x86_64"]["driver"]
        rc, calls, mocks = self._install(_env(versions=versions))
        self.assertEqual(rc, 3)
        self.assertEqual(calls, [])
        self.assert_logged(mocks["error"], "nvidia driver.version")

    def test_driver_package_failure_stops(self):
        fail = ["apt", "install", "nvidia-open", "-y"]
        rc, calls, mocks = self._install(_env(), fail_on=fail)
        self.assertEqual(rc, 3)
        self.assertEqual(calls[-1], ("exec", fail))
        self.assertEqual(self.recorded(mocks), [])
        self.assertFalse(os.path.exists(self.path("/etc/modprobe.d/nvprofiling.conf")))


class RhelTests(_InstallBase):
    def _expected(self, module_type, toolkit, excludes):
        return [
            ("download", "https://us.download.nvidia.com/tesla/580.126.20/"
                         "NVIDIA-Linux-x86_64-580.126.20.run", "abc123", self.work),
            ("exec", ["bash", "NVIDIA-Linux-x86_64-580.126.20.run", "--silent", "--dkms",
                      f"--kernel-module-type={module_type}"]),
            ("exec", ["dkms", "install", "--no-depmod", "-m", "nvidia", "-v", "580.126.20",
                      "-k", _KERNEL, "--force"]),
            ("exec", ["dnf", "config-manager", "--add-repo",
                      "https://developer.download.nvidia.com/compute/cuda/repos/rhel9/"
                      "x86_64/cuda-rhel9.repo"]),
            ("exec", ["dnf", "config-manager", "--save",
                      f"--setopt=cuda-rhel9-x86_64.excludepkgs={excludes}"]),
            ("exec", ["dnf", "clean", "expire-cache"]),
            ("exec", ["dnf", "install", "-y", toolkit]),
            _NVCC, _SAMPLES, *_TAIL_FM,
        ]

    def test_open_driver(self):
        rc, calls, mocks = self._install(_env("almalinux9.8"))
        self.assertEqual(rc, 0)
        self.assertEqual(calls, self._expected("open", "cuda-toolkit-13-0",
                                               "mft* kernel-mft*"))
        self.assertEqual(self.recorded(mocks), [("NVIDIA", "580.126.20"),
                                                ("CUDA", "13.0.88")])
        self.assertIn(("bash", self.work), mocks["cwd"])
        self.assert_not_logged(mocks["warn"], "hard-coded to x86_64")

    def test_v100_proprietary_and_cuda12_excludes(self):
        rc, calls, mocks = self._install(_env("almalinux9.8", sku="V100"))
        self.assertEqual(rc, 0)
        self.assertEqual(calls, self._expected("proprietary", "cuda-toolkit-12-9",
                                               "mft* kernel-mft* cccl-*"))

    def test_other_distro_skips_dkms(self):
        rc, calls, mocks = self._install(_env("sles15"))
        self.assertEqual(rc, 0)
        self.assertFalse(any(c[0] == "exec" and c[1][0] == "dkms" for c in calls))
        self.assert_logged(mocks["info"], "no dkms install")

    def test_non_x86_64_warns(self):
        versions = json.loads(json.dumps(_VERSIONS))
        versions["nvidia"]["almalinux9.8"]["aarch64"] = \
            versions["nvidia"]["almalinux9.8"]["x86_64"]
        versions["cuda"]["almalinux9.8"]["aarch64"] = versions["cuda"]["almalinux9.8"]["x86_64"]
        rc, _, mocks = self._install(_env("almalinux9.8", arch="aarch64", versions=versions))
        self.assertEqual(rc, 0)
        self.assert_logged(mocks["warn"], "hard-coded to x86_64")

    def test_download_failure_returns_3(self):
        rc, calls, mocks = self._install(_env("almalinux9.8"),
                                         download_error=OSError("boom"))
        self.assertEqual(rc, 3)
        self.assertEqual(len(calls), 1)
        self.assert_logged(mocks["error"], "download/verify of the driver failed")

    def test_missing_sha256_fails_up_front(self):
        versions = json.loads(json.dumps(_VERSIONS))
        del versions["nvidia"]["almalinux9.8"]["x86_64"]["driver"]["sha256"]
        rc, calls, mocks = self._install(_env("almalinux9.8", versions=versions))
        self.assertEqual(rc, 3)
        self.assertEqual(calls, [])

    def test_rackscale_is_unsupported(self):
        rc, calls, mocks = self._install(_env("almalinux9.8", NVLINK_RACKSCALE="true"))
        self.assertEqual(rc, 3)
        self.assertEqual(calls, [])
        self.assert_logged(mocks["error"], "Unsupported distribution for nvidia-imex")

    def test_aks_host_needs_no_cuda_config(self):
        versions = json.loads(json.dumps(_VERSIONS))
        del versions["cuda"]["almalinux9.8"]
        rc, calls, _ = self._install(_env("almalinux9.8", node_type="azure_vm_akshost",
                                          versions=versions))
        self.assertEqual(rc, 0)
        self.assertNotIn(_SAMPLES, calls)


class CommonTests(_InstallBase):
    def test_start_line_logs_inputs(self):
        _, _, mocks = self._install(_env(sku="GB200", arch="aarch64",
                                         NVLINK_RACKSCALE="true"))
        first = mocks["info"].call_args_list[0]
        self.assertEqual(first.args[1], "Installing the NVIDIA GPU driver")
        for key, value in [("distro", "ubuntu24.04"), ("arch", "aarch64"),
                           ("sku", "GB200"), ("node_type", "azure_vm_regular"),
                           ("rackscale", True), ("mrc_network", False),
                           ("cuda_toolkit", True), ("driver_version", "580.126.20"),
                           ("imex_version", "580.126.20-1")]:
            self.assertEqual(first.kwargs[key], value)
        last = mocks["info"].call_args_list[-1]
        self.assertEqual(last.args[1], "NVIDIA GPU driver installed")
        self.assertEqual(last.kwargs["imex_version"], "580.126.20-1")

    def test_baremetal_appends_grdma_option(self):
        rc, _, mocks = self._install(_env("almalinux9.8", node_type="baremetal_3p"))
        self.assertEqual(rc, 0)
        self.assertEqual(self.read("/etc/modprobe.d/nvidia.conf"), m._GRDMA_LINE + "\n")
        self.assert_logged(mocks["info"], "baremetal_3p: overriding the GPUDirect RDMA")

    def test_rerun_appends_again_and_warns(self):
        with open(self.path("/etc/modules-load.d/nvidia-peermem.conf"), "w") as handle:
            handle.write("nvidia_peermem\n")
        rc, _, mocks = self._install(_env())
        self.assertEqual(rc, 0)
        self.assertEqual(self.read("/etc/modules-load.d/nvidia-peermem.conf"),
                         "nvidia_peermem\nnvidia_peermem\n")
        self.assert_logged(mocks["warn"], "already contains 'nvidia_peermem'")

    def test_first_run_does_not_warn_about_appends(self):
        _, _, mocks = self._install(_env(node_type="baremetal_3p"))
        self.assert_not_logged(mocks["warn"], "already contains")

    def test_build_dir_cleanup_difference_is_warned(self):
        _, _, mocks = self._install(_env())
        self.assert_logged(mocks["warn"], "Not running bash's final cleanup")

    def test_work_dir_is_removed_on_success_and_failure(self):
        rc, _, _ = self._install(_env())
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(self.work))
        rc, _, _ = self._install(_env(), fail_on=["apt-get", "update"])
        self.assertEqual(rc, 3)
        self.assertFalse(os.path.exists(self.work))

    def test_nested_script_failure_stops(self):
        rc, calls, mocks = self._install(_env(), fail_on=_GDRCOPY[1])
        self.assertEqual(rc, 3)
        self.assertEqual(calls[-1], _GDRCOPY)
        self.assert_logged(mocks["error"], "nested script install_gdrcopy.sh failed")

    def test_fabric_manager_failure_stops(self):
        rc, calls, _ = self._install(_env(), fail_on=_FABRIC_MANAGER[1])
        self.assertEqual(rc, 3)
        self.assertEqual(calls[-1], _FABRIC_MANAGER)

    def test_samples_failure_stops(self):
        rc, calls, mocks = self._install(_env(), samples_rc=3)
        self.assertEqual(rc, 3)
        self.assertEqual(calls[-1], _SAMPLES)
        self.assert_logged(mocks["error"], "CUDA samples installation failed")

    def test_nvcc_failure_returns_3(self):
        rc, calls, mocks = self._install(_env(), capture={("bash", "-c"): (127, "")})
        self.assertEqual(rc, 3)
        self.assertEqual(calls[-1], _NVCC)
        self.assertEqual(self.recorded(mocks), [("NVIDIA", "580.178.04")])

    def test_imex_failure_stops(self):
        env = _env(sku="GB200", arch="aarch64", NVLINK_RACKSCALE="true")
        fail = ["apt-get", "install", "nvidia-imex", "-y"]
        rc, calls, mocks = self._install(env, fail_on=fail)
        self.assertEqual(rc, 3)
        self.assertEqual(calls[-1], ("exec", fail))
        self.assertNotIn(("IMEX", "580.126.20-1"), self.recorded(mocks))

    def test_initramfs_failure_stops(self):
        env = _env(sku="GB200", arch="aarch64", NVLINK_RACKSCALE="true")
        fail = ["update-initramfs", "-u", "-k", "all"]
        rc, calls, _ = self._install(env, fail_on=fail)
        self.assertEqual(rc, 3)
        self.assertEqual(calls[-1], ("exec", fail))

    def test_version_write_error_returns_3(self):
        rc, calls, mocks = self._install(_env(), write_error=PermissionError("denied"))
        self.assertEqual(rc, 3)
        self.assertEqual(calls[-1],
                         ("exec", ["apt", "install", "nvidia-open", "-y"]))
        self.assert_logged(mocks["error"], "denied")

    def test_file_write_error_returns_3(self):
        os.rmdir(self.path("/etc/profile.d"))
        rc, calls, mocks = self._install(_env())
        self.assertEqual(rc, 3)
        self.assertEqual(calls[-1], ("exec", ["apt", "install", "-y", "cuda-toolkit-13-0"]))
        self.assert_logged(mocks["error"], "could not write")


class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        for gpu in ["A100", "GB200", "V100"]:
            with self.subTest(gpu=gpu):
                cfg = BuildConfig(vendor="NVidia", gpu=gpu, os="Ubuntu24",
                                  fips=False, spec_path=None)
                steps = [s for s in build_plan(cfg) if s.op == _OP]
                self.assertEqual(len(steps), 1)
                self.assertIs(steps[0].action, m.install)
                self.assertEqual(steps[0].script, "install_nvidiagpudriver.sh")


if __name__ == "__main__":
    unittest.main()
