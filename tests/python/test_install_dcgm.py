"""Unit tests for components/python/install_dcgm.py.

  * Pure helpers are tested directly with realistic inputs. dcgmi_version(),
    exact_version_deps() and unit_exec_start() were also compared against the
    real awk/sed pipelines in WSL; the whole resolve_exact_version_deps walk
    against the bash function with a fake apt-cache; and the C probe source
    was compiled and run there.
  * Parity tests check the C probe source, the Azure Linux marker spec and
    the nv-hostengine drop-in against the heredocs in
    components/install_dcgm.sh.
  * install() is tested with exec_program, run_capture, config_for,
    write_component_version and the `screen` lookup mocked. Temporary
    directories (probe, marker rpmbuild), the unit file and the drop-in
    directory live inside a test-owned temporary directory.

Run from the repo root:
    python3 -m unittest tests.python.test_install_dcgm -v
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_dcgm
from utils.build_config import BuildConfig, build_plan

_OP = "install-dcgm"
_REPO = Path(__file__).resolve().parents[2]
_BASH = (_REPO / "components" / "install_dcgm.sh").read_text(encoding="utf-8")

_UBUNTU_DCGM = "1:4.7.0-2"
_AZL_DCGM = "4.7.0"
_DCGMI_OUT = "dcgmi  version: 4.4.1\n"
_SCREEN = "/usr/bin/screen"
_HOSTENGINE = "/usr/bin/nv-hostengine -n --service-account nvidia-dcgm"
_UNIT = f"[Unit]\nDescription=DCGM\n\n[Service]\nExecStart={_HOSTENGINE}\n"
_MARKER = "hpcx-provides-openmpi"

_CUDA_13 = {"driver": {"version": "13.0", "distribution": "ubuntu2404"}}
_CUDA_12 = {"driver": {"version": "12.9", "distribution": "ubuntu2404"}}

_DCGM = "datacenter-gpu-manager-4"
# Exact-version dependencies the fake apt-cache reports.
_APT_DEPS = {
    f"{_DCGM}-cuda13": f"Depends: {_DCGM}-core (= {{v}}), libc6 (>= 2.34)\n",
    f"{_DCGM}-core": "Pre-Depends: libdcgm4 (= {v})\nDepends: dcgmi (= {v}) | x (>= 1)\n",
    f"{_DCGM}-cuda12": f"Depends: {_DCGM}-core (= {{v}})\n",
}

_TAIL = [
    ["systemctl", "daemon-reload"],
    ["systemctl", "--now", "enable", "nvidia-dcgm"],
    ["systemctl", "restart", "nvidia-dcgm"],
    ["systemctl", "is-active", "--quiet", "nvidia-dcgm"],
]
_RHEL_INSTALL = ["dnf", "install", "--assumeyes", "--setopt=install_weak_deps=True"]
_RHEL_MULTINODE = ["dnf", "install", "--assumeyes"]


def _pinned(names, version=_UBUNTU_DCGM):
    return [f"{name}={version}" for name in names]


def _ubuntu13():
    return ["apt-get", "install", "-y", *_pinned(
        [f"{_DCGM}-cuda13", f"{_DCGM}-core", f"{_DCGM}-proprietary",
         f"{_DCGM}-proprietary-cuda13", f"{_DCGM}-multinode", f"{_DCGM}-multinode-cuda13",
         "dcgmi", "libdcgm4"])]


def _ubuntu12_extra():
    return ["apt-get", "install", "-y", *_pinned(
        [f"{_DCGM}-cuda12", f"{_DCGM}-proprietary-cuda12", f"{_DCGM}-multinode-cuda12",
         f"{_DCGM}-core", "dcgmi", "libdcgm4"])]


def _env(distribution="ubuntu24.04", **extra):
    env = {"DISTRIBUTION": distribution, "GPU": "NVIDIA", "SKU": "H100",
           "ARCHITECTURE": "x86_64", "TARGET_NODE_TYPE": "azure_vm_regular",
           "PATH": "/usr/bin"}
    env.update(extra)
    return env


def _heredoc(opening: str) -> str:
    start = _BASH.index(opening + "\n") + len(opening) + 1
    return _BASH[start:_BASH.index("\nEOF\n", start) + 1]


# --------------------------------------------------------------------------- #
# Pure helpers and parity with the bash heredocs.
# --------------------------------------------------------------------------- #
class HelperTests(unittest.TestCase):
    def test_parse_cuda_driver_version(self):
        self.assertEqual(install_dcgm.parse_cuda_driver_version("13000\n"), 13000)
        self.assertEqual(install_dcgm.parse_cuda_driver_version("12090"), 12090)
        for bad in ["", "\n", "0\n", "-1", "13.0", "abc", "13000 extra"]:
            with self.subTest(bad=bad):
                self.assertIsNone(install_dcgm.parse_cuda_driver_version(bad))

    def test_cuda_major(self):
        self.assertEqual(install_dcgm.cuda_major(13000), 13)
        self.assertEqual(install_dcgm.cuda_major(13020), 13)
        self.assertEqual(install_dcgm.cuda_major(12090), 12)
        self.assertEqual(install_dcgm.cuda_major(9020), 9)

    def test_sku_cuda_major(self):
        self.assertEqual(install_dcgm.sku_cuda_major(_CUDA_13), 13)
        self.assertEqual(install_dcgm.sku_cuda_major(_CUDA_12), 12)
        self.assertEqual(install_dcgm.sku_cuda_major({"driver": {"version": "13"}}), 13)
        for bad in [None, {}, {"driver": {}}, {"driver": "13.0"},
                    {"driver": {"version": "null"}}, {"driver": {"version": ""}}]:
            with self.subTest(bad=bad):
                self.assertIsNone(install_dcgm.sku_cuda_major(bad))

    def test_dcgmi_version(self):
        self.assertEqual(install_dcgm.dcgmi_version(_DCGMI_OUT), "4.4.1")
        self.assertEqual(install_dcgm.dcgmi_version("dcgmi  version: 4.4.1"), "4.4.1")
        # A shorter line first prints "" in awk, which word splitting drops.
        self.assertEqual(install_dcgm.dcgmi_version("banner line\n" + _DCGMI_OUT),
                         "4.4.1")
        self.assertIsNone(install_dcgm.dcgmi_version(""))
        self.assertIsNone(install_dcgm.dcgmi_version("a b\n"))

    def test_package_lists(self):
        self.assertEqual(install_dcgm.ubuntu_packages(13), [
            f"{_DCGM}-cuda13", f"{_DCGM}-core", f"{_DCGM}-proprietary",
            f"{_DCGM}-proprietary-cuda13", f"{_DCGM}-multinode", f"{_DCGM}-multinode-cuda13"])
        self.assertEqual(install_dcgm.ubuntu_extra_packages(12), [
            f"{_DCGM}-cuda12", f"{_DCGM}-proprietary-cuda12", f"{_DCGM}-multinode-cuda12"])
        self.assertEqual(install_dcgm.azure_linux_packages(13, "4.7.0"), [
            f"{_DCGM}-cuda13-4.7.0", f"{_DCGM}-multinode-4.7.0",
            f"{_DCGM}-multinode-cuda13-4.7.0"])

    def test_exact_version_deps_like_awk(self):
        show = ("Package: core\nVersion: 1:4.7.0-2\n"
                "Pre-Depends: libdcgm4 (= 1:4.7.0-2)\n"
                "Depends: dcgmi (= 1:4.7.0-2), nv-hostengine:amd64 (= 1:4.7.0-2) | "
                "foo (= 1:4.7.0-2), libc6 (>= 2.34), bar (= 1:4.6.0-1),\t baz (= 1:4.7.0-2)\t, "
                "qux(= 1:4.7.0-2), sp ace (= 1:4.7.0-2)\n"
                "Recommends: rec (= 1:4.7.0-2)\nDepends:zz (= 1:4.7.0-2)\n")
        self.assertEqual(install_dcgm.exact_version_deps(show, "1:4.7.0-2"),
                         ["baz", "dcgmi", "foo", "libdcgm4", "nv-hostengine", "zz"])
        self.assertEqual(install_dcgm.exact_version_deps("", "1"), [])

    def test_unit_exec_start_like_sed_tail(self):
        unit = ("[Service]\nExecStart=/usr/bin/old\nExecStartPre=/bin/x\n"
                f"ExecStart={_HOSTENGINE}\n ExecStart=/no\n")
        self.assertEqual(install_dcgm.unit_exec_start(unit), _HOSTENGINE)
        self.assertEqual(install_dcgm.unit_exec_start("[Service]\n"), "")

    def test_probe_source_matches_the_bash_heredoc(self):
        # The text between the `cat > ... <<'EOF'` line and the line "EOF".
        match = re.search(r"<<'EOF'\n(.*?\n)EOF\n", _BASH, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertEqual(install_dcgm._CUDA_PROBE_SOURCE, match.group(1))

    def test_marker_spec_matches_the_bash_heredoc(self):
        text = _heredoc('        cat > "${spec_file}" <<EOF')
        self.assertEqual(install_dcgm._AZURE_LINUX_MARKER_SPEC,
                         text.replace("${hpcx_provides_openmpi_marker}", _MARKER))

    def test_dropin_matches_the_bash_heredoc(self):
        text = _heredoc("cat > /etc/systemd/system/nvidia-dcgm.service.d/"
                        "99-mndiag-hpcx.conf <<EOF")
        expected = (text.replace("$(command -v screen)", _SCREEN)
                    .replace("\\$\\$", "$$")
                    .replace("${DCGM_HOSTENGINE_CMD}", _HOSTENGINE))
        self.assertEqual(install_dcgm.hostengine_dropin(_SCREEN, _HOSTENGINE), expected)
        self.assertIn('DCGM_MNDIAG_MPIRUN_PATH="$$(command -v mpirun)"', expected)


# --------------------------------------------------------------------------- #
# install(): external effects mocked.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.scratch = root / "scratch"
        self.scratch.mkdir()
        self.dropin_dir = root / "nvidia-dcgm.service.d"
        self.unit = root / "nvidia-dcgm.service"
        for name, value in (("_PROBE_PARENT_DIR", str(self.scratch)),
                            ("_DROPIN_DIR", str(self.dropin_dir))):
            patcher = mock.patch.object(install_dcgm, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _install(self, env, *, probe=(0, "13000\n"), dcgmi=(0, _DCGMI_OUT),
                 cuda_cfg=_CUDA_13, dcgm_cfg=None, fail_on=None, write_error=None,
                 apt_missing=(), dpkg_query=(0, "1:4.8.0-1"), marker_installed=False,
                 rpm_built=True, screen=True, screen_after_install=True,
                 unit_text=_UNIT, show=None):
        """Run install(env) with everything external mocked.

        probe: (rc, stdout) of the compiled probe; dcgmi: of `dcgmi --version`.
        fail_on: a command (list) for which exec_program returns 1.
        Returns (rc, mocks); mocks["probe_source"] is the C file's text as it
        was on disk when gcc ran.
        """
        if dcgm_cfg is None:
            dcgm_cfg = {"version": _UBUNTU_DCGM}
        if unit_text is not None:
            self.unit.write_text(unit_text)
        elif self.unit.exists():
            self.unit.unlink()
        show = show if show is not None else (0, f"{self.unit}\n")
        seen = {"screen": screen}

        def fake_exec(command, op, **kwargs):
            if command[0] == "gcc":
                seen["probe_source"] = Path(command[1]).read_text(encoding="utf-8")
                seen["probe_dir"] = os.path.dirname(command[1])
            if command == fail_on:
                return 1
            if command[:2] == ["rpm", "-q"]:
                return 0 if marker_installed else 1
            if command[0] == "rpmbuild":
                topdir = command[2].split(" ", 1)[1]
                seen["rpm_topdir"] = topdir
                seen["spec"] = Path(command[4]).read_text()
                if rpm_built:
                    rpms = Path(topdir, "RPMS", "noarch")
                    rpms.mkdir(parents=True)
                    (rpms / f"{_MARKER}-1.0-1.noarch.rpm").touch()
            if command[-1] == "screen":
                seen["screen"] = screen_after_install
            return 0

        def fake_capture(command, op, **kwargs):
            if command[0] == "dcgmi":
                return dcgmi
            if command[:2] == ["apt-cache", "show"]:
                package, version = command[2].split("=", 1)
                if package in apt_missing:
                    return (100, "")
                deps = _APT_DEPS.get(package, "").format(v=version)
                return (0, f"Package: {package}\nVersion: {version}\n{deps}")
            if command[:2] == ["systemctl", "show"]:
                return show
            if command[0] == "dpkg-query":
                return dpkg_query
            return probe

        def fake_config(component, env):
            return {"cuda": cuda_cfg, "dcgm": dcgm_cfg}[component]

        with mock.patch.object(install_dcgm, "exec_program",
                               side_effect=fake_exec) as exec_program, \
             mock.patch.object(install_dcgm, "run_capture",
                               side_effect=fake_capture) as run_capture, \
             mock.patch.object(install_dcgm, "config_for",
                               side_effect=fake_config) as config_for, \
             mock.patch.object(install_dcgm, "_command_path",
                               side_effect=lambda name, env: _SCREEN if seen["screen"]
                               else None), \
             mock.patch.object(install_dcgm, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(install_dcgm, "log_info") as log_info, \
             mock.patch.object(install_dcgm, "log_warn") as log_warn, \
             mock.patch.object(install_dcgm, "log_error") as log_error:
            rc = install_dcgm.install(env)
        return rc, {"exec": exec_program, "capture": run_capture,
                    "config_for": config_for, "record": record,
                    "info": log_info, "warn": log_warn, "error": log_error, **seen}

    @staticmethod
    def _commands(mocks):
        """exec_program commands, minus the gcc compile of the probe."""
        return [c.args[0] for c in mocks["exec"].call_args_list
                if c.args[0][0] != "gcc"]

    @staticmethod
    def _captures(mocks, first):
        return [c.args[0] for c in mocks["capture"].call_args_list if c.args[0][0] == first]

    def _dropin(self):
        return (self.dropin_dir / "99-mndiag-hpcx.conf").read_text()

    def _assert_probe_ran_and_was_cleaned_up(self, m):
        gcc = m["exec"].call_args_list[0].args[0]
        probe_dir = m["probe_dir"]
        self.assertEqual(gcc, ["gcc", f"{probe_dir}/cuda_driver_api_version.c",
                               "-ldl", "-o", f"{probe_dir}/cuda_driver_api_version"])
        self.assertEqual(m["capture"].call_args_list[0].args[0],
                         [f"{probe_dir}/cuda_driver_api_version"])
        self.assertEqual(m["probe_source"], install_dcgm._CUDA_PROBE_SOURCE)
        self.assertTrue(probe_dir.startswith(str(self.scratch)))
        self.assertEqual(os.listdir(self.scratch), [])

    # ---- Ubuntu ---------------------------------------------------------------
    def test_ubuntu_same_cuda_major(self):
        env = _env()
        rc, m = self._install(env)

        self.assertEqual(rc, 0)
        self._assert_probe_ran_and_was_cleaned_up(m)
        self.assertEqual(self._commands(m), [_ubuntu13(), *_TAIL])
        for call in m["exec"].call_args_list:
            self.assertEqual(call.args[1], _OP)
            self.assertIs(call.kwargs["env"], env)
        m["record"].assert_called_once_with("DCGM", _UBUNTU_DCGM)
        self.assertEqual(self._dropin(), install_dcgm.hostengine_dropin(_SCREEN, _HOSTENGINE))
        m["error"].assert_not_called()
        m["warn"].assert_not_called()

    def test_ubuntu_resolves_each_package_once_in_bash_order(self):
        rc, m = self._install(_env())
        self.assertEqual(rc, 0)
        shown = [c[2] for c in self._captures(m, "apt-cache")]
        self.assertEqual(shown, _ubuntu13()[3:])     # BFS order, no repeats

    def test_ubuntu_lower_sku_cuda_adds_packages(self):
        rc, m = self._install(_env(SKU="V100"), cuda_cfg=_CUDA_12)
        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m), [_ubuntu13(), _ubuntu12_extra(), *_TAIL])

    def test_ubuntu_compares_numerically(self):
        # 9 < 13 numerically (a text comparison would say "9" > "13").
        rc, m = self._install(_env(), cuda_cfg={"driver": {"version": "9.2"}})
        self.assertEqual(rc, 0)
        self.assertIn(f"{_DCGM}-cuda9={_UBUNTU_DCGM}", self._commands(m)[1])

    def test_ubuntu_higher_sku_cuda_adds_nothing(self):
        rc, m = self._install(_env(), probe=(0, "12090\n"))
        self.assertEqual(rc, 0)
        self.assertEqual(len(self._commands(m)), 1 + len(_TAIL))
        self.assertIn(f"{_DCGM}-cuda12={_UBUNTU_DCGM}", self._commands(m)[0])

    def test_ubuntu_unavailable_package_returns_3_before_installing(self):
        rc, m = self._install(_env(), apt_missing={"libdcgm4"})
        self.assertEqual(rc, 3)
        self.assertEqual(self._commands(m), [])
        self.assertEqual(m["error"].call_args_list[0].args[1],
                         f"libdcgm4={_UBUNTU_DCGM} is not available from the configured "
                         "apt repositories")

    def test_ubuntu26_04_installs_unpinned_and_reads_the_version(self):
        rc, m = self._install(_env("ubuntu26.04", SKU="V100"), cuda_cfg=_CUDA_12,
                              dcgm_cfg={})
        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m), [
            ["apt-get", "install", "-y", *install_dcgm.ubuntu_packages(13)],
            ["apt-get", "install", "-y", *install_dcgm.ubuntu_extra_packages(12)],
            *_TAIL])
        self.assertEqual(self._captures(m, "apt-cache"), [])
        self.assertEqual(self._captures(m, "dpkg-query"), [
            ["dpkg-query", "-W", "-f=${Version}", f"{_DCGM}-core"]])
        m["record"].assert_called_once_with("DCGM", "1:4.8.0-1")
        self.assertNotIn("dcgm", [c.args[0] for c in m["config_for"].call_args_list])

    def test_ubuntu26_04_empty_dpkg_query_returns_3(self):
        for result in ((0, ""), (1, "")):
            with self.subTest(result=result):
                rc, m = self._install(_env("ubuntu26.04"), dpkg_query=result)
                self.assertEqual(rc, 3)
                m["record"].assert_not_called()
                self.assertIn("dpkg-query", m["error"].call_args_list[0].args[1])

    # ---- Azure Linux ------------------------------------------------------------
    def test_azure_linux_builds_the_marker_and_installs_cuda13(self):
        # Mirrors the bash bug: even a V100 (CUDA 12 SKU) gets cuda13.
        rc, m = self._install(_env("azurelinux3.0", SKU="V100"), cuda_cfg=_CUDA_12,
                              probe=(0, "12090\n"), dcgm_cfg={"version": _AZL_DCGM})
        self.assertEqual(rc, 0)
        topdir = m["rpm_topdir"]
        self.assertEqual(self._commands(m), [
            ["rpm", "-q", _MARKER],
            ["rpmbuild", "--define", f"_topdir {topdir}", "-bb",
             os.path.join(topdir, f"{_MARKER}.spec")],
            ["rpm", "-Uvh", "--replacepkgs",
             os.path.join(topdir, "RPMS", "noarch", f"{_MARKER}-1.0-1.noarch.rpm")],
            ["dnf", "install", "-y", f"{_DCGM}-cuda13-4.7.0", f"{_DCGM}-multinode-4.7.0",
             f"{_DCGM}-multinode-cuda13-4.7.0"],
            *_TAIL,
        ])
        self.assertEqual(m["spec"], install_dcgm._AZURE_LINUX_MARKER_SPEC)
        self.assertTrue(topdir.startswith(str(self.scratch)))
        self.assertEqual(os.listdir(self.scratch), [])          # removed
        m["record"].assert_called_once_with("DCGM", _AZL_DCGM)
        m["warn"].assert_called_once()
        self.assertEqual(m["warn"].call_args.kwargs["sku"], "V100")

    def test_azure_linux_marker_already_installed(self):
        rc, m = self._install(_env("azurelinux3.0"), dcgm_cfg={"version": _AZL_DCGM},
                              marker_installed=True)
        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m)[:2], [
            ["rpm", "-q", _MARKER],
            ["dnf", "install", "-y", *install_dcgm.azure_linux_packages(13, _AZL_DCGM)]])
        m["warn"].assert_not_called()

    def test_azure_linux_marker_not_built_returns_3(self):
        rc, m = self._install(_env("azurelinux3.0"), dcgm_cfg={"version": _AZL_DCGM},
                              rpm_built=False)
        self.assertEqual(rc, 3)
        self.assertIn("failed to build", m["error"].call_args_list[0].args[1])
        self.assertEqual(self._commands(m)[-1][0], "rpmbuild")
        self.assertEqual(os.listdir(self.scratch), [])

    def test_azure_linux_does_not_need_the_sku_cuda_version(self):
        rc, m = self._install(_env("azurelinux3.0"), cuda_cfg=None,
                              dcgm_cfg={"version": _AZL_DCGM})
        self.assertEqual(rc, 0)

    # ---- RHEL family ------------------------------------------------------------
    def test_rhel_family_records_the_dcgmi_version(self):
        rc, m = self._install(_env("almalinux9.8", SKU="V100"), cuda_cfg=_CUDA_12)
        self.assertEqual(rc, 0)
        self._assert_probe_ran_and_was_cleaned_up(m)
        self.assertEqual(self._commands(m), [
            ["dnf", "clean", "expire-cache"],
            [*_RHEL_INSTALL, f"{_DCGM}-cuda13"],
            [*_RHEL_MULTINODE, f"{_DCGM}-multinode-cuda13"],
            [*_RHEL_INSTALL, f"{_DCGM}-cuda12"],
            [*_RHEL_MULTINODE, f"{_DCGM}-multinode-cuda12"],
            *_TAIL,
        ])
        self.assertEqual(self._captures(m, "dcgmi"), [["dcgmi", "--version"]])
        m["record"].assert_called_once_with("DCGM", "4.4.1")
        # The RHEL family never reads the dcgm config.
        self.assertEqual([c.args[0] for c in m["config_for"].call_args_list], ["cuda"])

    def test_rhel_family_same_cuda_major(self):
        for distro in ("rocky8.10", "rhel9.8"):
            with self.subTest(distro=distro):
                rc, m = self._install(_env(distro))
                self.assertEqual(rc, 0)
                self.assertEqual(self._commands(m), [
                    ["dnf", "clean", "expire-cache"],
                    [*_RHEL_INSTALL, f"{_DCGM}-cuda13"],
                    [*_RHEL_MULTINODE, f"{_DCGM}-multinode-cuda13"],
                    *_TAIL])

    # ---- nv-hostengine under screen ------------------------------------------
    def test_screen_installed_when_missing(self):
        for distro, command in (("ubuntu24.04", ["apt-get", "install", "-y", "screen"]),
                                ("almalinux9.8", ["dnf", "install", "-y", "screen"]),
                                ("azurelinux3.0", ["dnf", "install", "-y", "screen"])):
            with self.subTest(distro=distro):
                rc, m = self._install(_env(distro), screen=False,
                                      dcgm_cfg={"version": _AZL_DCGM}
                                      if distro == "azurelinux3.0" else None)
                self.assertEqual(rc, 0)
                self.assertEqual(self._commands(m)[-len(_TAIL) - 1:], [command, *_TAIL])

    def test_screen_still_missing_returns_3(self):
        rc, m = self._install(_env(), screen=False, screen_after_install=False)
        self.assertEqual(rc, 3)
        self.assertIn("screen is still not on PATH", m["error"].call_args_list[0].args[1])
        self.assertFalse(self.dropin_dir.exists())

    def test_unit_without_nv_hostengine_returns_3(self):
        for unit_text in ("[Service]\nExecStart=/usr/bin/other\n", None):
            with self.subTest(unit_text=unit_text):
                rc, m = self._install(_env(), unit_text=unit_text)
                self.assertEqual(rc, 3)
                self.assertIn("could not find the nv-hostengine ExecStart",
                              m["error"].call_args_list[0].args[1])
                self.assertNotIn(_TAIL[0], self._commands(m))
                self.assertFalse(self.dropin_dir.exists())

    def test_systemctl_show_failure_returns_3(self):
        rc, m = self._install(_env(), show=(1, ""))
        self.assertEqual(rc, 3)
        self.assertIn("unit file", m["error"].call_args_list[0].args[1])

    def test_dropin_created_then_replaced(self):
        rc, m = self._install(_env())
        dropin = str(self.dropin_dir / "99-mndiag-hpcx.conf")
        messages = {c.args[1]: c.kwargs for c in m["info"].call_args_list}
        self.assertEqual(messages[f"Created {dropin} (nv-hostengine runs under screen with "
                                  "mpi/hpcx loaded)"], {"path": dropin, "action": "created"})
        rc, m = self._install(_env())
        messages = {c.args[1]: c.kwargs for c in m["info"].call_args_list}
        self.assertEqual(messages[f"Replaced {dropin} (nv-hostengine runs under screen "
                                  "with mpi/hpcx loaded)"]["action"], "replaced")

    # ---- logging -----------------------------------------------------------
    def test_key_log_lines(self):
        rc, m = self._install(_env(SKU="V100"), cuda_cfg=_CUDA_12)
        self.assertEqual(rc, 0)

        calls = m["info"].call_args_list
        self.assertEqual(calls[0].args, (_OP, "Installing DCGM"))
        self.assertEqual(calls[0].kwargs["distro"], "ubuntu24.04")
        by_message = {c.args[1]: c.kwargs for c in calls}
        self.assertEqual(by_message["Probed CUDA driver API version 13000 (CUDA 13)"],
                         {"cuda_driver_api_version": 13000, "cuda_major": 13})
        decision = [c for c in calls if "also installing the cuda12" in c.args[1]]
        self.assertEqual(decision[0].kwargs, {"cuda_major": 13, "sku_cuda_major": 12})
        ubuntu = [c for c in calls if c.args[1].startswith("Ubuntu:")]
        self.assertEqual(ubuntu[0].kwargs["version_source"], "versions.json")
        resolved = by_message[f"Resolved 8 DCGM packages pinned to {_UBUNTU_DCGM}"]
        self.assertEqual(resolved["packages"], _ubuntu13()[3:])
        self.assertEqual(by_message[f"Recorded DCGM version {_UBUNTU_DCGM}"],
                         {"component": "DCGM", "version": _UBUNTU_DCGM})
        self.assertEqual(by_message[f"nv-hostengine command from {self.unit}: {_HOSTENGINE}"]
                         ["screen"], _SCREEN)
        self.assertIn("nvidia-dcgm service is active", by_message)
        self.assertEqual(calls[-1].args[1], "DCGM installed")

    # ---- failure paths -----------------------------------------------------
    def test_probe_failures_return_3_before_installing(self):
        for probe in [(1, ""), (0, ""), (0, "garbage\n")]:
            with self.subTest(probe=probe):
                rc, m = self._install(_env(), probe=probe)
                self.assertEqual(rc, 3)
                self.assertEqual(self._commands(m), [])
                m["record"].assert_not_called()
                m["error"].assert_called()
                self.assertEqual(os.listdir(self.scratch), [])

    def test_compile_failure_returns_3_and_cleans_up(self):
        def fail_gcc(command, op, **kwargs):
            return 1
        with mock.patch.object(install_dcgm, "exec_program", side_effect=fail_gcc), \
             mock.patch.object(install_dcgm, "run_capture") as run_capture, \
             mock.patch.object(install_dcgm, "log_info"), \
             mock.patch.object(install_dcgm, "log_error") as log_error:
            rc = install_dcgm.install(_env())
        self.assertEqual(rc, 3)
        run_capture.assert_not_called()
        self.assertIn("compile", log_error.call_args_list[0].args[1])
        self.assertEqual(os.listdir(self.scratch), [])

    def test_probe_dir_os_error_returns_3(self):
        with mock.patch.object(install_dcgm.tempfile, "mkdtemp",
                               side_effect=PermissionError("no tmp")), \
             mock.patch.object(install_dcgm, "exec_program") as exec_program, \
             mock.patch.object(install_dcgm, "log_info"), \
             mock.patch.object(install_dcgm, "log_error") as log_error:
            rc = install_dcgm.install(_env())
        self.assertEqual(rc, 3)
        exec_program.assert_not_called()
        self.assertIn("no tmp", log_error.call_args.args[1])

    def test_missing_dcgm_config_returns_3(self):
        for distribution in ["ubuntu22.04", "ubuntu24.04", "azurelinux3.0"]:
            with self.subTest(distribution=distribution):
                rc, m = self._install(_env(distribution), dcgm_cfg={})
                self.assertEqual(rc, 3)
                self.assertEqual(self._commands(m), [])
                self.assertIn("dcgm version", m["error"].call_args_list[0].args[1])

    def test_missing_sku_cuda_version_returns_3_on_ubuntu_and_rhel(self):
        for distribution in ["ubuntu24.04", "ubuntu26.04", "almalinux9.8"]:
            with self.subTest(distribution=distribution):
                rc, m = self._install(_env(distribution), cuda_cfg=None)
                self.assertEqual(rc, 3)
                self.assertEqual(self._commands(m), [])
                self.assertIn("cuda driver version",
                              m["error"].call_args_list[0].args[1])

    def test_each_failing_command_returns_3_and_stops(self):
        sequences = {
            "ubuntu24.04": [_ubuntu13(), _ubuntu12_extra()],
            "ubuntu26.04": [
                ["apt-get", "install", "-y", *install_dcgm.ubuntu_packages(13)],
                ["apt-get", "install", "-y", *install_dcgm.ubuntu_extra_packages(12)]],
            "almalinux9.8": [
                ["dnf", "clean", "expire-cache"],
                [*_RHEL_INSTALL, f"{_DCGM}-cuda13"],
                [*_RHEL_MULTINODE, f"{_DCGM}-multinode-cuda13"],
                [*_RHEL_INSTALL, f"{_DCGM}-cuda12"],
                [*_RHEL_MULTINODE, f"{_DCGM}-multinode-cuda12"],
            ],
        }
        for distribution, installs in sequences.items():
            sequence = installs + _TAIL
            for index, failing in enumerate(sequence):
                with self.subTest(distribution=distribution, failing=failing):
                    rc, m = self._install(_env(distribution), cuda_cfg=_CUDA_12,
                                          fail_on=failing)
                    self.assertEqual(rc, 3)
                    self.assertEqual(self._commands(m), sequence[:index + 1])
                    m["error"].assert_called()
                    # The version is recorded only after a successful install.
                    if index < len(installs):
                        m["record"].assert_not_called()
                    else:
                        m["record"].assert_called_once()

    def test_each_failing_azure_linux_command_returns_3(self):
        commands = ["rpmbuild", "rpm", "dnf"]
        for name in commands:
            with self.subTest(command=name):
                def fail(command, op, _name=name, **kwargs):
                    if command[0] == "gcc":
                        return 0
                    if command[:2] == ["rpm", "-q"]:
                        return 1
                    if command[0] == "rpmbuild":
                        rpms = Path(command[2].split(" ", 1)[1], "RPMS")
                        rpms.mkdir(parents=True, exist_ok=True)
                        (rpms / f"{_MARKER}-1.0-1.noarch.rpm").touch()
                    return 1 if command[0] == _name else 0
                with mock.patch.object(install_dcgm, "exec_program", side_effect=fail), \
                     mock.patch.object(install_dcgm, "run_capture",
                                       return_value=(0, "13000\n")), \
                     mock.patch.object(install_dcgm, "config_for",
                                       return_value={"version": _AZL_DCGM}), \
                     mock.patch.object(install_dcgm, "write_component_version") as record, \
                     mock.patch.object(install_dcgm, "log_info"), \
                     mock.patch.object(install_dcgm, "log_error") as error:
                    rc = install_dcgm.install(_env("azurelinux3.0"))
                self.assertEqual(rc, 3)
                record.assert_not_called()
                error.assert_called()
                self.assertEqual(os.listdir(self.scratch), [])

    def test_inactive_service_logs_dcgm_inactive(self):
        rc, m = self._install(_env(), fail_on=_TAIL[3])
        self.assertEqual(rc, 3)
        self.assertEqual(m["error"].call_args.args[1], "DCGM is inactive!")

    def test_dcgmi_failure_returns_3(self):
        for dcgmi in [(1, _DCGMI_OUT), (0, ""), (127, "")]:
            with self.subTest(dcgmi=dcgmi):
                rc, m = self._install(_env("almalinux9.8"), dcgmi=dcgmi)
                self.assertEqual(rc, 3)
                m["record"].assert_not_called()
                self.assertEqual(self._commands(m)[-1],
                                 [*_RHEL_MULTINODE, f"{_DCGM}-multinode-cuda13"])

    def test_unwritable_versions_file_returns_3(self):
        rc, m = self._install(_env(), write_error=PermissionError("denied"))
        self.assertEqual(rc, 3)
        self.assertEqual(len(self._commands(m)), 1)    # no systemctl after it
        self.assertIn("denied", m["error"].call_args.args[1])


# --------------------------------------------------------------------------- #
# Wiring.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "install-dcgm"]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, install_dcgm.install)
        self.assertEqual(steps[0].script, "install_dcgm.sh")


if __name__ == "__main__":
    unittest.main()
