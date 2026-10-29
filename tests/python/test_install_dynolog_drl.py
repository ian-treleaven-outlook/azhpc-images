"""Unit tests for components/python/install_dynolog_drl.py.

Two kinds of test:

  * The pure helpers (dependency commands, build scripts, the DCGM lookup and
    the unit file text) are tested directly. The DCGM lookup and the unit files
    were also compared with real awk and the real bash heredocs in WSL.
  * install() is tested with exec_program replaced by a fake that records each
    command and imitates its effect on disk: `git clone` creates the source
    directory and each `bash -c` build creates the binaries. run_capture
    (`ldconfig -p`) and write_component_version are mocked too. The install,
    source and systemd directories point into a temporary directory, so
    nothing is cloned, compiled or written under /opt, /tmp or /etc.

Run from the repo root:
    python3 -m unittest tests.python.test_install_dynolog_drl -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_dynolog_drl
from utils.build_config import BuildConfig, build_plan

_OP = "install-dynolog-drl"
_DYNOLOG = {"version": "1.0.0", "url": "https://github.com/jasontmsft/dynolog.git",
            "commit": "1f61fd5e6d7499e7bd7531830cbe1f81b43ff815"}
_DRL = {"version": "1.0.1", "url": "https://github.com/jasontmsft/dyno-relay-logger.git",
        "commit": "97ef384edacde60cde30f76f4c1b683087c3b24c"}
_DCGM_LIB = "/lib/x86_64-linux-gnu/libdcgm.so.4"
_LDCONFIG = ("1234 libs found in cache `/etc/ld.so.cache'\n"
             "\tlibdcgmmoduleprofiling.so.4 (libc6,x86-64) => "
             "/lib/x86_64-linux-gnu/libdcgmmoduleprofiling.so.4\n"
             f"\tlibdcgm.so.4 (libc6,x86-64) => {_DCGM_LIB}\n"
             "\tlibdcgm.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libdcgm.so\n")

# The dynolog.service bash writes for _DCGM_LIB (rendered by bash's heredoc).
_DYNOLOG_SERVICE = (
    "[Unit]\nDescription=dynolog\nAfter=nvidia-dcgm.service\n\n[Service]\n"
    'Environment="GLOG_logtostderr=1" "GLOG_minloglevel=2"\n'
    "ExecStart=/opt/dynolog/bin/dynolog -enable_ipc_monitor=true -enable_gpu_monitor=true "
    f"-kernel_monitor_reporting_interval_s=10 -dcgm_lib_path={_DCGM_LIB} "
    "-dcgm_reporting_interval_s=10 -use_udsrelay=true "
    '-dcgm_fields="100,155,204,1001,1005,1008,1009,1010,1011,1012,1013,1014"\n'
    "Restart=always\nRestartSec=60s\nUser=root\nGroup=root\n\n"
    "[Install]\nWantedBy=multi-user.target\n")
_DRL_SERVICE = (
    "[Unit]\nDescription=dyno-relay-logger\nAfter=nvidia-dcgm.service\n\n[Service]\n"
    'Environment="GLOG_logtostderr=1" "GLOG_minloglevel=2"\n'
    "ExecStart=/opt/dynolog/bin/dynorelaylogger --forward=aehubs\n"
    "Restart=always\nRestartSec=60s\nUser=root\nGroup=root\n\n"
    "[Install]\nWantedBy=multi-user.target\n")

_UBUNTU_DEPS = [
    ["apt-get", "install", "-y", "cmake", "rustc-1.82", "cargo-1.82", "ninja-build",
     "build-essential"],
    ["apt-get", "install", "-y", "g++", "pkg-config", "uuid-dev", "libssl-dev"],
]
_CMAKE = "cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release"


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
class HelperTests(unittest.TestCase):
    def test_dependency_commands_per_distro(self):
        m = install_dynolog_drl
        self.assertEqual(m.dependency_commands("azurelinux3.0"), [
            ["dnf", "install", "-y", "cmake", "rust", "cargo", "ninja-build",
             "build-essential"]])
        self.assertEqual(m.dependency_commands("ubuntu22.04"), _UBUNTU_DEPS)
        el = ["dnf", "install", "-y", "cmake", "rust", "cargo", "ninja-build",
              "libuuid-devel", "gcc-toolset-12"]
        for distribution in ("almalinux8.10", "rocky8.10"):
            self.assertEqual(m.dependency_commands(distribution),
                             [el, ["dnf", "install", "-y", "openssl3-devel"]])
        for distribution in ("almalinux9.6", "rocky9.7", "almalinux8.9"):
            self.assertEqual(m.dependency_commands(distribution),
                             [el, ["dnf", "install", "-y", "openssl-devel"]])
        self.assertEqual(m.dependency_commands("rhel9.4"), [])   # no bash branch
        self.assertIsNone(m.dependency_branch("rhel9.4"))

    def test_build_scripts_apply_the_distro_environment(self):
        m = install_dynolog_drl
        self.assertEqual(m.dynolog_build_script("ubuntu24.04"),
                         'set -e\nexport PATH="/usr/lib/rust-1.82/bin:$PATH"\n'
                         "./scripts/build.sh -DCMAKE_POLICY_VERSION_MINIMUM=3.5\n")
        self.assertEqual(m.dynolog_build_script("azurelinux3.0"),
                         "set -e\n./scripts/build.sh -DCMAKE_POLICY_VERSION_MINIMUM=3.5\n")
        self.assertEqual(m.drl_build_script("rocky9.7"),
                         "set -e\nsource /opt/rh/gcc-toolset-12/enable\n"
                         f"{_CMAKE}\ncmake --build . -j$(nproc)\n")
        self.assertEqual(m.drl_build_script("ubuntu24.04"),
                         'set -e\nexport PATH="/usr/lib/rust-1.82/bin:$PATH"\n'
                         "export OPENSSL_LIB_DIR=/usr/lib/$(dpkg-architecture "
                         "-qDEB_HOST_MULTIARCH)\n"
                         f"{_CMAKE}\ncmake --build . -j$(nproc)\n")

    def test_el8_drl_script_has_the_openssl3_workaround(self):
        self.assertEqual(
            install_dynolog_drl.drl_build_script("almalinux8.10"),
            "set -e\nsource /opt/rh/gcc-toolset-12/enable\n"
            "export OPENSSL_DIR=/usr\n"
            "export OPENSSL_INCLUDE_DIR=/usr/include/openssl3\n"
            "export OPENSSL_LIB_DIR=/usr/lib64/openssl3\n"
            f"{_CMAKE} -DOPENSSL_ROOT_DIR=/usr/include/openssl3"
            " -DOPENSSL_INCLUDE_DIR=/usr/include/openssl3"
            " -DOPENSSL_CRYPTO_LIBRARY=/usr/lib64/openssl3/libcrypto.so"
            " -DOPENSSL_SSL_LIBRARY=/usr/lib64/openssl3/libssl.so\n"
            "cmake --build . -j$(nproc)\n")

    def test_find_dcgm_lib(self):
        find = install_dynolog_drl.find_dcgm_lib
        # The first match wins; libdcgmmodule* lines don't match.
        self.assertEqual(find(_LDCONFIG), _DCGM_LIB)
        self.assertEqual(find("\tlibdcgm.so.3.1.8 (libc6,x86-64) => /usr/lib64/libdcgm.so.3.1.8\n"),
                         "/usr/lib64/libdcgm.so.3.1.8")
        self.assertIsNone(find("\tlibdcgm.so.4x (libc6) => /x\n\tlibdcgmx.so (libc6) => /y\n"))
        self.assertIsNone(find(""))

    def test_service_texts(self):
        self.assertEqual(install_dynolog_drl.dynolog_service(_DCGM_LIB), _DYNOLOG_SERVICE)
        self.assertEqual(install_dynolog_drl._DRL_SERVICE, _DRL_SERVICE)


# --------------------------------------------------------------------------- #
# install(): external effects faked.
# --------------------------------------------------------------------------- #
class _FakeExec:
    """Stands in for exec_program: records (command, cwd) and imitates effects.

    fail_at, when given, is the index of the call that should fail (exit 1).
    """

    def __init__(self, fail_at=None, clone_adds_build_dir=False, build_outputs=True):
        self.calls = []
        self.fail_at = fail_at
        self.clone_adds_build_dir = clone_adds_build_dir
        self.build_outputs = build_outputs

    def __call__(self, command, op, *, cwd=None, env=None):
        assert op == _OP
        self.calls.append((command, cwd))
        if len(self.calls) - 1 == self.fail_at:
            return 1
        if command[:2] == ["git", "clone"]:
            target = command[-1]
            if os.path.exists(target):
                return 128          # git: destination path already exists
            os.makedirs(target)
            if self.clone_adds_build_dir:
                os.makedirs(os.path.join(target, "build"))
        elif command[:2] == ["bash", "-c"] and self.build_outputs:
            if "scripts/build.sh" in command[2]:
                outputs = ("build/dynolog/src/dynolog", "build/release/dyno")
            else:
                outputs = ("dynorelaylogger", "dynorelayloggerinfo")
            for relative in outputs:
                path = Path(cwd, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"binary {relative}\n", encoding="utf-8")
        return 0

    @property
    def commands(self):
        return [command for command, _ in self.calls]


def _env(distribution="ubuntu24.04", *, gpu="NVIDIA", dynolog=_DYNOLOG, drl=_DRL):
    versions = {}
    if dynolog:
        versions["dynolog"] = {"common": dynolog}
    if drl:
        versions["dyno_relay_logger"] = {"common": drl}
    return {"DISTRIBUTION": distribution, "ARCHITECTURE": "x86_64", "GPU": gpu,
            "SKU": "H100", "COMPONENT_VERSIONS": json.dumps(versions)}


class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.install_dir = root / "opt" / "dynolog" / "bin"
        self.dynolog_src = root / "tmp" / "dynolog"
        self.drl_src = root / "tmp" / "dyno-relay-logger"
        self.systemd = root / "etc" / "systemd" / "system"
        self.systemd.mkdir(parents=True)
        (root / "tmp").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _install(self, env, *, fake=None, capture=(0, _LDCONFIG), write_error=None):
        """Run install(env) with every external effect faked; returns (rc, mocks)."""
        fake = fake or _FakeExec()
        m = install_dynolog_drl
        with mock.patch.object(m, "_INSTALL_DIR", str(self.install_dir)), \
             mock.patch.object(m, "_DYNOLOG_SRC", str(self.dynolog_src)), \
             mock.patch.object(m, "_DRL_SRC", str(self.drl_src)), \
             mock.patch.object(m, "_SYSTEMD_DIR", str(self.systemd)), \
             mock.patch.object(m, "exec_program", fake), \
             mock.patch.object(m, "run_capture", return_value=capture) as run_capture, \
             mock.patch.object(m, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(m, "log_info") as log_info, \
             mock.patch.object(m, "log_warn") as log_warn, \
             mock.patch.object(m, "log_error") as log_error:
            rc = m.install(env)
        return rc, {"exec": fake, "run_capture": run_capture, "record": record,
                    "log_info": log_info, "log_warn": log_warn, "log_error": log_error}

    def _expected_calls(self, distribution, deps):
        m = install_dynolog_drl
        dyn, drl = str(self.dynolog_src), str(self.drl_src)
        return ([(c, None) for c in deps] + [
            (["git", "clone", "--no-checkout", _DYNOLOG["url"], dyn], None),
            (["git", "checkout", _DYNOLOG["commit"]], dyn),
            (["git", "submodule", "update", "--init", "--recursive"], dyn),
            (["bash", "-c", m.dynolog_build_script(distribution)], dyn),
            (["ldconfig"], None),
            (["systemctl", "daemon-reload"], None),
            (["git", "clone", "--no-checkout", _DRL["url"], drl], None),
            (["git", "checkout", _DRL["commit"]], drl),
            (["git", "submodule", "update", "--init", "--recursive"], drl),
            (["bash", "-c", m.drl_build_script(distribution)], os.path.join(drl, "build")),
            (["systemctl", "daemon-reload"], None),
        ])

    # ---- success paths -----------------------------------------------------
    def test_ubuntu_builds_both_and_writes_disabled_services(self):
        env = _env("ubuntu24.04")
        rc, m = self._install(env)

        self.assertEqual(rc, 0)
        self.assertEqual(m["exec"].calls, self._expected_calls("ubuntu24.04", _UBUNTU_DEPS))
        m["run_capture"].assert_called_once_with(["ldconfig", "-p"], _OP, env=env)
        self.assertEqual(sorted(os.listdir(self.install_dir)),
                         ["dyno", "dynolog", "dynorelaylogger", "dynorelayloggerinfo"])
        self.assertEqual((self.systemd / "dynolog.service").read_text(encoding="utf-8"),
                         _DYNOLOG_SERVICE)
        self.assertEqual((self.systemd / "dyno-relay-logger.service").read_text(
            encoding="utf-8"), _DRL_SERVICE)
        self.assertFalse(self.dynolog_src.exists())
        self.assertFalse(self.drl_src.exists())
        # Never enabled (see the module docstring).
        self.assertNotIn("enable", [c[1] for c in m["exec"].commands if c[0] == "systemctl"])
        self.assertEqual(m["record"].call_args_list, [
            mock.call("dynolog", "1.0.0"), mock.call("dyno_relay_logger", "1.0.1")])
        m["log_error"].assert_not_called()

    def test_each_distro_uses_its_dependencies_and_build_environment(self):
        for distribution in ("azurelinux3.0", "almalinux8.10", "rocky8.10",
                             "almalinux9.6", "rocky9.7", "ubuntu22.04"):
            with self.subTest(distribution=distribution):
                self.tearDown()
                self.setUp()
                rc, m = self._install(_env(distribution))
                self.assertEqual(rc, 0)
                self.assertEqual(m["exec"].calls, self._expected_calls(
                    distribution, install_dynolog_drl.dependency_commands(distribution)))
                m["log_warn"].assert_not_called()

    def test_other_distros_build_without_dependencies_and_warn(self):
        rc, m = self._install(_env("rhel9.4"))
        self.assertEqual(rc, 0)
        self.assertEqual(m["exec"].calls, self._expected_calls("rhel9.4", []))
        m["log_warn"].assert_called_once()
        self.assertEqual(m["log_warn"].call_args.kwargs, {"distro": "rhel9.4"})

    def test_non_nvidia_gpu_is_skipped(self):
        rc, m = self._install(_env(gpu="AMD"))
        self.assertEqual(rc, 0)
        self.assertEqual(m["exec"].calls, [])
        m["record"].assert_not_called()
        self.assertFalse(self.install_dir.exists())
        self.assertEqual(m["log_info"].call_args.kwargs, {"gpu": "AMD", "skipped": True})

    # ---- failure paths -----------------------------------------------------
    def test_missing_config_returns_3_before_doing_anything(self):
        for env in (_env(dynolog=None), _env(drl=None),
                    _env(dynolog={"version": "1.0.0", "url": "u"})):   # no commit
            with self.subTest(env=env["COMPONENT_VERSIONS"]):
                rc, m = self._install(env)
                self.assertEqual(rc, 3)
                self.assertEqual(m["exec"].calls, [])
                self.assertFalse(self.install_dir.exists())
                m["log_error"].assert_called_once()

    def test_each_failing_command_returns_3_and_stops(self):
        total = len(self._expected_calls("ubuntu24.04", _UBUNTU_DEPS))
        for index in range(total):
            with self.subTest(fail_at=index):
                self.tearDown()
                self.setUp()
                fake = _FakeExec(fail_at=index)
                rc, m = self._install(_env("ubuntu24.04"), fake=fake)
                self.assertEqual(rc, 3)
                self.assertEqual(len(fake.calls), index + 1)
                m["record"].assert_not_called()
                m["log_error"].assert_called()

    def test_ldconfig_listing_failure_returns_3(self):
        rc, m = self._install(_env(), capture=(1, ""))
        self.assertEqual(rc, 3)
        self.assertFalse((self.systemd / "dynolog.service").exists())

    def test_missing_dcgm_returns_3(self):
        rc, m = self._install(_env(), capture=(0, "\tlibc.so.6 (libc6) => /lib/libc.so.6\n"))
        self.assertEqual(rc, 3)
        self.assertFalse((self.systemd / "dynolog.service").exists())
        self.assertIn("libdcgm.so not found", m["log_error"].call_args[0][1])

    def test_leftover_clone_dir_warns_and_fails_like_bash(self):
        self.dynolog_src.mkdir()
        rc, m = self._install(_env())
        self.assertEqual(rc, 3)
        m["log_warn"].assert_called_once()
        self.assertEqual(m["exec"].commands[-1][:2], ["git", "clone"])

    def test_existing_drl_build_dir_returns_3(self):
        rc, m = self._install(_env(), fake=_FakeExec(clone_adds_build_dir=True))
        self.assertEqual(rc, 3)
        self.assertNotIn("cmake", " ".join(m["exec"].commands[-1]))

    def test_missing_built_binary_returns_3(self):
        rc, m = self._install(_env(), fake=_FakeExec(build_outputs=False))
        self.assertEqual(rc, 3)
        self.assertIn("could not move", m["log_error"].call_args[0][1])

    def test_unwritable_service_file_returns_3(self):
        self.systemd = self.systemd / "missing"
        rc, m = self._install(_env())
        self.assertEqual(rc, 3)
        self.assertIn("dynolog.service", m["log_error"].call_args[0][1])

    def test_unwritable_versions_file_returns_3(self):
        rc, m = self._install(_env(), write_error=PermissionError("denied"))
        self.assertEqual(rc, 3)
        self.assertIn("dynolog=1.0.0", m["log_error"].call_args[0][1])

    # ---- logging -----------------------------------------------------------
    def test_key_log_lines(self):
        rc, m = self._install(_env("almalinux8.10"))
        self.assertEqual(rc, 0)
        calls = m["log_info"].call_args_list
        by_message = {c.args[1]: c.kwargs for c in calls}

        start = by_message["Installing dynolog and dyno-relay-logger"]
        self.assertEqual((start["dynolog_commit"], start["drl_version"]),
                         (_DYNOLOG["commit"], "1.0.1"))
        deps = by_message["Installing build dependencies (almalinux-rocky)"]
        self.assertEqual(deps["build_env"], "source /opt/rh/gcc-toolset-12/enable")
        self.assertEqual(by_message[f"Found the DCGM library: {_DCGM_LIB}"],
                         {"dcgm_lib": _DCGM_LIB})
        written = by_message[f"Wrote {self.systemd / 'dynolog.service'} (created)"]
        self.assertEqual(written["action"], "created")
        disabled = [c.kwargs for c in calls if c.kwargs.get("enabled") is False]
        self.assertEqual([d["service"] for d in disabled],
                         ["dynolog.service", "dyno-relay-logger.service"])
        self.assertIn({"component": "dyno_relay_logger", "version": "1.0.1"},
                      [c.kwargs for c in calls])
        self.assertEqual(calls[-1].kwargs["dcgm_lib"], _DCGM_LIB)


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        # The step has no `when`: AMD builds run it too, and it skips itself.
        for vendor, gpu in (("NVidia", "H100"), ("AMD", "MI300")):
            with self.subTest(vendor=vendor):
                cfg = BuildConfig(vendor=vendor, gpu=gpu, os="Ubuntu24",
                                  fips=False, spec_path=None)
                steps = [s for s in build_plan(cfg) if s.op == "install-dynolog-drl"]
                self.assertEqual(len(steps), 1)
                self.assertIs(steps[0].action, install_dynolog_drl.install)
                self.assertEqual(steps[0].script, "install_dynolog_drl.sh")


if __name__ == "__main__":
    unittest.main()
