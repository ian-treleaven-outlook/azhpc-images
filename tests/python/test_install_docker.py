"""Unit tests for components/python/install_docker.py.

  * The pure parse helpers are tested directly with realistic command output.
    Their results were also compared against the real bash pipelines
    (grep/awk/cut and bash word splitting) in WSL.
  * install() is tested with its external effects mocked: exec_program (in
    utils.process, under run() and systemctl(), and in utils.package_installer;
    one mock so the order of all commands is kept: apt-get, dnf, systemctl,
    ctr, the nested toolkit script), run_capture (docker --version, the
    package listings), write_component_version, config_for, time.sleep and
    utils.logger._log (every log line). The containerd socket path is pointed
    at a temporary directory; one test creates a real Unix socket there.

Run from the repo root:
    python3 -m unittest tests.python.test_install_docker -v
"""

from __future__ import annotations

import os
import socket
import tempfile
import unittest
from collections import namedtuple
from unittest import mock

from components.python import install_docker
from utils.build_config import BuildConfig, build_plan
from utils.distro import Distro

_OP = "install-docker"
# What install_packages() runs for bash's `apt-get install -y` / `dnf install -y`.
_APT = ["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0"]
_DNF = ["dnf", "install", "-y"]
_COMPONENT_DIR = "/repo/components"
_TOOLKIT = f"{_COMPONENT_DIR}/install_nvidia_container_toolkit.sh"

_DOCKER_OUT = "Docker version 29.1.4, build 0e5a6ab\n"
_APT_OUT = ("Listing...\n"
            "moby-buildx/now 0.30.1-ubuntu24.04u1 amd64 [installed,local]\n"
            "moby-cli/now 29.1.4-ubuntu24.04u1 amd64 [installed,local]\n"
            "moby-engine/now 29.1.4-ubuntu24.04u1 amd64 [installed,local]\n")
_DNF_OUT = ("Installed Packages\n"
            "moby-cli.x86_64        28.5.1-1.el9   @packages-microsoft-com-prod\n"
            "moby-engine.x86_64     28.5.1-1.el9   @packages-microsoft-com-prod\n")
_RPM_OUT = ("bash-5.2.15-3.azl3.x86_64\n"
            "moby-engine-25.0.3-11.azl3.x86_64\n"
            "moby-cli-25.0.3-11.azl3.x86_64\n")
_DPKG_OUT = "28.2.2-0ubuntu1"

_SERVICES = [
    ["systemctl", "enable", "docker"],
    ["systemctl", "restart", "docker"],
]
_CONTAINERD = [
    ["systemctl", "restart", "containerd"],
    ["ctr", "plugin", "ls"],
]


def _moby_command(distribution):
    return install_docker.moby_version_query(Distro(distribution))[0]


def _parse_moby(distribution, output):
    return install_docker.moby_version_query(Distro(distribution))[1](output)


def _env(distribution="ubuntu24.04", **extra):
    env = {"DISTRIBUTION": distribution, "COMPONENT_DIR": _COMPONENT_DIR,
           "GPU": "NVIDIA", "SKU": "H100", "ARCHITECTURE": "x86_64",
           "TARGET_NODE_TYPE": "azure_vm_regular"}
    env.update(extra)
    return env


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
class DockerVersionTests(unittest.TestCase):
    def test_drops_the_trailing_comma(self):
        self.assertEqual(install_docker.docker_version(_DOCKER_OUT), "29.1.4")

    def test_azure_style_version(self):
        self.assertEqual(install_docker.docker_version(
            "Docker version 25.0.3+azure-1, build 4debf41\n"), "25.0.3+azure-1")

    def test_extra_line_after_the_version_keeps_the_comma_like_bash(self):
        # awk prints "29.1.4," and "here"; ${v::-1} trims the last character of
        # the WHOLE capture ("here" -> "her"), so the first word keeps its
        # comma. Real bash gives "29.1.4," for this input too.
        self.assertEqual(install_docker.docker_version(
            "Docker version 29.1.4, build abc\nextra line here\n"), "29.1.4,")

    def test_empty_or_short_output_returns_none(self):
        self.assertIsNone(install_docker.docker_version(""))
        self.assertIsNone(install_docker.docker_version("Docker\n"))


class MobyVersionParseTests(unittest.TestCase):
    def test_apt_list(self):
        self.assertEqual(_parse_moby("ubuntu24.04", _APT_OUT),
                         "29.1.4-ubuntu24.04u1")

    def test_dnf_list(self):
        self.assertEqual(_parse_moby("almalinux9.8", _DNF_OUT),
                         "28.5.1-1.el9")

    def test_rpm_qa(self):
        self.assertEqual(_parse_moby("azurelinux3.0", _RPM_OUT),
                         "25.0.3-11.azl3.x86_64")

    def test_rpm_qa_records_the_first_moby_package_like_bash(self):
        # Bash quirk: grep moby also matches moby-containerd, and the unquoted
        # value's first word wins.
        output = "moby-containerd-2.0.0-3.azl3.x86_64\n" + _RPM_OUT
        self.assertEqual(install_docker.rpm_moby_version(output),
                         "2.0.0-3.azl3.x86_64")

    def test_rpm_line_without_dash_is_passed_through_like_cut(self):
        self.assertEqual(install_docker.rpm_moby_version("moby\n" + _RPM_OUT), "moby")

    def test_rpm_line_with_too_few_fields_is_skipped_like_word_splitting(self):
        self.assertEqual(install_docker.rpm_moby_version("moby-x\n" + _RPM_OUT),
                         "25.0.3-11.azl3.x86_64")

    def test_dnf_wrapped_line_gives_none(self):
        # dnf can wrap a long name: the version moves to a line grep doesn't keep.
        wrapped = "Installed Packages\nmoby-engine.x86_64\n      28.5.1-1.el9  @repo\n"
        self.assertIsNone(install_docker.grep_awk_version(wrapped, "moby-engine"))

    def test_no_match_gives_none(self):
        self.assertIsNone(_parse_moby("ubuntu24.04", "Listing...\n"))
        self.assertIsNone(_parse_moby("azurelinux3.0", ""))

    def test_version_commands_per_distro(self):
        self.assertEqual(_moby_command("ubuntu22.04"),
                         ["apt", "list", "--installed"])
        self.assertEqual(_moby_command("ubuntu24.04"),
                         ["apt", "list", "--installed"])
        # bash: dpkg-query -W -f='${Version}' docker.io (single-quoted, literal).
        self.assertEqual(_moby_command("ubuntu26.04"),
                         ["dpkg-query", "-W", "-f=${Version}", "docker.io"])
        self.assertEqual(_moby_command("azurelinux3.0"),
                         ["rpm", "-qa"])
        self.assertEqual(_moby_command("rocky8.10"),
                         ["dnf", "list", "installed"])

    def test_dpkg_query_version(self):
        # dpkg-query -W -f='${Version}' prints the bare version, no newline
        # (checked in WSL).
        self.assertEqual(_parse_moby("ubuntu26.04", _DPKG_OUT),
                         "28.2.2-0ubuntu1")
        self.assertIsNone(_parse_moby("ubuntu26.04", ""))


class WantsContainerToolkitTests(unittest.TestCase):
    def test_nvidia_unset_and_empty_mean_yes(self):
        self.assertTrue(install_docker.wants_container_toolkit({"GPU": "NVIDIA"}))
        self.assertTrue(install_docker.wants_container_toolkit({}))
        self.assertTrue(install_docker.wants_container_toolkit({"GPU": ""}))

    def test_amd_means_no(self):
        self.assertFalse(install_docker.wants_container_toolkit({"GPU": "AMD"}))


# --------------------------------------------------------------------------- #
# install(): external effects mocked.
# --------------------------------------------------------------------------- #
_Record = namedtuple("_Record", "level message fields")


class _Logs:
    """Stands in for utils.logger._log; filter by level with info/warn/error."""

    def __init__(self):
        self.records = []

    def __call__(self, level, op, message, fields=None, detail=None):
        assert op == _OP
        self.records.append(_Record(level, message, fields or {}))

    def at(self, level):
        return [record for record in self.records if record.level == level]


class InstallTests(unittest.TestCase):
    def setUp(self):
        # A temporary directory for the containerd socket; the socket path
        # constant is pointed into it for every test.
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.socket_path = os.path.join(self._tmp.name, "containerd.sock")
        patcher = mock.patch.object(install_docker, "_CONTAINERD_SOCKET", self.socket_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _install(self, env, *, fail_on=None, captures=None, moby_cfg=None,
                 write_error=None):
        """Run install(env) with everything external mocked.

        fail_on: a command (list) that exec_program reports as failing (rc 1).
        captures: {first word of command: (rc, stdout)} for run_capture.
        Returns (rc, mocks) where mocks is a dict of the fakes.
        """
        captures = {"docker": (0, _DOCKER_OUT), "apt": (0, _APT_OUT),
                    "dnf": (0, _DNF_OUT), "rpm": (0, _RPM_OUT),
                    "dpkg-query": (0, _DPKG_OUT),
                    **(captures or {})}

        def fake_exec(command, op, **kwargs):
            return 1 if command == fail_on else 0

        def fake_capture(command, op, **kwargs):
            return captures[command[0]]

        exec_program = mock.Mock(side_effect=fake_exec)
        logs = _Logs()
        with mock.patch("utils.process.exec_program", exec_program), \
             mock.patch("utils.package_installer.exec_program", exec_program), \
             mock.patch.object(install_docker, "run_capture",
                               side_effect=fake_capture) as run_capture, \
             mock.patch("utils.component_config.config_for",
                        return_value=moby_cfg) as config_for, \
             mock.patch.object(install_docker, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(install_docker.time, "sleep") as sleep, \
             mock.patch("utils.logger._log", logs):
            rc = install_docker.install(env)
        return rc, {"exec": exec_program, "capture": run_capture,
                    "config_for": config_for, "record": record, "sleep": sleep,
                    "info": logs.at("info"), "warn": logs.at("warn"),
                    "error": logs.at("error")}

    @staticmethod
    def _commands(mocks):
        return [c.args[0] for c in mocks["exec"].call_args_list]

    @staticmethod
    def _messages(records):
        return [r.message for r in records]

    # ---- success paths -----------------------------------------------------
    def test_ubuntu_nvidia(self):
        env = _env()
        rc, m = self._install(env)

        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m), [
            [*_APT, "moby-engine"],
            [*_APT, "moby-cli"],
            [*_APT, "moby-buildx"],
            [_TOOLKIT],
            *_SERVICES,
            *_CONTAINERD,
        ])
        # Every command runs under this step's op label, with the build env.
        for call in m["exec"].call_args_list:
            self.assertEqual(call.args[1], _OP)
            self.assertIs(call.kwargs["env"], env)
        self.assertEqual([c.args[0] for c in m["capture"].call_args_list],
                         [["docker", "--version"], ["apt", "list", "--installed"]])
        self.assertEqual(m["record"].call_args_list, [
            mock.call("DOCKER", "29.1.4"),
            mock.call("MOBY_ENGINE", "29.1.4-ubuntu24.04u1"),
        ])
        m["config_for"].assert_not_called()     # only read on baremetal_3p
        self.assertEqual(m["error"], [])

    def test_ubuntu_2604_installs_docker_io_on_every_node_type(self):
        # Checked before baremetal_3p in bash, so no moby pin and no config.
        for node_type in ("azure_vm_regular", "baremetal_3p"):
            with self.subTest(node_type=node_type):
                env = _env("ubuntu26.04", TARGET_NODE_TYPE=node_type)
                rc, m = self._install(env)

                self.assertEqual(rc, 0)
                self.assertEqual(self._commands(m), [
                    [*_APT, "docker.io", "docker-buildx"],
                    [_TOOLKIT],
                    *_SERVICES,
                    *_CONTAINERD,
                ])
                self.assertEqual(
                    [c.args[0] for c in m["capture"].call_args_list],
                    [["docker", "--version"],
                     ["dpkg-query", "-W", "-f=${Version}", "docker.io"]])
                self.assertEqual(m["record"].call_args_list, [
                    mock.call("DOCKER", "29.1.4"),
                    mock.call("MOBY_ENGINE", "28.2.2-0ubuntu1"),
                ])
                m["config_for"].assert_not_called()
                self.assertEqual(m["error"], [])
                messages = self._messages(m["info"])
                self.assertIn("Ubuntu 26.04: installing Canonical's docker.io and "
                              "docker-buildx (no Microsoft moby packages), on every "
                              "node type", messages)
                self.assertIn("Ubuntu 26.04: MOBY_ENGINE records the docker.io "
                              "package version, as bash does", messages)

    def test_ubuntu_2604_docker_io_failures_return_3(self):
        rc, m = self._install(
            _env("ubuntu26.04"),
            fail_on=[*_APT, "docker.io", "docker-buildx"])
        self.assertEqual(rc, 3)
        self.assertEqual(len(self._commands(m)), 1)
        m["record"].assert_not_called()

        # dpkg-query exits 1 when docker.io isn't installed (bash's set -e stops).
        rc, m = self._install(_env("ubuntu26.04"), captures={"dpkg-query": (1, "")})
        self.assertEqual(rc, 3)
        m["record"].assert_called_once_with("DOCKER", "29.1.4")

    def test_ubuntu_baremetal_3p_pins_moby_version(self):
        rc, m = self._install(_env(TARGET_NODE_TYPE="baremetal_3p"),
                              moby_cfg={"version": "29.1.4-ubuntu24.04u1"})

        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m)[:3], [
            [*_APT, "moby-engine=29.1.4-ubuntu24.04u1"],
            [*_APT, "moby-cli=29.1.4-ubuntu24.04u1"],
            [_TOOLKIT],
        ])
        m["config_for"].assert_called_once()
        self.assertEqual(m["config_for"].call_args.args[0], "moby")

    def test_azure_linux(self):
        rc, m = self._install(_env("azurelinux3.0"))

        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m)[:3], [
            [*_DNF, "moby-engine"],
            [*_DNF, "moby-cli"],
            [*_DNF, "docker-buildx"],
        ])
        self.assertEqual(m["capture"].call_args_list[1].args[0], ["rpm", "-qa"])
        m["record"].assert_called_with("MOBY_ENGINE", "25.0.3-11.azl3.x86_64")

    def test_azure_linux_warns_when_first_moby_match_is_not_moby_engine(self):
        # GPU=AMD skips containerd, so its socket warning can't interfere.
        output = "moby-containerd-2.0.0-3.azl3.x86_64\n" + _RPM_OUT
        rc, m = self._install(_env("azurelinux3.0", GPU="AMD"),
                              captures={"rpm": (0, output)})
        self.assertEqual(rc, 0)
        m["record"].assert_called_with("MOBY_ENGINE", "2.0.0-3.azl3.x86_64")
        self.assertEqual(len(m["warn"]), 1)
        self.assertEqual(m["warn"][-1].fields,
                         {"matched_package": "moby-containerd-2.0.0-3.azl3.x86_64",
                          "version": "2.0.0-3.azl3.x86_64"})

    def test_azure_linux_moby_engine_first_does_not_warn(self):
        rc, m = self._install(_env("azurelinux3.0", GPU="AMD"))
        self.assertEqual(rc, 0)
        self.assertEqual(m["warn"], [])

    def test_rpm_moby_source_line(self):
        self.assertEqual(install_docker.rpm_moby_source_line(_RPM_OUT),
                         "moby-engine-25.0.3-11.azl3.x86_64")
        # "moby-x" gives a blank cut result, so the next match is the source.
        self.assertEqual(install_docker.rpm_moby_source_line("moby-x\n" + _RPM_OUT),
                         "moby-engine-25.0.3-11.azl3.x86_64")
        self.assertIsNone(install_docker.rpm_moby_source_line("bash-5.2\n"))

    def test_rhel_family(self):
        rc, m = self._install(_env("almalinux9.8"))

        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m)[:3], [
            [*_DNF, "moby-engine"],
            [*_DNF, "moby-cli"],
            [*_DNF, "moby-buildx"],
        ])
        self.assertEqual(m["capture"].call_args_list[1].args[0],
                         ["dnf", "list", "installed"])
        m["record"].assert_called_with("MOBY_ENGINE", "28.5.1-1.el9")

    def test_baremetal_3p_only_affects_ubuntu(self):
        rc, m = self._install(_env("almalinux9.8", TARGET_NODE_TYPE="baremetal_3p"))
        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m)[2], [*_DNF, "moby-buildx"])
        m["config_for"].assert_not_called()

    def test_amd_skips_the_container_toolkit_and_containerd(self):
        rc, m = self._install(_env(GPU="AMD", SKU="MI300X"))

        self.assertEqual(rc, 0)
        self.assertEqual(self._commands(m), [
            [*_APT, "moby-engine"],
            [*_APT, "moby-cli"],
            [*_APT, "moby-buildx"],
            *_SERVICES,
        ])
        m["sleep"].assert_not_called()
        self.assertTrue(any("Skipping the NVIDIA Container Toolkit" in msg
                            for msg in self._messages(m["info"])))

    def test_unset_gpu_counts_as_nvidia(self):
        env = _env()
        del env["GPU"]
        rc, m = self._install(env)
        self.assertEqual(rc, 0)
        self.assertIn([_TOOLKIT], self._commands(m))

    # ---- containerd socket wait --------------------------------------------
    def test_missing_socket_waits_30_seconds_warns_and_continues(self):
        rc, m = self._install(_env())

        self.assertEqual(rc, 0)
        self.assertEqual(m["sleep"].call_count, 30)
        m["sleep"].assert_called_with(1)
        self.assertEqual(len(m["warn"]), 1)
        self.assertIn("did not appear", m["warn"][-1].message)
        self.assertEqual(m["warn"][-1].fields["waited_s"], 30)
        # Like bash, ctr plugin ls still runs.
        self.assertEqual(self._commands(m)[-1], ["ctr", "plugin", "ls"])

    def test_existing_socket_needs_no_wait(self):
        # A real Unix socket, so the `[ -S ... ]` check is genuinely exercised.
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(server.close)
        server.bind(self.socket_path)

        rc, m = self._install(_env())

        self.assertEqual(rc, 0)
        m["sleep"].assert_not_called()
        self.assertEqual(m["warn"], [])
        ready = [r for r in m["info"] if "socket ready" in r.message]
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].fields["waited_s"], 0)

    def test_regular_file_is_not_a_socket(self):
        with open(self.socket_path, "w", encoding="utf-8"):
            pass
        rc, m = self._install(_env())
        self.assertEqual(rc, 0)
        self.assertEqual(m["sleep"].call_count, 30)

    def test_socket_appearing_during_the_wait(self):
        # _is_socket is False twice, then True: two sleeps, "ready after 2s".
        with mock.patch.object(install_docker, "_is_socket",
                               side_effect=[False, False, True]):
            rc, m = self._install(_env())
        self.assertEqual(rc, 0)
        self.assertEqual(m["sleep"].call_count, 2)
        self.assertEqual(m["warn"], [])

    # ---- logging -----------------------------------------------------------
    def test_key_log_lines(self):
        rc, m = self._install(_env())
        self.assertEqual(rc, 0)

        start = m["info"][0]
        self.assertEqual(start.message, "Installing Docker (moby)")
        self.assertEqual(start.fields, {"distro": "ubuntu24.04", "gpu": "NVIDIA",
                                        "node_type": "azure_vm_regular",
                                        "container_toolkit": True})
        messages = self._messages(m["info"])
        self.assertIn("Installing moby-engine, moby-cli, moby-buildx", messages)
        self.assertIn("Enabling and restarting the docker service", messages)
        recorded = [r.fields for r in m["info"] if r.message.startswith("Recorded")]
        self.assertEqual(recorded, [
            {"component": "DOCKER", "version": "29.1.4"},
            {"component": "MOBY_ENGINE", "version": "29.1.4-ubuntu24.04u1"},
        ])
        summary = m["info"][-1]
        self.assertEqual(summary.message, "Docker installed")
        self.assertEqual(summary.fields["moby_engine_version"], "29.1.4-ubuntu24.04u1")

    # ---- failure paths -----------------------------------------------------
    def test_baremetal_3p_without_moby_version_returns_3(self):
        rc, m = self._install(_env(TARGET_NODE_TYPE="baremetal_3p"), moby_cfg=None)
        self.assertEqual(rc, 3)
        m["exec"].assert_not_called()
        self.assertIn("'moby'", m["error"][-1].message)
        self.assertEqual(m["error"][-1].fields["missing"], "version")

    def test_unknown_distribution_installs_nothing_and_returns_3(self):
        # Bash's else branch would try dnf; install_packages() knows no
        # manager for this distribution.
        rc, m = self._install(_env("sles15"))
        self.assertEqual(rc, 3)
        m["exec"].assert_not_called()
        m["record"].assert_not_called()
        self.assertIn("no package manager known", m["error"][-1].message)

    def test_each_failing_command_returns_3_and_stops(self):
        sequence = [
            [*_APT, "moby-engine"],
            [*_APT, "moby-cli"],
            [*_APT, "moby-buildx"],
            [_TOOLKIT],
            *_SERVICES,
            *_CONTAINERD,
        ]
        for index, failing in enumerate(sequence):
            with self.subTest(failing=failing):
                rc, m = self._install(_env(), fail_on=failing)
                self.assertEqual(rc, 3)
                # Nothing after the failing command ran, and nothing recorded.
                self.assertEqual(self._commands(m), sequence[:index + 1])
                m["capture"].assert_not_called()
                m["record"].assert_not_called()

    def test_docker_version_failure_returns_3(self):
        for result in [(1, _DOCKER_OUT), (0, "")]:
            with self.subTest(result=result):
                rc, m = self._install(_env(), captures={"docker": result})
                self.assertEqual(rc, 3)
                m["record"].assert_not_called()
                self.assertIn("docker version", m["error"][-1].message)

    def test_moby_version_failure_returns_3(self):
        for result in [(1, _APT_OUT), (0, "Listing...\n")]:
            with self.subTest(result=result):
                rc, m = self._install(_env(), captures={"apt": result})
                self.assertEqual(rc, 3)
                # DOCKER was already recorded, as in bash; MOBY_ENGINE never is.
                m["record"].assert_called_once_with("DOCKER", "29.1.4")
                self.assertIn("moby-engine version", m["error"][-1].message)

    def test_unwritable_versions_file_returns_3(self):
        rc, m = self._install(_env(), write_error=PermissionError("denied"))
        self.assertEqual(rc, 3)
        self.assertEqual(len(m["capture"].call_args_list), 1)  # stopped after DOCKER
        self.assertIn("denied", m["error"][-1].message)


# --------------------------------------------------------------------------- #
# Wiring.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "install-docker"]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, install_docker.install)
        self.assertEqual(steps[0].script, "install_docker.sh")


if __name__ == "__main__":
    unittest.main()
