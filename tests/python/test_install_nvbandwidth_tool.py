"""Unit tests for components/python/install_nvbandwidth_tool.py.

Two kinds of test:

  * The pure helpers (dependency_packages, configured_ref, configured_commit)
    are tested directly. configured_ref/configured_commit were
    also compared with real jq/bash in WSL on the same configs.
  * install() is tested with exec_program (in utils.process, behind run and
    run_script, and in utils.package_installer, which runs the dependency
    installs) replaced by one fake that records each command and imitates
    its effect on disk: `git clone` creates the source
    directory and the `bash -c` build creates the nvbandwidth binary.
    write_repo_source_info is mocked and recorded in the same list, so the
    order is checked. The install paths (_DEST_DIR, _WORK_DIR) point into a
    temporary directory, so the real file moves and cleanup run without
    touching /opt or /tmp, and nothing is installed, cloned or compiled.

(These replace the NvbandwidthTests that used to live in test_packaging.py.)

Run from the repo root:
    python3 -m unittest tests.python.test_install_nvbandwidth_tool -v
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_nvbandwidth_tool as m
from utils import component_config, package_installer, process
from utils.build_config import BuildConfig, build_plan
from utils.distro import Distro

_OP = "install-nvbandwidth"
_URL = "https://github.com/NVIDIA/nvbandwidth"
_COMMIT = "4a49bdae82d69e69330a88746cb12b449796ee8b"

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Package lists, one per bash install command.
_UBUNTU_PACKAGES = [["libboost-program-options-dev"]]
_DNF_BOOST_PACKAGES = [["boost-devel"]]
_AZL_ARM_PACKAGES = [["boost-devel", "boost-program-options"], ["cmake"]]
# The commands install_packages() runs for them.
_APT = ["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0"]
_DNF = ["dnf", "install", "-y"]
_UBUNTU_DEPS = [[*_APT, *p] for p in _UBUNTU_PACKAGES]
_DNF_BOOST = [[*_DNF, *p] for p in _DNF_BOOST_PACKAGES]
_AZL_ARM_DEPS = [[*_DNF, *p] for p in _AZL_ARM_PACKAGES]
# The one-shell build run_script() runs: HPC-X loaded, then cmake and make.
_BUILD_SCRIPT = ("set -e\n"
                 "source /etc/profile.d/modules.sh\n"
                 "module load mpi/hpcx\n"
                 "cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DMULTINODE=1 .\n"
                 "make\n")


class _FakeExec:
    """Stands in for exec_program: records calls and imitates their effects.

    fail_at, when given, is the index of the command that should fail (exit 1),
    counting commands only (not the write_repo_source_info markers).
    """

    def __init__(self, fail_at=None, clone_creates_source=True, build_creates_binary=True):
        self.calls = []          # (command, cwd) for every call, in order
        self.commands = 0
        self.fail_at = fail_at
        self.clone_creates_source = clone_creates_source
        self.build_creates_binary = build_creates_binary

    def __call__(self, command, op, *, cwd=None, env=None):
        assert op == _OP
        self.calls.append((command, cwd))
        self.commands += 1
        if self.commands - 1 == self.fail_at:
            return 1
        if command[:2] == ["git", "clone"]:
            source = os.path.join(cwd, "nvbandwidth")
            if os.path.exists(source):
                return 128       # what git does when the target already exists
            if self.clone_creates_source:
                os.makedirs(source)
        elif command[:2] == ["bash", "-c"] and self.build_creates_binary:
            Path(cwd, "nvbandwidth").write_text("built binary\n", encoding="utf-8")
        return 0


def _env(distribution="ubuntu24.04", arch="aarch64", *, config=None, node_type=None):
    """A minimal build env. config=None uses version 0.9, the GitHub url and a
    pinned commit (no ref: it defaults to v0.9); config={} leaves nvbandwidth
    out of COMPONENT_VERSIONS."""
    if config is None:
        config = {"version": "0.9", "url": _URL, "commit": _COMMIT}
    versions = {"nvbandwidth": {"common": config}} if config else {}
    env = {"DISTRIBUTION": distribution, "ARCHITECTURE": arch,
           "COMPONENT_VERSIONS": json.dumps(versions)}
    if node_type:
        env["TARGET_NODE_TYPE"] = node_type
    return env


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
class HelperTests(unittest.TestCase):
    def test_dependency_packages_follow_bash_branches(self):
        def deps(name, arch):
            return m.dependency_packages(Distro(name), arch)
        self.assertEqual(list(deps("ubuntu24.04", "aarch64")), _UBUNTU_PACKAGES)
        self.assertEqual(list(deps("ubuntu26.04", "x86_64")), _UBUNTU_PACKAGES)
        self.assertEqual(list(deps("almalinux9.8", "x86_64")), _DNF_BOOST_PACKAGES)
        self.assertEqual(list(deps("almalinux8.10", "x86_64")), _DNF_BOOST_PACKAGES)
        self.assertEqual(list(deps("azurelinux3.0", "x86_64")), _DNF_BOOST_PACKAGES)
        self.assertEqual(list(deps("azurelinux3.0", "aarch64")), _AZL_ARM_PACKAGES)
        for other in ("rocky9.8", "rhel9.8", "azurelinux3.1", ""):
            with self.subTest(distribution=other):
                self.assertEqual(deps(other, "x86_64"), ())

    def test_configured_ref_follows_jq_and_bash_defaults(self):
        ref = m.configured_ref
        self.assertEqual(ref({}, "0.9"), "v0.9")
        self.assertEqual(ref({"ref": "v0.8"}, "0.9"), "v0.8")
        self.assertEqual(ref({"ref": "v0.8", "branch": "main"}, "0.9"), "v0.8")
        self.assertEqual(ref({"branch": "main"}, "0.9"), "main")
        self.assertEqual(ref({"ref": None, "branch": "main"}, "0.9"), "main")
        self.assertEqual(ref({"ref": False, "branch": "main"}, "0.9"), "main")
        self.assertEqual(ref({"ref": "", "branch": "main"}, "0.9"), "v0.9")

    def test_configured_commit_is_empty_when_unset(self):
        self.assertEqual(m.configured_commit({"commit": _COMMIT}), _COMMIT)
        self.assertEqual(m.configured_commit({}), "")
        self.assertEqual(m.configured_commit({"commit": None}), "")
        self.assertEqual(m.configured_commit({"commit": False}), "")

    def test_build_lines_match_the_bash_script(self):
        script = (_REPO_ROOT / "components" / "install_nvbandwidth_tool.sh").read_text(
            encoding="utf-8")
        self.assertIn("module load mpi/hpcx\n" + "\n".join(m._BUILD_LINES) + "\n", script)

    def test_bash_script_still_matches(self):
        script = (_REPO_ROOT / "components" / "install_nvbandwidth_tool.sh").read_text(
            encoding="utf-8")
        for line in [
            "dest_dir=/opt/nvidia/nvbandwidth",
            "    apt install -y libboost-program-options-dev",
            "elif [[ $DISTRIBUTION == almalinux* ]]; then",
            'elif [[ $DISTRIBUTION == "azurelinux3.0" && $ARCHITECTURE != "aarch64" ]]; then',
            "    dnf -y install boost-devel",
            "    dnf install -y boost-devel boost-program-options",
            "    dnf install -y cmake",
            "NVBANDWIDTH_COMMIT=$(jq -r '.commit // empty' <<< $nvbandwidth_metadata)",
            "NVBANDWIDTH_REF=$(jq -r '.ref // .branch // empty' <<< $nvbandwidth_metadata)",
            "NVBANDWIDTH_REF=${NVBANDWIDTH_REF:-v${NVBANDWIDTH_VERSION}}",
            "git clone --branch ${NVBANDWIDTH_REF} ${NVBANDWIDTH_DOWNLOAD_URL}",
            "pushd nvbandwidth",
            "git checkout ${NVBANDWIDTH_COMMIT}",
            'write_repo_source_info "NVBANDWIDTH" . \\',
            '    --commit "${NVBANDWIDTH_COMMIT}" \\',
            '    --ref "${NVBANDWIDTH_REF}"',
            "source /etc/profile.d/modules.sh",
            "module load mpi/hpcx",
            "cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DMULTINODE=1 .",
            "\nmake\n",
            "mv ./nvbandwidth $dest_dir",
            "rm -rf ./nvbandwidth",
            'write_component_version "NVBANDWIDTH" ${NVBANDWIDTH_VERSION}',
        ]:
            with self.subTest(line=line):
                self.assertIn(line, script)


# --------------------------------------------------------------------------- #
# install(): exec_program faked, real file operations in a temp directory.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.dest = self.root / "opt" / "nvidia" / "nvbandwidth"
        self.work = self.root / "work"
        self.source = str(self.work / "nvbandwidth")

    def _install(self, env, fake=None, write_error=None, source_result=True):
        """Run install(env) with the externals faked and paths in the sandbox.

        Returns (rc, fake, mocks); fake.calls lists the commands and the
        write_repo_source_info markers in order.
        """
        fake = fake or _FakeExec()

        def fake_source(component, repo_dir, op, *, commit=None, ref=None, env=None):
            self.assertEqual(op, _OP)
            fake.calls.append((("repo_source", component, commit, ref), repo_dir))
            if isinstance(source_result, Exception):
                raise source_result
            return source_result

        with mock.patch.object(m, "_DEST_DIR", str(self.dest)), \
             mock.patch.object(m, "_WORK_DIR", str(self.work)), \
             mock.patch.object(process, "exec_program", side_effect=fake), \
             mock.patch.object(package_installer, "exec_program", side_effect=fake), \
             mock.patch.object(package_installer, "log_info"), \
             mock.patch.object(package_installer, "log_error") as pkg_error, \
             mock.patch.object(m, "write_repo_source_info", side_effect=fake_source), \
             mock.patch.object(m, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(m, "log_info") as log_info, \
             mock.patch.object(m, "log_warn") as log_warn, \
             mock.patch.object(component_config, "log_error") as log_error, \
             mock.patch.object(m, "log_error", log_error):
            rc = m.install(env)
        return rc, fake, {"record": record, "info": log_info, "warn": log_warn,
                          "error": log_error, "pkg_error": pkg_error}

    def _build_calls(self, *, ref="v0.9", commit=_COMMIT):
        calls = [(["git", "clone", "--branch", ref, _URL], str(self.work))]
        if commit:
            calls.append((["git", "checkout", commit], self.source))
        calls += [(("repo_source", "NVBANDWIDTH", commit or None, ref), self.source),
                  (["bash", "-c", _BUILD_SCRIPT], self.source)]
        return calls

    @staticmethod
    def _messages(log_mock):
        return [c.args[1] for c in log_mock.call_args_list]

    # ---- success paths -----------------------------------------------------
    def test_ubuntu_happy_path_mirrors_bash(self):
        rc, fake, mocks = self._install(_env())
        self.assertEqual(rc, 0)
        self.assertEqual(fake.calls, [(c, None) for c in _UBUNTU_DEPS] + self._build_calls())
        self.assertEqual((self.dest / "nvbandwidth").read_text(encoding="utf-8"),
                         "built binary\n")
        self.assertFalse(self.work.exists())          # bash's rm -rf ./nvbandwidth
        mocks["record"].assert_called_once_with("NVBANDWIDTH", "0.9")
        mocks["error"].assert_not_called()
        mocks["warn"].assert_not_called()
        start = mocks["info"].call_args_list[0].kwargs
        self.assertEqual((start["version"], start["ref"], start["commit"], start["source"]),
                         ("0.9", "v0.9", _COMMIT, _URL))
        self.assertIn(f"Installed {self.dest / 'nvbandwidth'}", self._messages(mocks["info"]))
        self.assertEqual(self._messages(mocks["info"])[-1],
                         f"nvbandwidth 0.9 installed to {self.dest}")

    def test_azure_linux_aarch64_installs_cmake_too(self):
        rc, fake, _ = self._install(_env("azurelinux3.0", "aarch64"))
        self.assertEqual(rc, 0)
        self.assertEqual(fake.calls, [(c, None) for c in _AZL_ARM_DEPS] + self._build_calls())

    def test_dnf_distros(self):
        for distribution, arch in [("azurelinux3.0", "x86_64"), ("almalinux9.8", "x86_64")]:
            with self.subTest(distribution=distribution):
                rc, fake, _ = self._install(_env(distribution, arch))
                self.assertEqual(rc, 0)
                self.assertEqual(fake.calls[0], (_DNF_BOOST[0], None))

    def test_distro_without_deps_branch_warns_and_builds(self):
        rc, fake, mocks = self._install(_env("rocky9.8", "x86_64"))
        self.assertEqual(rc, 0)
        self.assertEqual(fake.calls, self._build_calls())
        self.assertTrue(any("installs no dependencies on 'rocky9.8'" in msg
                            for msg in self._messages(mocks["warn"])))

    def test_configured_ref_is_cloned(self):
        for extra, ref in [({"ref": "v0.8"}, "v0.8"), ({"branch": "main"}, "main")]:
            with self.subTest(extra=extra):
                config = {"version": "0.9", "url": _URL, "commit": _COMMIT, **extra}
                rc, fake, _ = self._install(_env(config=config))
                self.assertEqual(rc, 0)
                self.assertEqual(fake.calls[1:], self._build_calls(ref=ref))

    def test_no_pinned_commit_skips_checkout_and_reads_it_from_git(self):
        config = {"version": "0.9", "url": _URL}
        rc, fake, mocks = self._install(_env(config=config))
        self.assertEqual(rc, 0)
        self.assertEqual(fake.calls[1:], self._build_calls(commit=""))
        self.assertTrue(any("pins no nvbandwidth commit" in msg
                            for msg in self._messages(mocks["warn"])))

    def test_rerun_replaces_a_leftover_checkout_and_binary(self):
        os.makedirs(self.source)
        self.dest.mkdir(parents=True)
        (self.dest / "nvbandwidth").write_text("old\n", encoding="utf-8")
        rc, _, mocks = self._install(_env())
        self.assertEqual(rc, 0)
        self.assertEqual((self.dest / "nvbandwidth").read_text(encoding="utf-8"),
                         "built binary\n")
        installed = [c for c in mocks["info"].call_args_list
                     if c.args[1].startswith("Installed ")]
        self.assertEqual(installed[0].kwargs["action"], "replaced")

    def test_real_versions_json_honours_the_baremetal_3p_override(self):
        versions = (_REPO_ROOT / "versions.json").read_text(encoding="utf-8")
        cases = {None: ("0.9", "4a49bdae82d69e69330a88746cb12b449796ee8b"),
                 "baremetal_3p": ("0.8", "66746a3bef61c8c2e12ab34955310da70b9e38cb")}
        for node_type, (version, commit) in cases.items():
            with self.subTest(node_type=node_type):
                env = _env(node_type=node_type)
                env.update(GPU="NVIDIA", SKU="GB200", COMPONENT_VERSIONS=versions)
                rc, fake, mocks = self._install(env)
                self.assertEqual(rc, 0)
                self.assertEqual(fake.calls[1:], self._build_calls(ref=f"v{version}",
                                                                   commit=commit))
                mocks["record"].assert_called_once_with("NVBANDWIDTH", version)

    # ---- failure paths: each must return 3 and log an error ----------------
    def test_missing_config_returns_3_before_any_change(self):
        for config in ({}, {"version": "0.9"}, {"url": _URL}):
            with self.subTest(config=config):
                rc, fake, mocks = self._install(_env(config=config))
                self.assertEqual(rc, 3)
                self.assertEqual(fake.calls, [])
                self.assertFalse(self.dest.exists())
                mocks["record"].assert_not_called()
                mocks["error"].assert_called_once()

    def test_each_failing_command_returns_3_and_stops(self):
        # Azure Linux aarch64 runs 5 commands: 2 deps, clone, checkout, build.
        for fail_at in range(5):
            with self.subTest(fail_at=fail_at):
                shutil.rmtree(self.work, ignore_errors=True)
                rc, fake, mocks = self._install(_env("azurelinux3.0", "aarch64"),
                                                _FakeExec(fail_at=fail_at))
                self.assertEqual(rc, 3)
                self.assertEqual(fake.commands, fail_at + 1)   # nothing after it
                self.assertIsInstance(fake.calls[-1][0], list)
                self.assertFalse((self.dest / "nvbandwidth").exists())
                mocks["record"].assert_not_called()
                # A failed command is logged where it ran: by install_packages()
                # for the dependencies, else by exec_program (faked here).
                if fail_at < 2:
                    mocks["pkg_error"].assert_called_once()
                mocks["error"].assert_not_called()

    def test_repo_source_failure_returns_3(self):
        for result in (False, PermissionError("denied")):
            with self.subTest(result=result):
                shutil.rmtree(self.work, ignore_errors=True)
                rc, fake, mocks = self._install(_env(), source_result=result)
                self.assertEqual(rc, 3)
                self.assertEqual(fake.calls[-1][0][0], "repo_source")
                mocks["record"].assert_not_called()
                self.assertIn("could not record the nvbandwidth source",
                              mocks["error"].call_args.args[1])

    def test_clone_without_a_source_directory_returns_3(self):
        rc, fake, mocks = self._install(_env(), _FakeExec(clone_creates_source=False))
        self.assertEqual(rc, 3)
        self.assertEqual(fake.commands, 2)     # deps, clone
        self.assertIn("did not create", mocks["error"].call_args.args[1])

    def test_build_without_a_binary_returns_3(self):
        rc, _, mocks = self._install(_env(), _FakeExec(build_creates_binary=False))
        self.assertEqual(rc, 3)
        mocks["record"].assert_not_called()
        self.assertIn("could not move", mocks["error"].call_args.args[1])

    def test_uncreatable_dest_dir_returns_3_without_raising(self):
        (self.root / "opt").write_text("not a directory\n", encoding="utf-8")
        rc, fake, mocks = self._install(_env())
        self.assertEqual(rc, 3)
        self.assertEqual(fake.calls, [])
        mocks["error"].assert_called_once()

    def test_unwritable_versions_file_returns_3_without_raising(self):
        rc, _, mocks = self._install(_env(), write_error=PermissionError("denied"))
        self.assertEqual(rc, 3)
        self.assertIn("denied", mocks["error"].call_args.args[1])


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="GB200", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == _OP]
        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, m.install)
        self.assertEqual(steps[0].script, "install_nvbandwidth_tool.sh")


if __name__ == "__main__":
    unittest.main()
