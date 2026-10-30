"""Unit tests for components/python/install_nvloom.py.

Two kinds of test:

  * The pure helpers (dependency_packages, configured_ref, patch_path,
    build_script) are tested directly. configured_ref was also compared with
    real jq/bash in WSL on the same configs.
  * install() is tested with exec_program (the module's and
    utils.package_installer's, which runs the dependency installs) replaced
    by one fake that records each command and imitates its effect on disk: `git clone` creates the source
    directory and the `bash -c` build creates the nvloom_cli binary. The
    install paths (_DEST_DIR, _WORK_DIR) point into a temporary directory, so
    the real file moves and cleanup run without touching /opt or /tmp, and
    nothing is installed, cloned or compiled. write_repo_source_info is
    mocked too; its calls are recorded in the same list as the commands, so
    the order (clone, checkout, record, patch, build) is checked.

Run from the repo root:
    python3 -m unittest tests.python.test_install_nvloom -v
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_nvloom
from utils import package_installer
from utils.build_config import BuildConfig, build_plan

_OP = "install-nvloom"
_URL = "https://github.com/NVIDIA/nvloom.git"
_COMMIT = "bc0983828805d9ef2a912768fdae1465d581ef7c"

# tests/python/<this file> -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_AZL_PACKAGES = [
    ["build-essential"],
    ["boost-devel", "boost-program-options"],
    ["cmake"],
]
_UBUNTU_PACKAGES = [
    ["build-essential"],
    ["libboost-program-options-dev"],
    ["cmake"],
]
# The commands install_packages() runs for them.
_AZL_DEPS = [["dnf", "install", "-y", *p] for p in _AZL_PACKAGES]
_UBUNTU_DEPS = [["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0", *p]
                for p in _UBUNTU_PACKAGES]


class _FakeExec:
    """Stands in for exec_program: records calls and imitates their effects.

    fail_at, when given, is the index of the command that should fail (exit 1),
    counting commands only (not the write_repo_source_info markers).
    """

    def __init__(self, fail_at=None, clone_creates_source=True, build_creates_binary=True):
        self.calls = []          # (command, cwd) for every call, in order
        self.commands = 0        # how many commands (not other markers) ran
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
            source = os.path.join(cwd, "nvloom")
            if os.path.exists(source):
                return 128       # what git does when the target already exists
            if self.clone_creates_source:
                os.makedirs(source)
        elif command[:2] == ["bash", "-c"] and self.build_creates_binary:
            Path(cwd, "nvloom_cli").write_text("built binary\n", encoding="utf-8")
        return 0


class _Sandbox:
    """A temporary directory holding the dest dir, work dir and a fake repo
    checkout (TOP_DIR) that contains the Azure Linux patch."""

    def __init__(self, root):
        self.root = Path(root)
        self.dest = self.root / "opt" / "nvidia" / "nvloom"
        self.work = self.root / "work"
        self.top_dir = self.root / "repo"
        self.patch = self.top_dir / "distros" / "azurelinux3.0" / "azurelinux_nvloom.patch"
        self.patch.parent.mkdir(parents=True)
        self.patch.write_text("diff --git a/CMakeLists.txt b/CMakeLists.txt\n",
                              encoding="utf-8")


def _env(distribution, sandbox, *, config=None, node_type=None):
    """A minimal build env. config=None uses version 1.2.0, the GitHub url and
    a pinned commit (no ref: it defaults to v1.2.0); config={} leaves nvloom
    out of COMPONENT_VERSIONS."""
    if config is None:
        config = {"version": "1.2.0", "url": _URL, "commit": _COMMIT}
    versions = {"nvloom": {distribution: {"aarch64": config}}} if config else {}
    env = {
        "DISTRIBUTION": distribution,
        "ARCHITECTURE": "aarch64",
        "TOP_DIR": str(sandbox.top_dir),
        "COMPONENT_VERSIONS": json.dumps(versions),
    }
    if node_type:
        env["TARGET_NODE_TYPE"] = node_type
    return env


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
class HelperTests(unittest.TestCase):
    def test_dependency_packages_follow_bash_distro_tests(self):
        packages = install_nvloom.dependency_packages
        self.assertEqual(list(packages("azurelinux3.0")), _AZL_PACKAGES)
        self.assertEqual(list(packages("ubuntu24.04")), _UBUNTU_PACKAGES)
        self.assertEqual(list(packages("ubuntu22.04")), _UBUNTU_PACKAGES)
        for other in ("almalinux9.8", "rocky8.10", "azurelinux3.1", ""):
            with self.subTest(distribution=other):
                self.assertIsNone(install_nvloom.dependency_packages(other))

    def test_configured_ref_follows_jq_and_bash_defaults(self):
        ref = install_nvloom.configured_ref
        self.assertEqual(ref({}, "1.2.0"), "v1.2.0")
        self.assertEqual(ref({"ref": "v9"}, "1.2.0"), "v9")
        self.assertEqual(ref({"ref": "v9", "branch": "main"}, "1.2.0"), "v9")
        self.assertEqual(ref({"branch": "main"}, "1.2.0"), "main")
        self.assertEqual(ref({"ref": None, "branch": "main"}, "1.2.0"), "main")
        self.assertEqual(ref({"ref": False, "branch": "main"}, "1.2.0"), "main")
        # jq keeps "" (it hides branch); bash's ${REF:-...} then defaults it.
        self.assertEqual(ref({"ref": "", "branch": "main"}, "1.2.0"), "v1.2.0")

    def test_bash_script_still_matches(self):
        script = (_REPO_ROOT / "components" / "install_nvloom.sh").read_text(encoding="utf-8")
        for line in [
            "NVLOOM_COMMIT=$(jq -r '.commit' <<< $nvloom_metadata)",
            "NVLOOM_REF=$(jq -r '.ref // .branch // empty' <<< $nvloom_metadata)",
            "NVLOOM_REF=${NVLOOM_REF:-v${NVLOOM_VERSION}}",
            "git clone $NVLOOM_DOWNLOAD_URL --branch $NVLOOM_REF",
            "git checkout $NVLOOM_COMMIT",
            'write_repo_source_info "NVLOOM" . \\',
            '--commit "${NVLOOM_COMMIT}" \\',
            '--ref "${NVLOOM_REF}"',
            'cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="100" .',
            'write_component_version "NVLOOM" ${NVLOOM_VERSION}',
        ]:
            with self.subTest(line=line):
                self.assertIn(line, script)
        # Both distro branches clone, check out and record the source.
        self.assertEqual(script.count("git checkout $NVLOOM_COMMIT"), 2)
        self.assertEqual(script.count('write_repo_source_info "NVLOOM"'), 2)

    def test_patch_path_is_inside_the_repo_checkout(self):
        self.assertEqual(
            install_nvloom.patch_path("/src/azhpc-images"),
            "/src/azhpc-images/distros/azurelinux3.0/azurelinux_nvloom.patch")

    def test_the_patch_exists_in_this_repo(self):
        self.assertTrue(os.path.isfile(install_nvloom.patch_path(str(_REPO_ROOT))))

    def test_build_script_loads_hpcx_then_builds_in_one_shell(self):
        self.assertEqual(install_nvloom.build_script().splitlines(), [
            "set -e",
            "source /etc/profile.d/modules.sh",
            "module load mpi/hpcx",
            'cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc '
            '-DCMAKE_CUDA_ARCHITECTURES="100" .',
            "make -j $(nproc)",
            "module unload mpi/hpcx",
        ])


# --------------------------------------------------------------------------- #
# install(): exec_program faked, real file operations in a temp directory.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        # addCleanup runs after each test, even a failing one.
        self.addCleanup(temp.cleanup)
        self.box = _Sandbox(temp.name)

    def _install(self, env, fake=None, write_error=None, source_result=True):
        """Run install(env) with exec_program faked and paths in the sandbox.

        source_result: what the mocked write_repo_source_info returns (or an
        exception it raises).
        Returns the exit code, the fake (its .calls list) and the mocks for
        write_component_version and log_error.
        """
        fake = fake or _FakeExec()

        def fake_source(component, repo_dir, op, *, commit=None, ref=None, env=None):
            fake.calls.append((("repo_source", component, commit, ref), repo_dir))
            if isinstance(source_result, Exception):
                raise source_result
            return source_result

        with mock.patch.object(install_nvloom, "_DEST_DIR", str(self.box.dest)), \
             mock.patch.object(install_nvloom, "_WORK_DIR", str(self.box.work)), \
             mock.patch.object(install_nvloom, "exec_program", side_effect=fake), \
             mock.patch.object(package_installer, "exec_program", side_effect=fake), \
             mock.patch.object(package_installer, "log_info"), \
             mock.patch.object(package_installer, "log_error") as self.pkg_error, \
             mock.patch.object(install_nvloom, "write_repo_source_info",
                               side_effect=fake_source), \
             mock.patch.object(install_nvloom, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(install_nvloom, "log_info") as self.log_info, \
             mock.patch.object(install_nvloom, "log_error") as log_error:
            rc = install_nvloom.install(env)
        return rc, fake, record, log_error

    def _expected_build_calls(self, *, with_patch, ref="v1.2.0", commit=_COMMIT):
        source = str(self.box.work / "nvloom")
        calls = [(["git", "clone", _URL, "--branch", ref], str(self.box.work)),
                 (["git", "checkout", commit], source),
                 (("repo_source", "NVLOOM", commit, ref), source)]
        if with_patch:
            calls.append((["git", "apply", str(self.box.patch)], source))
        calls.append((["bash", "-c", install_nvloom.build_script()], source))
        return calls

    # ---- success paths -----------------------------------------------------
    def test_azure_linux_builds_with_the_repo_patch(self):
        rc, fake, record, log_error = self._install(_env("azurelinux3.0", self.box))

        self.assertEqual(rc, 0)
        self.assertEqual(fake.calls,
                         [(c, None) for c in _AZL_DEPS]
                         + self._expected_build_calls(with_patch=True))
        self.assertEqual((self.box.dest / "nvloom_cli").read_text(encoding="utf-8"),
                         "built binary\n")
        self.assertFalse(self.box.work.exists())     # bash's rm -rf ./nvloom
        record.assert_called_once_with("NVLOOM", "1.2.0")
        log_error.assert_not_called()
        start = self.log_info.call_args_list[0].kwargs
        self.assertEqual((start["version"], start["ref"], start["commit"], start["source"]),
                         ("1.2.0", "v1.2.0", _COMMIT, _URL))
        self.assertTrue(any("Applied azurelinux_nvloom.patch" in c.args[1]
                            for c in self.log_info.call_args_list))

    def test_ubuntu_builds_without_a_patch(self):
        rc, fake, record, log_error = self._install(_env("ubuntu24.04", self.box))

        self.assertEqual(rc, 0)
        self.assertEqual(fake.calls,
                         [(c, None) for c in _UBUNTU_DEPS]
                         + self._expected_build_calls(with_patch=False))
        self.assertTrue((self.box.dest / "nvloom_cli").is_file())
        record.assert_called_once_with("NVLOOM", "1.2.0")
        log_error.assert_not_called()

    def test_configured_ref_and_branch_are_cloned(self):
        for extra, ref in [({"ref": "v1.2.0-rc"}, "v1.2.0-rc"), ({"branch": "main"}, "main")]:
            with self.subTest(extra=extra):
                config = {"version": "1.2.0", "url": _URL, "commit": _COMMIT, **extra}
                rc, fake, _, _ = self._install(_env("ubuntu24.04", self.box, config=config))
                self.assertEqual(rc, 0)
                self.assertEqual(fake.calls[3:],
                                 self._expected_build_calls(with_patch=False, ref=ref))

    def test_checkout_failure_stops_before_recording(self):
        rc, fake, record, log_error = self._install(_env("ubuntu24.04", self.box),
                                                    _FakeExec(fail_at=4))
        self.assertEqual(rc, 3)
        self.assertEqual(fake.calls[-1][0], ["git", "checkout", _COMMIT])
        record.assert_not_called()

    def test_repo_source_failure_returns_3(self):
        for result in (False, PermissionError("denied")):
            with self.subTest(result=result):
                shutil.rmtree(self.box.work, ignore_errors=True)
                rc, fake, record, log_error = self._install(
                    _env("azurelinux3.0", self.box), source_result=result)
                self.assertEqual(rc, 3)
                self.assertEqual(fake.calls[-1][0][0], "repo_source")   # nothing after it
                record.assert_not_called()
                self.assertIn("could not record the nvloom source",
                              log_error.call_args[0][1])

    def test_patch_is_never_read_from_hpcuser_home(self):
        # Regression: bash hardcodes /home/hpcuser/azhpc-images/....
        _, fake, _, _ = self._install(_env("azurelinux3.0", self.box))

        commands = [cmd for cmd, _ in fake.calls if isinstance(cmd, list)]
        applied = [cmd for cmd in commands if cmd[:2] == ["git", "apply"]]
        self.assertEqual(applied, [["git", "apply", str(self.box.patch)]])
        self.assertFalse(any("/home/hpcuser" in " ".join(cmd) for cmd in commands))

    def test_rerun_replaces_a_leftover_checkout_and_binary(self):
        # A failed earlier run left a checkout behind; a successful one left
        # a binary. Bash's git clone would fail on the leftover checkout.
        (self.box.work / "nvloom").mkdir(parents=True)
        self.box.dest.mkdir(parents=True)
        (self.box.dest / "nvloom_cli").write_text("old\n", encoding="utf-8")

        rc, _, _, _ = self._install(_env("ubuntu24.04", self.box))

        self.assertEqual(rc, 0)
        self.assertEqual((self.box.dest / "nvloom_cli").read_text(encoding="utf-8"),
                         "built binary\n")

    def test_real_versions_json_honours_the_baremetal_3p_override(self):
        versions = (_REPO_ROOT / "versions.json").read_text(encoding="utf-8")
        cases = {None: ("2.0.0", "7426efd0f6a9feb8cab70e9c5745c1a9eefaf352"),
                 "baremetal_3p": ("1.2.0", "bc0983828805d9ef2a912768fdae1465d581ef7c")}
        for node_type, (expected, commit) in cases.items():
            with self.subTest(node_type=node_type):
                env = _env("ubuntu24.04", self.box, node_type=node_type)
                env.update(GPU="NVIDIA", SKU="GB200", COMPONENT_VERSIONS=versions)
                rc, fake, record, _ = self._install(env)
                self.assertEqual(rc, 0)
                self.assertIn(f"v{expected}", fake.calls[3][0])
                self.assertEqual(fake.calls[4][0], ["git", "checkout", commit])
                self.assertEqual(fake.calls[5][0],
                                 ("repo_source", "NVLOOM", commit, f"v{expected}"))
                record.assert_called_once_with("NVLOOM", expected)

    # ---- failure paths: each must return 3 and log an error ----------------
    def test_missing_config_returns_3_before_any_change(self):
        for config in ({}, {"version": "1.2.0"}, {"url": _URL},
                       {"version": "1.2.0", "url": _URL},
                       {"version": "1.2.0", "url": _URL, "commit": ""}):
            with self.subTest(config=config):
                rc, fake, record, log_error = self._install(
                    _env("ubuntu24.04", self.box, config=config))
                self.assertEqual(rc, 3)
                self.assertEqual(fake.calls, [])
                self.assertFalse(self.box.dest.exists())
                record.assert_not_called()
                log_error.assert_called_once()

    def test_unsupported_distro_returns_3_without_recording(self):
        # Bash would skip the build but still record NVLOOM.
        rc, fake, record, log_error = self._install(_env("almalinux9.8", self.box))

        self.assertEqual(rc, 3)
        self.assertEqual(fake.calls, [])
        self.assertFalse(self.box.dest.exists())
        record.assert_not_called()
        self.assertIn("almalinux9.8", log_error.call_args[0][1])

    def test_azure_linux_without_top_dir_returns_3(self):
        env = _env("azurelinux3.0", self.box)
        del env["TOP_DIR"]
        rc, fake, record, log_error = self._install(env)

        self.assertEqual(rc, 3)
        self.assertEqual(fake.calls, [])
        record.assert_not_called()
        log_error.assert_called_once()

    def test_azure_linux_with_missing_patch_returns_3(self):
        self.box.patch.unlink()
        rc, fake, record, log_error = self._install(_env("azurelinux3.0", self.box))

        self.assertEqual(rc, 3)
        self.assertEqual(fake.calls, [])
        record.assert_not_called()
        self.assertIn("azurelinux_nvloom.patch", log_error.call_args[0][1])

    def test_each_failing_command_returns_3_and_stops(self):
        # Azure Linux runs 7 commands: 3 deps, clone, checkout, apply, build.
        # (write_repo_source_info runs between checkout and apply; it is not
        # a command, so it doesn't count for fail_at.)
        for fail_at in range(7):
            with self.subTest(fail_at=fail_at):
                # Each sub-test starts like a fresh build area. (install()
                # would also clear it; this keeps the sub-tests independent.)
                shutil.rmtree(self.box.work, ignore_errors=True)
                rc, fake, record, log_error = self._install(
                    _env("azurelinux3.0", self.box), _FakeExec(fail_at=fail_at))
                self.assertEqual(rc, 3)
                commands = [c for c, _ in fake.calls if isinstance(c, list)]
                self.assertEqual(len(commands), fail_at + 1)   # nothing after it
                self.assertIsInstance(fake.calls[-1][0], list)
                self.assertFalse((self.box.dest / "nvloom_cli").exists())
                record.assert_not_called()
                # install_packages() logs a failing dependency install itself.
                errors = self.pkg_error if fail_at < 3 else log_error
                errors.assert_called_once()

    def test_clone_without_an_nvloom_directory_returns_3(self):
        rc, fake, record, log_error = self._install(
            _env("ubuntu24.04", self.box), _FakeExec(clone_creates_source=False))

        self.assertEqual(rc, 3)
        self.assertEqual(len(fake.calls), 4)     # stopped after the clone
        record.assert_not_called()
        log_error.assert_called_once()

    def test_build_without_a_binary_returns_3(self):
        rc, _, record, log_error = self._install(
            _env("ubuntu24.04", self.box), _FakeExec(build_creates_binary=False))

        self.assertEqual(rc, 3)
        record.assert_not_called()
        self.assertIn("nvloom_cli", log_error.call_args[0][1])

    def test_uncreatable_dest_dir_returns_3_without_raising(self):
        # A regular file where a parent directory should be makes makedirs
        # raise an OSError (NotADirectoryError).
        (self.box.root / "opt").write_text("not a directory\n", encoding="utf-8")
        rc, fake, record, log_error = self._install(_env("ubuntu24.04", self.box))

        self.assertEqual(rc, 3)
        self.assertEqual(fake.calls, [])
        record.assert_not_called()
        log_error.assert_called_once()

    def test_unwritable_versions_file_returns_3_without_raising(self):
        rc, _, _, log_error = self._install(
            _env("ubuntu24.04", self.box),
            write_error=PermissionError("/opt/azurehpc: permission denied"))

        self.assertEqual(rc, 3)
        self.assertIn("permission denied", log_error.call_args[0][1])


# --------------------------------------------------------------------------- #
# Wiring: the build plan must call this module, keeping the bash fallback.
# --------------------------------------------------------------------------- #
class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="GB200", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "install-nvloom"]

        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, install_nvloom.install)
        self.assertEqual(steps[0].script, "install_nvloom.sh")


if __name__ == "__main__":
    unittest.main()
