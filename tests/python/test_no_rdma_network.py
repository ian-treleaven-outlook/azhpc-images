"""Unit tests for components/python/no_rdma_network.py (install.sh's no-RDMA branch)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import no_rdma_network as m
from utils import package_installer
from utils.build_config import BuildConfig, build_plan

_INSTALL_SH = (Path(__file__).resolve().parents[2] / "distros" / "ubuntu24.04"
               / "install.sh").read_text(encoding="utf-8")


class BashSourceTests(unittest.TestCase):
    def test_commands_match_install_sh(self):
        self.assertIn("apt-get install -y " + " ".join(m.RDMA_CORE_PACKAGES), _INSTALL_SH)
        self.assertIn(f'echo "{m.BLACKLIST_TEXT.strip()}" | tee {m.BLACKLIST_FILE}',
                      _INSTALL_SH)


class InstallRdmaCoreTests(unittest.TestCase):
    _ENV = {"SKU": "NCv6", "DISTRIBUTION": "ubuntu24.04"}

    def _run(self, rc):
        with mock.patch.object(package_installer, "exec_program", return_value=rc) as run, \
             mock.patch.object(package_installer, "log_info"), \
             mock.patch.object(package_installer, "log_error") as error, \
             mock.patch.object(m, "log_info"):
            result = m.install_rdma_core(dict(self._ENV))
        return result, run, error

    def test_success(self):
        rc, run, error = self._run(0)
        self.assertEqual(rc, 0)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0],
                         ["apt-get", "install", "-y", "-o", "Dpkg::Use-Pty=0",
                          "rdma-core", "libibverbs-dev", "ibverbs-utils",
                          "librdmacm-dev", "pkg-config"])
        self.assertEqual(run.call_args.args[1], "install-rdma-core")
        self.assertEqual(run.call_args.kwargs["env"], self._ENV)
        error.assert_not_called()

    def test_failure_returns_3(self):
        # install_packages() logs the failure (manager, packages, exit code).
        rc, _, error = self._run(100)
        self.assertEqual(rc, 3)
        self.assertEqual(error.call_args.kwargs["exit_code"], 100)


class BlacklistManaIbTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "blacklist-mana-ib.conf"
        patcher = mock.patch.object(m, "BLACKLIST_FILE", str(self.path))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _run(self):
        with mock.patch.object(m, "log_info") as info, \
             mock.patch.object(m, "log_error") as error:
            rc = m.blacklist_mana_ib({"SKU": "NCv6"})
        return rc, info, error

    def test_writes_the_exact_file(self):
        rc, info, _ = self._run()
        self.assertEqual(rc, 0)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "blacklist mana_ib\n")
        self.assertEqual(info.call_args.kwargs["action"], "created")

    def test_rerun_overwrites_like_tee(self):
        self.path.write_text("old\nblacklist other\n", encoding="utf-8")
        rc, info, _ = self._run()
        self.assertEqual(rc, 0)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "blacklist mana_ib\n")
        self.assertEqual(info.call_args.kwargs["action"], "replaced")

    def test_write_error_returns_3(self):
        with mock.patch.object(Path, "write_text", side_effect=PermissionError("denied")):
            rc, _, error = self._run()
        self.assertEqual(rc, 3)
        error.assert_called_once()


class BuildPlanWiringTests(unittest.TestCase):
    def _ops(self, gpu):
        cfg = BuildConfig(vendor="NVidia", gpu=gpu, os="Ubuntu24", fips=False, spec_path=None)
        return [s.op for s in build_plan(cfg) if s.when(cfg)]

    def test_ncv6_runs_rdma_core_then_libfabric_then_blacklist(self):
        ops = self._ops("NCv6")
        self.assertLess(ops.index("install-rdma-core"), ops.index("install-libfabric"))
        self.assertLess(ops.index("install-libfabric"), ops.index("blacklist-mana-ib"))
        # ...and all three come right at the start, where install.sh has them.
        self.assertEqual(ops[:3], ["install-rdma-core", "install-libfabric",
                                   "blacklist-mana-ib"])

    def test_infiniband_skus_skip_them(self):
        for gpu in ("A100", "GB200"):
            with self.subTest(gpu=gpu):
                ops = self._ops(gpu)
                self.assertNotIn("install-rdma-core", ops)
                self.assertNotIn("blacklist-mana-ib", ops)

    def test_actions_are_wired(self):
        cfg = BuildConfig(vendor="NVidia", gpu="NCv6", os="Ubuntu24", fips=False, spec_path=None)
        steps = {s.op: s for s in build_plan(cfg)}
        self.assertIs(steps["install-rdma-core"].action, m.install_rdma_core)
        self.assertIs(steps["blacklist-mana-ib"].action, m.blacklist_mana_ib)


if __name__ == "__main__":
    unittest.main()
