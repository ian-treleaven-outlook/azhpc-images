"""Unit tests for components/python/install_mpis.py.

  * The modulefiles are compared with fixtures in
    tests/python/fixtures/install_mpis/<scenario>/. Those were written by
    running the modulefile section of the real components/install_mpis.sh
    (sourcing the real utils/utilities.sh for sku_uses_ucx/sku_network_mode)
    in bash, for a UCX SKU, NCv6 (non-UCX), an EL8 AMD target (rebuilt UCX,
    Intel MPI EL8 append) and GB200 on aarch64 (no MVAPICH, no Intel MPI).
    symlinks.txt lists each symlink bash made as "<name> -> <target name>".
  * Pure helpers are tested directly; drop_last_two's expected values come
    from bash's ${x:0:-2}.
  * install() runs with every external effect mocked (config lookup,
    download, exec_program, nproc, versions file, dnf pinning) and every path
    pointed into a temporary directory, so nothing touches /opt, /etc, /usr or
    the network.

Run from the repo root:
    python3 -m unittest tests.python.test_install_mpis -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from components.python import install_mpis as mpis
from utils.build_config import BuildConfig, build_plan
from utils.sku import sku_network_mode
from utils.version_validate import active_hpcx_component

_OP = "install-mpis"
_FIXTURES = Path(__file__).parent / "fixtures" / "install_mpis"
_NPROC = 64

_HPCX_URL = ("https://content.mellanox.com/hpc/hpc-x/v2.25.1_cuda13/"
             "hpcx-v2.25.1-gcc-doca_ofed-ubuntu24.04-cuda13-x86_64.tbz")
_INBOX_URL = ("https://content.mellanox.com/hpc/hpc-x/v2.25.1_cuda13/"
              "hpcx-v2.25.1-gcc-inbox-ubuntu24.04-cuda13-x86_64.tbz")
_AMD_URL = ("https://content.mellanox.com/hpc/hpc-x/v2.18/"
            "hpcx-v2.18-gcc-mlnx_ofed-redhat8-cuda12-x86_64.tbz")
_MVAPICH_URL = "https://mvapich.cse.ohio-state.edu/download/mvapich/mv2/mvapich-4.1.tar.gz"
_OMPI_URL = "https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.10.tar.gz"
_IMPI_URL = ("https://registrationcenter-download.intel.com/akdlm/IRC_NAS/x/"
             "intel-mpi-2021.16.1.804_offline.sh")

_CONFIGS = {
    "pmix": {"version": "4.2.9-2"},
    "hpcx": {"version": "2.25.1", "sha256": "h1", "url": _HPCX_URL},
    "hpcx_inbox": {"version": "2.25.1", "sha256": "h2", "url": _INBOX_URL},
    "hpcx_amd": {"version": "2.18", "sha256": "h3", "url": _AMD_URL},
    "mvapich": {"version": "4.1", "sha256": "m1", "url": _MVAPICH_URL},
    "ompi": {"version": "5.0.10", "sha256": "o1", "url": _OMPI_URL},
    "impi": {"version": "2021.16.1", "sha256": "i1", "url": _IMPI_URL},
}


def _read(path) -> str:
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
class PureHelperTests(unittest.TestCase):
    def test_drop_last_two_matches_bash_substring(self):
        # bash: x="$v"; echo "${x:0:-2}"  (fewer than 2 chars is a bash error)
        self.assertEqual(mpis.drop_last_two("4.2.9-2"), "4.2.9")
        self.assertEqual(mpis.drop_last_two("5.0.1-1"), "5.0.1")
        self.assertEqual(mpis.drop_last_two("2021.16.1"), "2021.16")
        self.assertIsNone(mpis.drop_last_two("ab"))     # bash gives "" -> refused
        self.assertIsNone(mpis.drop_last_two("a"))      # bash: substring expression < 0
        self.assertIsNone(mpis.drop_last_two(""))

    def test_url_basename_like_basename(self):
        self.assertEqual(mpis.url_basename(_HPCX_URL),
                         "hpcx-v2.25.1-gcc-doca_ofed-ubuntu24.04-cuda13-x86_64.tbz")
        self.assertEqual(mpis.url_basename(_HPCX_URL, ".tbz"),
                         "hpcx-v2.25.1-gcc-doca_ofed-ubuntu24.04-cuda13-x86_64")
        self.assertEqual(mpis.url_basename(_OMPI_URL, ".tar.gz"), "openmpi-5.0.10")
        self.assertEqual(mpis.url_basename("http://x/a.tgz", ".tar.gz"), "a.tgz")
        self.assertEqual(mpis.url_basename("http://x/.tbz", ".tbz"), ".tbz")
        self.assertEqual(mpis.url_basename("http://x/dir/"), "dir")

    def test_hpcx_variant_agrees_with_version_validate(self):
        for gpu in ("NVIDIA", "AMD"):
            for sku in ("A100", "H100", "NCv6", "GB200", "MI300X"):
                for node_type in ("azure_vm_regular", "baremetal_1p", "baremetal_3p", ""):
                    with self.subTest(gpu=gpu, sku=sku, node_type=node_type):
                        component, reason = mpis.hpcx_variant(
                            gpu, sku_network_mode(sku, node_type))
                        self.assertEqual(component, active_hpcx_component(gpu, sku))
                        self.assertTrue(reason)

    def test_distro_predicates(self):
        for distro in ("almalinux8.10", "almalinux9.8", "rocky8.10", "rocky9.8",
                       "rhel8.10", "azurelinux3.0"):
            self.assertTrue(mpis.is_rpm_distro(distro), distro)
        for distro in ("ubuntu22.04", "ubuntu24.04", "azurelinux3.1", ""):
            self.assertFalse(mpis.is_rpm_distro(distro), distro)

        self.assertTrue(mpis.skips_mvapich("ubuntu24.04", "gb-family"))
        self.assertTrue(mpis.skips_mvapich("azurelinux3.0", "gb-family"))
        self.assertFalse(mpis.skips_mvapich("ubuntu22.04", "gb-family"))
        self.assertFalse(mpis.skips_mvapich("ubuntu24.04", "GB200"))
        self.assertFalse(mpis.skips_mvapich("ubuntu24.04", "A100"))

        for distro in ("almalinux8.10", "rocky8.10", "rhel8.10", "rhel8"):
            self.assertTrue(mpis.is_el8(distro), distro)
        for distro in ("almalinux8.9", "rocky9.8", "almalinux9.8", "ubuntu22.04"):
            self.assertFalse(mpis.is_el8(distro), distro)

        self.assertTrue(mpis.uses_bundled_pmix("azurelinux3.0", "azure_vm_regular"))
        self.assertTrue(mpis.uses_bundled_pmix("ubuntu24.04", "baremetal_3p"))
        self.assertFalse(mpis.uses_bundled_pmix("ubuntu24.04", "baremetal_1p"))
        self.assertFalse(mpis.uses_bundled_pmix("ubuntu24.04", ""))

        self.assertEqual(mpis.impi_fi_provider("no_rdma"), "tcp")
        self.assertEqual(mpis.impi_fi_provider("standard_ib"), "mlx")
        self.assertEqual(mpis.impi_fi_provider("mrc"), "mlx")

    def test_hpcx_rebuild_variants(self):
        rebuild = "/opt/h/utils/hpcx_rebuild.sh"
        rocm = ("--rebuild-ucx", "--ucx-extra-config", "--with-rocm=/opt/rocm")
        self.assertEqual(
            mpis.hpcx_rebuild_command("/opt/h", bundled_pmix=True, uses_ucx=True,
                                      ucx_args=rocm, pmix_path="/opt/pmix/4.2.9"),
            [rebuild, "--with-hcoll", *rocm, "--ompi-extra-config",
             "--with-pmix --enable-orterun-prefix-by-default"])
        self.assertEqual(
            mpis.hpcx_rebuild_command("/opt/h", bundled_pmix=False, uses_ucx=False,
                                      ucx_args=(), pmix_path="/opt/pmix/4.2.9"),
            [rebuild, "--ompi-extra-config",
             "--with-pmix=/opt/pmix/4.2.9 --enable-orterun-prefix-by-default "
             "--without-ucx --with-ofi=/opt/libfabric"])
        self.assertEqual(
            mpis.hpcx_rebuild_command("/opt/h", bundled_pmix=False, uses_ucx=True,
                                      ucx_args=(), pmix_path="/opt/pmix/4.2.9"),
            [rebuild, "--with-hcoll", "--ompi-extra-config",
             "--with-pmix=/opt/pmix/4.2.9 --enable-orterun-prefix-by-default"])

    def test_configure_commands(self):
        cmd, lib = mpis.mvapich_configure_command("ubuntu22.04", "4.1", uses_ucx=True,
                                                  ucx_path="/opt/h/ucx")
        self.assertEqual(cmd, ["./configure", "FFLAGS=-fallow-argument-mismatch",
                               "--prefix=/opt/mvapich-4.1", "--enable-g=none",
                               "--enable-fast=yes", "--with-ucx=/opt/h/ucx"])
        self.assertEqual(lib, "/opt/h/ucx/lib")
        cmd, lib = mpis.mvapich_configure_command("rocky9.8", "4.0", uses_ucx=False,
                                                  ucx_path="/opt/h/ucx")
        self.assertEqual(cmd, ["./configure", "--prefix=/opt/mvapich-4.0", "--enable-g=none",
                               "--enable-fast=yes", "--with-device=ch4:ofi",
                               "--with-libfabric=/opt/libfabric"])
        self.assertEqual(lib, "/opt/libfabric/lib")

        cmd, lib = mpis.ompi_configure_command(
            "5.0.10", uses_ucx=True, pmix_flag="--with-pmix=/opt/pmix/4.2.9",
            ucx_path="/opt/h/ucx", hcoll_path="/opt/h/hcoll", ld_library_path="")
        self.assertEqual(cmd, [
            "./configure", "LD_LIBRARY_PATH=:/opt/h/hcoll/lib", "--prefix=/opt/openmpi-5.0.10",
            "--with-ucx=/opt/h/ucx", "--with-hcoll=/opt/h/hcoll",
            "--with-pmix=/opt/pmix/4.2.9", "--enable-mpirun-prefix-by-default",
            "--with-platform=contrib/platform/mellanox/optimized"])
        self.assertEqual(lib, "/opt/h/ucx/lib")
        cmd, lib = mpis.ompi_configure_command(
            "5.0.10", uses_ucx=False, pmix_flag="--with-pmix", ucx_path="/opt/h/ucx",
            hcoll_path="/opt/h/hcoll", ld_library_path="/x")
        self.assertEqual(cmd, ["./configure", "--prefix=/opt/openmpi-5.0.10", "--without-ucx",
                               "--with-ofi=/opt/libfabric", "--with-pmix",
                               "--enable-mpirun-prefix-by-default"])
        self.assertEqual(lib, "/opt/libfabric/lib")


# --------------------------------------------------------------------------- #
# Modulefiles, compared with what the real bash writes.
# --------------------------------------------------------------------------- #
# scenario -> (DISTRIBUTION, ARCHITECTURE, SKU, SKU_FAMILY, HPC-X version,
#              HPC-X folder, UCX dir under HPC-X); the same values were used to
#              render the fixtures with bash.
_SCENARIOS = {
    "ucx": ("ubuntu24.04", "x86_64", "A100", "A100", "2.25.1",
            "hpcx-v2.25.1-gcc-doca_ofed-ubuntu24.04-cuda13-x86_64", "ucx"),
    "non_ucx": ("ubuntu24.04", "x86_64", "NCv6", "NCv6", "2.25.1",
                "hpcx-v2.25.1-gcc-inbox-ubuntu24.04-cuda13-x86_64", "ucx"),
    "el8_amd": ("almalinux8.10", "x86_64", "MI300X", "MI300X", "2.18",
                "hpcx-v2.18-gcc-mlnx_ofed-redhat8-cuda12-x86_64", "ucx/hpcx-rebuild"),
    "gb_aarch64": ("ubuntu24.04", "aarch64", "GB200", "gb-family", "2.25.1",
                   "hpcx-v2.25.1-gcc-doca_ofed-ubuntu24.04-cuda13-aarch64", "ucx"),
}


class ModulefileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _render(self, name):
        distro, arch, sku, family, hpcx_version, folder, ucx_sub = _SCENARIOS[name]
        hpcx_path = f"/opt/{folder}"
        hpcx = mpis._Hpcx(path=hpcx_path, hcoll=f"{hpcx_path}/hcoll",
                          sharp=f"{hpcx_path}/sharp", ucx=f"{hpcx_path}/{ucx_sub}")
        uses_ucx = sku != "NCv6"
        mvapich = None if mpis.skips_mvapich(distro, family) else {"version": "4.1"}
        impi = None if arch == "aarch64" else {"version": "2021.16.1"}
        inp = mpis._Inputs(
            distribution=distro, architecture=arch, gpu="", sku=sku, sku_family=family,
            node_type="azure_vm_regular", network_mode=sku_network_mode(sku),
            uses_ucx=uses_ucx, module_dir=os.path.join(self._tmp.name, name),
            hpcx_component="", hpcx_reason="", hpcx={"version": hpcx_version},
            pmix_version="", pmix_path="", mvapich=mvapich, ompi={"version": "5.0.10"},
            impi=impi, impi_2021_version="2021.16" if impi else None)
        _, mvapich_lib = mpis.mvapich_configure_command(distro, "4.1", uses_ucx=uses_ucx,
                                                        ucx_path=hpcx.ucx)
        _, ompi_lib = mpis.ompi_configure_command(
            "5.0.10", uses_ucx=uses_ucx, pmix_flag="", ucx_path=hpcx.ucx,
            hcoll_path=hpcx.hcoll, ld_library_path="")
        with mock.patch.object(mpis, "log_info"), \
             mock.patch.object(mpis, "log_warn") as warn:
            mpis._write_modulefiles(inp, hpcx, mvapich_lib if mvapich else None, ompi_lib,
                                    mpis._Report())
        warn.assert_not_called()       # first run: no rerun warnings (incl. EL8 append)
        return os.path.join(inp.module_dir, "mpi")

    def test_modulefiles_and_symlinks_match_bash(self):
        for name in _SCENARIOS:
            with self.subTest(scenario=name):
                mpi_dir = self._render(name)
                fixture_dir = _FIXTURES / name
                expected_files = sorted(p.name for p in fixture_dir.iterdir()
                                        if p.name != "symlinks.txt")
                files = sorted(n for n in os.listdir(mpi_dir)
                               if not os.path.islink(os.path.join(mpi_dir, n)))
                self.assertEqual(files, expected_files)
                for file in files:
                    self.assertEqual(_read(os.path.join(mpi_dir, file)),
                                     _read(fixture_dir / file), file)
                links = sorted(
                    f"{n} -> {os.path.basename(os.readlink(os.path.join(mpi_dir, n)))}\n"
                    for n in os.listdir(mpi_dir) if os.path.islink(os.path.join(mpi_dir, n)))
                self.assertEqual("".join(links), _read(fixture_dir / "symlinks.txt"))
                # Links are absolute, like bash's `ln -s ${MPI_MODULE_FILES_DIRECTORY}/...`.
                self.assertEqual(os.readlink(os.path.join(mpi_dir, "hpcx")),
                                 os.path.join(mpi_dir, f"hpcx-{_SCENARIOS[name][4]}"))


# --------------------------------------------------------------------------- #
# install(), with every external effect mocked.
# --------------------------------------------------------------------------- #
class InstallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = self.root = self._tmp.name
        self.opt = os.path.join(root, "opt")
        self.work = os.path.join(root, "work")
        self.rocm = os.path.join(root, "rocm")
        self.ld_dir = os.path.join(root, "ld.so.conf.d")
        self.pc_lib = os.path.join(root, "usr", "local", "lib", "pkgconfig")
        self.pc_lib64 = os.path.join(root, "usr", "local", "lib64", "pkgconfig")
        self.modules = os.path.join(root, "modulefiles")
        os.makedirs(self.opt)
        os.makedirs(self.ld_dir)
        for name, value in (("_INSTALL_PREFIX", self.opt), ("_WORK_DIR", self.work),
                            ("_ROCM_DIR", self.rocm), ("_LD_SO_CONF_DIR", self.ld_dir),
                            ("_PKGCONFIG_DIRS", (self.pc_lib,)),
                            ("_PKGCONFIG_DIRS_RPM", (self.pc_lib, self.pc_lib64))):
            patcher = mock.patch.object(mpis, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    # ---- harness -------------------------------------------------------------
    def _fake_exec(self, command, op, *, cwd=None, env=None):
        self.assertEqual(op, _OP)
        self.calls.append((command, cwd, env))
        if command == self.fail_on:
            return 2
        # tar creates the source folder the configure/make steps run in.
        if command[:2] == ["tar", "-xvf"] and command[2].endswith(".tar.gz"):
            os.makedirs(os.path.join(cwd, command[2][:-len(".tar.gz")]), exist_ok=True)
        return 0

    def _install(self, distro="ubuntu22.04", arch="x86_64", gpu="NVIDIA", sku="A100",
                 family=None, node_type="azure_vm_regular", configs=None,
                 fail_on=None, nproc=(0, f"{_NPROC}\n"), pin_ok=True, download_error=None,
                 write_error=None, extra_env=None):
        self.calls, self.fail_on = [], fail_on
        configs = dict(_CONFIGS if configs is None else configs)
        env = {"DISTRIBUTION": distro, "ARCHITECTURE": arch, "GPU": gpu, "SKU": sku,
               "SKU_FAMILY": family or sku, "TARGET_NODE_TYPE": node_type,
               "MODULE_FILES_DIRECTORY": self.modules, "PATH": "/usr/bin",
               "LD_LIBRARY_PATH": "/caller/lib"}
        env.update(extra_env or {})

        def fake_download(url, sha256, dest_dir="."):
            if download_error:
                raise download_error
            path = Path(dest_dir) / Path(url).name
            path.write_bytes(b"x")
            return path

        with mock.patch.object(mpis, "config_for",
                               side_effect=lambda c, e: configs.get(c)) as config, \
             mock.patch.object(mpis, "download_and_verify",
                               side_effect=fake_download) as download, \
             mock.patch.object(mpis, "exec_program", side_effect=self._fake_exec), \
             mock.patch.object(mpis, "run_capture", return_value=nproc), \
             mock.patch.object(mpis, "write_component_version",
                               side_effect=write_error) as record, \
             mock.patch.object(mpis, "dnf_pin_packages", return_value=pin_ok) as pin, \
             mock.patch.object(mpis, "log_info") as info, \
             mock.patch.object(mpis, "log_warn") as warn, \
             mock.patch.object(mpis, "log_error") as error:
            result = mpis.install(env)
        self.env = env
        self.config, self.download, self.record, self.pin = config, download, record, pin
        self.info, self.warn, self.error = info, warn, error
        return result

    def _commands(self):
        return [c[0] for c in self.calls]

    def _messages(self, logger):
        return [c.args[1] for c in logger.call_args_list]

    def _hpcx_path(self, url):
        return f"{self.opt}/{mpis.url_basename(url, '.tbz')}"

    def _hpcx_commands(self, url, rebuild):
        path = self._hpcx_path(url)
        return [
            ["tar", "-xvf", mpis.url_basename(url)],
            ["mv", mpis.url_basename(url, ".tbz"), self.opt],
            [f"{path}/utils/hpcx_fix_ladir.sh"],
            rebuild,
            ["cp", "-r", f"{path}/ompi/tests", f"{path}/hpcx-rebuild"],
            [f"{path}/utils/hpcx_fix_ladir.sh"],
            ["ldconfig"],
        ]

    def _impi_commands(self):
        modulefiles = f"{self.opt}/intel/oneapi/mpi/2021.16/etc/modulefiles"
        return [["bash", "intel-mpi-2021.16.1.804_offline.sh", "-s", "-a", "-s",
                 "--eula", "accept"],
                ["mv", f"{modulefiles}/mpi", f"{modulefiles}/impi"]]

    def _mpi_dir(self):
        return os.path.join(self.modules, "mpi")

    # ---- success paths ------------------------------------------------------
    def test_ubuntu_ucx_sku_full_command_list(self):
        self.assertEqual(self._install(), 0)
        path = self._hpcx_path(_HPCX_URL)
        pmix = f"{self.opt}/pmix/4.2.9"
        expected = self._hpcx_commands(_HPCX_URL, [
            f"{path}/utils/hpcx_rebuild.sh", "--with-hcoll", "--ompi-extra-config",
            f"--with-pmix={pmix} --enable-orterun-prefix-by-default"]) + [
            ["tar", "-xvf", "mvapich-4.1.tar.gz"],
            ["./configure", "FFLAGS=-fallow-argument-mismatch",
             f"--prefix={self.opt}/mvapich-4.1", "--enable-g=none", "--enable-fast=yes",
             f"--with-ucx={path}/ucx"],
            ["make", f"-j{_NPROC}"], ["make", "install"],
            ["tar", "-xvf", "openmpi-5.0.10.tar.gz"],
            ["./configure", f"LD_LIBRARY_PATH=/caller/lib:{path}/hcoll/lib",
             f"--prefix={self.opt}/openmpi-5.0.10", f"--with-ucx={path}/ucx",
             f"--with-hcoll={path}/hcoll", f"--with-pmix={pmix}",
             "--enable-mpirun-prefix-by-default",
             "--with-platform=contrib/platform/mellanox/optimized"],
            ["make", f"-j{_NPROC}"], ["make", "install"],
        ] + self._impi_commands()
        self.assertEqual(self._commands(), expected)

        # cwd: work dir for downloads/tar/scripts, source dir for the builds.
        cwds = [c[1] for c in self.calls]
        self.assertEqual(cwds[:7], [self.work] * 7)
        self.assertEqual(cwds[7:11], [self.work] + [f"{self.work}/mvapich-4.1"] * 3)
        self.assertEqual(cwds[11:15], [self.work] + [f"{self.work}/openmpi-5.0.10"] * 3)
        # HPCX_DIR only for the two hpcx_fix_ladir.sh runs.
        for command, _, env in self.calls:
            if command[0].endswith("hpcx_fix_ladir.sh"):
                self.assertEqual(env["HPCX_DIR"], path)
            else:
                self.assertNotIn("HPCX_DIR", env)
                self.assertEqual(env["PATH"], "/usr/bin")

        self.assertEqual([c.args for c in self.download.call_args_list],
                         [(_HPCX_URL, "h1"), (_MVAPICH_URL, "m1"), (_OMPI_URL, "o1"),
                          (_IMPI_URL, "i1")])
        self.assertEqual([c.args for c in self.record.call_args_list],
                         [("HPCX", "2.25.1"), ("MVAPICH", "4.1"), ("OMPI", "5.0.10"),
                          ("IMPI", "2021.16.1")])
        self.pin.assert_not_called()                    # not an RPM distro
        self.assertEqual(_read(os.path.join(self.ld_dir, "hpcx-sharp.conf")),
                         f"{path}/sharp/lib\n")
        self.assertEqual(_read(os.path.join(self.ld_dir, "hpcx-ucx.conf")),
                         f"{path}/ucx/lib\n")
        self.assertTrue(os.path.isdir(self.pc_lib))
        self.assertFalse(os.path.exists(self.pc_lib64))
        self.assertEqual(sorted(os.listdir(self._mpi_dir())), [
            "hpcx", "hpcx-2.25.1", "hpcx-pmix", "hpcx-pmix-2.25.1", "impi-2021",
            "impi_2021.16", "mvapich", "mvapich-4.1", "openmpi", "openmpi-5.0.10"])
        self.assertIn(f"{path}/hcoll/lib:{path}/ucx/lib\n",
                      _read(os.path.join(self._mpi_dir(), "openmpi-5.0.10")))
        self.assertIn("FI_PROVIDER     mlx\n",
                      _read(os.path.join(self._mpi_dir(), "impi_2021.16")))
        self.assertFalse(os.path.exists(self.work))
        self.error.assert_not_called()

    def test_logs_decisions_versions_and_summary(self):
        self.assertEqual(self._install(), 0)
        messages = self._messages(self.info)
        start = self.info.call_args_list[0]
        self.assertTrue(start.args[1].startswith("Installing MPIs on ubuntu22.04"))
        self.assertEqual((start.kwargs["hpcx_component"], start.kwargs["network_mode"],
                          start.kwargs["pmix_path"], start.kwargs["mvapich_version"],
                          start.kwargs["impi_version"]),
                         ("hpcx", "standard_ib", f"{self.opt}/pmix/4.2.9", "4.1", "2021.16.1"))
        self.assertIn("HPC-X variant hpcx: default HPC-X (InfiniBand network)", messages)
        self.assertIn(f"PMIx path for the MPI builds: {self.opt}/pmix/4.2.9", messages)
        self.assertTrue(any(m.startswith("Rebuilding HPC-X Open MPI: UCX SKU") for m in messages))
        for component, version in (("HPCX", "2.25.1"), ("MVAPICH", "4.1"),
                                   ("OMPI", "5.0.10"), ("IMPI", "2021.16.1")):
            self.assertIn(f"Recorded {component}={version}", messages)
        ld_conf = os.path.join(self.ld_dir, "hpcx-ucx.conf")
        self.assertIn(f"Wrote {ld_conf} (created)", messages)
        self.assertTrue(any(m.startswith("Created symlink") and "/mpi/hpcx ->" in m
                            for m in messages))
        summary = self.info.call_args_list[-1]
        self.assertEqual(summary.args[1], "MPIs installed: HPCX=2.25.1, MVAPICH=4.1, "
                                          "OMPI=5.0.10, IMPI=2021.16.1")
        self.assertEqual(summary.kwargs["symlinks"], 5)
        # Warnings: `set CC=` always; `rm -rf -- */` on success; no rerun warnings.
        warnings = self._messages(self.warn)
        self.assertEqual(len(warnings), 2)
        self.assertIn("set CC=/usr/bin/gcc", warnings[0])
        self.assertIn("rm -rf -- */", warnings[1])

    def test_ncv6_uses_inbox_hpcx_and_libfabric(self):
        self.assertEqual(self._install(distro="ubuntu24.04", sku="NCv6"), 0)
        path = self._hpcx_path(_INBOX_URL)
        pmix = f"{self.opt}/pmix/4.2.9"
        expected = self._hpcx_commands(_INBOX_URL, [
            f"{path}/utils/hpcx_rebuild.sh", "--ompi-extra-config",
            f"--with-pmix={pmix} --enable-orterun-prefix-by-default --without-ucx "
            "--with-ofi=/opt/libfabric"]) + [
            ["tar", "-xvf", "mvapich-4.1.tar.gz"],
            ["./configure", "FFLAGS=-fallow-argument-mismatch",
             f"--prefix={self.opt}/mvapich-4.1", "--enable-g=none", "--enable-fast=yes",
             "--with-device=ch4:ofi", "--with-libfabric=/opt/libfabric"],
            ["make", f"-j{_NPROC}"], ["make", "install"],
            ["tar", "-xvf", "openmpi-5.0.10.tar.gz"],
            ["./configure", f"--prefix={self.opt}/openmpi-5.0.10", "--without-ucx",
             "--with-ofi=/opt/libfabric", f"--with-pmix={pmix}",
             "--enable-mpirun-prefix-by-default"],
            ["make", f"-j{_NPROC}"], ["make", "install"],
        ] + self._impi_commands()
        self.assertEqual(self._commands(), expected)
        self.assertEqual(self.config.call_args_list[1].args[0], "hpcx_inbox")
        self.assertIn("HPC-X variant hpcx_inbox: network mode no_rdma: no DOCA-OFED on "
                      "non-IB SKUs, so use the inbox HPC-X (UCX linked against rdma-core)",
                      self._messages(self.info))
        mpi_dir = self._mpi_dir()
        self.assertIn("FI_PROVIDER     tcp\n", _read(os.path.join(mpi_dir, "impi_2021.16")))
        self.assertTrue(_read(os.path.join(mpi_dir, "mvapich-4.1")).endswith(
            "/opt/mvapich-4.1\nsetenv          FI_PROVIDER tcp\n"
            "setenv          MPIR_CVAR_CH4_CMA_ENABLE 0\n"))

    def test_amd_el8_rebuilds_ucx_pins_and_links_pkgconfig(self):
        os.makedirs(self.rocm)
        path = self._hpcx_path(_AMD_URL)
        # .pc files bash's loop would link (and things it would skip).
        for tree in ("hcoll", "sharp", "ucx/hpcx-rebuild"):
            os.makedirs(f"{path}/{tree}/lib/pkgconfig")
        Path(f"{path}/hcoll/lib/pkgconfig/hcoll.pc").write_text("hcoll")
        Path(f"{path}/sharp/lib/pkgconfig/sharp.pc").write_text("sharp")
        Path(f"{path}/sharp/lib/pkgconfig/notes.txt").write_text("not .pc")
        os.makedirs(f"{path}/sharp/lib/pkgconfig/dir.pc")       # [[ -f ]] skips it
        Path(f"{path}/ucx/hpcx-rebuild/lib/pkgconfig/ucx.pc").write_text("ucx")
        os.makedirs(self.pc_lib)
        Path(self.pc_lib, "ucx.pc").write_text("old")           # ln -sf replaces it

        self.assertEqual(self._install(distro="almalinux8.10", gpu="AMD", sku="MI300X"), 0)
        pmix = f"{self.opt}/pmix/4.2.9"
        ucx = f"{path}/ucx/hpcx-rebuild"
        commands = self._commands()
        self.assertEqual(commands[:7], self._hpcx_commands(_AMD_URL, [
            f"{path}/utils/hpcx_rebuild.sh", "--with-hcoll", "--rebuild-ucx",
            "--ucx-extra-config", "--with-rocm=/opt/rocm", "--ompi-extra-config",
            f"--with-pmix={pmix} --enable-orterun-prefix-by-default"]))
        # No FFLAGS on EL; the rebuilt UCX is used from here on.
        self.assertEqual(commands[8], ["./configure", f"--prefix={self.opt}/mvapich-4.1",
                                       "--enable-g=none", "--enable-fast=yes",
                                       f"--with-ucx={ucx}"])
        self.assertIn(f"--with-ucx={ucx}", commands[12])
        self.assertEqual(_read(os.path.join(self.ld_dir, "hpcx-ucx.conf")), f"{ucx}/lib\n")
        self.assertEqual([c.args[0] for c in self.pin.call_args_list],
                         [["ucx*"], ["openmpi", "perftest"]])
        self.assertEqual(self.pin.call_args_list[0].args[1], _OP)
        for pc_dir in (self.pc_lib, self.pc_lib64):
            self.assertEqual(sorted(os.listdir(pc_dir)), ["hcoll.pc", "sharp.pc", "ucx.pc"])
            self.assertEqual(os.readlink(os.path.join(pc_dir, "ucx.pc")),
                             f"{ucx}/lib/pkgconfig/ucx.pc")
            self.assertEqual(os.readlink(os.path.join(pc_dir, "hcoll.pc")),
                             f"{path}/hcoll/lib/pkgconfig/hcoll.pc")
        impi = _read(os.path.join(self._mpi_dir(), "impi_2021.16"))
        self.assertTrue(impi.endswith("FI_PROVIDER     mlx\n# see https://community.intel.com"
                                      "/t5/Intel-MPI-Library/Suspected-unfixed-Intel-MPI-race-"
                                      "condition-in-collectives/td-p/1693452 for Intel MPI bug\n"
                                      "setenv          I_MPI_STARTUP_MODE         pmi_shm\n"))
        self.assertEqual(self.config.call_args_list[1].args[0], "hpcx_amd")
        self.error.assert_not_called()

    def test_amd_without_rocm_fails_after_recording_hpcx(self):
        self.assertEqual(self._install(distro="almalinux8.10", gpu="AMD", sku="MI300X"), 3)
        self.assertEqual(len(self.calls), 3)          # tar, mv, fix_ladir
        self.record.assert_called_once_with("HPCX", "2.18")
        self.assertEqual(self.error.call_args.args[1],
                         "ROCm must be installed before rebuilding HPC-X UCX with "
                         "ROCm support.")
        self.assertFalse(os.path.exists(self.work))

    def test_amd_non_ucx_does_not_need_rocm(self):
        self.assertEqual(self._install(distro="ubuntu24.04", gpu="AMD", sku="NCv6"), 0)
        self.assertNotIn("--rebuild-ucx", self._commands()[3])

    def test_azure_linux_gb200_bundled_pmix_no_mvapich_no_impi(self):
        configs = dict(_CONFIGS, pmix={"version": "5.0.1-1"})
        self.assertEqual(self._install(distro="azurelinux3.0", arch="aarch64", sku="GB200",
                                       family="gb-family", configs=configs), 0)
        path = self._hpcx_path(_HPCX_URL)
        expected = self._hpcx_commands(_HPCX_URL, [
            f"{path}/utils/hpcx_rebuild.sh", "--with-hcoll", "--ompi-extra-config",
            "--with-pmix --enable-orterun-prefix-by-default"]) + [
            ["tar", "-xvf", "openmpi-5.0.10.tar.gz"],
            ["./configure", f"LD_LIBRARY_PATH=/caller/lib:{path}/hcoll/lib",
             f"--prefix={self.opt}/openmpi-5.0.10", f"--with-ucx={path}/ucx",
             f"--with-hcoll={path}/hcoll", "--with-pmix",
             "--enable-mpirun-prefix-by-default",
             "--with-platform=contrib/platform/mellanox/optimized"],
            ["make", f"-j{_NPROC}"], ["make", "install"],
        ]
        self.assertEqual(self._commands(), expected)
        self.assertNotIn("mvapich", [c.args[0] for c in self.config.call_args_list])
        self.assertNotIn("impi", [c.args[0] for c in self.config.call_args_list])
        self.assertEqual([c.args for c in self.record.call_args_list],
                         [("HPCX", "2.25.1"), ("OMPI", "5.0.10")])
        self.assertEqual([c.args[0] for c in self.pin.call_args_list],
                         [["ucx*"], ["openmpi", "perftest"]])
        self.assertTrue(os.path.isdir(self.pc_lib64))
        messages = self._messages(self.info)
        self.assertTrue(any(m.startswith("Skipping MVAPICH: not supported on azurelinux3.0 "
                                         "for gb-family SKUs") for m in messages))
        self.assertIn("Skipping Intel MPI on aarch64 (x86_64 only)", messages)
        self.assertTrue(any(m.startswith("Rebuilding HPC-X Open MPI: azurelinux3.0/") and
                            "PMIx bundled in HPC-X" in m for m in messages))
        self.assertEqual(sorted(os.listdir(self._mpi_dir())), [
            "hpcx", "hpcx-2.25.1", "hpcx-pmix", "hpcx-pmix-2.25.1", "openmpi",
            "openmpi-5.0.10"])

    def test_baremetal_3p_uses_bundled_pmix_on_ubuntu(self):
        self.assertEqual(self._install(distro="ubuntu24.04", arch="aarch64", sku="GB200",
                                       family="gb-family", node_type="baremetal_3p"), 0)
        commands = self._commands()
        self.assertEqual(commands[3][-1], "--with-pmix --enable-orterun-prefix-by-default")
        self.assertIn("--with-pmix", commands[8])
        self.assertNotIn(f"--with-pmix={self.opt}/pmix/4.2.9", commands[8])
        self.pin.assert_not_called()

    def test_gb200_on_ubuntu22_still_builds_mvapich(self):
        self.assertEqual(self._install(distro="ubuntu22.04", arch="aarch64", sku="GB200",
                                       family="gb-family"), 0)
        self.assertIn(["tar", "-xvf", "mvapich-4.1.tar.gz"], self._commands())

    def test_empty_caller_ld_library_path(self):
        self.assertEqual(self._install(extra_env={"LD_LIBRARY_PATH": ""}), 0)
        path = self._hpcx_path(_HPCX_URL)
        self.assertEqual(self._commands()[12][1], f"LD_LIBRARY_PATH=:{path}/hcoll/lib")

    # ---- reruns -------------------------------------------------------------
    def test_rerun_appends_with_warning_then_fails_at_first_symlink(self):
        mpi_dir = self._mpi_dir()
        os.makedirs(mpi_dir)
        Path(mpi_dir, "hpcx-2.25.1").write_text("old\n")
        os.symlink(os.path.join(mpi_dir, "mvapich-4.1"), os.path.join(mpi_dir, "mvapich"))

        self.assertEqual(self._install(), 3)
        # Appended (not replaced), as bash's `cat >>`.
        self.assertTrue(_read(os.path.join(mpi_dir, "hpcx-2.25.1")).startswith(
            "old\n#%Module 1.0\n"))
        rerun = [c for c in self.warn.call_args_list if "already exists" in c.args[1]]
        self.assertEqual(len(rerun), 1)
        self.assertEqual(rerun[0].kwargs["path"], os.path.join(mpi_dir, "hpcx-2.25.1"))
        self.assertIn("ln -s` (without -f) fails here too", self.error.call_args.args[1])
        # bash stops at that ln -s: no openmpi modulefile, no later symlinks.
        self.assertFalse(os.path.exists(os.path.join(mpi_dir, "openmpi-5.0.10")))
        self.assertTrue(os.path.exists(os.path.join(mpi_dir, "mvapich-4.1")))
        self.assertFalse(any("rm -rf -- */" in m for m in self._messages(self.warn)))

    # ---- missing or unusable config -------------------------------------------
    def test_missing_config_fails_before_downloading(self):
        cases = {
            "pmix missing": dict(_CONFIGS, pmix=None),
            "pmix too short": dict(_CONFIGS, pmix={"version": "ab"}),
            "hpcx missing": dict(_CONFIGS, hpcx=None),
            "hpcx no sha256": dict(_CONFIGS, hpcx={"version": "2.25.1", "url": _HPCX_URL}),
            "mvapich missing": dict(_CONFIGS, mvapich=None),
            "ompi no url": dict(_CONFIGS, ompi={"version": "5.0.10", "sha256": "o1"}),
            "impi missing": dict(_CONFIGS, impi=None),
            "impi too short": dict(_CONFIGS, impi={"version": "21", "sha256": "i",
                                                   "url": _IMPI_URL}),
        }
        for name, configs in cases.items():
            with self.subTest(case=name):
                self.assertEqual(self._install(configs=configs), 3)
                self.download.assert_not_called()
                self.assertEqual(self.calls, [])
                self.error.assert_called_once()

    def test_missing_skipped_configs_are_not_needed(self):
        configs = dict(_CONFIGS, mvapich=None, impi=None)
        self.assertEqual(self._install(distro="ubuntu24.04", arch="aarch64", sku="GB200",
                                       family="gb-family", configs=configs), 0)

    def test_missing_module_dir_fails(self):
        self.assertEqual(self._install(extra_env={"MODULE_FILES_DIRECTORY": ""}), 3)
        self.config.assert_not_called()

    # ---- failures -------------------------------------------------------------
    def test_each_failing_command_stops_the_step(self):
        self._install()
        commands = self._commands()
        shutil.rmtree(self.modules)
        # The command index after which each version is recorded.
        recorded_after = {"HPCX": 2, "MVAPICH": 10, "OMPI": 14, "IMPI": 16}
        for index, command in enumerate(commands):
            if command in commands[:index]:
                continue        # repeated commands (make, fix_ladir) fail at the first one
            with self.subTest(command=command):
                self.assertEqual(self._install(fail_on=command), 3)
                self.assertEqual(self._commands(), commands[:index + 1])
                self.assertEqual([c.args[0] for c in self.record.call_args_list],
                                 [k for k, after in recorded_after.items() if after < index])
                self.error.assert_called()
                self.assertFalse(os.path.exists(self.work))
                self.assertFalse(os.path.exists(self._mpi_dir()))

    def test_download_failure_returns_3(self):
        self.assertEqual(self._install(download_error=OSError("boom")), 3)
        self.assertEqual(self.calls, [])
        self.assertIn("download/verify of HPC-X failed", self.error.call_args.args[1])

    def test_pin_failure_returns_3(self):
        self.assertEqual(self._install(distro="rocky9.8", pin_ok=False), 3)
        self.assertEqual(self._commands()[-1], ["ldconfig"])     # stopped after HPC-X
        self.record.assert_called_once_with("HPCX", "2.25.1")

    def test_nproc_failure_returns_3(self):
        self.assertEqual(self._install(nproc=(1, "")), 3)
        self.assertEqual(self.calls, [])

    def test_missing_source_folder_returns_3(self):
        original = self._fake_exec

        def no_tar_output(command, op, *, cwd=None, env=None):
            if command[:2] == ["tar", "-xvf"]:
                self.calls.append((command, cwd, env))
                return 0
            return original(command, op, cwd=cwd, env=env)

        self._fake_exec = no_tar_output
        self.assertEqual(self._install(), 3)
        self.assertEqual(self._commands()[-1], ["tar", "-xvf", "mvapich-4.1.tar.gz"])
        self.assertIn("was not created by tar", self.error.call_args.args[1])

    def test_ld_so_conf_write_error_returns_3(self):
        os.rmdir(self.ld_dir)
        self.assertEqual(self._install(), 3)
        self.assertNotIn(["ldconfig"], self._commands())
        self.assertIn("file operation failed", self.error.call_args.args[1])
        self.assertFalse(os.path.exists(self.work))

    def test_unwritable_versions_file_returns_3(self):
        self.assertEqual(self._install(write_error=PermissionError("denied")), 3)
        self.assertEqual(len(self.calls), 3)


class BuildPlanWiringTests(unittest.TestCase):
    def test_step_runs_the_python_action_and_keeps_its_bash_script(self):
        cfg = BuildConfig(vendor="NVidia", gpu="A100", os="Ubuntu24",
                          fips=False, spec_path=None)
        steps = [s for s in build_plan(cfg) if s.op == "install-mpis"]
        self.assertEqual(len(steps), 1)
        self.assertIs(steps[0].action, mpis.install)
        self.assertEqual(steps[0].script, "install_mpis.sh")


if __name__ == "__main__":
    unittest.main()
