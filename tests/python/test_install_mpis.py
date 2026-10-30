"""Unit tests for components/python/install_mpis.py.

  * The modulefiles are compared with fixtures in
    tests/python/fixtures/install_mpis/<scenario>/. Those were written by
    running the real components/install_mpis.sh in bash (sourcing the real
    utils/utilities.sh for sku_uses_ucx/sku_network_mode, with tar, make,
    configure, the hpcx scripts, ldd, pkg-config, ... replaced by logging
    stubs and /opt, /etc, /usr/local moved into a scratch directory) for a
    UCX SKU, NCv6 (non-UCX), an EL8 AMD target (rebuilt UCX, Intel MPI EL8
    append), GB200 on aarch64 (no MVAPICH, no Intel MPI), Ubuntu 22.04
    (rebuilt, netlink-patched UCX), Azure Linux 3.0 GB200 (internal PMIx, no
    HPC-X rebuild) and Ubuntu 26.04 (no MVAPICH/Open MPI). symlinks.txt lists
    each symlink bash made as "<name> -> <target name>". The same run, with
    the Python port next to it, gave identical command sequences for 14
    targets; the expected command lists below come from it.
  * Pure helpers are tested directly; drop_last_two's expected values come
    from bash's ${x:0:-2}, ldd_libmpi's from the real awk.
  * install() runs with every external effect mocked (config lookup,
    download, exec_program, run_capture, versions file, dnf pinning) and every
    path pointed into a temporary directory, so nothing touches /opt, /etc,
    /usr or the network.

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
from utils.sku import sku_network_mode, sku_uses_ucx
from utils.version_validate import active_hpcx_component

_OP = "install-mpis"
_FIXTURES = Path(__file__).parent / "fixtures" / "install_mpis"
_NPROC = 64
_PMIX_INTERNAL = "5.0.6"

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

_VENDOR_OMPI = ("--enable-mpi1-compatibility --without-xpmem --with-slurm "
                "--enable-orterun-prefix-by-default")
_INTERNAL = "--with-pmix=internal --with-hwloc=internal --with-libevent=internal"
_PLATFORM = "--with-platform=contrib/platform/mellanox/optimized"
_SED_ARGS = ["sed", "-i", "--follow-symlinks", "-E",
             "-e", r"s|/hpcx-rebuild\$\{mpi_version\}|/hpcx-rebuild|g",
             "-e", r"s|/hpcx-rebuild\$mpi_version|/hpcx-rebuild|g",
             "-e", "s|/hpcx-rebuild[0-9]+|/hpcx-rebuild|g"]
_BUGGY_NETLINK = ("    msg_done = ucs_netlink_is_msg_done("
                  "(const struct nlmsghdr *)recv_msg);\n")
_FIXED_NETLINK = "    msg_done = ucs_netlink_is_dump_done(recv_msg, recv_msg_len);\n"


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
                       "rhel8.10", "rhel9.6", "rhel10.0", "azurelinux3.0"):
            self.assertTrue(mpis.is_rpm_distro(distro), distro)
        for distro in ("ubuntu22.04", "ubuntu24.04", "ubuntu26.04", "azurelinux3.1", ""):
            self.assertFalse(mpis.is_rpm_distro(distro), distro)

        self.assertTrue(mpis.skips_mvapich("ubuntu24.04", "gb-family"))
        self.assertTrue(mpis.skips_mvapich("azurelinux3.0", "gb-family"))
        self.assertTrue(mpis.skips_mvapich("ubuntu26.04", "H100"))
        self.assertTrue(mpis.skips_mvapich("ubuntu26.04", "gb-family"))
        self.assertFalse(mpis.skips_mvapich("ubuntu22.04", "gb-family"))
        self.assertFalse(mpis.skips_mvapich("ubuntu24.04", "GB200"))
        self.assertFalse(mpis.skips_mvapich("ubuntu24.04", "A100"))

        self.assertTrue(mpis.skips_ompi("ubuntu26.04"))
        for distro in ("ubuntu24.04", "ubuntu22.04", "azurelinux3.0", "rhel10.0"):
            self.assertFalse(mpis.skips_ompi(distro), distro)

        for distro in ("almalinux8.10", "rocky8.10", "rhel8.10", "rhel8"):
            self.assertTrue(mpis.is_el8(distro), distro)
        for distro in ("almalinux8.9", "rocky9.8", "almalinux9.8", "ubuntu22.04"):
            self.assertFalse(mpis.is_el8(distro), distro)

        self.assertEqual(mpis.impi_fi_provider("no_rdma"), "tcp")
        self.assertEqual(mpis.impi_fi_provider("standard_ib"), "mlx")
        self.assertEqual(mpis.impi_fi_provider("mrc"), "mlx")

    def test_internal_pmix(self):
        for distro in ("ubuntu26.04", "azurelinux3.0", "almalinux10.0", "almalinux10",
                       "rocky10.1", "rhel10.0"):
            with self.subTest(distro=distro):
                self.assertTrue(mpis.uses_internal_pmix(distro, "azure_vm_regular"))
                self.assertTrue(mpis.internal_pmix_reason(distro, ""))
        for distro in ("ubuntu24.04", "ubuntu22.04", "almalinux9.8", "rocky8.10",
                       "rhel9.6", "almalinux1", "azurelinux3.1", "ubuntu26.10"):
            with self.subTest(distro=distro):
                self.assertFalse(mpis.uses_internal_pmix(distro, "azure_vm_regular"))
                self.assertIsNone(mpis.internal_pmix_reason(distro, ""))
        self.assertTrue(mpis.uses_internal_pmix("ubuntu24.04", "baremetal_3p"))
        self.assertIn("baremetal_3p", mpis.internal_pmix_reason("ubuntu22.04", "baremetal_3p"))
        self.assertFalse(mpis.uses_internal_pmix("ubuntu24.04", "baremetal_1p"))

    def test_ompi_deps_and_pmix_flags(self):
        internal = ["--with-pmix=internal", "--with-hwloc=internal", "--with-libevent=internal"]
        self.assertEqual(mpis.ompi_deps_args(True, None), internal)
        self.assertEqual(mpis.ompi_deps_args(False, "/opt/pmix/4.2.9"),
                         ["--with-pmix=/opt/pmix/4.2.9"])
        self.assertEqual(mpis.ompi_pmix_flags(True, "azurelinux3.0", "", None), internal)
        self.assertEqual(mpis.ompi_pmix_flags(False, "ubuntu24.04", "", "/opt/pmix/4.2.9"),
                         ["--with-pmix=/opt/pmix/4.2.9"])
        # bash's unreachable elif, mirrored (internal PMIx is checked first).
        self.assertEqual(mpis.ompi_pmix_flags(False, "azurelinux3.0", "", "/p"), ["--with-pmix"])
        self.assertEqual(mpis.ompi_pmix_flags(False, "ubuntu24.04", "baremetal_3p", "/p"),
                         ["--with-pmix"])

    def test_rebuild_decisions(self):
        self.assertTrue(mpis.rebuild_ucx_reason("AMD", "almalinux8.10", True))
        self.assertTrue(mpis.rebuild_ucx_reason("NVIDIA", "ubuntu22.04", True))
        self.assertTrue(mpis.rebuild_ucx_reason("", "ubuntu22.04", True))
        self.assertIsNone(mpis.rebuild_ucx_reason("AMD", "almalinux8.10", False))
        self.assertIsNone(mpis.rebuild_ucx_reason("NVIDIA", "ubuntu22.04", False))
        self.assertIsNone(mpis.rebuild_ucx_reason("NVIDIA", "ubuntu24.04", True))

        # Only NVIDIA + UCX + internal PMIx + no UCX rebuild keeps the vendor build.
        self.assertFalse(mpis.needs_hpcx_rebuild(rebuild_ucx=False, internal_pmix=True,
                                                 gpu="NVIDIA", uses_ucx=True))
        for kwargs in (dict(rebuild_ucx=True, internal_pmix=True, gpu="NVIDIA", uses_ucx=True),
                       dict(rebuild_ucx=False, internal_pmix=False, gpu="NVIDIA", uses_ucx=True),
                       dict(rebuild_ucx=False, internal_pmix=True, gpu="AMD", uses_ucx=True),
                       dict(rebuild_ucx=False, internal_pmix=True, gpu="NVIDIA", uses_ucx=False),
                       dict(rebuild_ucx=False, internal_pmix=True, gpu="", uses_ucx=True)):
            with self.subTest(**kwargs):
                self.assertTrue(mpis.needs_hpcx_rebuild(**kwargs))

    def test_hpcx_rebuild_command_variants(self):
        script = "/opt/h/utils/hpcx_rebuild.sh"
        external = ["--with-pmix=/opt/pmix/4.2.9"]
        internal = mpis.ompi_deps_args(True, None)
        cases = [
            # NVIDIA, Ubuntu 22.04: rebuilt UCX needs CUDA_HOME.
            (dict(gpu="NVIDIA", uses_ucx=True, rebuild_ucx=True, deps_args=external),
             [script, "--rebuild-ucx", "--cuda", "--with-hcoll", "--ompi-extra-config",
              f"{_VENDOR_OMPI} --with-pmix=/opt/pmix/4.2.9 {_PLATFORM}"],
             {"CUDA_HOME": "/usr/local/cuda"}),
            # NVIDIA UCX, no UCX rebuild.
            (dict(gpu="NVIDIA", uses_ucx=True, rebuild_ucx=False, deps_args=external),
             [script, "--cuda", "--with-hcoll", "--ompi-extra-config",
              f"{_VENDOR_OMPI} --with-pmix=/opt/pmix/4.2.9 {_PLATFORM}"], {}),
            # NVIDIA non-UCX (NCv6).
            (dict(gpu="NVIDIA", uses_ucx=False, rebuild_ucx=False, deps_args=external),
             [script, "--cuda", "--ompi-extra-config",
              f"{_VENDOR_OMPI} --with-pmix=/opt/pmix/4.2.9 --without-ucx "
              "--with-ofi=/opt/libfabric"], {}),
            # AMD UCX with internal PMIx (e.g. rhel10).
            (dict(gpu="AMD", uses_ucx=True, rebuild_ucx=True, deps_args=internal),
             [script, "--rebuild-ucx", "--ucx-extra-config", "--with-rocm=/opt/rocm",
              "--with-hcoll", "--ompi-extra-config",
              f"{_VENDOR_OMPI} {_INTERNAL} {_PLATFORM} --with-rocm=/opt/rocm"], {}),
            # AMD non-UCX: no UCX args, no ROCm Open MPI option.
            (dict(gpu="AMD", uses_ucx=False, rebuild_ucx=False, deps_args=external),
             [script, "--ompi-extra-config",
              f"{_VENDOR_OMPI} --with-pmix=/opt/pmix/4.2.9 --without-ucx "
              "--with-ofi=/opt/libfabric"], {}),
            # Neither vendor (GPU unset): no --cuda.
            (dict(gpu="", uses_ucx=True, rebuild_ucx=True, deps_args=external),
             [script, "--rebuild-ucx", "--with-hcoll", "--ompi-extra-config",
              f"{_VENDOR_OMPI} --with-pmix=/opt/pmix/4.2.9 {_PLATFORM}"], {}),
        ]
        for kwargs, command, extra_env in cases:
            with self.subTest(**{k: v for k, v in kwargs.items() if k != "deps_args"}):
                self.assertEqual(mpis.hpcx_rebuild_command("/opt/h", **kwargs),
                                 (command, extra_env))

    def test_ldd_libmpi_matches_awk(self):
        # Expected values are what  $(printf ... | awk '$1 ~ /^libmpi\.so/ {print $3}')
        # gives in bash.
        self.assertEqual(mpis.ldd_libmpi(
            "\tlinux-vdso.so.1 (0x00007ffd)\n"
            "\tlibmpi.so.40 => /opt/h/hpcx-rebuild/lib/libmpi.so.40 (0x00007f00)\n"
            "\tlibmpi_mpifh.so.40 => /x/libmpi_mpifh.so.40 (0x1)\n"),
            "/opt/h/hpcx-rebuild/lib/libmpi.so.40")
        self.assertEqual(mpis.ldd_libmpi("\tlibmpi.so.40 => not found\n"), "not")
        self.assertEqual(mpis.ldd_libmpi(
            "libmpi.so.40 => /a/lib/libmpi.so.40 (0x1)\nlibmpi.so.12\n"
            "libmpi.so.40 => /b/lib/libmpi.so.40\n"),
            "/a/lib/libmpi.so.40\n\n/b/lib/libmpi.so.40")
        self.assertEqual(mpis.ldd_libmpi("  libmpi.so.40   =>   /c/libmpi.so (0x1)\n"),
                         "/c/libmpi.so")
        self.assertEqual(mpis.ldd_libmpi("\tnot a dynamic executable\n"), "")
        self.assertEqual(mpis.ldd_libmpi(""), "")

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
            "5.0.10", uses_ucx=True, pmix_flags=["--with-pmix=/opt/pmix/4.2.9"],
            ucx_path="/opt/h/ucx", hcoll_path="/opt/h/hcoll", ld_library_path="")
        self.assertEqual(cmd, [
            "./configure", "LD_LIBRARY_PATH=:/opt/h/hcoll/lib", "--prefix=/opt/openmpi-5.0.10",
            "--with-ucx=/opt/h/ucx", "--with-hcoll=/opt/h/hcoll",
            "--with-pmix=/opt/pmix/4.2.9", "--enable-mpirun-prefix-by-default", _PLATFORM])
        self.assertEqual(lib, "/opt/h/ucx/lib")
        cmd, lib = mpis.ompi_configure_command(
            "5.0.10", uses_ucx=False, pmix_flags=mpis.ompi_deps_args(True, None),
            ucx_path="/opt/h/ucx", hcoll_path="/opt/h/hcoll", ld_library_path="/x")
        self.assertEqual(cmd, ["./configure", "--prefix=/opt/openmpi-5.0.10", "--without-ucx",
                               "--with-ofi=/opt/libfabric", *_INTERNAL.split(),
                               "--enable-mpirun-prefix-by-default"])
        self.assertEqual(lib, "/opt/libfabric/lib")

    def test_hpcx_module(self):
        self.assertEqual(mpis.hpcx_module("/opt/h", True), "/opt/h/modulefiles/hpcx-rebuild")
        self.assertEqual(mpis.hpcx_module("/opt/h", False), "/opt/h/modulefiles/hpcx")


# --------------------------------------------------------------------------- #
# Modulefiles, compared with what the real bash writes.
# --------------------------------------------------------------------------- #
# scenario -> (DISTRIBUTION, ARCHITECTURE, GPU, SKU, SKU_FAMILY, TARGET_NODE_TYPE,
#              HPC-X version, HPC-X folder); the same values were used to render
#              the fixtures with bash.
_SCENARIOS = {
    "ucx": ("ubuntu24.04", "x86_64", "NVIDIA", "A100", "A100", "azure_vm_regular", "2.25.1",
            "hpcx-v2.25.1-gcc-doca_ofed-ubuntu24.04-cuda13-x86_64"),
    "non_ucx": ("ubuntu24.04", "x86_64", "NVIDIA", "NCv6", "NCv6", "azure_vm_regular", "2.25.1",
                "hpcx-v2.25.1-gcc-inbox-ubuntu24.04-cuda13-x86_64"),
    "el8_amd": ("almalinux8.10", "x86_64", "AMD", "MI300X", "MI300X", "azure_vm_regular", "2.18",
                "hpcx-v2.18-gcc-mlnx_ofed-redhat8-cuda12-x86_64"),
    "gb_aarch64": ("ubuntu24.04", "aarch64", "NVIDIA", "GB200", "gb-family", "azure_vm_regular",
                   "2.25.1", "hpcx-v2.25.1-gcc-doca_ofed-ubuntu24.04-cuda13-aarch64"),
    "u22_ucx": ("ubuntu22.04", "x86_64", "NVIDIA", "A100", "A100", "azure_vm_regular", "2.25.1",
                "hpcx-v2.25.1-gcc-doca_ofed-ubuntu24.04-cuda13-x86_64"),
    "azl3_gb": ("azurelinux3.0", "aarch64", "NVIDIA", "GB200", "gb-family", "azure_vm_regular",
                "2.25.1", "hpcx-v2.25.1-gcc-doca_ofed-ubuntu24.04-cuda13-aarch64"),
    "u26": ("ubuntu26.04", "x86_64", "NVIDIA", "H100", "H100", "azure_vm_regular", "2.25.1",
            "hpcx-v2.25.1-gcc-doca_ofed-ubuntu24.04-cuda13-x86_64"),
}


class ModulefileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _render(self, name):
        """Write the modulefiles in bash's order, with bash's decisions."""
        distro, arch, gpu, sku, family, node_type, hpcx_version, folder = _SCENARIOS[name]
        uses_ucx = sku_uses_ucx(sku)
        internal = mpis.uses_internal_pmix(distro, node_type)
        rebuild_ucx = mpis.rebuild_ucx_reason(gpu, distro, uses_ucx) is not None
        hpcx_path = f"/opt/{folder}"
        hpcx = mpis._Hpcx(path=hpcx_path, hcoll=f"{hpcx_path}/hcoll",
                          sharp=f"{hpcx_path}/sharp",
                          ucx=f"{hpcx_path}/ucx{'/hpcx-rebuild' if rebuild_ucx else ''}")
        mvapich = None if mpis.skips_mvapich(distro, family) else {"version": "4.1"}
        ompi = None if mpis.skips_ompi(distro) else {"version": "5.0.10"}
        impi = None if arch == "aarch64" else {"version": "2021.16.1"}
        inp = mpis._Inputs(
            distribution=distro, architecture=arch, gpu=gpu, sku=sku, sku_family=family,
            node_type=node_type, network_mode=sku_network_mode(sku, node_type),
            uses_ucx=uses_ucx, module_dir=os.path.join(self._tmp.name, name),
            component_dir="", hpcx_component="", hpcx_reason="",
            hpcx={"version": hpcx_version}, internal_pmix=internal, internal_pmix_reason=None,
            pmix_version=None, pmix_path="/opt/pmix/4.2.9", rebuild_ucx=rebuild_ucx,
            rebuild_ucx_reason=None,
            rebuild_hpcx=mpis.needs_hpcx_rebuild(rebuild_ucx=rebuild_ucx, internal_pmix=internal,
                                                 gpu=gpu, uses_ucx=uses_ucx),
            mvapich=mvapich, ompi=ompi, impi=impi,
            impi_2021_version="2021.16" if impi else None)
        os.makedirs(inp.mpi_dir)
        report = mpis._Report()
        with mock.patch.object(mpis, "log_info"), \
             mock.patch.object(mpis, "log_warn") as warn:
            mpis._write_hpcx_modulefiles(inp, hpcx, report)
            if mvapich:
                _, lib = mpis.mvapich_configure_command(distro, "4.1", uses_ucx=uses_ucx,
                                                        ucx_path=hpcx.ucx)
                mpis._write_mvapich_modulefile(inp, "4.1", lib, report)
            if ompi:
                _, lib = mpis.ompi_configure_command(
                    "5.0.10", uses_ucx=uses_ucx, pmix_flags=[], ucx_path=hpcx.ucx,
                    hcoll_path=hpcx.hcoll, ld_library_path="")
                mpis._write_ompi_modulefile(inp, "5.0.10", hpcx, lib, report)
            if impi:
                mpis._write_impi_modulefile(inp, report)
        warn.assert_not_called()       # first run: no rerun warnings (incl. EL8 append)
        return inp.mpi_dir

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
                version = _SCENARIOS[name][6]
                self.assertEqual(os.readlink(os.path.join(mpi_dir, "hpcx")),
                                 os.path.join(mpi_dir, f"hpcx-{version}"))
                self.assertEqual(os.readlink(os.path.join(mpi_dir, f"hpcx-pmix-{version}")),
                                 os.path.join(mpi_dir, f"hpcx-{version}"))


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
        self.patch_tmp = os.path.join(root, "patch-tmp")
        self.components = os.path.join(root, "components")
        os.makedirs(self.opt)
        os.makedirs(self.ld_dir)
        os.makedirs(self.patch_tmp)
        os.makedirs(os.path.join(self.components, "patches"))
        Path(self.components, "patches", "ucx-netlink-dump-done.patch").write_text("patch")
        for name, value in (("_INSTALL_PREFIX", self.opt), ("_WORK_DIR", self.work),
                            ("_ROCM_DIR", self.rocm), ("_LD_SO_CONF_DIR", self.ld_dir),
                            ("_PKGCONFIG_DIRS", (self.pc_lib,)),
                            ("_PKGCONFIG_DIRS_RPM", (self.pc_lib, self.pc_lib64)),
                            ("_PATCH_TMP_PARENT", self.patch_tmp)):
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
        # tar -xzf <ucx tarball> -C <dir>: unpack netlink.c (buggy, fixed or absent).
        if command[:2] == ["tar", "-xzf"] and self.netlink is not None:
            name = os.path.basename(command[2])[:-len(".tar.gz")]
            source = Path(command[4], name, "src", "ucs", "sys")
            source.mkdir(parents=True)
            (source / "netlink.c").write_text(self.netlink)
        return 0

    def _fake_capture(self, command, op, *, cwd=None, env=None, failure_is_error=True):
        self.assertEqual(op, _OP)
        self.captures.append((command, cwd, env, failure_is_error))
        if command == ["nproc"]:
            return self.nproc
        if command[0].endswith("/ucx_info"):
            return self.ucx_info
        if command[0] == "ldd":
            if self.ldd is not None:
                return self.ldd
            rebuild = command[1][:-len("/bin/ompi_info")]
            return (0, f"\tlinux-vdso.so.1 (0x1)\n\tlibmpi.so.40 => {rebuild}/lib/"
                       "libmpi.so.40 (0x00007f00)\n")
        if command[0] == "pkg-config":
            return self.pkg_config
        raise AssertionError(f"unexpected run_capture {command}")

    def _install(self, distro="ubuntu22.04", arch="x86_64", gpu="NVIDIA", sku="A100",
                 family=None, node_type="azure_vm_regular", configs=None,
                 fail_on=None, nproc=(0, f"{_NPROC}\n"), pin_ok=True, download_error=None,
                 write_error=None, extra_env=None, netlink=_BUGGY_NETLINK,
                 ucx_tarballs=("ucx-1.20.0.tar.gz",), ucx_info=(0, "# Library version: 1.20.0\n"),
                 ldd=None, pkg_config=(0, f"{_PMIX_INTERNAL}\n")):
        self.calls, self.captures, self.fail_on = [], [], fail_on
        self.netlink, self.ucx_info, self.ldd, self.pkg_config = netlink, ucx_info, ldd, pkg_config
        self.nproc = nproc
        self.record_points = []
        configs = dict(_CONFIGS if configs is None else configs)
        env = {"DISTRIBUTION": distro, "ARCHITECTURE": arch, "GPU": gpu, "SKU": sku,
               "SKU_FAMILY": family or sku, "TARGET_NODE_TYPE": node_type,
               "MODULE_FILES_DIRECTORY": self.modules, "COMPONENT_DIR": self.components,
               "PATH": "/usr/bin", "LD_LIBRARY_PATH": "/caller/lib"}
        env.update(extra_env or {})
        # The HPC-X trees' bundled UCX source (mv itself is mocked).
        for url in (_HPCX_URL, _INBOX_URL, _AMD_URL):
            sources = Path(self._hpcx_path(url), "sources")
            sources.mkdir(parents=True, exist_ok=True)
            for name in ucx_tarballs:
                (sources / name).write_bytes(b"tgz")

        def fake_download(url, sha256, dest_dir="."):
            if download_error:
                raise download_error
            path = Path(dest_dir) / Path(url).name
            path.write_bytes(b"x")
            return path

        def fake_record(component, version):
            self.record_points.append((component, len(self.calls)))
            if write_error:
                raise write_error

        def fake_mkdtemp(dir=None):
            # A fixed name, so a command list from one run matches the next.
            self.assertEqual(dir, self.patch_tmp)
            path = os.path.join(dir, "tmp.ucx")
            os.mkdir(path)
            return path

        with mock.patch.object(mpis.tempfile, "mkdtemp", side_effect=fake_mkdtemp), \
             mock.patch.object(mpis, "config_for",
                               side_effect=lambda c, e: configs.get(c)) as config, \
             mock.patch.object(mpis, "download_and_verify",
                               side_effect=fake_download) as download, \
             mock.patch.object(mpis, "exec_program", side_effect=self._fake_exec), \
             mock.patch.object(mpis, "run_capture", side_effect=self._fake_capture), \
             mock.patch.object(mpis, "write_component_version",
                               side_effect=fake_record) as record, \
             mock.patch.object(mpis, "dnf_pin_packages", return_value=pin_ok) as pin, \
             mock.patch.object(mpis, "log_info") as info, \
             mock.patch.object(mpis, "log_warn") as warn, \
             mock.patch.object(mpis, "log_error") as error, \
             mock.patch.object(mpis, "log_error_detail") as error_detail:
            result = mpis.install(env)
        self.env = env
        self.config, self.download, self.record, self.pin = config, download, record, pin
        self.info, self.warn, self.error, self.error_detail = info, warn, error, error_detail
        return result

    def _commands(self):
        return [c[0] for c in self.calls]

    def _captured(self):
        return [c[0] for c in self.captures]

    def _messages(self, logger):
        return [c.args[1] for c in logger.call_args_list]

    def _records(self):
        return [c.args for c in self.record.call_args_list]

    def _hpcx_path(self, url):
        return f"{self.opt}/{mpis.url_basename(url, '.tbz')}"

    def _call_env(self, predicate):
        return next(env for command, _, env in self.calls if predicate(command))

    def _hpcx_start(self, url):
        path = self._hpcx_path(url)
        return [["tar", "-xvf", mpis.url_basename(url)],
                ["mv", mpis.url_basename(url, ".tbz"), self.opt],
                [f"{path}/utils/hpcx_fix_ladir.sh"]]

    def _patch_commands(self, url):
        tarball = f"{self._hpcx_path(url)}/sources/ucx-1.20.0.tar.gz"
        tar_x = ["tar", "-xzf", tarball, "-C"]
        return tar_x, tarball

    def _hpcx_end(self, url):
        path = self._hpcx_path(url)
        return [[f"{path}/utils/hpcx_fix_ladir.sh"], ["ldconfig"]]

    def _rebuild(self, url, *args):
        return [f"{self._hpcx_path(url)}/utils/hpcx_rebuild.sh", *args]

    def _cp_tests(self, url):
        path = self._hpcx_path(url)
        return ["cp", "-r", f"{path}/ompi/tests", f"{path}/hpcx-rebuild"]

    def _mvapich_commands(self, configure_tail, fflags=True):
        return [["tar", "-xvf", "mvapich-4.1.tar.gz"],
                ["./configure", *(["FFLAGS=-fallow-argument-mismatch"] if fflags else []),
                 f"--prefix={self.opt}/mvapich-4.1", "--enable-g=none", "--enable-fast=yes",
                 *configure_tail],
                ["make", f"-j{_NPROC}"], ["make", "install"]]

    def _ompi_commands(self, configure):
        return [["tar", "-xvf", "openmpi-5.0.10.tar.gz"], configure,
                ["make", f"-j{_NPROC}"], ["make", "install"]]

    def _ompi_ucx_configure(self, url, ucx_sub, pmix_flags):
        path = self._hpcx_path(url)
        return ["./configure", f"LD_LIBRARY_PATH=/caller/lib:{path}/hcoll/lib",
                f"--prefix={self.opt}/openmpi-5.0.10", f"--with-ucx={path}/{ucx_sub}",
                f"--with-hcoll={path}/hcoll", *pmix_flags,
                "--enable-mpirun-prefix-by-default", _PLATFORM]

    def _impi_commands(self):
        modulefiles = f"{self.opt}/intel/oneapi/mpi/2021.16/etc/modulefiles"
        return [["bash", "intel-mpi-2021.16.1.804_offline.sh", "-s", "-a", "-s",
                 "--eula", "accept"],
                ["mv", f"{modulefiles}/mpi", f"{modulefiles}/impi"]]

    def _mpi_dir(self):
        return os.path.join(self.modules, "mpi")

    def _assert_tmp_cleaned(self):
        self.assertEqual(os.listdir(self.patch_tmp), [])
        self.assertFalse(os.path.exists(self.work))

    # ---- success paths ------------------------------------------------------
    def test_ubuntu22_patches_and_rebuilds_ucx_full_command_list(self):
        self.assertEqual(self._install(), 0)
        path = self._hpcx_path(_HPCX_URL)
        pmix = f"{self.opt}/pmix/4.2.9"
        ucx = f"{path}/ucx/hpcx-rebuild"
        commands = self._commands()
        tar_x, tarball = self._patch_commands(_HPCX_URL)
        # The scratch dir comes from mkdtemp; take it from the tar command.
        work_dir = commands[3][4]
        self.assertEqual(os.path.dirname(work_dir), self.patch_tmp)
        patch_file = os.path.join(self.components, "patches", "ucx-netlink-dump-done.patch")
        expected = self._hpcx_start(_HPCX_URL) + [
            [*tar_x, work_dir],
            ["patch", "-d", f"{work_dir}/ucx-1.20.0", "-p1", "-i", patch_file],
            ["tar", "-czf", tarball, "-C", work_dir, "ucx-1.20.0"],
            self._rebuild(_HPCX_URL, "--rebuild-ucx", "--cuda", "--with-hcoll",
                          "--ompi-extra-config",
                          f"{_VENDOR_OMPI} --with-pmix={pmix} {_PLATFORM}"),
            self._cp_tests(_HPCX_URL),
        ] + self._hpcx_end(_HPCX_URL) + self._mvapich_commands([f"--with-ucx={ucx}"]) + \
            self._ompi_commands(self._ompi_ucx_configure(
                _HPCX_URL, "ucx/hpcx-rebuild", [f"--with-pmix={pmix}"])) + self._impi_commands()
        self.assertEqual(commands, expected)
        self.assertEqual(self._captured(), [
            ["nproc"], [f"{ucx}/bin/ucx_info", "-v"],
            ["ldd", f"{path}/hpcx-rebuild/bin/ompi_info"]])
        # ucx_info runs against the rebuilt UCX's libs only; ldd's exit code is an answer.
        self.assertEqual(self.captures[1][2]["LD_LIBRARY_PATH"], f"{ucx}/lib")
        self.assertFalse(self.captures[2][3])

        # cwd: work dir for downloads/tar/scripts, source dir for the builds.
        cwds = [c[1] for c in self.calls]
        self.assertEqual(cwds[:10], [self.work] * 10)
        self.assertEqual(cwds[10:14], [self.work] + [f"{self.work}/mvapich-4.1"] * 3)
        self.assertEqual(cwds[14:18], [self.work] + [f"{self.work}/openmpi-5.0.10"] * 3)
        # HPCX_DIR only for the two hpcx_fix_ladir.sh runs; CUDA_HOME only for the rebuild.
        for command, _, env in self.calls:
            self.assertEqual(env["PATH"], "/usr/bin")
            if command[0].endswith("hpcx_fix_ladir.sh"):
                self.assertEqual(env["HPCX_DIR"], path)
            else:
                self.assertNotIn("HPCX_DIR", env)
            if command[0].endswith("hpcx_rebuild.sh"):
                self.assertEqual(env["CUDA_HOME"], "/usr/local/cuda")
            else:
                self.assertNotIn("CUDA_HOME", env)

        self.assertEqual([c.args for c in self.download.call_args_list],
                         [(_HPCX_URL, "h1"), (_MVAPICH_URL, "m1"), (_OMPI_URL, "o1"),
                          (_IMPI_URL, "i1")])
        self.assertEqual(self._records(), [("HPCX", "2.25.1"), ("MVAPICH", "4.1"),
                                           ("OMPI", "5.0.10"), ("IMPI", "2021.16.1")])
        self.pin.assert_not_called()                    # not an RPM distro
        self.assertEqual(_read(os.path.join(self.ld_dir, "hpcx-sharp.conf")),
                         f"{path}/sharp/lib\n")
        self.assertEqual(_read(os.path.join(self.ld_dir, "hpcx-ucx.conf")), f"{ucx}/lib\n")
        self.assertTrue(os.path.isdir(self.pc_lib))
        self.assertFalse(os.path.exists(self.pc_lib64))
        self.assertEqual(sorted(os.listdir(self._mpi_dir())), [
            "hpcx", "hpcx-2.25.1", "hpcx-pmix", "hpcx-pmix-2.25.1", "impi-2021",
            "impi_2021.16", "mvapich", "mvapich-4.1", "openmpi", "openmpi-5.0.10"])
        self.assertTrue(os.path.islink(os.path.join(self._mpi_dir(), "hpcx-pmix-2.25.1")))
        self.assertIn(f"module load {path}/modulefiles/hpcx-rebuild\n",
                      _read(os.path.join(self._mpi_dir(), "hpcx-2.25.1")))
        self.assertIn(f"{path}/hcoll/lib:{ucx}/lib\n",
                      _read(os.path.join(self._mpi_dir(), "openmpi-5.0.10")))
        self.assertIn("FI_PROVIDER     mlx\n",
                      _read(os.path.join(self._mpi_dir(), "impi_2021.16")))
        self._assert_tmp_cleaned()
        self.error.assert_not_called()

    def test_logs_decisions_versions_and_summary(self):
        self.assertEqual(self._install(), 0)
        messages = self._messages(self.info)
        start = self.info.call_args_list[0]
        self.assertTrue(start.args[1].startswith("Installing MPIs on ubuntu22.04"))
        self.assertEqual((start.kwargs["hpcx_component"], start.kwargs["network_mode"],
                          start.kwargs["pmix_path"], start.kwargs["internal_pmix"],
                          start.kwargs["rebuild_ucx"], start.kwargs["rebuild_hpcx"],
                          start.kwargs["mvapich_version"], start.kwargs["ompi_version"],
                          start.kwargs["impi_version"]),
                         ("hpcx", "standard_ib", f"{self.opt}/pmix/4.2.9", False, True, True,
                          "4.1", "5.0.10", "2021.16.1"))
        self.assertIn("HPC-X variant hpcx: default HPC-X (InfiniBand network)", messages)
        self.assertIn(f"PMIx path for the MPI builds: {self.opt}/pmix/4.2.9", messages)
        self.assertTrue(any(m.startswith("HPC-X rebuild: yes, UCX rebuild: ubuntu22.04")
                            for m in messages))
        self.assertTrue(any(m.startswith("Patched ucx-1.20.0 for the netlink route-dump hang")
                            for m in messages))
        rebuild = next(c for c in self.info.call_args_list
                       if c.args[1].startswith("Rebuilding HPC-X Open MPI and UCX"))
        self.assertEqual(rebuild.kwargs["rebuild_env"], "CUDA_HOME=/usr/local/cuda")
        self.assertIn("--rebuild-ucx --cuda --with-hcoll", rebuild.kwargs["rebuild_args"])
        self.assertTrue(any(m.startswith("Rebuilt UCX at ") and m.endswith(
            "runs: # Library version: 1.20.0") for m in messages))
        self.assertTrue(any(m.startswith("Rebuilt ompi_info loads ") for m in messages))
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
        self.assertEqual(summary.kwargs["symlinks"], 6)
        self.assertEqual(len(summary.kwargs["patched"]), 1)
        self.assertTrue(summary.kwargs["rebuilt_hpcx"])
        # Warnings: `set CC=` always; `rm -rf -- */` on success; no rerun warnings.
        warnings = self._messages(self.warn)
        self.assertEqual(len(warnings), 2)
        self.assertIn("set CC=/usr/bin/gcc", warnings[0])
        self.assertIn("rm -rf -- */", warnings[1])

    def test_ubuntu24_ucx_sku_rebuilds_openmpi_only(self):
        self.assertEqual(self._install(distro="ubuntu24.04"), 0)
        path = self._hpcx_path(_HPCX_URL)
        pmix = f"{self.opt}/pmix/4.2.9"
        expected = self._hpcx_start(_HPCX_URL) + [
            self._rebuild(_HPCX_URL, "--cuda", "--with-hcoll", "--ompi-extra-config",
                          f"{_VENDOR_OMPI} --with-pmix={pmix} {_PLATFORM}"),
            self._cp_tests(_HPCX_URL),
        ] + self._hpcx_end(_HPCX_URL) + self._mvapich_commands([f"--with-ucx={path}/ucx"]) + \
            self._ompi_commands(self._ompi_ucx_configure(
                _HPCX_URL, "ucx", [f"--with-pmix={pmix}"])) + self._impi_commands()
        self.assertEqual(self._commands(), expected)
        self.assertNotIn("CUDA_HOME", self.calls[3][2])
        # No UCX rebuild: no patch, no ucx_info.
        self.assertEqual(self._captured(), [["nproc"],
                                            ["ldd", f"{path}/hpcx-rebuild/bin/ompi_info"]])
        self.assertEqual(_read(os.path.join(self.ld_dir, "hpcx-ucx.conf")), f"{path}/ucx/lib\n")

    def test_ncv6_uses_inbox_hpcx_and_libfabric(self):
        self.assertEqual(self._install(distro="ubuntu24.04", sku="NCv6"), 0)
        pmix = f"{self.opt}/pmix/4.2.9"
        expected = self._hpcx_start(_INBOX_URL) + [
            self._rebuild(_INBOX_URL, "--cuda", "--ompi-extra-config",
                          f"{_VENDOR_OMPI} --with-pmix={pmix} --without-ucx "
                          "--with-ofi=/opt/libfabric"),
            self._cp_tests(_INBOX_URL),
        ] + self._hpcx_end(_INBOX_URL) + self._mvapich_commands(
            ["--with-device=ch4:ofi", "--with-libfabric=/opt/libfabric"]) + \
            self._ompi_commands(["./configure", f"--prefix={self.opt}/openmpi-5.0.10",
                                 "--without-ucx", "--with-ofi=/opt/libfabric",
                                 f"--with-pmix={pmix}", "--enable-mpirun-prefix-by-default"]) + \
            self._impi_commands()
        self.assertEqual(self._commands(), expected)
        self.assertEqual(self.config.call_args_list[0].args[0], "hpcx_inbox")
        self.assertIn("HPC-X variant hpcx_inbox: network mode no_rdma: no DOCA-OFED on "
                      "non-IB SKUs, so use the inbox HPC-X (UCX linked against rdma-core)",
                      self._messages(self.info))
        mpi_dir = self._mpi_dir()
        self.assertIn("FI_PROVIDER     tcp\n", _read(os.path.join(mpi_dir, "impi_2021.16")))
        self.assertTrue(_read(os.path.join(mpi_dir, "mvapich-4.1")).endswith(
            "/opt/mvapich-4.1\nsetenv          FI_PROVIDER tcp\n"
            "setenv          MPIR_CVAR_CH4_CMA_ENABLE 0\n"))

    def test_ubuntu22_ncv6_does_not_rebuild_ucx(self):
        self.assertEqual(self._install(sku="NCv6"), 0)
        rebuild = self._commands()[3]
        self.assertTrue(rebuild[0].endswith("hpcx_rebuild.sh"))
        self.assertNotIn("--rebuild-ucx", rebuild)
        self.assertNotIn("CUDA_HOME", self.calls[3][2])

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

        # HPC-X 2.18's UCX doesn't have the netlink bug: no patch.
        self.assertEqual(self._install(distro="almalinux8.10", gpu="AMD", sku="MI300X",
                                       netlink=_FIXED_NETLINK), 0)
        pmix = f"{self.opt}/pmix/4.2.9"
        ucx = f"{path}/ucx/hpcx-rebuild"
        commands = self._commands()
        tar_x, _ = self._patch_commands(_AMD_URL)
        self.assertEqual(commands[:8], self._hpcx_start(_AMD_URL) + [
            [*tar_x, commands[3][4]],
            self._rebuild(_AMD_URL, "--rebuild-ucx", "--ucx-extra-config",
                          "--with-rocm=/opt/rocm", "--with-hcoll", "--ompi-extra-config",
                          f"{_VENDOR_OMPI} --with-pmix={pmix} {_PLATFORM} --with-rocm=/opt/rocm"),
            self._cp_tests(_AMD_URL)] + self._hpcx_end(_AMD_URL))
        self.assertNotIn("CUDA_HOME", self.calls[4][2])
        self.assertIn("ucx-1.20.0 does not have the netlink dump hang; skipping UCX patch",
                      self._messages(self.info))
        # No FFLAGS on EL; the rebuilt UCX is used from here on.
        self.assertEqual(commands[8:12], self._mvapich_commands([f"--with-ucx={ucx}"],
                                                                fflags=False))
        self.assertIn(f"--with-ucx={ucx}", commands[13])
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
        self.assertEqual(self.config.call_args_list[0].args[0], "hpcx_amd")
        self._assert_tmp_cleaned()
        self.error.assert_not_called()

    def test_amd_without_rocm_fails_after_the_patch_check(self):
        self.assertEqual(self._install(distro="almalinux8.10", gpu="AMD", sku="MI300X",
                                       netlink=_FIXED_NETLINK), 3)
        self.assertEqual(len(self.calls), 4)          # tar, mv, fix_ladir, tar -xzf
        self.record.assert_called_once_with("HPCX", "2.18")
        self.assertEqual(self.error.call_args.args[1],
                         "ROCm must be installed before rebuilding HPC-X UCX with "
                         "ROCm support.")
        self._assert_tmp_cleaned()

    def test_amd_non_ucx_rebuilds_without_ucx_but_still_needs_rocm(self):
        # bash now checks /opt/rocm for every AMD rebuild, not only UCX ones.
        self.assertEqual(self._install(distro="ubuntu24.04", gpu="AMD", sku="NCv6"), 3)
        self.assertEqual(len(self.calls), 3)
        os.makedirs(self.rocm)
        shutil.rmtree(self.modules)
        self.assertEqual(self._install(distro="ubuntu24.04", gpu="AMD", sku="NCv6"), 0)
        self.assertEqual(self._commands()[3], self._rebuild(
            _AMD_URL, "--ompi-extra-config",
            f"{_VENDOR_OMPI} --with-pmix={self.opt}/pmix/4.2.9 --without-ucx "
            "--with-ofi=/opt/libfabric"))

    def test_azure_linux_gb200_internal_pmix_uses_vendor_hpcx(self):
        # No pmix config needed: Open MPI uses its internal PMIx.
        configs = dict(_CONFIGS, pmix=None)
        self.assertEqual(self._install(distro="azurelinux3.0", arch="aarch64", sku="GB200",
                                       family="gb-family", configs=configs), 0)
        path = self._hpcx_path(_HPCX_URL)
        expected = self._hpcx_start(_HPCX_URL) + self._hpcx_end(_HPCX_URL) + \
            self._ompi_commands(self._ompi_ucx_configure(_HPCX_URL, "ucx", _INTERNAL.split()))
        self.assertEqual(self._commands(), expected)
        self.assertEqual(self._captured(), [["nproc"], ["pkg-config", "--modversion", "pmix"]])
        self.assertEqual(self.captures[1][2]["PKG_CONFIG_PATH"], f"{path}/ompi5/lib/pkgconfig")
        config_names = [c.args[0] for c in self.config.call_args_list]
        for name in ("pmix", "mvapich", "impi"):
            self.assertNotIn(name, config_names)
        self.assertEqual(self._records(), [("HPCX", "2.25.1"), ("PMIX", _PMIX_INTERNAL),
                                           ("OMPI", "5.0.10")])
        self.assertEqual([c.args[0] for c in self.pin.call_args_list],
                         [["ucx*"], ["openmpi", "perftest"]])
        self.assertTrue(os.path.isdir(self.pc_lib64))
        self.assertIn(f"module load {path}/modulefiles/hpcx\n",
                      _read(os.path.join(self._mpi_dir(), "hpcx-2.25.1")))
        messages = self._messages(self.info)
        self.assertTrue(any(m.startswith("Skipping MVAPICH: not supported on azurelinux3.0 "
                                         "for gb-family SKUs") for m in messages))
        self.assertIn("Skipping Intel MPI on aarch64 (x86_64 only)", messages)
        self.assertTrue(any(m.startswith("Not rebuilding HPC-X") for m in messages))
        self.assertTrue(any(m.startswith("Open MPI uses its internal PMIx") and
                            "azurelinux3.0" in m for m in messages))
        self.assertEqual(sorted(os.listdir(self._mpi_dir())), [
            "hpcx", "hpcx-2.25.1", "hpcx-pmix", "hpcx-pmix-2.25.1", "openmpi",
            "openmpi-5.0.10"])

    def test_ubuntu26_skips_mvapich_and_openmpi(self):
        configs = dict(_CONFIGS, pmix=None, mvapich=None, ompi=None)
        self.assertEqual(self._install(distro="ubuntu26.04", sku="H100", configs=configs), 0)
        path = self._hpcx_path(_HPCX_URL)
        self.assertEqual(self._commands(), self._hpcx_start(_HPCX_URL) +
                         self._hpcx_end(_HPCX_URL) + self._impi_commands())
        self.assertEqual(self.captures[1][2]["PKG_CONFIG_PATH"], f"{path}/ompi5/lib/pkgconfig")
        self.assertEqual(self._records(), [("HPCX", "2.25.1"), ("PMIX", _PMIX_INTERNAL),
                                           ("IMPI", "2021.16.1")])
        self.pin.assert_not_called()
        self.assertEqual(sorted(os.listdir(self._mpi_dir())), [
            "hpcx", "hpcx-2.25.1", "hpcx-pmix", "hpcx-pmix-2.25.1", "impi-2021",
            "impi_2021.16"])
        messages = self._messages(self.info)
        self.assertTrue(any(m.startswith("Skipping MVAPICH: MVAPICH 4.1 does not build with "
                                         "ubuntu26.04's gcc 15") for m in messages))
        self.assertTrue(any(m.startswith("Skipping the standalone Open MPI on ubuntu26.04")
                            for m in messages))
        self.assertEqual(self.info.call_args_list[0].kwargs["ompi_version"], "skipped")

    def test_baremetal_3p_uses_internal_pmix_on_ubuntu(self):
        self.assertEqual(self._install(distro="ubuntu24.04", arch="aarch64", sku="GB200",
                                       family="gb-family", node_type="baremetal_3p"), 0)
        commands = self._commands()
        self.assertEqual(commands[:5], self._hpcx_start(_HPCX_URL) + self._hpcx_end(_HPCX_URL))
        self.assertEqual(commands[6], self._ompi_ucx_configure(_HPCX_URL, "ucx",
                                                               _INTERNAL.split()))
        self.assertNotIn("pmix", [c.args[0] for c in self.config.call_args_list])
        self.assertEqual(self._records()[1], ("PMIX", _PMIX_INTERNAL))
        self.pin.assert_not_called()

    def test_rhel10_amd_rebuilds_with_internal_pmix(self):
        os.makedirs(self.rocm)
        configs = dict(_CONFIGS, pmix=None)
        self.assertEqual(self._install(distro="rhel10.0", gpu="AMD", sku="MI300X",
                                       configs=configs), 0)
        path = self._hpcx_path(_AMD_URL)
        commands = self._commands()
        self.assertEqual(commands[6], self._rebuild(
            _AMD_URL, "--rebuild-ucx", "--ucx-extra-config", "--with-rocm=/opt/rocm",
            "--with-hcoll", "--ompi-extra-config",
            f"{_VENDOR_OMPI} {_INTERNAL} {_PLATFORM} --with-rocm=/opt/rocm"))
        self.assertEqual(self._captured()[-1], ["pkg-config", "--modversion", "pmix"])
        self.assertEqual(self.captures[-1][2]["PKG_CONFIG_PATH"],
                         f"{path}/hpcx-rebuild/lib/pkgconfig")
        self.assertEqual(self._records()[:2], [("HPCX", "2.18"), ("PMIX", _PMIX_INTERNAL)])
        self.assertIn(self._ompi_ucx_configure(_AMD_URL, "ucx/hpcx-rebuild", _INTERNAL.split())
                      [2:], [c[2:] for c in commands])
        self.assertEqual([c.args[0] for c in self.pin.call_args_list],
                         [["ucx*"], ["openmpi", "perftest"]])

    def test_rhel_rpm_pins_and_el8_impi(self):
        self.assertEqual(self._install(distro="rhel8.10", sku="H100"), 0)
        self.assertEqual([c.args[0] for c in self.pin.call_args_list],
                         [["ucx*"], ["openmpi", "perftest"]])
        self.assertIn("I_MPI_STARTUP_MODE", _read(os.path.join(self._mpi_dir(), "impi_2021.16")))

    def test_gb200_on_ubuntu22_still_builds_mvapich(self):
        self.assertEqual(self._install(distro="ubuntu22.04", arch="aarch64", sku="GB200",
                                       family="gb-family"), 0)
        self.assertIn(["tar", "-xvf", "mvapich-4.1.tar.gz"], self._commands())

    def test_empty_caller_ld_library_path(self):
        self.assertEqual(self._install(extra_env={"LD_LIBRARY_PATH": ""}), 0)
        path = self._hpcx_path(_HPCX_URL)
        configure = [c for c in self._commands() if c[0] == "./configure"][1]
        self.assertEqual(configure[1], f"LD_LIBRARY_PATH=:{path}/hcoll/lib")

    # ---- HPC-X rebuild details -------------------------------------------------
    def test_rebuild_entrypoints_are_fixed_with_sed(self):
        path = self._hpcx_path(_HPCX_URL)
        os.makedirs(f"{path}/modulefiles")
        Path(f"{path}/modulefiles/hpcx-rebuild").write_text("x")
        Path(f"{path}/hpcx-rebuild.sh").write_text("x")
        self.assertEqual(self._install(distro="ubuntu24.04"), 0)
        commands = self._commands()
        self.assertEqual(commands[4:7], [
            [*_SED_ARGS, f"{path}/modulefiles/hpcx-rebuild"],
            [*_SED_ARGS, f"{path}/hpcx-rebuild.sh"],
            self._cp_tests(_HPCX_URL)])
        self.assertIn(f"Fixed the hpcx-rebuild prefix in {path}/hpcx-rebuild.sh "
                      "(edited in place)", self._messages(self.info))

    def test_only_existing_rebuild_entrypoints_are_fixed(self):
        path = self._hpcx_path(_HPCX_URL)
        Path(path).mkdir(parents=True)
        Path(f"{path}/hpcx-rebuild.sh").write_text("x")
        self.assertEqual(self._install(distro="ubuntu24.04"), 0)
        self.assertEqual([c for c in self._commands() if c[0] == "sed"],
                         [[*_SED_ARGS, f"{path}/hpcx-rebuild.sh"]])

    def test_rebuild_failure_shows_log_tails(self):
        path = self._hpcx_path(_HPCX_URL)
        ucx_log = Path(f"{path}/sources/ucx-1.20.0/ucx-1.20.0-configure.log")
        ompi_log = Path(f"{path}/sources/openmpi-gitclone/ompi-make.log")
        for log in (ucx_log, ompi_log):
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("".join(f"{log.name} line {i}\n" for i in range(50)))
        os.makedirs(f"{path}/sources/ucx-1.20.0/ucx-dir.log")      # [[ -f ]] skips it
        rebuild = self._rebuild(_HPCX_URL, "--cuda", "--with-hcoll", "--ompi-extra-config",
                                f"{_VENDOR_OMPI} --with-pmix={self.opt}/pmix/4.2.9 {_PLATFORM}")
        self.assertEqual(self._install(distro="ubuntu24.04", fail_on=rebuild), 3)
        self.assertEqual(self._commands()[-1], rebuild)
        details = self.error_detail.call_args_list
        self.assertEqual([c.args[1] for c in details],
                         [f"===== tail of {ucx_log}", f"===== tail of {ompi_log}"])
        lines = details[0].args[2].split("\n")
        self.assertEqual(len(lines), 40)
        self.assertEqual((lines[0], lines[-1]), (f"{ucx_log.name} line 10",
                                                 f"{ucx_log.name} line 49"))
        self.assertIn("hpcx_rebuild.sh failed", self.error.call_args.args[1])
        self.assertEqual(self._captured(), [["nproc"]])

    def test_rebuild_failure_without_logs_warns(self):
        rebuild = self._rebuild(_HPCX_URL, "--cuda", "--with-hcoll", "--ompi-extra-config",
                                f"{_VENDOR_OMPI} --with-pmix={self.opt}/pmix/4.2.9 {_PLATFORM}")
        self.assertEqual(self._install(distro="ubuntu24.04", fail_on=rebuild), 3)
        self.error_detail.assert_not_called()
        self.assertTrue(any("no hpcx_rebuild.sh UCX/Open MPI logs" in m
                            for m in self._messages(self.warn)))

    def test_unusable_rebuilt_ucx_fails(self):
        self.assertEqual(self._install(ucx_info=(127, "")), 3)
        self.assertTrue(self._commands()[-1][0].endswith("hpcx_rebuild.sh"))
        ucx = f"{self._hpcx_path(_HPCX_URL)}/ucx/hpcx-rebuild"
        self.assertEqual(self.error.call_args.args[1],
                         f"Rebuilt HPC-X UCX at {ucx} is not usable.")
        self.assertEqual(self._records(), [("HPCX", "2.25.1")])

    def test_rebuilt_ompi_info_must_load_the_rebuilt_libmpi(self):
        path = self._hpcx_path(_HPCX_URL)
        cases = {
            "vendor libmpi": ((0, f"\tlibmpi.so.40 => {path}/ompi/lib/libmpi.so.40 (0x1)\n"),
                              f"{path}/ompi/lib/libmpi.so.40"),
            "not found": ((0, "\tlibmpi.so.40 => not found\n"), "not"),
            "ldd failed": ((1, ""), "<unresolved>"),
        }
        for name, (ldd, shown) in cases.items():
            with self.subTest(case=name):
                shutil.rmtree(self.modules, ignore_errors=True)
                self.assertEqual(self._install(distro="ubuntu24.04", ldd=ldd), 3)
                self.assertEqual(self.error.call_args.args[1],
                                 f"Rebuilt HPC-X ompi_info loads {shown} instead of the "
                                 "rebuilt libmpi.")
                self.assertEqual(self.error.call_args.kwargs["ldd_exit_code"], ldd[0])
                self.assertNotIn("cp", [c[0] for c in self._commands()])

    def test_internal_pmix_version_failures(self):
        for name, pkg_config in (("pkg-config fails", (1, "")), ("empty", (0, "\n"))):
            with self.subTest(case=name):
                shutil.rmtree(self.modules, ignore_errors=True)
                self.assertEqual(self._install(distro="azurelinux3.0", arch="aarch64",
                                               sku="GB200", family="gb-family",
                                               pkg_config=pkg_config), 3)
                self.assertIn("could not read the internal PMIx version",
                              self.error.call_args.args[1])
                self.assertEqual(self._records(), [("HPCX", "2.25.1")])
                self.assertEqual(len(self.calls), 3)        # stopped before fix_ladir

    # ---- UCX netlink patch -----------------------------------------------------
    def test_missing_ucx_source_tarball_fails(self):
        self.assertEqual(self._install(ucx_tarballs=()), 3)
        self.assertEqual(len(self.calls), 3)
        self.assertIn("HPC-X UCX source tarball not found under", self.error.call_args.args[1])

    def test_several_ucx_tarballs_patches_the_first_with_a_warning(self):
        self.assertEqual(self._install(ucx_tarballs=("ucx-1.20.0.tar.gz",
                                                     "ucx-1.21.0.tar.gz")), 0)
        self.assertTrue(self._commands()[3][2].endswith("/sources/ucx-1.20.0.tar.gz"))
        self.assertTrue(any("2 UCX source tarballs" in m for m in self._messages(self.warn)))

    def test_missing_netlink_source_fails(self):
        # bash's grep would fail and rebuild UCX unpatched; we stop.
        self.assertEqual(self._install(netlink=None), 3)
        self.assertEqual(self._commands()[-1][:2], ["tar", "-xzf"])
        self.assertIn("can't tell whether this UCX has the netlink", self.error.call_args.args[1])
        self._assert_tmp_cleaned()

    def test_missing_patch_file_fails(self):
        self.assertEqual(self._install(extra_env={"COMPONENT_DIR": ""}), 3)
        self.assertEqual(self._commands()[-1][:2], ["tar", "-xzf"])
        self.assertIn("UCX patch patches/ucx-netlink-dump-done.patch not found",
                      self.error.call_args.args[1])

    def test_patch_failure_stops_and_cleans_up(self):
        work_dir = os.path.join(self.patch_tmp, "tmp.ucx")
        patch = ["patch", "-d", f"{work_dir}/ucx-1.20.0", "-p1", "-i",
                 os.path.join(self.components, "patches", "ucx-netlink-dump-done.patch")]
        self.assertEqual(self._install(fail_on=patch), 3)
        self.assertEqual(self._commands()[-1], patch)
        self.assertIn("applying patches/ucx-netlink-dump-done.patch to ucx-1.20.0 failed",
                      self.error.call_args.args[1])
        self._assert_tmp_cleaned()

    # ---- reruns -------------------------------------------------------------
    def test_rerun_appends_with_warning_then_fails_at_first_symlink(self):
        mpi_dir = self._mpi_dir()
        os.makedirs(mpi_dir)
        Path(mpi_dir, "hpcx-2.25.1").write_text("old\n")
        os.symlink(os.path.join(mpi_dir, "hpcx-2.25.1"), os.path.join(mpi_dir, "hpcx-pmix-2.25.1"))

        self.assertEqual(self._install(distro="ubuntu24.04"), 3)
        # Appended (not replaced), as bash's `cat >>`.
        self.assertTrue(_read(os.path.join(mpi_dir, "hpcx-2.25.1")).startswith(
            "old\n#%Module 1.0\n"))
        rerun = [c for c in self.warn.call_args_list if "already exists" in c.args[1]]
        self.assertEqual(len(rerun), 1)
        self.assertEqual(rerun[0].kwargs["path"], os.path.join(mpi_dir, "hpcx-2.25.1"))
        self.assertIn("ln -s` (without -f) fails here too", self.error.call_args.args[1])
        # bash stops at that ln -s: no hpcx link, no MVAPICH.
        self.assertFalse(os.path.lexists(os.path.join(mpi_dir, "hpcx")))
        self.assertNotIn(["tar", "-xvf", "mvapich-4.1.tar.gz"], self._commands())
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
                self.assertFalse(os.path.exists(self._mpi_dir()))

    def test_missing_skipped_configs_are_not_needed(self):
        configs = dict(_CONFIGS, mvapich=None, impi=None)
        self.assertEqual(self._install(distro="ubuntu24.04", arch="aarch64", sku="GB200",
                                       family="gb-family", configs=configs), 0)
        configs = dict(_CONFIGS, pmix=None, mvapich=None, ompi=None)
        shutil.rmtree(self.modules)
        self.assertEqual(self._install(distro="ubuntu26.04", sku="H100", configs=configs), 0)

    def test_missing_module_dir_fails(self):
        self.assertEqual(self._install(extra_env={"MODULE_FILES_DIRECTORY": ""}), 3)
        self.config.assert_not_called()

    # ---- failures -------------------------------------------------------------
    def test_each_failing_command_stops_the_step(self):
        for distro, netlink in (("ubuntu24.04", _FIXED_NETLINK), ("ubuntu22.04", _BUGGY_NETLINK)):
            self._install(distro=distro, netlink=netlink)
            commands = self._commands()
            # A version recorded after n commands needs commands[:n] to succeed.
            recorded_after = list(self.record_points)
            shutil.rmtree(self.modules)
            for index, command in enumerate(commands):
                if command in commands[:index]:
                    continue    # repeated commands (make, fix_ladir) fail at the first one
                with self.subTest(distro=distro, command=command):
                    self.assertEqual(self._install(distro=distro, netlink=netlink,
                                                   fail_on=command), 3)
                    self.assertEqual(self._commands(), commands[:index + 1])
                    self.assertEqual([c.args[0] for c in self.record.call_args_list],
                                     [k for k, n in recorded_after if n <= index])
                    self.error.assert_called()
                    self._assert_tmp_cleaned()
                    # Modulefiles are only written after ldconfig.
                    if index <= commands.index(["ldconfig"]):
                        self.assertEqual(os.listdir(self._mpi_dir()), [])
                    shutil.rmtree(self.modules)

    def test_download_failure_returns_3(self):
        self.assertEqual(self._install(download_error=OSError("boom")), 3)
        self.assertEqual(self.calls, [])
        self.assertIn("download/verify of HPC-X failed", self.error.call_args.args[1])

    def test_pin_failure_returns_3(self):
        self.assertEqual(self._install(distro="rocky9.8", pin_ok=False), 3)
        self.assertEqual(self._commands()[-1], ["ldconfig"])     # stopped after HPC-X
        self.record.assert_called_once_with("HPCX", "2.25.1")
        self.assertEqual(os.listdir(self._mpi_dir()), [])        # pin comes before the modulefile

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
        self.assertEqual(self._install(distro="ubuntu24.04"), 3)
        self.assertEqual(self._commands()[-1], ["tar", "-xvf", "mvapich-4.1.tar.gz"])
        self.assertIn("was not created by tar", self.error.call_args.args[1])

    def test_ld_so_conf_write_error_returns_3(self):
        os.rmdir(self.ld_dir)
        self.assertEqual(self._install(), 3)
        self.assertNotIn(["ldconfig"], self._commands())
        self.assertIn("file operation failed", self.error.call_args.args[1])
        self._assert_tmp_cleaned()

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
