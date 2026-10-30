"""almalinux.py — the build plan for distros/almalinux8.10/install.sh and
distros/almalinux9.x/install.sh.

The two install.sh scripts are the same except that AlmaLinux 9's ends with
`dnf update -y` (a test pins that), so one plan serves both. Read it next to
either install.sh: the steps are in the same order. (set_properties.sh's
system changes and install_utils.sh run first, in ImageBuilder's bootstrap.)

What differs from Ubuntu 24.04:
  * NVIDIA only: install.sh exits on AMD (DistroPlan.vendors in PLANS).
  * fix_setools_cyclecloud.sh runs right after install_utils.sh.
  * install.sh tests no SKU, so every SKU runs every step: DOCA and
    install_nvidiagpudriver.sh on NCv6 too (no rdma-core or GRID path), no
    GB200 NVLink tools, and the AMD/Intel libraries, AZNFS, hpcdiag,
    monitoring tools and health checks on GB200 as well.
  * Lustre and mpifileutils come straight after MPI, before NCCL.
  * hpcdiag, AZNFS and the monitoring tools come before persistent RDMA
    naming, and copy_test_file.sh before the health checks.
  * No udev rules or interface-renaming step. ./network-config.sh runs near
    the end, then AlmaLinux 9 upgrades every package.
  * clear_history.sh is commented out in install.sh, so it isn't run here
    (Packer runs it after the build).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.plans import steps

if TYPE_CHECKING:
    from utils.build_config import BuildConfig
    from utils.plans.step import Step

# install.sh's message for AMD, without its "Error: " prefix.
VENDOR_ERROR = "AMD GPU support is not implemented yet for AlmaLinux."


def almalinux9(c):
    return c.os == "Alma9"


def build(cfg: BuildConfig) -> list[Step]:
    return [
        # "Fix legacy SETools version lookup for CycleCloud compatibility"
        steps.setools_cyclecloud(),
        steps.doca(),
        # "Install CUDA before MPI so HPC-X can rebuild Open MPI with CUDA support."
        steps.nv_driver(),

        steps.pmix(),
        steps.mpis(),
        # Lustre "must run after install_doca + install_mpis so the
        # build-from-source path can use /usr/src/ofa_kernel/default and HPC-X";
        # mpifileutils builds against it.
        steps.lustre(),
        steps.mpifileutils(),

        steps.nccl(),
        steps.docker(),
        steps.dcgm(),
        steps.amd_libs(),
        steps.intel_libs(),

        # Either folder: almalinux8.10 or almalinux9.x.
        steps.cleanup_downloads(cfg.distro_dir),

        steps.tuning(),
        steps.waagent(),
        steps.hpcdiag(),
        steps.aznfs(),
        steps.monitoring_tools(),
        steps.persistent_rdma_naming(),
        steps.copy_tests(),
        steps.health_checks(cfg.gpu_arg),
        steps.kernel_os_version(),
        steps.azsecpack_prereqs(),
        steps.cloudinit(),
        steps.sku_customizations(),
        steps.trivy(),
        steps.rhel_family_network_config(),

        # Only almalinux9.x/install.sh ends with `dnf update -y`.
        steps.system_upgrade(when=almalinux9),
    ]
