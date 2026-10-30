"""rhel.py — the build plan for distros/rhel8.10/install.sh and
distros/rhel9.x/install.sh.

The two install.sh scripts are identical (a test pins that), so one plan
serves both. Read it next to either: the steps are in the same order.
(set_properties.sh's system changes and install_utils.sh run first, in
ImageBuilder's bootstrap.)

RHEL's install.sh is Rocky's (utils/plans/rocky.py) for NVIDIA only: it
refuses any other vendor up front (DistroPlan.vendors in PLANS), so it has
no ROCm or RCCL step and doesn't gate the NVIDIA ones.

What differs from Ubuntu 24.04:
  * NVIDIA only.
  * fix_setools_cyclecloud.sh runs right after install_utils.sh.
  * install.sh tests no SKU: DOCA and install_nvidiagpudriver.sh on NCv6
    too (no rdma-core or GRID path), no GB200 NVLink tools, and the
    AMD/Intel libraries, AZNFS, hpcdiag, monitoring tools and health checks
    on GB200 as well.
  * Lustre and mpifileutils come straight after MPI, before NCCL.
  * hpcdiag, AZNFS and the monitoring tools come before persistent RDMA
    naming, and copy_test_file.sh before the health checks.
  * No udev rules or interface-renaming step. ./network-config.sh runs near
    the end, then $UTILS_DIR/clear_history.sh (as bash; see
    steps.clear_history). Packer runs clear_history.sh again after the build.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.plans import steps

if TYPE_CHECKING:
    from utils.build_config import BuildConfig
    from utils.plans.step import Step

# install.sh's message for any vendor but NVIDIA, without its "Error: " prefix.
VENDOR_ERROR = "Only NVIDIA GPU support is implemented for RHEL."


def build(cfg: BuildConfig) -> list[Step]:
    return [
        # "Fix legacy SETools version lookup for CycleCloud compatibility"
        steps.setools_cyclecloud(),
        steps.doca(),

        # "Install CUDA before MPI so HPC-X can rebuild Open MPI with CUDA support."
        # install.sh passes "$SKU"; the script doesn't read it (it uses the
        # exported SKU), but a step that runs as bash passes it the same way.
        steps.nv_driver(args=(cfg.gpu,)),

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

        # Either folder: rhel8.10 or rhel9.x.
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
        # "clear history (cleanup logs, caches, and build artifacts)"
        steps.clear_history(),
    ]
