"""ubuntu24_04.py — the build plan for distros/ubuntu24.04/install.sh.

Read it next to that install.sh: the steps are in the same order, with the
same conditions. (set_properties.sh's system changes and install_utils.sh
run first, in ImageBuilder's bootstrap.)

Any step marked resync="ADO <id>" has a Python port that is behind its bash
script and runs bash until that ticket updates the port (see Step.resync).
None are marked today: every port matches upstream 0c2b8f9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.plans import steps
from utils.sku import sku_has_infiniband

if TYPE_CHECKING:
    from utils.build_config import BuildConfig
    from utils.plans.step import Step

DISTRO_DIR = "distros/ubuntu24.04"


def nvidia(c):
    return c.vendor == "NVidia"


def amd(c):
    return c.vendor == "AMD"


def infiniband(c):
    # install.sh: [[ "$(sku_network_mode)" != "no_rdma" ]] (today: not NCv6)
    return sku_has_infiniband(c.gpu)


def no_rdma(c):
    return not sku_has_infiniband(c.gpu)


def build(cfg: BuildConfig) -> list[Step]:
    return [
        # DOCA-OFED on InfiniBand SKUs. No-RDMA SKUs (NCv6) instead get the
        # inline rdma-core install, then libfabric, then the mana_ib blacklist.
        steps.doca(when=infiniband),
        steps.rdma_core(when=no_rdma),
        steps.libfabric(when=no_rdma),
        steps.blacklist_mana_ib(when=no_rdma),

        # AMD: "Install ROCm before MPI so HPC-X can rebuild UCX with ROCm support."
        steps.rocm(when=amd),

        # NVIDIA: "Install CUDA before MPI so HPC-X can rebuild Open MPI with CUDA
        # support." One shared driver script for every NVIDIA SKU except NCv6,
        # which gets the GRID driver. GB200/GB300/VR200 differences (IMEX, CDMM,
        # no fabric manager) are inside install_nvidiagpudriver.sh, switched on by
        # the NVLINK_RACKSCALE env var that packer sets.
        steps.nv_driver(when=lambda c: nvidia(c) and c.gpu != "NCv6"),
        steps.nv_grid(when=lambda c: nvidia(c) and c.gpu == "NCv6"),

        steps.pmix(),
        steps.mpis(),

        # GB200 NVLink tools (need the driver and HPC-X)
        steps.nvshmem(when=lambda c: nvidia(c) and c.gpu == "GB200"),
        steps.nvloom(when=lambda c: nvidia(c) and c.gpu == "GB200"),
        steps.nvbandwidth(when=lambda c: nvidia(c) and c.gpu == "GB200"),

        steps.nccl(when=nvidia),
        steps.docker(),                       # every vendor, as in install.sh
        steps.dcgm(when=nvidia),
        steps.rccl(when=amd),                 # ROCm itself ran before MPI

        # Lustre client, then mpifileutils, which builds against it.
        steps.lustre(),
        steps.mpifileutils(),

        steps.amd_libs(when=lambda c: c.architecture == "x86_64"),
        steps.intel_libs(when=lambda c: c.architecture == "x86_64"),

        # (install_dynolog_drl.sh was removed upstream in #627.)
        steps.cleanup_downloads(DISTRO_DIR),

        steps.tuning(),
        steps.waagent(),
        steps.persistent_rdma_naming(),

        # if [[ "$SKU" != "GB200" ]]; then ... (NHC also skips NCv6)
        steps.aznfs(when=lambda c: c.gpu != "GB200"),
        steps.hpcdiag(when=lambda c: c.gpu != "GB200"),
        steps.monitoring_tools(when=lambda c: c.gpu != "GB200"),
        steps.health_checks(cfg.gpu_arg, when=lambda c: c.gpu not in {"GB200", "NCv6"}),

        steps.kernel_os_version(),
        steps.azsecpack_prereqs(),
        steps.udev_rules(),
        steps.copy_tests(),
        steps.cloudinit(),
        steps.sku_customizations(),
        steps.trivy(),
        # (disable_auto_upgrade.sh was removed upstream from Ubuntu 22.04 and
        # 24.04 in #623; Packer handles it now. Only Ubuntu 26.04 still runs it.)
        steps.predictive_interface_renaming(),
    ]
