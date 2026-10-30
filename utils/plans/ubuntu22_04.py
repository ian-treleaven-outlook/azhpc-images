"""ubuntu22_04.py — the build plan for distros/ubuntu22.04/install.sh.

Read it next to that install.sh: the steps are in the same order, with the
same conditions. (set_properties.sh's system changes, then pre_bootstrap()'s
steps, then install_utils.sh run first, in ImageBuilder's bootstrap.)

Unlike Ubuntu 24.04, this install.sh has no per-SKU gating: every NVIDIA SKU
gets DOCA-OFED and the shared driver script (no no-RDMA path, no GRID driver,
no GB200 NVLink tools), the AMD and Intel libraries are not limited to
x86_64, and every SKU gets AZNFS, hpcdiag, Moneo and the health checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.plans import steps

if TYPE_CHECKING:
    from utils.build_config import BuildConfig
    from utils.plans.step import Step

DISTRO_DIR = "distros/ubuntu22.04"


def nvidia(c):
    return c.vendor == "NVidia"


def amd(c):
    return c.vendor == "AMD"


def pre_bootstrap(cfg: BuildConfig) -> list[Step]:
    return [
        # "remove packages requiring Ubuntu Pro for security updates"
        steps.unused_packages(),
    ]


def build(cfg: BuildConfig) -> list[Step]:
    return [
        steps.doca(),

        # AMD: "Install ROCm before MPI so HPC-X can rebuild UCX with ROCm support."
        steps.rocm(when=amd),
        # NVIDIA: "Install CUDA before MPI so HPC-X can rebuild Open MPI with CUDA support."
        steps.nv_driver(when=nvidia),

        steps.pmix(),
        steps.mpis(),

        steps.nccl(when=nvidia),
        steps.docker(),                       # every vendor, as in install.sh
        steps.dcgm(when=nvidia),
        steps.rccl(when=amd),                 # ROCm itself ran before MPI

        # Lustre client, then mpifileutils, which builds against it.
        steps.lustre(),
        steps.mpifileutils(),

        steps.amd_libs(),
        steps.intel_libs(),

        steps.cleanup_downloads(DISTRO_DIR),

        steps.tuning(),
        steps.waagent(),
        steps.aznfs(),
        steps.hpcdiag(),
        steps.monitoring_tools(),
        steps.persistent_rdma_naming(),
        steps.udev_rules(),
        steps.copy_tests(),
        steps.health_checks(cfg.gpu_arg),
        steps.kernel_os_version(),
        steps.azsecpack_prereqs(),
        steps.cloudinit(),
        steps.sku_customizations(),
        steps.trivy(),
        steps.predictive_interface_renaming(),
    ]
