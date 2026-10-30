"""azurelinux3_0.py — the build plan for distros/azurelinux3.0/install.sh.

Read it next to that install.sh: the steps are in the same order, with the
same conditions. (set_properties.sh's system changes and install_utils.sh
run first, in ImageBuilder's bootstrap.)

What differs from Ubuntu 24.04: MOFED instead of DOCA, on every SKU; no
Lustre client; every NVIDIA SKU gets install_nvidiagpudriver.sh (no GRID
branch); the GB200 NVLink tools come after NCCL; hpcdiag, the monitoring
tools and persistent RDMA naming run on every SKU, after AzSecPack's
prerequisites; no cloud-init or interface-renaming steps; and
./network-config.sh runs last.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.plans import steps

if TYPE_CHECKING:
    from utils.build_config import BuildConfig
    from utils.plans.step import Step

DISTRO_DIR = "distros/azurelinux3.0"


def nvidia(c):
    return c.vendor == "NVidia"


def amd(c):
    return c.vendor == "AMD"


def gb200(c):
    return nvidia(c) and c.gpu == "GB200"


def x86_64(c):
    return c.architecture == "x86_64"


def build(cfg: BuildConfig) -> list[Step]:
    return [
        steps.mofed(),

        # AMD: "Install ROCm before MPI so HPC-X can rebuild UCX with ROCm support."
        steps.rocm(when=amd),
        # NVIDIA: "Install CUDA before MPI so HPC-X can rebuild Open MPI with CUDA
        # support." The one driver script for every NVIDIA SKU.
        steps.nv_driver(when=nvidia),

        steps.pmix(),
        steps.mpis(),
        steps.mpifileutils(),

        steps.nccl(when=nvidia),
        steps.nvshmem(when=gb200),
        steps.nvloom(when=gb200),
        steps.nvbandwidth(when=gb200),

        steps.docker(),                       # every vendor, as in install.sh
        steps.dcgm(when=nvidia),
        steps.rccl(when=amd),

        steps.amd_libs(when=x86_64),
        steps.intel_libs(when=x86_64),

        steps.cleanup_downloads(DISTRO_DIR),

        steps.tuning(),
        steps.waagent(),

        # if [[ "$SKU" != "GB200" ]]; then ...
        steps.aznfs(when=lambda c: c.gpu != "GB200"),
        steps.health_checks(cfg.gpu_arg, when=lambda c: c.gpu != "GB200"),

        steps.kernel_os_version(),
        steps.azsecpack_prereqs(),
        steps.hpcdiag(),
        steps.monitoring_tools(),
        steps.persistent_rdma_naming(),
        steps.udev_rules(),
        steps.copy_tests(),
        steps.sku_customizations(),
        steps.trivy(),
        steps.azurelinux_network_config(),
    ]
