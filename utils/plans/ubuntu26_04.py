"""ubuntu26_04.py — the build plan for distros/ubuntu26.04/install.sh.

Read it next to that install.sh: the steps are in the same order, with the
same conditions. (set_properties.sh's system changes and install_utils.sh
run first, in ImageBuilder's bootstrap.)

Ubuntu 26.04's install.sh differs from 24.04's: no PMIx step (HPC-X 2.51
brings PMIx 5), MPI before the NVIDIA driver, NCCL/RCCL before Docker, DOCA
on every SKU, no GB200-only steps or gating, and it still runs
disable_auto_upgrade.sh. It also refuses some SKUs outright (see
reject_sku()).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.plans import steps
from utils.plans.step import Step

if TYPE_CHECKING:
    from utils.build_config import BuildConfig

DISTRO_DIR = "distros/ubuntu26.04"


def reject_sku(sku: str) -> str:
    """install.sh's error for a SKU it refuses before doing anything, or "".

    DistroPlan.reject_sku: resolve_config() fails the build with it, before
    any system change, as install.sh's `exit 1`. The messages are install.sh's
    word for word (without the "##[error]" pipeline prefix).
    """
    if sku == "V100":
        return ("V100 is not supported on Ubuntu 26.04 because NVIDIA does not "
                "provide a compatible CUDA toolkit.")
    # "These SKUs need driver, architecture, or network paths that have not
    # been validated on Ubuntu 26.04 yet." VR200 is headed for 26.04 (it needs
    # kernel 7.0; upstream's gb200-ppa-kernel-7.0 branch). When install.sh
    # stops refusing it, test_plan_ubuntu26_04 fails until this list follows.
    if sku in ("GB200", "VR200", "NCv6"):
        return f"{sku} is not supported on Ubuntu 26.04 yet."
    return ""


def nvidia(c):
    return c.vendor == "NVidia"


def amd(c):
    return c.vendor == "AMD"


def build(cfg: BuildConfig) -> list[Step]:
    return [
        steps.doca(),
        steps.rocm(when=amd),

        # "HPC-X 2.51 supplies the Open MPI 5, PMIx 5, hwloc, and libevent
        # stack used on Ubuntu 26.04": no install_pmix.sh. MPI runs before the
        # NVIDIA driver; on NVIDIA, install_mpis.sh keeps HPC-X's own Open MPI
        # here instead of rebuilding it with CUDA.
        steps.mpis(),

        steps.nv_driver(when=nvidia),
        steps.nccl(when=nvidia),
        steps.rccl(when=amd),
        steps.docker(),                       # every vendor, as in install.sh
        steps.dcgm(when=nvidia),

        # Lustre client (it skips kernel 7.0 itself), then mpifileutils, which
        # builds against it when it is there.
        steps.lustre(),
        steps.mpifileutils(),

        steps.amd_libs(when=lambda c: c.architecture == "x86_64"),
        steps.intel_libs(when=lambda c: c.architecture == "x86_64"),

        steps.cleanup_downloads(DISTRO_DIR),

        steps.tuning(),
        steps.waagent(),
        steps.persistent_rdma_naming(),
        steps.aznfs(),
        steps.hpcdiag(),
        steps.monitoring_tools(),
        steps.health_checks(cfg.gpu_arg),
        steps.kernel_os_version(),
        steps.azsecpack_prereqs(),
        steps.udev_rules(),
        steps.copy_tests(),
        steps.cloudinit(),
        steps.sku_customizations(),
        steps.trivy(),
        steps.auto_upgrade(),
        steps.predictive_interface_renaming(),
    ]
