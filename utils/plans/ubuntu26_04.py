"""ubuntu26_04.py — the build plan for distros/ubuntu26.04/install.sh.

Read it next to that install.sh: the steps are in the same order, with the
same conditions. (set_properties.sh's system changes and install_utils.sh
run first, in ImageBuilder's bootstrap.)

Ubuntu 26.04's install.sh differs from 24.04's: no PMIx step (HPC-X 2.51
brings PMIx 5), MPI before the NVIDIA driver, NCCL/RCCL before Docker, DOCA
on every SKU, no GB200-only steps or gating, and it still runs
disable_auto_upgrade.sh. It also refuses some SKUs outright (see
unsupported_sku_error()).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.logger import log_error
from utils.plans import steps
from utils.plans.step import Step

if TYPE_CHECKING:
    from utils.build_config import BuildConfig

DISTRO_DIR = "distros/ubuntu26.04"

# install.sh's messages for the SKUs it refuses, word for word (without the
# "##[error]" pipeline prefix; log_error marks the line as an error).
_V100_ERROR = ("V100 is not supported on Ubuntu 26.04 because NVIDIA does not provide "
               "a compatible CUDA toolkit.")
# "These SKUs need driver, architecture, or network paths that have not been
# validated on Ubuntu 26.04 yet."
_NOT_YET_SUPPORTED = ("GB200", "VR200", "NCv6")
UNSUPPORTED_SKUS = frozenset({"V100", *_NOT_YET_SUPPORTED})


def unsupported_sku_error(sku: str) -> str | None:
    """install.sh's error for a SKU it refuses to build, or None."""
    if sku == "V100":
        return _V100_ERROR
    if sku in _NOT_YET_SUPPORTED:
        return f"{sku} is not supported on Ubuntu 26.04 yet."
    return None


def _reject_unsupported_sku(env: dict[str, str]) -> int:
    """Fail the build, as install.sh's `exit 1` before anything else."""
    sku = env.get("SKU", "")
    log_error("reject-unsupported-sku", unsupported_sku_error(sku) or
              f"{sku} is not supported on Ubuntu 26.04", sku=sku)
    return 3


def nvidia(c):
    return c.vendor == "NVidia"


def amd(c):
    return c.vendor == "AMD"


def unsupported(c):
    return c.gpu in UNSUPPORTED_SKUS


def pre_bootstrap(cfg: BuildConfig) -> list[Step]:
    # TODO(toolkit): DistroPlan should list the SKUs install.sh rejects, and
    # resolve_config() refuse them like a rejected vendor. As a step, the
    # rejection comes after set_properties.sh's system changes (apt upgrade,
    # jq), which bash never reaches for these SKUs. VR200 arrives here as
    # GB200 (SKU_ALIASES), so it gets GB200's message.
    return [
        Step("reject-unsupported-sku", action=_reject_unsupported_sku, when=unsupported),
    ]


def build(cfg: BuildConfig) -> list[Step]:
    if unsupported(cfg):
        return []           # install.sh has already exited (see pre_bootstrap)
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
