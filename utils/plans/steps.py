"""steps.py — the shared step library every distro plan is built from.

Each function returns the Step for one line of an install.sh: the log label
(`op`), the bash script it mirrors, and the Python action that runs it. A
distro plan (e.g. utils/plans/ubuntu24_04.py) lists them in its install.sh's
order and adds that distro's `when=` gating:

    steps.doca(when=infiniband),
    steps.rocm(when=amd),

Keyword arguments are passed on to Step, so a plan can also set `args=`,
`skip_warning=` or `resync=`. Defining each step once here means the action
wiring and log labels are the same on every distro.
"""

from __future__ import annotations

import functools
from pathlib import Path

from components.python import (
    add_udev_rules, copy_test_file, disable_auto_upgrade, disable_cloudinit,
    disable_predictive_interface_renaming, hpc_tuning, install_amd_libs,
    install_azsecpack_prereqs, install_azure_persistent_rdma_naming, install_aznfs,
    install_dcgm, install_doca, install_docker, install_health_checks, install_hpcdiag,
    install_intel_libs, install_libfabric, install_lustre_client, install_mofed,
    install_monitoring_tools, install_mpifileutils, install_mpis, install_nccl,
    install_nvbandwidth_tool, install_nvidiagpudriver, install_nvidiagriddriver,
    install_nvloom, install_nvshmem, install_pmix, install_rccl, install_rocm,
    install_waagent, no_rdma_network, remove_unused_packages, setup_sku_customizations,
    trivy_scan, write_kernel_os_version,
)
from components.python import azurelinux_network_config as azl_network_config
from components.python import fix_setools_cyclecloud
# Imported under a short name: the step function below is called
# rhel_family_network_config, and would hide the module's own name.
from components.python import rhel_family_network_config as rhel_net_config
from utils.logger import log_info
from utils.package_installer import upgrade_all_packages
from utils.plans.step import Step
from utils.process import exec_program


# --- networking --------------------------------------------------------------
def doca(**kw):
    return Step("install-doca", "install_doca.sh", action=install_doca.install, **kw)

def rdma_core(**kw):
    """Inline in install.sh (no-RDMA SKUs): apt-get install rdma-core ..."""
    return Step("install-rdma-core", action=no_rdma_network.install_rdma_core, **kw)

def libfabric(**kw):
    return Step("install-libfabric", "install_libfabric.sh",
                action=install_libfabric.install, **kw)

def blacklist_mana_ib(**kw):
    """Inline in install.sh (no-RDMA SKUs): the mana_ib modprobe blacklist."""
    return Step("blacklist-mana-ib", action=no_rdma_network.blacklist_mana_ib, **kw)


# --- GPU drivers and runtimes -----------------------------------------------
def rocm(**kw):
    return Step("install-rocm", "install_rocm.sh", action=install_rocm.install, **kw)

def nv_driver(**kw):
    return Step("install-nv-driver", "install_nvidiagpudriver.sh",
                action=install_nvidiagpudriver.install, **kw)

def nv_grid(**kw):
    return Step("install-nv-grid", "install_nvidiagriddriver.sh",
                action=install_nvidiagriddriver.install, **kw)


# --- MPI -----------------------------------------------------------------------
def pmix(**kw):
    return Step("install-pmix", "install_pmix.sh", action=install_pmix.install, **kw)

def mpis(**kw):
    return Step("install-mpis", "install_mpis.sh", action=install_mpis.install, **kw)


# --- NVLink tools, collectives, containers ------------------------------------
def nvshmem(**kw):
    return Step("install-nvshmem", "install_nvshmem.sh", action=install_nvshmem.install, **kw)

def nvloom(**kw):
    return Step("install-nvloom", "install_nvloom.sh", action=install_nvloom.install, **kw)

def nvbandwidth(**kw):
    return Step("install-nvbandwidth", "install_nvbandwidth_tool.sh",
                action=install_nvbandwidth_tool.install, **kw)

def nccl(**kw):
    return Step("install-nccl", "install_nccl.sh", action=install_nccl.install, **kw)

def docker(**kw):
    return Step("install-docker", "install_docker.sh", action=install_docker.install, **kw)

def dcgm(**kw):
    return Step("install-dcgm", "install_dcgm.sh", action=install_dcgm.install, **kw)

def rccl(**kw):
    return Step("install-rccl", "install_rccl.sh", action=install_rccl.install, **kw)


# --- storage and libraries ---------------------------------------------------
def lustre(**kw):
    return Step("install-lustre", "install_lustre_client.sh",
                action=install_lustre_client.install, **kw)

def mpifileutils(**kw):
    return Step("install-mpifileutils", "install_mpifileutils.sh",
                action=install_mpifileutils.install, **kw)

def amd_libs(**kw):
    return Step("install-amd-libs", "install_amd_libs.sh",
                action=install_amd_libs.install, **kw)

def intel_libs(**kw):
    return Step("install-intel-libs", "install_intel_libs.sh",
                action=install_intel_libs.install, **kw)


# --- cleanup (inline in every install.sh) ------------------------------------
# install.sh's inline cleanup, word for word. It is the same in every distro's
# install.sh (a test checks each plan's install.sh contains it).
CLEANUP_SCRIPT = """\
rm -rf *.tgz *.bz2 *.tbz *.tar.gz *.run *.deb *_offline.sh
rm -rf /tmp/MLNX_OFED_LINUX* /tmp/*conf*
rm -rf /var/intel/
(
    shopt -s dotglob nullglob
    rm -rf -- /var/cache/* || true
    rm -Rf -- */ || true
)
"""


def cleanup_downloads_action(env: dict[str, str], distro_dir: str) -> int:
    """Free space by removing downloaded tarballs/installers and extracted dirs.

    Runs install.sh's inline cleanup (CLEANUP_SCRIPT) word for word, with
    `bash -e` like install.sh's `set -e`, in the distro folder install.sh runs
    from (`distro_dir`, e.g. "distros/ubuntu24.04"). The subshell's `dotglob
    nullglob` makes the globs include hidden entries and expand to nothing
    when nothing matches; its `|| true` keeps those two removals from failing
    the build. Destructive by design; only ever runs during a real build on
    the build VM.
    """
    build_dir = Path(env.get("TOP_DIR", ".")) / distro_dir
    log_info("cleanup-downloads", f"Removing downloads and extracted folders in {build_dir}",
             build_dir=str(build_dir))
    return exec_program(["bash", "-ec", CLEANUP_SCRIPT], "cleanup-downloads",
                        cwd=str(build_dir), env=env)


def cleanup_downloads(distro_dir: str, **kw):
    # functools.partial fills in distro_dir now, so the action still takes
    # just `env`, like every other action.
    return Step("cleanup-downloads",
                action=functools.partial(cleanup_downloads_action, distro_dir=distro_dir),
                **kw)


# --- tuning, agents, monitoring ----------------------------------------------
def tuning(**kw):
    return Step("hpc-tuning", "hpc-tuning.sh", action=hpc_tuning.install, **kw)

def waagent(**kw):
    return Step("install-waagent", "install_waagent.sh", action=install_waagent.install, **kw)

def persistent_rdma_naming(**kw):
    return Step("install-persistent-rdma-naming", "install_azure_persistent_rdma_naming.sh",
                action=install_azure_persistent_rdma_naming.install, **kw)

def aznfs(**kw):
    return Step("install-aznfs", "install_aznfs.sh", action=install_aznfs.install, **kw)

def hpcdiag(**kw):
    return Step("install-hpcdiag", "install_hpcdiag.sh", action=install_hpcdiag.install, **kw)

def monitoring_tools(**kw):
    return Step("install-monitoring-tools", "install_monitoring_tools.sh",
                action=install_monitoring_tools.install, **kw)

def health_checks(gpu_arg: str, **kw):
    """install_health_checks.sh "$GPU" (NVIDIA or AMD)."""
    return Step("install-health-checks", "install_health_checks.sh",
                action=install_health_checks.install, args=(gpu_arg,), **kw)


# --- final configuration ------------------------------------------------------
def kernel_os_version(**kw):
    return Step("write-kernel-os-version", "write_kernel_os_version.sh",
                action=write_kernel_os_version.install, **kw)

def azsecpack_prereqs(**kw):
    return Step("install-azsecpack-prereqs", "install_azsecpack_prereqs.sh",
                action=install_azsecpack_prereqs.install, **kw)

def udev_rules(**kw):
    return Step("add-udev-rules", "add-udev-rules.sh", action=add_udev_rules.install, **kw)

def copy_tests(**kw):
    return Step("copy-test-file", "copy_test_file.sh", action=copy_test_file.install, **kw)

def cloudinit(**kw):
    return Step("disable-cloudinit", "disable_cloudinit.sh",
                action=disable_cloudinit.install, **kw)

def sku_customizations(**kw):
    return Step("setup-sku-customizations", "setup_sku_customizations.sh",
                action=setup_sku_customizations.install, **kw)

def trivy(**kw):
    return Step("trivy-scan", "trivy_scan.sh", action=trivy_scan.install, **kw)


# --- whole-system steps at the end of some install.sh scripts -----------------
def system_upgrade_action(env: dict[str, str]) -> int:
    """Upgrade every installed package. Returns 0, or 3 on failure."""
    return 0 if upgrade_all_packages(env, "upgrade-all-packages") else 3


def system_upgrade(**kw):
    """Inline in install.sh: `dnf update -y` (today only AlmaLinux 9's, near the end)."""
    return Step("upgrade-all-packages", action=system_upgrade_action, **kw)


def clear_history(**kw):
    """$UTILS_DIR/clear_history.sh, the last line of Rocky's and RHEL's install.sh.

    It stays bash, on purpose. It is the image's deprovisioning (245 lines:
    remove Defender, clean logs, caches, cloud-init state and shell history,
    zero free disk space), and Packer's "Clear history and deprovision" step
    runs the same script again after the build and its tests. Porting it means
    porting both runs, a separate task.

    Note that it truncates every *.log under /var/log, including this CLI's own
    log file (/var/log/azhpc by default), so that file keeps only the lines
    logged after this step; the console output Packer records keeps them all.
    """
    return Step("clear-history", "clear_history.sh", base="utils", **kw)


# --- scripts in the distro folder (./name.sh) ---------------------------------
def auto_upgrade(**kw):
    """./disable_auto_upgrade.sh (only Ubuntu 26.04's install.sh runs it now)."""
    return Step("disable-auto-upgrade", "disable_auto_upgrade.sh", base="distro",
                action=disable_auto_upgrade.install, **kw)

def predictive_interface_renaming(**kw):
    """./disable_predictive_interface_renaming.sh (Ubuntu distro folders only)."""
    return Step("disable-predictive-interface-renaming",
                "disable_predictive_interface_renaming.sh", base="distro",
                action=disable_predictive_interface_renaming.install, **kw)

def unused_packages(**kw):
    """./remove_unused_packages.sh (Ubuntu 22.04 only; runs before install_utils.sh)."""
    return Step("remove-unused-packages", "remove_unused_packages.sh", base="distro",
                action=remove_unused_packages.install, **kw)


# --- Azure Linux ---------------------------------------------------------------
def mofed(**kw):
    """Azure Linux's InfiniBand stack (MOFED), in place of DOCA."""
    return Step("install-mofed", "install_mofed.sh", action=install_mofed.install, **kw)

def azurelinux_network_config(**kw):
    """./network-config.sh in distros/azurelinux3.0 (the RHEL family's differs)."""
    return Step("network-config", "network-config.sh", base="distro",
                action=azl_network_config.install, **kw)


# --- AlmaLinux, Rocky Linux and RHEL -------------------------------------------
def setools_cyclecloud(**kw):
    """fix_setools_cyclecloud.sh, right after install_utils.sh."""
    return Step("fix-setools-cyclecloud", "fix_setools_cyclecloud.sh",
                action=fix_setools_cyclecloud.install, **kw)

def rhel_family_network_config(**kw):
    """./network-config.sh (the same script in all six AlmaLinux, Rocky and RHEL folders)."""
    return Step("network-config", "network-config.sh", base="distro",
                action=rhel_net_config.install, **kw)
