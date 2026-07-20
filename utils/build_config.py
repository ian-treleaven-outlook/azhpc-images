from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from pathlib import Path
import json
from utils.logger import log_info, log_warn, log_error
from utils.process import exec_program
import os
import platform
import subprocess

from components.python import install_mpifileutils, install_cmake, install_libfabric, install_intel_libs, install_hpcdiag, install_monitoring_tools, install_nvbandwidth_tool, install_aznfs
from utils.sku import sku_has_infiniband

MODULE_DIRS = {
    "ubuntu": "/usr/share/modules/modulefiles",
}
DEFAULT_MODULE_DIR = "/usr/share/Modules/modulefiles"  # alma/rocky/azure/rhel

VALID_GPUS = {
    "NVidia": {"GB200", "GB300", "NCv6", "V100", "A100", "H100", "H200", "VR200"},
    "AMD": {"MI300", "MI400", "MI500"},
}
DEDICATED_PATH_GPUS = {"GB200", "GB300", "NCv6", "V100"}
DISTRO_DIRS = {
    "Ubuntu24": "distros/ubuntu24.04",
    "Ubuntu22": "distros/ubuntu22.04",
    "Alma9": "distros/almalinux9.7",
    "Alma8": "distros/almalinux8.10",
    "Rocky9": "distros/rocky9.7",
    "Rocky8": "distros/rocky8.10",
    "Azure3": "distros/azurelinux3.0",
}
GPU_ARGS = {"NVidia": "NVIDIA", "AMD": "AMD"}

def _detect_distribution() -> str:
    """Mirror: . /etc/os-release; echo $ID$VERSION_ID  -> e.g. 'ubuntu24.04'."""
    data: dict[str, str] = {}
    with open("/etc/os-release", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, _, val = line.partition("=")
                data[key] = val.strip().strip('"')
    return f"{data.get('ID', '')}{data.get('VERSION_ID', '')}"

def _detect_arch_distro(distribution: str) -> str:
    """Mirror: dpkg --print-architecture (deb) or rpm --eval %{_arch} (rpm)."""
    if "ubuntu" in distribution:
        cmd = ["dpkg", "--print-architecture"]
    else:
        cmd = ["rpm", "--eval", "%{_arch}"]
    return subprocess.run(cmd, capture_output
                          =True, text=True).stdout.strip()

def component_env(repo_root: Path, cfg: BuildConfig) -> dict[str, str]:
    """The env vars set_properties.sh exported that the component scripts read.

    Pure environment setup only — no apt/yum side effects (those stay in the
    bootstrap step for now).
    """
    distribution = _detect_distribution()

    env = os.environ.copy()
    env["TOP_DIR"]       = str(repo_root)
    env["COMPONENT_DIR"] = str(repo_root / "components")
    env["UTILS_DIR"]     = str(repo_root / "utils")
    env["TEST_DIR"]      = str(repo_root / "tests")

    env["DISTRIBUTION"]        = distribution
    env["ARCHITECTURE"]        = platform.machine()              # uname -m
    env["ARCHITECTURE_DISTRO"] = _detect_arch_distro(distribution)

    env["GPU"]        = cfg.gpu_arg                              # NVIDIA / AMD
    env["SKU"]        = cfg.gpu                                  # H100, MI300, ...
    env["SKU_FAMILY"] = "gb-family" if cfg.gpu in {"GB200", "GB300"} else cfg.gpu
    env["NODE_TYPE"]  = os.environ.get("NODE_TYPE", "azure-vm")

    env["MODULE_FILES_DIRECTORY"] = (
        MODULE_DIRS["ubuntu"] if "ubuntu" in distribution else DEFAULT_MODULE_DIR
    )

    versions = repo_root / "versions.json"
    if versions.is_file():
        env["COMPONENT_VERSIONS"] = versions.read_text(encoding="utf-8")

    return env

@dataclass(frozen=True)
class BuildConfig:
    vendor: str
    gpu: str
    os: str
    fips: bool
    spec_path: Path | None

    @property
    def distro_dir(self) -> str:
        return DISTRO_DIRS[self.os]
    
    @property
    def gpu_arg(self) -> str:
        return GPU_ARGS[self.vendor]
    
    @property
    def has_dedicated_path(self) -> bool:
        return self.gpu in DEDICATED_PATH_GPUS

    @property
    def architecture(self) -> str:
        """Target CPU architecture, derived from the SKU (Grace SKUs are ARM)."""
        return "aarch64" if self.gpu in {"GB200", "GB300"} else "x86_64"

class ConfigError(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code

def resolve_config(args) -> BuildConfig:
    # required args
    if not args.vendor:
        raise ConfigError("--vendor is required", 1)
    if not args.gpu:
        raise ConfigError("--gpu is required", 1)
    if not args.os:
        raise ConfigError("--os is required", 1)

    # vendor / gpu / os validation 
    if args.vendor not in VALID_GPUS:
        raise ConfigError(f"unsupported vendor '{args.vendor}' (NVidia|AMD)", 1)
    if args.gpu not in VALID_GPUS[args.vendor]:
        raise ConfigError(f"GPU '{args.gpu}' not valid for vendor {args.vendor}", 1)
    if args.os not in DISTRO_DIRS:
        raise ConfigError(
            f"unsupported os '{args.os}' "
            "(Ubuntu24|Ubuntu22|Azure3|Alma9|Alma8|Rocky9|Rocky8)", 1)

    # spec file validation
    spec_path: Path | None = None
    if args.spec:
        spec_path = Path(args.spec)
        if not spec_path.is_file():
            raise ConfigError(f"spec file not found: {args.spec}", 2)
        try:
            json.loads(spec_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            raise ConfigError(f"spec file is malformed JSON: {args.spec}", 2)

    # build and return the immutable config object
    return BuildConfig(
        vendor=args.vendor,
        gpu=args.gpu,
        os=args.os,
        fips=bool(args.fips),
        spec_path=spec_path,
    )

@dataclass(frozen=True)
class Step:
    op: str                       # log label, e.g. "install-rocm"
    script: str = ""              # canonical bash script name (parity + bash exec)
    action: Callable[[dict[str, str]], int] | None = None  # optional python action
    args: tuple[str, ...] = ()
    base: str = "component"       # "component" (components/) or "distro" (build_dir)
    when: Callable[[BuildConfig], bool] = lambda cfg: True  # condition

def _cleanup_downloads(env: dict[str, str]) -> int:
    """Free space by removing downloaded tarballs/installers and extracted dirs.

    Mirrors the inline cleanup in install.sh between install_dynolog_drl and
    hpc-tuning. Runs the exact bash rm in the distro build dir. Destructive by
    design; only ever runs during a real build() on the build VM.
    """
    build_dir = Path(env.get("TOP_DIR", ".")) / "distros" / env.get("DISTRIBUTION", "")
    cleanup = (
        "rm -rf *.tgz *.bz2 *.tbz *.tar.gz *.run *.deb *_offline.sh; "
        "rm -rf /tmp/MLNX_OFED_LINUX* /tmp/*conf*; "
        "rm -rf /var/intel/; "
        "rm -rf /var/cache/* || true; "
        "rm -Rf -- */"
    )
    return exec_program(["bash", "-c", cleanup], "cleanup-downloads",
                        cwd=str(build_dir), env=env)


def build_plan(cfg: BuildConfig) -> list[Step]:
    """The ordered list of component steps, mirroring distros/<os>/install.sh.

    Every step carries its canonical bash script name (for parity), plus an
    optional python `action` used to run it natively. Order and `when` gates
    follow distros/ubuntu24.04/install.sh exactly.

    (System prep / install_utils.sh runs separately via ImageBuilder._bootstrap
    before this plan executes. Inline shell in install.sh that isn't a component
    script — the non-IB rdma-core install + mana_ib blacklist, the AMD moby
    docker setup, and the tarball cleanup — is not represented here yet.)
    """
    def nvidia(c):
        return c.vendor == "NVidia"

    def amd(c):
        return c.vendor == "AMD"

    return [
        # update cmake (skip GB200)
        Step("install-cmake", "install_cmake.sh", action=install_cmake.install,
             when=lambda c: c.gpu != "GB200"),

        # Lustre clients
        Step("install-lustre", "install_lustre_client.sh"),

        # DOCA-OFED on InfiniBand SKUs; libfabric on non-IB (NCv6)
        Step("install-doca", "install_doca.sh",
             when=lambda c: sku_has_infiniband(c.gpu)),
        Step("install-libfabric", "install_libfabric.sh",
             action=install_libfabric.install,
             when=lambda c: not sku_has_infiniband(c.gpu)),

        # PMIX + MPI libraries + mpifileutils
        Step("install-pmix", "install_pmix.sh"),
        Step("install-mpis", "install_mpis.sh"),
        Step("install-mpifileutils", "install_mpifileutils.sh",
             action=install_mpifileutils.install),

        # --- NVIDIA driver branch (mutually exclusive by SKU) ---
        Step("install-nv-driver-gb200", "install_nvidiagpudriver_gb200.sh",
             base="distro", when=lambda c: nvidia(c) and c.gpu == "GB200"),
        Step("install-nvshmem", "install_nvshmem.sh",
             when=lambda c: nvidia(c) and c.gpu == "GB200"),
        Step("install-nvloom", "install_nvloom.sh",
             when=lambda c: nvidia(c) and c.gpu == "GB200"),
        Step("install-nvbandwidth", "install_nvbandwidth_tool.sh",
             action=install_nvbandwidth_tool.install,
             when=lambda c: nvidia(c) and c.gpu == "GB200"),
        Step("install-nv-grid", "install_nvidiagriddriver.sh",
             when=lambda c: nvidia(c) and c.gpu == "NCv6"),
        Step("install-nv-driver", "install_nvidiagpudriver.sh",
             when=lambda c: nvidia(c) and c.gpu not in {"GB200", "NCv6"}),

        # NCCL, docker, DCGM (NVIDIA)
        Step("install-nccl", "install_nccl.sh", when=nvidia),
        Step("install-docker", "install_docker.sh", when=nvidia),
        Step("install-dcgm", "install_dcgm.sh", when=nvidia),

        # --- AMD branch ---
        Step("install-rocm", "install_rocm.sh", when=amd),
        Step("install-rccl", "install_rccl.sh", when=amd),

        # --- x86_64 libraries ---
        Step("install-amd-libs", "install_amd_libs.sh",
             when=lambda c: c.architecture == "x86_64"),
        Step("install-intel-libs", "install_intel_libs.sh",
             action=install_intel_libs.install,
             when=lambda c: c.architecture == "x86_64"),

        # dynolog + dyno-relay-logger
        Step("install-dynolog-drl", "install_dynolog_drl.sh"),

        # free space: remove downloaded tarballs/installers (inline rm in install.sh)
        Step("cleanup-downloads", action=_cleanup_downloads),

        # optimizations + persistent RDMA naming
        Step("hpc-tuning", "hpc-tuning.sh"),
        Step("install-persistent-rdma-naming",
             "install_azure_persistent_rdma_naming.sh"),

        # not-GB200 group
        Step("install-aznfs", "install_aznfs.sh", action=install_aznfs.install,
             when=lambda c: c.gpu != "GB200"),
        Step("install-hpcdiag", "install_hpcdiag.sh",
             action=install_hpcdiag.install, when=lambda c: c.gpu != "GB200"),
        Step("install-monitoring-tools", "install_monitoring_tools.sh",
             action=install_monitoring_tools.install,
             when=lambda c: c.gpu != "GB200"),
        Step("install-health-checks", "install_health_checks.sh",
             args=(cfg.gpu_arg,),
             when=lambda c: c.gpu not in {"GB200", "NCv6"}),

        # final configuration steps
        Step("add-udev-rules", "add-udev-rules.sh"),
        Step("copy-test-file", "copy_test_file.sh"),
        Step("disable-cloudinit", "disable_cloudinit.sh"),
        Step("setup-sku-customizations", "setup_sku_customizations.sh"),
        Step("trivy-scan", "trivy_scan.sh"),
        Step("disable-auto-upgrade", "disable_auto_upgrade.sh", base="distro"),
        Step("disable-predictive-interface-renaming",
             "disable_predictive_interface_renaming.sh", base="distro"),
    ]

class ImageBuilder:
    def __init__(self, repo_root: Path, config: BuildConfig):
        self.repo_root = repo_root
        self.config = config

    def _bootstrap(self, build_dir: Path, env: dict[str, str]) -> int:
        """Run system prep that must happen before any component.

        Mirrors install_utils.sh (Microsoft repo, base packages, IB modules).
        Still bash for now — it's heavily distro-specific, side-effecting work.
        """
        install_utils = build_dir / "install_utils.sh"
        log_info("bootstrap", "Preparing system (install_utils.sh)")
        return exec_program([str(install_utils)], "bootstrap",
                            cwd=str(build_dir), env=env)

    def build(self) -> int:
        cfg = self.config

        # Warn (but continue) if this GPU uses the generic path rather than a
        # dedicated one (GB200/NCv6/... have special handling; A100 does not).
        if not cfg.has_dedicated_path:
            log_warn("resolve-config",
                     f"GPU '{cfg.gpu}' has no dedicated build path; "
                     f"using generic {cfg.vendor} build")

        # --- Phase 1: setup ---
        # build_dir  = the distro folder the bash scripts run from
        # components = where the component scripts live
        # env        = vars handed to every step (this loads versions.json)
        build_dir = self.repo_root / cfg.distro_dir
        components = self.repo_root / "components"
        env = component_env(self.repo_root, cfg)

        log_info("build-image", f"Building {cfg.os} for {cfg.gpu}")

        # --- Phase 2: bootstrap (system prep) ---
        # Runs install_utils.sh first; abort the build if it fails.
        rc = self._bootstrap(build_dir, env)
        if rc != 0:
            log_error("bootstrap", f"system prep failed with exit code {rc}")
            return 3

        # --- Phase 3: run each step of the plan, in order ---
        for step in build_plan(cfg):
            # (a) skip steps that don't apply to this target (e.g. rocm on NVIDIA)
            if not step.when(cfg):
                log_info(step.op, "skipped")
                continue

            log_info(step.op, "starting")

            # (b) run it: a python action, or the bash script via exec_program
            if step.action is not None:
                rc = step.action(env)
            else:
                # "distro" scripts live in build_dir; the rest in components/
                base_dir = build_dir if step.base == "distro" else components
                rc = exec_program([str(base_dir / step.script), *step.args],
                                  step.op, cwd=str(build_dir), env=env)

            # (c) stop the whole build on the first failure
            if rc != 0:
                log_error(step.op, f"failed with exit code {rc}")
                return 3
            log_info(step.op, "done")

        # --- Phase 4: every step passed ---
        log_info("build-image", "Image built successfully")
        return 0

        