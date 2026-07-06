from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from pathlib import Path
import json
from utils.logger import log_info, log_warn, log_error
from utils.process import exec_program
from utils.package_installer import PackageInstaller
import os
import platform
import subprocess

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
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

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
    script: str = ""                                    # bash script in components/
    action: Callable[[dict[str, str]], int] | None = None  # OR a python action
    args: tuple[str, ...] = ()
    when: Callable[[BuildConfig], bool] = lambda cfg: True  # condition

# install_mpifileutils build dependencies, keyed by package manager.
# This replaces the if/elif/else distro branching that used to live in
# components/install_mpifileutils.sh — detection is handled by
# detect_package_manager(), so the only per-distro knowledge left is the
# package names themselves.
_MPIFILEUTILS_DEPS = {
    "apt-get": ["libbz2-dev", "libattr1-dev", "libarchive-dev", "libssl-dev", "libcap-dev"],
    "apt":     ["libbz2-dev", "libattr1-dev", "libarchive-dev", "libssl-dev", "libcap-dev"],
    "tdnf":    ["bzip2-devel", "libattr-devel", "libarchive-devel"],
    "yum":     ["bzip2-devel", "libattr-devel", "libarchive-devel"],
    "dnf":     ["bzip2-devel", "libattr-devel", "libarchive-devel"],
}

def install_mpifileutils_deps(env: dict[str, str]) -> int:
    """Install mpifileutils build dependencies via PackageInstaller.

    Replaces the package-install slice of install_mpifileutils.sh.
    """
    installer = PackageInstaller()
    if installer.manager is None:
        return 3
    deps = _MPIFILEUTILS_DEPS.get(installer.manager.name, [])
    return 0 if installer.install_package(deps) else 3

def build_plan(cfg: BuildConfig) -> list[Step]:
    """The ordered list of component steps, mirroring distros/<os>/install.sh."""
    return [
        Step("bootstrap", "../distros/<...>/install_utils.sh"),
        Step("install-cmake",   "install_cmake.sh",         when=lambda c: c.gpu != "GB200"),
        Step("install-lustre",  "install_lustre_client.sh"),
        Step("install-doca",    "install_doca.sh", when=lambda c: c.gpu != "NCv6"),         # TODO: gate on sku_has_infiniband (runtime)
        Step("install-pmix",    "install_pmix.sh"),
        Step("install-mpis",    "install_mpis.sh"),
        Step("install-mpifileutils-deps", action=install_mpifileutils_deps),
        Step("install-mpifileutils", "install_mpifileutils.sh"),

        # NVIDIA branch
        Step("install-nv-driver", "install_nvidiagpudriver.sh",
             when=lambda c: c.vendor == "NVidia" and c.gpu not in {"GB200", "NCv6"}),
        Step("install-nv-grid",   "install_nvidiagriddriver.sh",
             when=lambda c: c.vendor == "NVidia" and c.gpu == "NCv6"),
        Step("install-nccl",      "install_nccl.sh",   when=lambda c: c.vendor == "NVidia"),
        Step("install-docker",    "install_docker.sh", when=lambda c: c.vendor == "NVidia"),
        Step("install-dcgm",      "install_dcgm.sh",   when=lambda c: c.vendor == "NVidia"),

        # AMD branch
        Step("install-rocm", "install_rocm.sh", when=lambda c: c.vendor == "AMD"),
        Step("install-rccl", "install_rccl.sh", when=lambda c: c.vendor == "AMD"),
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
        if not cfg.has_dedicated_path:
            log_warn("resolve-config",
                     f"GPU '{cfg.gpu}' has no dedicated build path; "
                     f"using generic {cfg.vendor} build")

        build_dir = self.repo_root / cfg.distro_dir
        components = self.repo_root / "components"
        env = component_env(self.repo_root, cfg)

        log_info("build-image", f"Building {cfg.os} for {cfg.gpu}")

        rc = self._bootstrap(build_dir, env)
        if rc != 0:
            log_error("bootstrap", f"system prep failed with exit code {rc}")
            return 3

        for step in build_plan(cfg):
            if not step.when(cfg):
                log_info(step.op, "skipped")
                continue
            log_info(step.op, "starting")
            if step.action is not None:
                rc = step.action(env)
            else:
                rc = exec_program([str(components / step.script), *step.args],
                                  step.op, cwd=str(build_dir), env=env)
            if rc != 0:
                log_error(step.op, f"failed with exit code {rc}")
                return 3
            log_info(step.op, "done")

        log_info("build-image", "Image built successfully")
        return 0
        