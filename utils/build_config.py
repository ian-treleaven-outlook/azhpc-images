from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from pathlib import Path
import json
from utils.logger import log_info, log_warn, log_error
from utils.process import exec_program

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
    script: str                   # filename in components/
    args: tuple[str, ...] = ()
    when: Callable[[BuildConfig], bool] = lambda cfg: True  # condition

def build_plan(cfg: BuildConfig) -> list[Step]:
    """The ordered list of component steps, mirroring distros/<os>/install.sh."""
    return [
        Step("install-cmake",   "install_cmake.sh",         when=lambda c: c.gpu != "GB200"),
        Step("install-lustre",  "install_lustre_client.sh"),
        Step("install-doca",    "install_doca.sh"),         # TODO: gate on sku_has_infiniband (runtime)
        Step("install-pmix",    "install_pmix.sh"),
        Step("install-mpis",    "install_mpis.sh"),
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
    
    def build(self) -> int:
        cfg = self.config
        if not cfg.has_dedicated_path:
            log_warn("resolve-config",
                    f"GPU '{cfg.gpu}' has no dedicated build path; "
                    f"using generic {cfg.vendor} build")

        build_dir = self.repo_root / cfg.distro_dir
        components = self.repo_root / "components"

        log_info("build-image", f"Building {cfg.os} for {cfg.gpu}")
        for step in build_plan(cfg):
            if not step.when(cfg):
                log_info(step.op, "skipped")
                continue
            log_info(step.op, "starting")
            rc = exec_program([str(components / step.script), *step.args],
                            step.op, cwd=str(build_dir))
            if rc != 0:
                log_error(step.op, f"failed with exit code {rc}")
                return 3
            log_info(step.op, "done")

        log_info("build-image", "Image built successfully")
        return 0
        