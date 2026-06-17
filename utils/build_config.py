from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


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
    if not args.os_name:
        raise ConfigError("--os is required", 1)

    # vendor / gpu / os validation 
    if args.vendor not in VALID_GPUS:
        raise ConfigError(f"unsupported vendor '{args.vendor}' (NVidia|AMD)", 1)
    if args.gpu not in VALID_GPUS[args.vendor]:
        raise ConfigError(f"GPU '{args.gpu}' not valid for vendor {args.vendor}", 1)
    if args.os_name not in DISTRO_DIRS:
        raise ConfigError(
            f"unsupported os '{args.os_name}' "
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
        os=args.os_name,
        fips=bool(args.fips),
        spec_path=spec_path,
    )

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
        install = build_dir / "install.sh"

        log_info("build-image", f"Building {cfg.os_name} for {cfg.gpu}")
        rc = exec_program([str(install), cfg.gpu_arg, cfg.gpu],
                          "build-image", cwd=str(build_dir))
        if rc != 0:
            log_error("build-image", f"build failed in {cfg.distro_dir}/install.sh")
            return 3
        log_info("build-image", "Image built successfully")
        return 0