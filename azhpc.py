"""Python implementation of the Azure HPC CLI tool."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

VERSION = "0.1.0"
BUILD_COMMIT = "dev"
BUILD_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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

def print_help():
    print("""azhpc - Build HPC/AI Linux images for Azure

USAGE:
    azhpc [OPTIONS]
    azhpc --help

OPTIONS:
    --spec <PATH>           Path to versions.json (optional)
    --vendor <VENDOR>       Hardware vendor: NVidia | AMD
    --gpu <SKU>             Target GPU. Values with dedicated build paths
                            are marked (*); others use the generic path.
                                NVidia : GB200* | GB300* | NCv6* | V100*
                                         A100 | H100 | H200 | VR200
                                AMD    : MI300 | MI400 | MI500
    --os <OS>               Target OS: Ubuntu24 | Ubuntu22 | Azure3 |
                                       Alma9 | Alma8 | Rocky9 | Rocky8
    --fips                  Build a FIPS-compliant image (default: non-FIPS)
    --dry-run               Validate args and print build plan; do not build
    -v, --verbose           Verbose (debug-level) logging
    --log-format <FORMAT>   Log format: text | json
                            (default: text if stdout is a TTY, else json)
    --version               Print version information and exit
    -h, --help              Show this help message and exit

EXIT STATUS:
    0  Image built successfully
    1  Invalid arguments or unsupported vendor/GPU combination
    2  Specification file missing or malformed
    3  Build failure""")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="azhpc", add_help=True,
                                description="Build HPC/AI Linux images for Azure")
    p.add_argument("--spec")
    p.add_argument("--vendor")
    p.add_argument("--gpu")
    p.add_argument("--os", dest="os_name")
    p.add_argument("--fips", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--log-format", choices=["text", "json"])
    p.add_argument("--version", action="store_true")
    return p

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # --version 
    if args.version:
        plat = f"{sys.platform}/{sys.implementation.name}"
        print(f"azhpc {VERSION} (python, {plat})")
        print(f"  commit:  {BUILD_COMMIT}")
        print(f"  built:   {BUILD_DATE}")
        return 0
    
    # configure logger before importing it since it reads env at import
    if args.log_format:
        os.environ["LOG_FORMAT"] = args.log_format
    if args.verbose:
        os.environ["LOG_LEVEL"] = "debug"
    os.environ["VERSION"] = VERSION
    os.environ["BUILD_COMMIT"] = BUILD_COMMIT
    os.environ["VENDOR"] = args.vendor or ""
    os.environ["GPU"] = args.gpu or ""
    os.environ["OS"] = args.os_name or ""
    os.environ["FIPS"] = "true" if args.fips else "false"

    from utils.logger import log_info, log_warn, log_debug, log_error

    def die(message: str, code: int = 1) -> int:
        log_error("cli", message)
        return code
    
    # required args
    if not args.vendor:
        return die("--vendor is required", 1)
    if not args.gpu:
        return die("--gpu is required", 1)
    if not args.os_name:
        return die("--os is required", 1)

    # vendor / gpu validation
    if args.vendor not in VALID_GPUS:
        return die(f"unsupported vendor '{args.vendor}' (NVidia|AMD)", 1)
    if args.gpu not in VALID_GPUS[args.vendor]:
        return die(f"GPU '{args.gpu}' not valid for vendor {args.vendor}", 1)

    # warn when the GPU has no dedicated build path
    if args.gpu not in DEDICATED_PATH_GPUS:
        log_warn("resolve-config",
                 f"GPU '{args.gpu}' has no dedicated build path; "
                 f"using generic {args.vendor} build")

    # os validation
    if args.os_name not in DISTRO_DIRS:
        return die(f"unsupported os '{args.os_name}' "
                   "(Ubuntu24|Ubuntu22|Azure3|Alma9|Alma8|Rocky9|Rocky8)", 1)
    
    # spec file validation
    effective_spec = "<azhpc-images default versions.json>"
    if args.spec:
        spec_path = Path(args.spec)
        if not spec_path.is_file():
            return die(f"spec file not found: {args.spec}", 2)
        try:
            json.loads(spec_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return die(f"spec file is malformed JSON: {args.spec}", 2)
        effective_spec = args.spec

    log_info("resolve-config",
             f"Effective build: vendor={args.vendor} gpu={args.gpu} "
             f"os={args.os_name} fips={args.fips} spec={effective_spec}")
    log_debug("resolve-config",
              f"log_format={os.environ.get('LOG_FORMAT')} dry_run={args.dry_run}")
    
    if args.dry_run:
        log_info("dry-run", "Dry-run requested; not executing build steps")
        return 0

    # translate flags into build inputs
    distro_dir = DISTRO_DIRS[args.os_name]
    gpu_arg = GPU_ARGS[args.vendor]
    repo_root = Path(__file__).resolve().parent
    install = repo_root / distro_dir / "install.sh"

    log_info("build-image", f"Invoking {distro_dir}/install.sh {gpu_arg} {args.gpu}")
    completed = subprocess.run([str(install), gpu_arg, args.gpu],
                               cwd=str(repo_root / distro_dir))
    if completed.returncode != 0:
        return die(f"build failed in {distro_dir}/install.sh", 3)

    log_info("build-image", "Image built successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())