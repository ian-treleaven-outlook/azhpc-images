"""Python implementation of the Azure HPC CLI tool."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from utils.build_config import resolve_config, ConfigError, ImageBuilder
from utils import logger

VERSION = "0.1.0"
BUILD_COMMIT = "dev"
BUILD_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def print_version():
    plat = f"{sys.platform}/{sys.implementation.name}"
    print(f"azhpc {VERSION} (python, {plat})")
    print(f"  commit:  {BUILD_COMMIT}")
    print(f"  built:   {BUILD_DATE}")

def print_help():
    print("""azhpc - Build HPC/AI Linux images for Azure

USAGE:
    azhpc --vendor <VENDOR> --gpu <SKU> --os <OS> [OPTIONS]
    azhpc install <PACKAGE>... [-v]
    azhpc --help

COMMANDS:
    install <PACKAGE>...    Install one or more packages using the detected
                            system package manager (apt-get, dnf, tdnf, ...)

REQUIRED:
    --vendor <VENDOR>       Hardware vendor: NVidia | AMD
    --gpu <SKU>             Target GPU. Values with dedicated build
                            paths are marked (*); others use the generic path.
                                NVidia : GB200* | GB300* | NCv6* | V100*
                                        A100 | H100 | H200 | VR200
                                AMD    : MI300 | MI400 | MI500
    --os <OS>               Target OS: Ubuntu24 | Ubuntu22 | Azure3 |
                                    Alma9 | Alma8 | Rocky9 | Rocky8

OPTIONS:
    --spec <PATH>           Path to versions.json
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

def configure_logging(args) -> None:
    logger.configure(
        log_format = args.log_format,
        log_level = "debug" if args.verbose else None,
        version = VERSION, commit = BUILD_COMMIT,
        vendor = args.vendor, gpu = args.gpu, os = args.os,
        fips = args.fips,
    )

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="azhpc", add_help=False,
                                description="Build HPC/AI Linux images for Azure")
    p.add_argument("-h", "--help", action="store_true")
    p.add_argument("--spec")
    p.add_argument("--vendor")
    p.add_argument("--gpu")
    p.add_argument("--os", dest="os")
    p.add_argument("--fips", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--log-format", choices=["text", "json"])
    p.add_argument("--version", action="store_true")
    return p

def cmd_install(args_list) -> int:
    """`azhpc install <pkg>... [-v]` — install packages via the detected manager."""
    verbose = False
    packages = []
    for a in args_list:
        if a in ("-v", "--verbose"):
            verbose = True
        else:
            packages.append(a)

    logger.configure(log_level="debug" if verbose else None,
                     version=VERSION, commit=BUILD_COMMIT)

    if not packages:
        logger.log_error("install", "no packages specified")
        return 1

    from utils.package_installer import PackageInstaller
    ok = PackageInstaller().install_package(packages)
    return 0 if ok else 3

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "install":
        return cmd_install(argv[1:])

    args = build_parser().parse_args(argv)
    if args.help:
        print_help()
        return 0
    if args.version:
        print_version()
        return 0

    configure_logging(args)         
    try:
        config = resolve_config(args)
    except ConfigError as e:
        logger.log_error("cli", str(e))
        return e.code

    logger.log_info("resolve-config", f"Effective build: {config}")
    if args.dry_run:
        logger.log_info("dry-run", "Dry-run requested; not building")
        return 0

    repo_root = Path(__file__).resolve().parent
    return ImageBuilder(repo_root, config).build()
    
if __name__ == "__main__":
    sys.exit(main())