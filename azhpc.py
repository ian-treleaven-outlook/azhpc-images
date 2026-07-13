"""Python implementation of the Azure HPC CLI tool."""

from __future__ import annotations

import argparse
import sys
import json
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
    versions                Print the resolved component versions that would be
                            installed for a target (--os/--vendor/--gpu). Does
                            not build anything.
    validate                Check versions.json for internal consistency for a
                            target (e.g. CUDA-major agreement). Exit 0 if clean,
                            1 if issues are found.

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

def cmd_versions(args_list) -> int:
    """`azhpc versions --os <OS> --vendor <V> --gpu <SKU>` — print the resolved
    component versions for a target without building anything."""
    from utils.build_config import DISTRO_DIRS, GPU_ARGS, VALID_GPUS
    from utils.version_report import resolve_report, format_report

    p = argparse.ArgumentParser(prog="azhpc versions")
    p.add_argument("--os", required=True, choices=sorted(DISTRO_DIRS))
    p.add_argument("--vendor", required=True, choices=sorted(GPU_ARGS))
    p.add_argument("--gpu", required=True)
    p.add_argument("--arch", default="x86_64")
    p.add_argument("--node-type", dest="node_type", default="azure-vm")
    p.add_argument("--spec", default="versions.json")
    ns = p.parse_args(args_list)

    if ns.gpu not in VALID_GPUS[ns.vendor]:
        print(f"error: GPU '{ns.gpu}' not valid for vendor {ns.vendor}",
              file=sys.stderr)
        return 1

    spec = Path(ns.spec)
    if not spec.is_file():
        print(f"error: spec file not found: {ns.spec}", file=sys.stderr)
        return 2
    versions = json.loads(spec.read_text(encoding="utf-8"))

    distribution = DISTRO_DIRS[ns.os].split("/")[-1]
    report = resolve_report(
        versions, distribution=distribution, architecture=ns.arch,
        gpu=GPU_ARGS[ns.vendor], sku=ns.gpu, node_type=ns.node_type,
    )
    print(f"# Resolved component versions for {ns.vendor}/{ns.gpu} "
          f"on {ns.os} ({ns.arch}, {ns.node_type})")
    print(format_report(report))
    return 0

def cmd_validate(args_list) -> int:
    """`azhpc validate --os <OS> --vendor <V> --gpu <SKU>` — check versions.json
    for internal consistency for a target. Exit 0 if clean, 1 if issues."""
    from utils.build_config import DISTRO_DIRS, GPU_ARGS, VALID_GPUS
    from utils.version_validate import check_cuda_consistency

    p = argparse.ArgumentParser(prog="azhpc validate")
    p.add_argument("--os", required=True, choices=sorted(DISTRO_DIRS))
    p.add_argument("--vendor", required=True, choices=sorted(GPU_ARGS))
    p.add_argument("--gpu", required=True)
    p.add_argument("--arch", default="x86_64")
    p.add_argument("--node-type", dest="node_type", default="azure-vm")
    p.add_argument("--spec", default="versions.json")
    ns = p.parse_args(args_list)

    if ns.gpu not in VALID_GPUS[ns.vendor]:
        print(f"error: GPU '{ns.gpu}' not valid for vendor {ns.vendor}",
              file=sys.stderr)
        return 1

    spec = Path(ns.spec)
    if not spec.is_file():
        print(f"error: spec file not found: {ns.spec}", file=sys.stderr)
        return 2
    versions = json.loads(spec.read_text(encoding="utf-8"))

    distribution = DISTRO_DIRS[ns.os].split("/")[-1]
    issues = check_cuda_consistency(
        versions, distribution=distribution, architecture=ns.arch,
        gpu=GPU_ARGS[ns.vendor], sku=ns.gpu, node_type=ns.node_type,
    )
    target = f"{ns.vendor}/{ns.gpu} on {ns.os} ({ns.arch}, {ns.node_type})"
    if not issues:
        print(f"OK: no version-consistency issues for {target}")
        return 0
    print(f"FAIL: {len(issues)} version-consistency issue(s) for {target}:",
          file=sys.stderr)
    for issue in issues:
        print(f"  - {issue}", file=sys.stderr)
    return 1

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "install":
        return cmd_install(argv[1:])
    if argv and argv[0] == "versions":
        return cmd_versions(argv[1:])
    if argv and argv[0] == "validate":
        return cmd_validate(argv[1:])

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