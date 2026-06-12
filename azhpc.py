"""Python implementation of the Azure HPC CLI tool."""

from datetime import datetime, timezone

VERSION = "0.1.0"
BUILD_COMMIT = "dev"
BUILD_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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
