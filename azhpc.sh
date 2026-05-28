#!/usr/bin/env bash
# Short-flag implementation of the azhpc CLI spec.

set -euo pipefail

VERSION="0.1.0"
BUILD_COMMIT="dev"
BUILD_DATE="$(date -u +%Y-%m-%d)"

spec=""
vendor=""
gpu=""
os=""
fips=false
dry_run=false
verbose=false
log_format=""
show_version=false

print_help() {
    cat <<'EOF'
azhpc - Build HPC/AI Linux images for Azure

USAGE:
    azhpc [OPTIONS]
    azhpc -h

OPTIONS:
    -s <PATH>     Path to versions.json (optional; default: azhpc-images versions.json)
    -m <VENDOR>   Hardware vendor (required): NVidia | AMD | Microsoft
    -g <GPU>      GPU model (required). Valid values depend on -m:
                    NVidia    : A100+ | GB200 | GB300 | VR200
                    AMD       : MI300 | MI400 | MI500
                    Microsoft : MAIA200
    -o <OS>       Target OS (required): Ubuntu24 | Ubuntu22 | Azure3 | Alma9 | Nix25
    -f            Build a FIPS-compliant image (default: non-FIPS)
    -n            Dry-run: validate args and print build plan; do not build
    -v            Verbose (debug-level) logging
    -l <FORMAT>   Log format: text | json (default: text if stdout is a TTY, else json)
    -r            Print version information and exit
    -h            Show this help message and exit

EXIT STATUS:
    0  Image built successfully
    1  Invalid arguments or unsupported vendor/GPU combination
    2  Specification file missing or malformed
    3  Build failure
EOF
}

die() { 
    echo "azhpc: error: $*" >&2; exit "${2:-1}"; 
}

while getopts ":s:m:g:o:fnvl:rh" opt; do
    case "$opt" in
        s) spec="$OPTARG" ;;
        m) vendor="$OPTARG" ;;
        g) gpu="$OPTARG" ;;
        o) os="$OPTARG" ;;
        f) fips=true ;;
        n) dry_run=true ;;
        v) verbose=true ;;
        l) log_format="$OPTARG" ;;
        r) show_version=true ;;
        h) print_help; exit 0 ;;
        :) die "option -$OPTARG requires an argument" 1 ;;
        \?) die "invalid option -$OPTARG (use -h for help)" 1 ;;
    esac
done

if [[ "$show_version" == true ]]; then
    echo "azhpc $VERSION (commit $BUILD_COMMIT, built $BUILD_DATE)"
    exit 0
fi

# Required args
[[ -n "$vendor" ]] || die "-m <vendor> is required" 1
[[ -n "$gpu"    ]] || die "-g <gpu> is required" 1
[[ -n "$os"     ]] || die "-o <os> is required" 1

# Vendor / GPU validation
case "$vendor" in
    NVidia)
        case "$gpu" in A100+|GB200|GB300|VR200) ;;
            *) die "GPU '$gpu' not valid for vendor NVidia" 1 ;; esac ;;
    AMD)
        case "$gpu" in MI300|MI400|MI500) ;;
            *) die "GPU '$gpu' not valid for vendor AMD" 1 ;; esac ;;
    Microsoft)
        [[ "$gpu" == "MAIA200" ]] || die "GPU '$gpu' not valid for vendor Microsoft" 1 ;;
    *) die "unsupported vendor '$vendor' (NVidia|AMD|Microsoft)" 1 ;;
esac

# OS validation
case "$os" in
    Ubuntu24|Ubuntu22|Azure3|Alma9|Nix25) ;;
    *) die "unsupported os '$os' (Ubuntu24|Ubuntu22|Azure3|Alma9|Nix25)" 1 ;;
esac

# Log format default + validation
if [[ -z "$log_format" ]]; then
    if [[ -t 1 ]]; then log_format=text; else log_format=json; fi
fi
case "$log_format" in
    text|json) ;;
    *) die "invalid log format '$log_format' (text|json)" 1 ;;
esac

# Spec file validation (exit 2 per spec)
if [[ -n "$spec" ]]; then
    [[ -f "$spec" ]] || die "spec file not found: $spec" 2
    if command -v jq >/dev/null 2>&1; then
        jq empty "$spec" >/dev/null 2>&1 || die "spec file is malformed JSON: $spec" 2
    fi
fi

effective_spec="${spec:-<azhpc-images default versions.json>}"

if [[ "$verbose" == true || "$dry_run" == true ]]; then
    cat >&2 <<EOF
Build plan:
  vendor      = $vendor
  gpu         = $gpu
  os          = $os
  fips        = $fips
  spec        = $effective_spec
  log_format  = $log_format
  verbose     = $verbose
  dry_run     = $dry_run
EOF
fi

if [[ "$dry_run" == true ]]; then
    exit 0
fi

# --- real build would go here; exit 3 on failure ---
# run_build || die "build failed" 3
exit 0