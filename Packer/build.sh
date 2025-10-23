#!/usr/bin/env bash
set -euo pipefail

# HPC Image Builder - Main Build Script
# Author: ian-treleaven-outlook
# Date: 2025-10-23

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TARGET="docker"
OS_FAMILY="ubuntu"
OS_VERSION="22.04"
CPU_ARCH="x86_64"
GPU_TYPE="none"
BUILD_STAGE="all"
PUSH_IMAGE=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help function
show_help() {
    cat << EOF
HPC Image Builder
Usage: $(basename "$0") [OPTIONS]

OPTIONS:
    -t, --target TARGET      Build target: docker, azure, hyperv, qemu (default: docker)
    -o, --os OS              OS family: ubuntu, alma, azurelinux (default: ubuntu)
    -v, --version VERSION    OS version (default: 22.04)
    -a, --arch ARCH          CPU architecture: x86_64, arm64 (default: x86_64)
    -g, --gpu GPU            GPU type: none, nvidia-a100, nvidia-h100, amd-mi300x (default: none)
    -s, --stage STAGE        Build stage: all, base, hpc, gpu (default: all)
    -p, --push               Push image to registry (Docker only)
    -V, --verbose            Enable verbose output
    -h, --help               Show this help message

EXAMPLES:
    # Build Ubuntu 22.04 with NVIDIA A100 for Docker
    $(basename "$0") -t docker -o ubuntu -v 22.04 -g nvidia-a100

    # Build AlmaLinux 9 for Azure with AMD MI300X
    $(basename "$0") -t azure -o alma -v 9 -g amd-mi300x

    # Build base image only for Hyper-V
    $(basename "$0") -t hyperv -o ubuntu -v 24.04 -s base

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        -o|--os)
            OS_FAMILY="$2"
            shift 2
            ;;
        -v|--version)
            OS_VERSION="$2"
            shift 2
            ;;
        -a|--arch)
            CPU_ARCH="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_TYPE="$2"
            shift 2
            ;;
        -s|--stage)
            BUILD_STAGE="$2"
            shift 2
            ;;
        -p|--push)
            PUSH_IMAGE=true
            shift
            ;;
        -V|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check for Packer
    if ! command -v packer &> /dev/null; then
        log_error "Packer is not installed"
        echo "Install Packer from: https://www.packer.io/downloads"
        exit 1
    fi
    
    # Check for target-specific requirements
    case ${BUILD_TARGET} in
        docker)
            if ! command -v docker &> /dev/null; then
                log_error "Docker is not installed"
                exit 1
            fi
            if ! docker info &> /dev/null; then
                log_error "Docker daemon is not running"
                exit 1
            fi
            ;;
        azure)
            if [[ -z "${AZURE_CLIENT_ID:-}" ]]; then
                log_warn "Azure credentials not found in environment"
                log_info "Running in pipeline mode - credentials will be provided by ADO"
            fi
            ;;
        hyperv)
            if [[ ! "$OSTYPE" == "msys" ]] && [[ ! "$OSTYPE" == "win32" ]]; then
                log_warn "Hyper-V builds should be run on Windows"
            fi
            ;;
    esac
    
    log_info "Prerequisites check completed"
}

# Initialize Packer
init_packer() {
    log_info "Initializing Packer..."
    
    cd "${SCRIPT_DIR}"
    
    if [[ "${VERBOSE}" == true ]]; then
        PACKER_LOG=1 packer init hpc.pkr.hcl
    else
        packer init hpc.pkr.hcl > /dev/null 2>&1
    fi
    
    if [[ $? -eq 0 ]]; then
        log_info "Packer initialized successfully"
    else
        log_error "Packer initialization failed"
        exit 1
    fi
}

# Build image
build_image() {
    log_info "Starting image build..."
    log_info "Configuration:"
    echo "  Target: ${BUILD_TARGET}"
    echo "  OS: ${OS_FAMILY} ${OS_VERSION}"
    echo "  Architecture: ${CPU_ARCH}"
    echo "  GPU: ${GPU_TYPE}"
    echo "  Stage: ${BUILD_STAGE}"
    echo ""
    
    cd "${SCRIPT_DIR}"
    
    # Prepare Packer command
    PACKER_CMD="packer build"
    
    if [[ "${VERBOSE}" == true ]]; then
        export PACKER_LOG=1
    fi
    
    # Add variables
    PACKER_CMD="${PACKER_CMD} -var build_target=${BUILD_TARGET}"
    PACKER_CMD="${PACKER_CMD} -var os_family=${OS_FAMILY}"
    PACKER_CMD="${PACKER_CMD} -var os_version=${OS_VERSION}"
    PACKER_CMD="${PACKER_CMD} -var cpu_arch=${CPU_ARCH}"
    PACKER_CMD="${PACKER_CMD} -var gpu_type=${GPU_TYPE}"
    
    # Add stage filter if not "all"
    if [[ "${BUILD_STAGE}" != "all" ]]; then
        PACKER_CMD="${PACKER_CMD} -only=${BUILD_STAGE}.*"
    fi
    
    # Add HCL file
    PACKER_CMD="${PACKER_CMD} hpc.pkr.hcl"
    
    # Execute build
    log_info "Executing: ${PACKER_CMD}"
    
    if eval ${PACKER_CMD}; then
        log_info "Build completed successfully!"
        
        # Push Docker image if requested
        if [[ "${BUILD_TARGET}" == "docker" ]] && [[ "${PUSH_IMAGE}" == true ]]; then
            log_info "Pushing Docker image..."
            IMAGE_TAG="hpc-images/${OS_FAMILY}:${OS_VERSION}-${GPU_TYPE}-latest"
            docker push ${IMAGE_TAG}
            log_info "Image pushed: ${IMAGE_TAG}"
        fi
    else
        log_error "Build failed"
        exit 1
    fi
}

# Main execution
main() {
    echo "=========================================="
    echo "HPC Image Builder"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "User: ian-treleaven-outlook"
    echo "=========================================="
    echo ""
    
    check_prerequisites
    init_packer
    build_image
    
    echo ""
    echo "=========================================="
    echo "Build Summary"
    echo "=========================================="
    echo "Target: ${BUILD_TARGET}"
    echo "Image: ${OS_FAMILY} ${OS_VERSION} (${CPU_ARCH})"
    echo "GPU: ${GPU_TYPE}"
    echo "Status: SUCCESS"
    echo "=========================================="
}

# Run main function
main