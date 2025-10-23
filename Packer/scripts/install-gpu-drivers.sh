#!/usr/bin/env bash
set -euo pipefail

# GPU Driver Installation Script
# Author: ian-treleaven-outlook
# Date: 2025-10-23

echo "Installing GPU drivers..."
echo "GPU Type: ${GPU_TYPE}"

if [[ "${GPU_TYPE}" == "none" ]]; then
    echo "No GPU specified, skipping driver installation"
    exit 0
fi

case ${GPU_TYPE} in
    nvidia-*)
        echo "Installing NVIDIA drivers..."
        
        # Install NVIDIA driver based on OS
        case ${OS_FAMILY} in
            ubuntu)
                # Add NVIDIA package repositories
                wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
                dpkg -i cuda-keyring_1.1-1_all.deb
                apt-get update
                
                # Install CUDA and drivers
                apt-get install -y cuda-toolkit-12-3
                apt-get install -y nvidia-driver-545
                ;;
            alma|azurelinux)
                # RHEL-based NVIDIA installation
                yum install -y kernel-devel kernel-headers
                wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
                sh cuda_12.3.0_545.23.06_linux.run --silent --driver --toolkit
                ;;
        esac
        
        # Install NVIDIA Container Toolkit
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
        apt-get update && apt-get install -y nvidia-container-toolkit
        ;;
        
    amd-*)
        echo "Installing AMD ROCm drivers..."
        # AMD GPU driver installation
        wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
        apt-get install -y ./amdgpu-install_6.0.60000-1_all.deb
        amdgpu-install -y --usecase=rocm
        ;;
esac

echo "GPU driver installation completed"