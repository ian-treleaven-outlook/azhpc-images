#!/usr/bin/env bash
set -euo pipefail

# Base OS Setup Script
# Author: ian-treleaven-outlook
# Date: 2025-10-23

echo "Setting up base OS..."
echo "OS Family: ${OS_FAMILY}"
echo "OS Version: ${OS_VERSION}"
echo "CPU Architecture: ${CPU_ARCH}"

# Update package manager
case ${OS_FAMILY} in
    ubuntu)
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get upgrade -y
        apt-get install -y \
            build-essential \
            curl \
            wget \
            git \
            vim \
            htop \
            tmux \
            python3 \
            python3-pip
        ;;
    alma|azurelinux)
        yum update -y
        yum groupinstall -y "Development Tools"
        yum install -y \
            curl \
            wget \
            git \
            vim \
            htop \
            tmux \
            python3 \
            python3-pip
        ;;
esac

echo "Base OS setup completed"