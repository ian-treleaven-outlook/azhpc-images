#!/usr/bin/env bash
set -euo pipefail

# Infiniband/RDMA Setup Script
# Author: ian-treleaven-outlook
# Date: 2025-10-23

echo "Setting up Infiniband/RDMA..."

case ${OS_FAMILY} in
    ubuntu)
        apt-get install -y \
            rdma-core \
            infiniband-diags \
            ibverbs-utils \
            perftest \
            libibverbs-dev \
            librdmacm-dev
        ;;
    alma|azurelinux)
        yum install -y \
            rdma-core \
            infiniband-diags \
            libibverbs-utils \
            perftest \
            libibverbs-devel \
            librdmacm-devel
        ;;
esac

# Enable RDMA services
systemctl enable rdma || true

echo "Infiniband/RDMA setup completed"