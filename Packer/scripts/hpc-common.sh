#!/usr/bin/env bash
set -euo pipefail

# HPC Common Tools Installation
# Author: ian-treleaven-outlook
# Date: 2025-10-23

echo "Installing HPC common tools..."

# Install OpenMPI
case ${OS_FAMILY} in
    ubuntu)
        apt-get install -y openmpi-bin openmpi-common libopenmpi-dev
        ;;
    alma|azurelinux)
        yum install -y openmpi openmpi-devel
        ;;
esac

# Install common HPC tools
if command -v apt-get &> /dev/null; then
    apt-get install -y \
        nfs-common \
        pdsh \
        hwloc \
        numactl \
        iperf3 \
        fio
else
    yum install -y \
        nfs-utils \
        pdsh \
        hwloc \
        numactl \
        iperf3 \
        fio
fi

echo "HPC common tools installed"