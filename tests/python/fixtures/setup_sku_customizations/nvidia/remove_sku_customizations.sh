#!/bin/bash

# Stop nvidia fabric manager
if systemctl is-active --quiet nvidia-fabricmanager
then
    systemctl stop nvidia-fabricmanager
    systemctl disable nvidia-fabricmanager
fi

# Remove NVIDIA peer memory module
if lsmod | grep nvidia_peermem &> /dev/null
then
    rmmod nvidia_peermem
fi

# Clear topo and graph files
rm -rf /opt/microsoft/ncv4
rm -rf /opt/microsoft/ndv2
rm -rf /opt/microsoft/ndv4
rm -rf /opt/microsoft/ndv5
rm -rf /opt/microsoft/ncv6

# Clear contents of nccl.conf
cat /dev/null > /etc/nccl.conf

