#!/bin/bash

metadata_endpoint="http://169.254.169.254/metadata/instance?api-version=2019-06-04"
vmSize=$(curl -H Metadata:true $metadata_endpoint | jq -r ".compute.vmSize")
vmSize=$(echo "$vmSize" | awk '{print tolower($0)}')

## Topo file setup based on SKU
case $vmSize in
    
    standard_nd*v4)
        /opt/azurehpc/customizations/ndv4_rocm.sh;;

    standard_nd96is*_mi300x_v5)
        /opt/azurehpc/customizations/ndv5_rocm.sh;;

    *) echo "No SKU customization for $vmSize";;
esac
