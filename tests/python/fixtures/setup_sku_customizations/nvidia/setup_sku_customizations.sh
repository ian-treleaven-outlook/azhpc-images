#!/bin/bash

# NCCL_TOPO_FAMILY may be pre-set by the caller via the environment
# to skip the Azure IMDS lookup on non-Azure nodes.
if [[ -n "${NCCL_TOPO_FAMILY:-}" ]]; then
    topoFamily="$NCCL_TOPO_FAMILY"
else
    metadata_endpoint="http://169.254.169.254/metadata/instance?api-version=2019-06-04"
    vmSize=$(curl -H Metadata:true "$metadata_endpoint" | jq -r ".compute.vmSize")
    vmSize=$(echo "$vmSize" | awk '{print tolower($0)}')

    ## Derive topology family from VM size
    case $vmSize in
        standard_nc96ads_a100_v4)          topoFamily="ncv4" ;;
        standard_nd*v4)                    topoFamily="ndv4" ;;
        standard_nd40rs_v2)               topoFamily="ndv2" ;;
        standard_hb176*v4)                topoFamily="hbv4" ;;
        standard_nc80adis_h100_v5)        topoFamily="ncv5" ;;
        standard_nd96is*_h[1-2]00_v5)    topoFamily="ndv5" ;;
        standard_nd128is*_gb[2-3]00_v6)  topoFamily="gb-family" ;;
        standard_nc*_rtxpro6000bse_v6)    topoFamily="ncv6" ;;
        *) topoFamily="" ;;
    esac
fi

## Topo file setup based on family
case $topoFamily in
    ncv4)      /opt/azurehpc/customizations/ncv4.sh;;
    ndv4)      /opt/azurehpc/customizations/ndv4.sh;;
    ndv2)      /opt/azurehpc/customizations/ndv2.sh;;
    hbv4)      /opt/azurehpc/customizations/hbv4.sh;;
    ncv5)      /opt/azurehpc/customizations/ncv5.sh;;
    ndv5)      /opt/azurehpc/customizations/ndv5.sh;;
    gb-family) /opt/azurehpc/customizations/ndv6.sh;;
    ncv6)      /opt/azurehpc/customizations/ncv6.sh;;
    *)         echo "No SKU customization for topoFamily='$topoFamily'";;
esac
