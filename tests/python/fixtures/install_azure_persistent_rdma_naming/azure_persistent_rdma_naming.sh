#!/bin/bash

rdma_rename=/usr/sbin/rdma_rename_56.1

an_index=0
ib_index=0

for old_device in $(ibdev2netdev -v | sort -n | cut -f2 -d' '); do

	link_layer=$(ibv_devinfo -d $old_device | sed -n 's/^[\ \t]*link_layer:[\ \t]*\([a-zA-Z]*\)$/\1/p')
	
	if [ "$link_layer" = "InfiniBand" ]; then
		
		$rdma_rename $old_device NAME_FIXED mlx5_ib${ib_index}
		ib_index=$(($ib_index + 1))
		
	elif [ "$link_layer" = "Ethernet" ]; then
	
		$rdma_rename $old_device NAME_FIXED mlx5_an${an_index}
		an_index=$(($an_index + 1))
		
	else
	
		echo "Unknown device type for $old_device - $device_type."
		
	fi
	
done
