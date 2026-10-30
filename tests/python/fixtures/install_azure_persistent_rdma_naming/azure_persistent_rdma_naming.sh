#!/bin/bash

rdma_rename=/usr/sbin/rdma_rename_56.1

mapfile -t all_devices < <(ibdev2netdev -v | sort -n | awk '{print $2}')

next_index() {
	local prefix=$1
	local max_index=-1
	local dev suffix

	for dev in "${all_devices[@]}"; do
		suffix=${dev#$prefix}
		if [[ "$dev" == "$prefix"* && "$suffix" =~ ^[0-9]+$ && "$suffix" -gt "$max_index" ]]; then
			max_index=$suffix
		fi
	done

	echo $((max_index + 1))
}

an_index=$(next_index mlx5_an)
ib_index=$(next_index mlx5_ib)

for old_device in "${all_devices[@]}"; do

	case "$old_device" in
		mlx5_ib*|mlx5_an*) continue ;;
	esac

	link_layer=$(ibv_devinfo -d $old_device | sed -n 's/^[\ \t]*link_layer:[\ \t]*\([a-zA-Z]*\)$/\1/p')
	
	if [ "$link_layer" = "InfiniBand" ]; then
		
		$rdma_rename $old_device NAME_FIXED mlx5_ib${ib_index}
		ib_index=$(($ib_index + 1))
		
	elif [ "$link_layer" = "Ethernet" ]; then
	
		$rdma_rename $old_device NAME_FIXED mlx5_an${an_index}
		an_index=$(($an_index + 1))
		
	else
	
		echo "Unknown device type for $old_device."
		
	fi
	
done
