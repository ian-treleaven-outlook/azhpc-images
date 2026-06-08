#!/bin/bash
set -ex

source ${UTILS_DIR}/utilities.sh
source ${UTILS_DIR}/logger.sh

log_info install-nvidiagpudriver "starting NVIDIA driver + CUDA installation for ${DISTRIBUTION} (${ARCHITECTURE}, SKU=${SKU:-unknown})"

# Install NVIDIA driver
nvidia_metadata=$(get_component_config "nvidia")
cuda_metadata=$(get_component_config "cuda")

if [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
    log_info install-nvidiagpudriver "installing NVIDIA driver on Azure Linux 3.0 via tdnf"
    if [ "$SKU" = "V100" ]; then
        # V100 requires proprietary kernel modules
        AL3_GPU_DRIVER_PACKAGES="cuda"
        log_info install-nvidiagpudriver "V100 SKU: using proprietary kernel modules (cuda package)"
    elif [ "$ARCHITECTURE" = "aarch64" ]; then
        AL3_GPU_DRIVER_PACKAGES="cuda-open-hwe"
        log_info install-nvidiagpudriver "aarch64: using open kernel modules with HWE kernel (cuda-open-hwe)"
    else
        AL3_GPU_DRIVER_PACKAGES="cuda-open"
        log_info install-nvidiagpudriver "x86_64: using open kernel modules (cuda-open)"
    fi

    log_info install-nvidiagpudriver "registering Azure Linux NVIDIA prod and CUDA repos"
    if [[ "$ARCHITECTURE" == "aarch64" ]]; then
        curl https://packages.microsoft.com/azurelinux/3.0/prod/nvidia/aarch64/config.repo > /etc/yum.repos.d/azurelinux-nvidia-prod.repo
        curl https://developer.download.nvidia.com/compute/cuda/repos/azl3/sbsa/cuda-azl3.repo > /etc/yum.repos.d/cuda-azl3.repo
    else
        curl https://packages.microsoft.com/azurelinux/3.0/prod/nvidia/x86_64/config.repo > /etc/yum.repos.d/azurelinux-nvidia-prod.repo
        curl https://developer.download.nvidia.com/compute/cuda/repos/azl3/x86_64/cuda-azl3.repo > /etc/yum.repos.d/cuda-azl3.repo
    fi

    # Disable the NVIDIA CUDA repo during driver install — all driver
    # packages come from PMC and the CUDA repo has an identically-named
    # 'cuda' meta-package that would conflict.
    # Do not use this before bugfixed tdnf lands (https://github.com/vmware/tdnf/pull/553/commits/a418054b02c4cac787184f973dac4d6790344ef3)
    # or before switching to dnf
    # tdnf install -y --disablerepo=cuda-azl3* $AL3_GPU_DRIVER_PACKAGES
    log_info install-nvidiagpudriver "installing ${AL3_GPU_DRIVER_PACKAGES} from PMC (CUDA repo disabled to avoid 'cuda' name collision)"
    tdnf install -y --disablerepo=cuda-azl3-x86_64 --disablerepo=cuda-azl3-sbsa $AL3_GPU_DRIVER_PACKAGES
    NVIDIA_DRIVER_VERSION=$(tdnf list installed | grep "^${AL3_GPU_DRIVER_PACKAGES}\." | sed 's/.*\s\+\([0-9.]\+-[0-9]\+\)_.*/\1/')
    log_debug install-nvidiagpudriver "resolved NVIDIA driver version ${NVIDIA_DRIVER_VERSION}"

    # Temp disable NVIDIA driver updates
    mkdir -p /etc/tdnf/locks.d
    echo cuda >> /etc/tdnf/locks.d/nvidia.conf
    log_info install-nvidiagpudriver "locked 'cuda' package against tdnf updates (driver/CUDA must move in lockstep)"
elif [[ $DISTRIBUTION == *"ubuntu"* ]]; then
    log_info install-nvidiagpudriver "installing NVIDIA driver on Ubuntu via apt"
    # APT-based NVIDIA driver installation for Ubuntu
    NVIDIA_DRIVER_VERSION=$(jq -r '.driver.version' <<< $nvidia_metadata)
    CUDA_DRIVER_DISTRIBUTION=$(jq -r '.driver.distribution' <<< $cuda_metadata)
    log_info install-nvidiagpudriver "target NVIDIA driver version ${NVIDIA_DRIVER_VERSION} (CUDA distro ${CUDA_DRIVER_DISTRIBUTION})"

    # Add NVIDIA CUDA APT repo (provides both driver and toolkit packages)
    log_info install-nvidiagpudriver "registering NVIDIA CUDA apt repo (driver + toolkit)"
    wget https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DRIVER_DISTRIBUTION}/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i ./cuda-keyring_1.1-1_all.deb
    apt-get update

    # Pin the driver version and install via APT packages
    log_info install-nvidiagpudriver "pinning NVIDIA driver to ${NVIDIA_DRIVER_VERSION}"
    apt install nvidia-driver-pinning-${NVIDIA_DRIVER_VERSION} -y
    if [ "$SKU" = "V100" ]; then
        # V100 requires proprietary kernel modules
        log_info install-nvidiagpudriver "V100 SKU: installing cuda-drivers (proprietary kernel modules)"
        apt install cuda-drivers -y
    else
        # A100, H100, H200 use open kernel modules
        log_info install-nvidiagpudriver "installing nvidia-open (open kernel modules; A100/H100/H200)"
        apt install nvidia-open -y
    fi

    # Remove unused configuration file if created by the NVIDIA driver package
    rm -f /etc/modprobe.d/nvidia-graphics-drivers-kms.conf

    # Apply nvprofiling settings
    log_info install-nvidiagpudriver "enabling nvprofiling for non-admin users"
    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | tee /etc/modprobe.d/nvprofiling.conf

    # nvidia-peermem is NOT modprobe'd at build time. Loading it before the
    # first reboot is fragile across the matrix of distros / kernels we
    # support (e.g. Ubuntu 26.04 needs DOCA-OFED's patched ib_core in
    # /lib/modules/$(uname -r)/updates/dkms/ which is not active in the
    # build kernel; general-purpose build SKUs have no IB hardware to load
    # against; baremetal builds reboot before IB is fully up). The module is
    # queued for first boot via /etc/modules-load.d/nvidia-peermem.conf
    # written below and via the openibd ExecStartPost drop-in installed by
    # setup_sku_customizations.sh.
else
    log_info install-nvidiagpudriver "installing NVIDIA driver on RHEL-family via .run installer"
    # RHEL-family: AlmaLinux, Rocky Linux, RHEL - .run file installation
    NVIDIA_DRIVER_VERSION=$(jq -r '.driver.version' <<< $nvidia_metadata)
    NVIDIA_DRIVER_SHA256=$(jq -r '.driver.sha256' <<< $nvidia_metadata)
    NVIDIA_DRIVER_URL=https://us.download.nvidia.com/tesla/${NVIDIA_DRIVER_VERSION}/NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}.run
    CUDA_DRIVER_DISTRIBUTION=$(jq -r '.driver.distribution' <<< $cuda_metadata)
    log_info install-nvidiagpudriver "target NVIDIA driver version ${NVIDIA_DRIVER_VERSION}"

    if [ "$SKU" = "V100" ]; then
        KERNEL_MODULE_TYPE="proprietary"
        log_info install-nvidiagpudriver "V100 SKU: building proprietary kernel modules"
    else
        KERNEL_MODULE_TYPE="open"
        log_info install-nvidiagpudriver "building open kernel modules (A100/H100/H200)"
    fi

    log_info install-nvidiagpudriver "downloading and verifying NVIDIA .run installer"
    download_and_verify $NVIDIA_DRIVER_URL ${NVIDIA_DRIVER_SHA256}
    log_info install-nvidiagpudriver "running NVIDIA installer (silent, dkms, kernel-module-type=${KERNEL_MODULE_TYPE})"
    bash NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}.run --silent --dkms --kernel-module-type=${KERNEL_MODULE_TYPE}
    if [[ $DISTRIBUTION == almalinux* ]] || [[ $DISTRIBUTION == rocky* ]] || [[ $DISTRIBUTION == rhel* ]]; then
        log_info install-nvidiagpudriver "force-rebuilding nvidia dkms module against running kernel"
        dkms install --no-depmod -m nvidia -v ${NVIDIA_DRIVER_VERSION} -k `uname -r` --force
    fi
    # nvidia-peermem is NOT modprobe'd at build time -- see comment in the
    # Ubuntu branch above. The module is queued for first boot via
    # /etc/modules-load.d/nvidia-peermem.conf written below and via the
    # openibd ExecStartPost drop-in installed by setup_sku_customizations.sh.
fi
write_component_version "NVIDIA" ${NVIDIA_DRIVER_VERSION}
log_info install-nvidiagpudriver "installed NVIDIA driver ${NVIDIA_DRIVER_VERSION}"

touch /etc/modules-load.d/nvidia-peermem.conf
echo "nvidia_peermem" >> /etc/modules-load.d/nvidia-peermem.conf
log_info install-nvidiagpudriver "queued nvidia_peermem for autoload on first boot"

if [[ "$DISTRIBUTION" != *-aks ]]; then
    # Install CUDA toolkit
    CUDA_DRIVER_VERSION=$(jq -r '.driver.version' <<< $cuda_metadata)
    CUDA_SAMPLES_VERSION=$(jq -r '.samples.version' <<< $cuda_metadata)
    CUDA_SAMPLES_SHA256=$(jq -r '.samples.sha256' <<< $cuda_metadata)
    log_info install-nvidiagpudriver "installing CUDA toolkit ${CUDA_DRIVER_VERSION}"

    if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
        # NVIDIA APT repo already configured during driver installation
        apt install -y cuda-toolkit-${CUDA_DRIVER_VERSION//./-}
    elif [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
        tdnf install -y cuda-toolkit-${CUDA_DRIVER_VERSION//./-}
    else
        # RHEL-family: AlmaLinux, Rocky Linux, RHEL, etc.
        log_info install-nvidiagpudriver "registering CUDA dnf repo (${CUDA_DRIVER_DISTRIBUTION})"
        dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DRIVER_DISTRIBUTION}/x86_64/cuda-${CUDA_DRIVER_DISTRIBUTION}.repo

        # DOCA ships mft tied to the kernel-mft-dkms it built; cuda-rhel9
        # ships mft on a different cadence (sometimes newer). Letting
        # cuda-rhel9 offer mft causes 'dnf check-update' to flag a
        # stale-package upgrade in verify_package_updates and risks an
        # accidental upgrade that breaks compat with the DOCA-built
        # kernel-mft-dkms. mft must track DOCA, not CUDA. Same pattern as
        # install_nvidia_fabric_manager.sh excluding nvidia-fabricmanager*
        # from cuda-azl3 on AzureLinux 3, and a per-repo replacement for
        # the (removed) global DOCA pin in install_doca.sh.
        log_info install-nvidiagpudriver "excluding mft*/kernel-mft* from cuda repo (mft must track DOCA, not CUDA)"
        dnf config-manager --save \
            --setopt="cuda-${CUDA_DRIVER_DISTRIBUTION}-x86_64.excludepkgs=mft* kernel-mft*" >/dev/null

        dnf clean expire-cache
        dnf install -y cuda-toolkit-${CUDA_DRIVER_VERSION//./-}
    fi

    log_info install-nvidiagpudriver "writing /etc/profile.d/cuda.sh (PATH + LD_LIBRARY_PATH for /usr/local/cuda)"
    echo 'export PATH="${PATH:+$PATH:}/usr/local/cuda/bin"' | tee /etc/profile.d/cuda.sh > /dev/null
    echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/cuda/lib64"' | tee -a /etc/profile.d/cuda.sh > /dev/null

    # Ensure proper permissions
    chmod 644 /etc/profile.d/cuda.sh

    cuda_version=$(source /etc/profile; nvcc --version | grep release | awk '{print $6}' | cut -c2-)
    write_component_version "CUDA" ${cuda_version}
    log_info install-nvidiagpudriver "installed CUDA toolkit ${cuda_version}"

    log_info install-nvidiagpudriver "installing CUDA samples"
    $COMPONENT_DIR/install_cuda_samples.sh

else
    log_info install-nvidiagpudriver "AKS distribution: skipping CUDA toolkit install"
fi

log_info install-nvidiagpudriver "installing GDRCopy"
$COMPONENT_DIR/install_gdrcopy.sh

if [[ "$ARCHITECTURE" != "aarch64" ]]; then
    # Install nvidia fabric manager (required for ND96asr_v4)
    log_info install-nvidiagpudriver "installing NVIDIA Fabric Manager (required for ND96asr_v4)"
    $COMPONENT_DIR/install_nvidia_fabric_manager.sh
else
    log_info install-nvidiagpudriver "aarch64: configuring CDMM mode + NVIDIA IMEX (Grace-Hopper path)"
    # Apply nvprofiling settings
    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | tee /etc/modprobe.d/nvprofiling.conf

    # Enable CDMM mode
    echo 'options nvidia NVreg_CoherentGPUMemoryMode=driver' | tee /etc/modprobe.d/nvidia-openrm.conf

    # Install NVIDIA IMEX
    nvidia_imex_metadata=$(jq -r '.imex' <<< $nvidia_metadata)
    IMEX_VERSION=$(jq -r '.version' <<< $nvidia_imex_metadata)
    log_info install-nvidiagpudriver "installing nvidia-imex-${IMEX_VERSION}"
    tdnf install -y nvidia-imex-${IMEX_VERSION}

    # Add configuration to /etc/modprobe.d/nvidia.conf
    cat <<EOF >> /etc/modprobe.d/nvidia.conf
options nvidia NVreg_CreateImexChannel0=1
EOF

    grep -q 'RMBug5172204War=4' /etc/modprobe.d/nvidia.conf 2>/dev/null || \
        echo 'options nvidia NVreg_RegistryDwords="RMBug5172204War=4"' | tee -a /etc/modprobe.d/nvidia.conf

    # Ensure modprobe settings are available when nvidia module loads on next boot
    log_info install-nvidiagpudriver "rebuilding initramfs (dracut --force) so nvidia modprobe options apply at boot"
    dracut --force

    # Configuring nvidia-imex service
    systemctl enable nvidia-imex.service
    log_info install-nvidiagpudriver "enabled nvidia-imex.service for first boot"

fi

log_info install-nvidiagpudriver "configuring NVIDIA persistence daemon"
$COMPONENT_DIR/configure_nvidia_persistence.sh

# cleanup downloaded files
rm -rf *.run *.tar.gz *.rpm
rm -rf -- */

log_info install-nvidiagpudriver "NVIDIA driver + CUDA installation complete"