#!/bin/bash
set -ex

source ${UTILS_DIR}/utilities.sh
source ${UTILS_DIR}/logger.sh

log_info install-pmix "starting PMIx installation for ${DISTRIBUTION} (${ARCHITECTURE})"

pmix_metadata=$(get_component_config "pmix")
PMIX_VERSION=$(jq -r '.version' <<< $pmix_metadata)
log_info install-pmix "target PMIx version ${PMIX_VERSION}"

if [[ $DISTRIBUTION == *"ubuntu"* ]]; then
    log_info install-pmix "installing PMIx on Ubuntu via Microsoft slurm apt repo"
    UBUNTU_VERSION=$(cat /etc/os-release | grep VERSION_ID | cut -d= -f2 | cut -d\" -f2)
    if [ $UBUNTU_VERSION == 24.04 ]; then
        REPO=slurm-ubuntu-noble
        SIGNED_BY="/usr/share/keyrings/microsoft-prod.gpg"
    elif [ $UBUNTU_VERSION == 22.04 ]; then
        REPO=slurm-ubuntu-jammy
        SIGNED_BY="/etc/apt/trusted.gpg.d/microsoft-prod.gpg"
    else
        log_error install-pmix "$DISTRIBUTION not supported for pmix installation"
    fi
    log_debug install-pmix "using slurm repo ${REPO} (signed-by ${SIGNED_BY})"
    echo "deb [arch=$ARCHITECTURE_DISTRO signed-by=$SIGNED_BY] https://packages.microsoft.com/repos/$REPO/ insiders main" > /etc/apt/sources.list.d/slurm.list

    cp ${COMPONENT_DIR}/slurm-repo/slurm-u.pin /etc/apt/preferences.d/slurm-repository-pin-990
    log_info install-pmix "pinned slurm repo (priority 990) so insiders track wins over distro packages"

    ## This package is pre-installed in all hpc images used by cyclecloud, but if customer wants to
    ## use generic ubuntu marketplace image then this package sets up the right gpg keys for PMC.
    if [ ! -e /etc/apt/sources.list.d/microsoft-prod.list ]; then
        log_info install-pmix "registering Microsoft package repo (PMC gpg keys) — generic Ubuntu image path"
        curl -sSL -O https://packages.microsoft.com/config/ubuntu/$UBUNTU_VERSION/packages-microsoft-prod.deb
        dpkg -i packages-microsoft-prod.deb
        rm packages-microsoft-prod.deb
    fi
    apt update
    log_info install-pmix "installing pmix=${PMIX_VERSION} and dev deps (libevent, libhwloc)"
    apt install -y pmix=${PMIX_VERSION} libevent-dev libhwloc-dev # libmunge-dev
    # Hold versions of packages to prevent accidental updates. Packages can still be upgraded explictly by
    # '--allow-change-held-packages' flag.
    apt-mark hold pmix=${PMIX_VERSION} libevent-dev libhwloc-dev # libmunge-dev
    log_info install-pmix "marked pmix and dev deps on hold (no accidental upgrades)"
elif [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
    log_info install-pmix "installing PMIx on Azure Linux 3.0 via tdnf"
    tdnf -y install pmix pmix-devel pmix-tools
    tdnf -y install hwloc-devel libevent-devel munge-devel
    if [ "$ARCHITECTURE" = "aarch64" ]; then
        postfix="aarch64"
    else
        postfix="x86_64"
    fi
    PMIX_VERSION=$(tdnf list installed | grep -i pmix.${postfix} | sed 's/.*[[:space:]]\([0-9.]*-[0-9]*\)\..*/\1/')
    log_debug install-pmix "resolved installed PMIx version to ${PMIX_VERSION}"
else
    log_info install-pmix "installing PMIx on RHEL-family via dnf/yum"
    # RHEL-family: AlmaLinux, Rocky Linux, RHEL, etc.
    OS_MAJOR_VERSION=$(sed -n 's/^VERSION_ID="\([0-9]\+\).*/\1/p' /etc/os-release)
    cp ${COMPONENT_DIR}/slurm-repo/slurm-el${OS_MAJOR_VERSION}.repo /etc/yum.repos.d/slurm.repo
    log_info install-pmix "registered slurm-el${OS_MAJOR_VERSION} yum repo"

    if [ ! -e /etc/yum.repos.d/microsoft-prod.repo ];then
        log_info install-pmix "registering Microsoft package repo (PMC gpg keys) — generic RHEL image path"
        curl -sSL -O https://packages.microsoft.com/config/rhel/${OS_MAJOR_VERSION}/packages-microsoft-prod.rpm
        rpm -i packages-microsoft-prod.rpm
        rm packages-microsoft-prod.rpm
    fi

    if [[ $OS_MAJOR_VERSION == "9" ]]; then
        log_info install-pmix "enabling crb repo (provides hwloc-devel/libevent-devel on EL9)"
        dnf config-manager --set-enabled crb
    elif  [[ $OS_MAJOR_VERSION == "8" ]]; then
        log_info install-pmix "enabling powertools repo (provides hwloc-devel/libevent-devel on EL8)"
        dnf config-manager --set-enabled powertools
    fi
    log_info install-pmix "refreshing all yum packages before pmix install"
    yum update -y
    log_info install-pmix "installing pmix-${PMIX_VERSION}.el${OS_MAJOR_VERSION} and dev deps"
    yum -y install pmix-${PMIX_VERSION}.el${OS_MAJOR_VERSION} hwloc-devel libevent-devel munge-devel
fi

write_component_version "PMIX" ${PMIX_VERSION}
log_info install-pmix "PMIx ${PMIX_VERSION} installation complete"