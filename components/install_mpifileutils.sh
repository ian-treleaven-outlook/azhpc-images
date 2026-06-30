#!/bin/bash
set -ex

# Install mpifileutils - MPI-based file utilities for parallel file operations
# Provides tools like dbcast, dcp, drm, dsync, dfind, dwalk, dcmp, dtar

source ${UTILS_DIR}/utilities.sh

mpifileutils_metadata=$(get_component_config "mpifileutils")
MPIFILEUTILS_VERSION=$(jq -r '.version' <<< $mpifileutils_metadata)
MPIFILEUTILS_URL=$(jq -r '.url' <<< $mpifileutils_metadata)
MPIFILEUTILS_SHA256=$(jq -r '.sha256' <<< $mpifileutils_metadata)

INSTALL_PREFIX="/opt/mpifileutils"
BUILD_DIR="/tmp/mpifileutils-build"
SRC_DIR="/tmp/mpifileutils-src"
LUSTRE_CMAKE_FLAG="ON"

if [[ $DISTRIBUTION == "azurelinux3.0" ]]; then
    LUSTRE_CMAKE_FLAG="OFF"
fi

echo "=== Installing mpifileutils ${MPIFILEUTILS_VERSION} ==="

# Build dependencies (libbz2-dev, libattr1-dev, libarchive-dev, libssl-dev,
# libcap-dev) are installed by the azhpc orchestrator via PackageInstaller
# (see install_mpifileutils_deps in utils/build_config.py) before this script
# runs, so the package-manager branching that used to live here is gone.

# Create directories
mkdir -p "${INSTALL_PREFIX}"
mkdir -p "${BUILD_DIR}"
mkdir -p "${SRC_DIR}"

# Download and verify
cd "${SRC_DIR}"
TARBALL="mpifileutils-v${MPIFILEUTILS_VERSION}.tgz"
download_and_verify "${MPIFILEUTILS_URL}" "${MPIFILEUTILS_SHA256}"

# Extract
tar -xzf "${TARBALL}"

# Load HPC-X MPI for building
source /etc/profile.d/modules.sh
module load mpi/hpcx

# Build with CMake
cd "${BUILD_DIR}"
cmake "${SRC_DIR}/mpifileutils-v${MPIFILEUTILS_VERSION}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DENABLE_XATTRS=ON \
    -DENABLE_LIBARCHIVE=ON \
    -DENABLE_LUSTRE="${LUSTRE_CMAKE_FLAG}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

make -j$(nproc)
make install

module unload mpi/hpcx

# Cleanup build artifacts
rm -rf "${BUILD_DIR}" "${SRC_DIR}"

echo "=== mpifileutils installed to ${INSTALL_PREFIX} ==="
ls -la "${INSTALL_PREFIX}/bin/"

write_component_version "MPIFILEUTILS" ${MPIFILEUTILS_VERSION}
