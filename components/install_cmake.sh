#!/bin/bash
set -eo pipefail

source ${UTILS_DIR}/utilities.sh
source ${UTILS_DIR}/logger.sh

op="install-cmake"

#update CMAKE
cmake_metadata=$(get_component_config "cmake")
cmake_version=$(jq -r '.version' <<< $cmake_metadata)
cmake_url=$(jq -r '.url' <<< $cmake_metadata)
cmake_sha256=$(jq -r '.sha256' <<< $cmake_metadata)
TARBALL="cmake-${cmake_version}-linux-x86_64.tar.gz"

log_info  "$op" "Installing CMake ${cmake_version}"
log_debug "$op" "url=${cmake_url} sha256=${cmake_sha256:0:12}…"

download_and_verify "${cmake_url}" "${cmake_sha256}"
tar -xzf "${TARBALL}"

pushd "cmake-${cmake_version}-linux-x86_64" >/dev/null
if ! cp -f bin/{ccmake,cmake,cpack,ctest} /usr/local/bin \
  || ! cp -rf share/cmake-* /usr/local/share/; then
    popd >/dev/null
    log_error "$op" "Failed to install CMake ${cmake_version} binaries into /usr/local"
    exit 1
fi
popd >/dev/null
hash -r

write_component_version "CMAKE" ${cmake_version}

# Remove installation files
rm -rf cmake-${cmake_version}-linux-x86_64*

log_info  "$op" "Installed CMake ${cmake_version} into /usr/local"
