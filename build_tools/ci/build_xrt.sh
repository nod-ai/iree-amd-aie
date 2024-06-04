#!/bin/bash

set -eu -o errtrace

repo_dir="$(cd $(dirname $0)/../.. && pwd)"
src_dir="${repo_dir}/third_party/XRT"
build_dir="$1"
install_dir="$2"

echo "Building XRT"
echo "============"
echo "Source directory: $src_dir"
echo "Build directory: $build_dir"
echo "Install directory: $install_dir"
mkdir -p "${build_dir}"
mkdir -p "${install_dir}"

# Note that all of the install prefixes and DESTDIR are required.
# XRT is hard-coded to install to some absolute locations regardless.
cmake -GNinja \
  "-S${src_dir}" \
  "-B${build_dir}" \
  -DDISABLE_ABI_CHECK=ON \
  -DHAS_PYTHON=OFF \
  -DCMAKE_INSTALL_PREFIX=/opt/xilinx \
  -DXRT_INSTALL_PREFIX=/opt/xilinx \
  -DCMAKE_BUILD_TYPE=Release 

cmake --build "$build_dir" -- -k 0
DESTDIR=$install_dir cmake --build "$build_dir" --target install
