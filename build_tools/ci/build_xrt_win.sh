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

export XRT_BUILD_DIR="$build_dir"
export XRT_INSTALL_DIR=/c/opt/xilinx
export DESTDIR=/c/opt/xilinx

pushd $src_dir

sed -i "s/\/Qspectre//g" src/CMake/nativeWin.cmake
git submodule update --recursive --init
python ./src/runtime_src/tools/scripts/xrtdeps-win19.py --icd --opencl
cd build
./build-win19.sh -j 10 -boost $BOOST_ROOT -noabi -cmake "$(which cmake)"

popd
