#!/bin/bash

set -eux -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/../.. && pwd)"
iree_dir="$(cd $repo_root/../iree && pwd)"
build_dir="$repo_root/iree-build"
install_dir="$repo_root/iree-install"
mkdir -p "$build_dir"
build_dir="$(cd $build_dir && pwd)"
cache_dir="${cache_dir:-}"

# Setup cache dir.
if [ -z "${cache_dir}" ]; then
  cache_dir="${repo_root}/.build-cache"
  mkdir -p "${cache_dir}"
  cache_dir="$(cd ${cache_dir} && pwd)"
fi
echo "Caching to ${cache_dir}"
mkdir -p "${cache_dir}/ccache"
mkdir -p "${cache_dir}/pip"

python="$(which python)"
echo "Using python: $python"

export CCACHE_DIR="${cache_dir}/ccache"
export CCACHE_MAXSIZE="2000M"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Clear ccache stats.
ccache -z

XRT_BUILD_DIR=$repo_root/xrt-build
XRT_INSTALL_DIR=$repo_root/xrt-install
bash $this_dir/build_xrt_win.sh $XRT_BUILD_DIR $XRT_INSTALL_DIR

echo "Building IREE"
echo "============="
echo '{
    "version": 4,
    "cmakeMinimumRequired": {
      "major": 3,
      "minor": 23,
      "patch": 0
    },
    "include": [
        "build_tools/cmake/presets/all.json"
    ]
}' > $iree_dir/CMakeUserPresets.json

cd $iree_dir

cmake -S "$iree_dir" -B "$build_dir" \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DIREE_ENABLE_ASSERTIONS=ON \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DIREE_BUILD_BINDINGS_TFLITE=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
  -DIREE_INPUT_TOSA=OFF \
  -DIREE_INPUT_STABLEHLO=OFF \
  -DCMAKE_OBJECT_PATH_MAX=4096 \
  -DIREE_CMAKE_PLUGIN_PATHS=../iree-amd-aie \
  -DIREE_EXTERNAL_HAL_DRIVERS=xrt \
  -DXRT_DIR=$XRT_INSTALL_DIR/a/iree-amd-aie/iree-amd-aie/xrt-build/xilinx/xrt/share/cmake/XRT

echo "Building all"
echo "------------"
cmake --build "$build_dir" -- -k 0

echo "Installing"
echo "----------"
echo "Install to: $install_dir"
cmake --build "$build_dir" --target iree-install-dist

echo "CTest"
echo "-----"
ctest --test-dir "$build_dir" -R amd-aie --output-on-failure -j

# Show ccache stats.
ccache --show-stats
