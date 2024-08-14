#!/bin/bash

set -eu -o errtrace

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

export CCACHE_DIR="${cache_dir}"
export CCACHE_MAXSIZE="2000M"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Clear ccache stats.
ccache -z

## Build XRT.
#XRT_BUILD_DIR=$repo_root/xrt-build
#XRT_INSTALL_DIR=$repo_root/xrt-install
#$this_dir/build_xrt.sh $XRT_BUILD_DIR $XRT_INSTALL_DIR

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

pip download -q mlir -f https://makslevental.github.io/wheels
unzip -q mlir*whl
pip install "numpy<2" pyyaml "pybind11[global]==2.10.4" nanobind

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
  -DIREE_INPUT_TORCH=OFF \
  -DIREE_CMAKE_PLUGIN_PATHS=../iree-amd-aie \
  -DCMAKE_OBJECT_PATH_MAX=4096 \
  -DBoost_INCLUDE_DIR=${BOOST_ROOT}/include\
  -DBoost_LIBRARY_DIRS=${BOOST_ROOT}/lib \
  -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF \
  -DIREE_BUILD_BUNDLED_LLVM=OFF \
  -DCMAKE_PREFIX_PATH=$PWD/mlir \
  -DPython3_EXECUTABLE=$(which python) \
  -DHAVE_STD_REGEX=ON \
  -DIREE_EMBED_ENABLE_WINDOWS_DLL_DECLSPEC=1 \
  -DIREE_EMBED_BUILDING_LIBRARY=1 \
  -DMLIR_CAPI_ENABLE_WINDOWS_DLL_DECLSPEC=1 \
  -DMLIR_CAPI_BUILDING_LIBRARY=1 \
  -DCMAKE_CXX_FLAGS="-DIREE_EMBED_ENABLE_WINDOWS_DLL_DECLSPEC=1 -DIREE_EMBED_BUILDING_LIBRARY=1 -DMLIR_CAPI_ENABLE_WINDOWS_DLL_DECLSPEC=1 -DMLIR_CAPI_BUILDING_LIBRARY=1 /EHsc" \
  -DCMAKE_C_FLAGS="-DIREE_EMBED_ENABLE_WINDOWS_DLL_DECLSPEC=1 -DIREE_EMBED_BUILDING_LIBRARY=1 -DMLIR_CAPI_ENABLE_WINDOWS_DLL_DECLSPEC=1 -DMLIR_CAPI_BUILDING_LIBRARY=1 /EHsc" \
  -DPYBIND11_FINDPYTHON=ON

# Unknown CMake command "python_add_library" -> -DPYBIND11_FINDPYTHON=ON
#  -DIREE_EXTERNAL_HAL_DRIVERS=xrt \
#  -DXRT_DIR=$XRT_INSTALL_DIR/opt/xilinx/xrt/share/cmake/XRT

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
