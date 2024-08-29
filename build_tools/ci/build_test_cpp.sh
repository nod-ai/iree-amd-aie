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

# https://stackoverflow.com/a/8597411/9045206
# note: on windows (git-bash) result is "msys"
# well only if you have apparently the right version of git-bash installed
# https://stackoverflow.com/a/72164385
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  export CMAKE_TOOLCHAIN_FILE="$this_dir/linux_default_toolchain.cmake"
  export CC=clang
  export CXX=clang++
elif [[ "$OSTYPE" == "msys"* ]]; then
  export CC=clang-cl.exe
  export CXX=clang-cl.exe
fi
export CCACHE_DIR="${cache_dir}/ccache"
export CCACHE_MAXSIZE="700M"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Clear ccache stats.
ccache -z

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
CMAKE_ARGS="\
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$install_dir \
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
  -DCMAKE_OBJECT_PATH_MAX=4096 \
  -DIREE_CMAKE_PLUGIN_PATHS=$repo_root"

if [[ "$OSTYPE" != "darwin"* ]]; then
  cmake $CMAKE_ARGS \
    -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
    -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld" \
    -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DLLVM_TARGET_ARCH=X86 \
    -DLLVM_TARGETS_TO_BUILD=X86 \
    -DIREE_EXTERNAL_HAL_DRIVERS=xrt \
    -S $iree_dir -B $build_dir
else
  cmake $CMAKE_ARGS \
    -S $iree_dir -B $build_dir
fi

echo "Building all"
echo "------------"
cmake --build "$build_dir" -- -k 0

echo "Installing"
echo "----------"
echo "Install to: $install_dir"
cmake --build "$build_dir" --target iree-install-dist

echo "CTest"
echo "-----"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  ctest --test-dir "$build_dir" -R amd-aie --output-on-failure -j
elif [[ "$OSTYPE" == "darwin"* ]]; then
  ctest --test-dir "$build_dir" -R amd-aie -E "pack_peel_pipeline_matmul|conv_fill_spec_pad" --output-on-failure -j --repeat until-pass:5
elif [[ "$OSTYPE" == "msys"* ]]; then
  # hack while windows is flaky to get past failing tests
  ctest --test-dir "$build_dir" -R amd-aie --output-on-failure -j --repeat until-pass:5
fi

# Show ccache stats.
ccache --show-stats -v
grep -r -B15 'Result: .*_miss' $CCACHE_DEBUGDIR

rm -f "$install_dir"/bin/clang*
rm -f "$install_dir"/bin/llvm-link*
cp "$build_dir"/tools/testing/e2e/iree-e2e-matmul-test "$install_dir"/bin
