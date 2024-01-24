#!/bin/bash

set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/../.. && pwd)"
iree_dir="$(cd $repo_root/../iree && pwd)"
build_dir="$repo_root/../iree-build"
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

export CMAKE_TOOLCHAIN_FILE="$this_dir/linux_default_toolchain.cmake"
export CC=clang
export CXX=clang++
export CCACHE_DIR="${cache_dir}/ccache"
export CCACHE_MAXSIZE="700M"
export CCACHE_DEBUG=1
export CCACHE_LOGFILE="${cache_dir}/ccache.log"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Clear ccache stats.
ccache -z

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
cmake --preset new-linux-minimal -B "$build_dir" \
  -DIREE_CMAKE_PLUGIN_PATHS=../iree-amd-aie \
  -DIREE_BUILD_COMPILER=OFF # DO NOT SUBMIT.

echo "Building all"
echo "------------"
cmake --build "$build_dir" -- -k 0

echo "CTest"
echo "-----"
ctest --test-dir "$build_dir" -R amd-aie

# Show ccache stats.
ccache --show-stats

cat $CCACHE_LOGFILE
