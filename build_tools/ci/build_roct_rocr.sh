#!/bin/bash

set -eux -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/../.. && pwd)"

roct_dir="$(cd $repo_root/third_party/ROCT-Thunk-Interface && pwd)"
rocr_dir="$(cd $repo_root/third_party/ROCR-Runtime && pwd)"

build_roct_dir="$repo_root/roct-build"
roct_install_dir="$repo_root/roct-install"
mkdir -p "$build_roct_dir"
build_roct_dir="$(cd $build_roct_dir && pwd)"

build_rocr_dir="$repo_root/rocr-build"
rocr_install_dir="$repo_root/rocr-install"
mkdir -p "$build_rocr_dir"
build_rocr_dir="$(cd $build_rocr_dir && pwd)"

cache_dir="${cache_dir:-}"

if [ -z "${cache_dir}" ]; then
  cache_dir="${repo_root}/.build-cache"
  mkdir -p "${cache_dir}"
  cache_dir="$(cd ${cache_dir} && pwd)"
fi
echo "Caching to ${cache_dir}"
mkdir -p "${cache_dir}/ccache"

if [[ "$OSTYPE" == "msys"* ]]; then
  export CC=clang-cl.exe
  export CXX=clang-cl.exe
fi
export CCACHE_DIR="${cache_dir}/ccache"
export CCACHE_MAXSIZE="700M"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

cd $roct_dir
cmake -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$roct_install_dir" \
      -S "$roct_dir" -B "$build_roct_dir"
cmake --build "$build_roct_dir" --target install

cd $rocr_dir
cmake -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$rocr_install_dir" \
      -DCMAKE_PREFIX_PATH="$roct_install_dir" \
      -DIMAGE_SUPPORT=OFF \
      -S "$rocr_dir/src" -B "$build_rocr_dir"
cmake --build "$build_rocr_dir" --target install
