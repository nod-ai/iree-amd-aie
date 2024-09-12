#!/bin/bash
#
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -eux -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/.. && pwd)"
llvm_dir="$(cd $repo_root/third_party/iree/third_party/llvm-project/llvm && pwd)"
build_dir="$repo_root/llvm-build"
install_dir="$repo_root/llvm-install"
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
if [[ "$OSTYPE" == "linux"* ]]; then
  export CMAKE_TOOLCHAIN_FILE="$this_dir/linux_default_toolchain.cmake"
  export CC="${CC:-clang}"
  export CXX="${CXX:-clang++}"
fi

export CCACHE_DIR="${cache_dir}/ccache"
export CCACHE_MAXSIZE="700M"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CCACHE_SLOPPINESS=include_file_ctime,include_file_mtime,time_macros

# Clear ccache stats.
ccache -z

# https://discourse.cmake.org/t/yet-another-command-line-spaces-in-arguments-problem-is-this-really-2022/5829
CMAKE_ARGS=(
  -GNinja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="$install_dir"
  -DCMAKE_OBJECT_PATH_MAX=4096
  -DLLVM_INCLUDE_EXAMPLES=OFF
  -DLLVM_INCLUDE_TESTS=OFF
  -DLLVM_INCLUDE_BENCHMARKS=OFF
  -DLLVM_APPEND_VC_REV=OFF
  -DLLVM_ENABLE_ASSERTIONS=ON
  -DLLVM_ENABLE_IDE=ON
  -DLLVM_ENABLE_BINDINGS=OFF
  -DLLVM_ENABLE_LIBEDIT=OFF
  -DLLVM_ENABLE_LIBXML2=OFF
  -DLLVM_ENABLE_TERMINFO=OFF
  -DLLVM_ENABLE_ZLIB=OFF
  -DLLVM_ENABLE_ZSTD=OFF
  -DLLVM_FORCE_ENABLE_STATS=ON
  -DLLVM_INSTALL_UTILS=ON
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON
  -DLLVM_ENABLE_PROJECTS="mlir;clang;lld"
)

clang_llvm_tools_not_to_build="$this_dir/clang_llvm_tools_not_to_build.txt"
if [ -f "$clang_llvm_tools_not_to_build" ]; then
  IFS=$'\n'
  set +x
  for tool in `cat $clang_llvm_tools_not_to_build`; do
    CMAKE_ARGS+=("-D${tool}_BUILD=OFF")
  done
  set -x
fi

if [[ "$OSTYPE" == "linux"* ]]; then
  CMAKE_ARGS+=(
    -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld"
    -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld"
    -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld"
    -DCMAKE_C_COMPILER="${CC}"
    -DCMAKE_CXX_COMPILER="${CXX}"
    -DLLVM_TARGET_ARCH=X86
    -DLLVM_TARGETS_TO_BUILD=X86
    -S
    "$llvm_dir"
    -B
    "$build_dir"
  )
elif [[ "$OSTYPE" == "darwin"* ]]; then
  CMAKE_ARGS+=(
    -DLLVM_TARGET_ARCH="X86;AArch64"
    -DLLVM_TARGETS_TO_BUILD="X86;AArch64"
    -S
    "$llvm_dir"
    -B
    "$build_dir"
  )
fi

cmake "${CMAKE_ARGS[@]}"

echo "Building all"
echo "------------"
cmake --build "$build_dir" -- -k 0

echo "Installing"
echo "----------"
echo "Install to: $install_dir"
cmake --build "$build_dir" --target install
