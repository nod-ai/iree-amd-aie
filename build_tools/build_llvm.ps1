# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Enable strict mode
$ErrorActionPreference = 'Stop'

$this_dir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
$repo_root = Resolve-Path -Path "$this_dir/.."
$llvm_dir = Resolve-Path -Path "$repo_root/third_party/iree/third_party/llvm-project/llvm"
echo "llvm_dir $llvm_dir"
$build_dir = "$repo_root/llvm-build"
echo "build_dir $build_dir"
$install_dir = "$repo_root/llvm-install"

if (-not (Test-Path "$build_dir"))
{
    New-Item -Path $build_dir -ItemType Directory | Out-Null
}
$build_dir = Resolve-Path -Path $build_dir
$cache_dir = $env:cache_dir

if (-not $cache_dir)
{
    $cache_dir = "$repo_root/.build-cache"
    if (-not (Test-Path "$cache_dir"))
    {
        New-Item -Path $cache_dir -ItemType Directory | Out-Null
    }
    $cache_dir = Resolve-Path -Path $cache_dir
}
echo "Caching to $cache_dir"

if (-not (Test-Path "$cache_dir/ccache"))
{
    New-Item -Path "$cache_dir/ccache" -ItemType Directory | Out-Null
}
if (-not (Test-Path "$cache_dir/pip"))
{
    New-Item -Path "$cache_dir/pip" -ItemType Directory | Out-Null
}

$python = (Get-Command python -ErrorAction SilentlyContinue).Source
echo "Using python: $python"

$env:CC = 'clang-cl.exe'
$env:CXX = 'clang-cl.exe'
$env:CCACHE_DIR = "$cache_dir/ccache"
$env:CCACHE_MAXSIZE = '700M'
$env:CMAKE_C_COMPILER_LAUNCHER = 'ccache'
$env:CMAKE_CXX_COMPILER_LAUNCHER = 'ccache'
$env:CCACHE_SLOPPINESS = 'include_file_ctime,include_file_mtime,time_macros'

& ccache -z

# on windows python bindings don't for split build because
# i can't figure out MLIR_CAPI_EXPORTED and MLIR_CAPI_BUILDING_LIBRARY
# which somehow disables exceptions (blocking bootgen and xrt)

$CMAKE_ARGS = @(
  "-GNinja"
  "-DCMAKE_BUILD_TYPE=Release"
  "-DCMAKE_INSTALL_PREFIX=$install_dir"
  "-DCMAKE_OBJECT_PATH_MAX=4096"
  "-DCMAKE_EXE_LINKER_FLAGS_INIT=-fuse-ld=lld"
  "-DCMAKE_SHARED_LINKER_FLAGS_INIT=-fuse-ld=lld"
  "-DCMAKE_MODULE_LINKER_FLAGS_INIT=-fuse-ld=lld"
  "-DCMAKE_C_COMPILER=$env:CC"
  "-DCMAKE_CXX_COMPILER=$env:CXX"
  "-DCMAKE_C_FLAGS=-DMLIR_CAPI_ENABLE_WINDOWS_DLL_DECLSPEC=1 -DMLIR_CAPI_BUILDING_LIBRARY=1"
  "-DCMAKE_CXX_FLAGS=-DMLIR_CAPI_ENABLE_WINDOWS_DLL_DECLSPEC=1 -DMLIR_CAPI_BUILDING_LIBRARY=1 -D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING=1"
  "-DLLVM_INCLUDE_EXAMPLES=OFF"
  "-DLLVM_INCLUDE_TESTS=OFF"
  "-DLLVM_INCLUDE_BENCHMARKS=OFF"
  "-DLLVM_APPEND_VC_REV=OFF"
  "-DLLVM_ENABLE_ASSERTIONS=ON"
  "-DLLVM_ENABLE_IDE=ON"
  "-DLLVM_ENABLE_BINDINGS=OFF"
  "-DLLVM_ENABLE_LIBEDIT=OFF"
  "-DLLVM_ENABLE_LIBXML2=OFF"
  "-DLLVM_ENABLE_TERMINFO=OFF"
  "-DLLVM_ENABLE_ZLIB=OFF"
  "-DLLVM_ENABLE_ZSTD=OFF"
  "-DLLVM_FORCE_ENABLE_STATS=ON"
  "-DLLVM_INSTALL_UTILS=ON"
  "-DMLIR_ENABLE_BINDINGS_PYTHON=OFF"
  "-DLLVM_ENABLE_PROJECTS=mlir;clang;lld"
  "-DLLVM_TARGET_ARCH=X86"
  "-DLLVM_TARGETS_TO_BUILD=X86"
  "-S"
  "$llvm_dir"
  "-B"
  "$build_dir"
)


$clang_llvm_tools_not_to_build="$this_dir/clang_llvm_tools_not_to_build.txt"
if (Test-Path "$clang_llvm_tools_not_to_build")
{
    Get-Content -Path $clang_llvm_tools_not_to_build | ForEach-Object {
       $CMAKE_ARGS += "-D${_}_BUILD=OFF"
    }
}

& cmake $CMAKE_ARGS

echo "Building all"
echo "------------"
cmake --build $build_dir -- -k 0

echo "Installing"
echo "----------"
echo "Install to: $install_dir"
cmake --build $build_dir --target install
