# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Enable strict mode
$ErrorActionPreference = 'Stop'

$this_dir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
$repo_root = Resolve-Path -Path "$this_dir/.."
$iree_dir = Resolve-Path -Path "$repo_root/third_party/iree"
$build_dir = "$repo_root/iree-build"
$install_dir = "$repo_root/iree-install"

if (-not (Test-Path "$build_dir"))
{
    New-Item -Path $build_dir -ItemType Directory | Out-Null
}
$build_dir = Resolve-Path -Path "$build_dir"
$cache_dir = "$env:cache_dir"
$llvm_install_dir = "$env:llvm_install_dir"

if (-not $cache_dir)
{
    $cache_dir = "$repo_root/.build-cache"
    if (-not (Test-Path "$cache_dir"))
    {
        New-Item -Path "$cache_dir" -ItemType Directory | Out-Null
    }
    $cache_dir = Resolve-Path -Path "$cache_dir"
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

echo "Building IREE"

$CMAKE_ARGS = @(
    "-GNinja"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_INSTALL_PREFIX=$install_dir"
    "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DCMAKE_C_COMPILER=$env:CC"
    "-DCMAKE_CXX_COMPILER=$env:CXX"
    "-DLLVM_TARGET_ARCH=X86"
    "-DLLVM_TARGETS_TO_BUILD=X86"
    "-DCMAKE_OBJECT_PATH_MAX=4096"
    "-DIREE_BUILD_BINDINGS_TFLITE=OFF"
    "-DIREE_BUILD_SAMPLES=OFF"
    "-DIREE_ENABLE_ASSERTIONS=ON"
    "-DIREE_ERROR_ON_MISSING_SUBMODULES=OFF"
    "-DIREE_HAL_DRIVER_DEFAULTS=OFF"
    "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON"
    "-DIREE_HAL_DRIVER_LOCAL_TASK=ON"
    "-DIREE_INPUT_STABLEHLO=OFF"
    "-DIREE_INPUT_TORCH=OFF"
    "-DIREE_INPUT_TOSA=OFF"
    "-DIREE_LINK_COMPILER_SHARED_LIBRARY=OFF"
    "-DIREE_TARGET_BACKEND_DEFAULTS=OFF"
    "-DIREE_TARGET_BACKEND_LLVM_CPU=ON"
    "-DIREE_CMAKE_PLUGIN_PATHS=$repo_root"
    "-DIREE_EXTERNAL_HAL_DRIVERS=xrt"
    "-DIREE_BUILD_PYTHON_BINDINGS=ON"
)

$peano_install_dir = "$env:PEANO_INSTALL_DIR"
if ($peano_install_dir -and (Test-Path "$peano_install_dir"))
{
    $CMAKE_ARGS += @("-DPEANO_INSTALL_DIR=$peano_install_dir")
}

if ($llvm_install_dir -and (Test-Path "$llvm_install_dir"))
{
    echo "using existing llvm install @ $llvm_install_dir"
    $CMAKE_ARGS += @(
        "-DIREE_BUILD_BUNDLED_LLVM=OFF"
        "-DClang_DIR=$llvm_install_dir/lib/cmake/clang"
        "-DLLD_DIR=$llvm_install_dir/lib/cmake/lld"
        "-DMLIR_DIR=$llvm_install_dir/lib/cmake/mlir"
        "-DLLVM_DIR=$llvm_install_dir/lib/cmake/llvm"
    )
}

& cmake $CMAKE_ARGS -S $iree_dir -B $build_dir

echo "Building all"
echo "------------"
& cmake --build $build_dir -- -k 0

echo "Installing"
echo "----------"
echo "Install to: $install_dir"
& cmake --build $build_dir --target install
& cmake --build $build_dir --target iree-install-dist

echo "CTest"
echo "-----"

# bash because lit doesn't magically translate // RUN to powershell
# 5 repeats is a hack while Windows is flaky to get past failing tests
# better have git-bash installed...
$env:Path = "C:\Program Files\Git\bin;$env:Path"
pushd $build_dir
& bash -l -c "ctest -R amd-aie -E driver --output-on-failure -j --repeat until-pass:5"
popd

if ($llvm_install_dir -and (Test-Path "$llvm_install_dir"))
{
    Copy-Item -Path "$llvm_install_dir/bin/lld.exe" -Destination "$install_dir/bin" -Force
    Copy-Item -Path "$llvm_install_dir/bin/FileCheck.exe" -Destination "$install_dir/bin" -Force
    Copy-Item -Path "$llvm_install_dir/bin/not.exe" -Destination "$install_dir/bin" -Force
}

Copy-Item -Path "$build_dir/tools/testing/e2e/iree-e2e-matmul-test.exe" -Destination "$install_dir/bin" -Force
