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
& bash -l -c "ctest -R amd-aie --output-on-failure -j --repeat until-pass:5"
popd

if ($llvm_install_dir -and (Test-Path "$llvm_install_dir"))
{
    Copy-Item -Path "$llvm_install_dir/bin/lld.exe" -Destination "$install_dir/bin" -Force
    Copy-Item -Path "$llvm_install_dir/bin/FileCheck.exe" -Destination "$install_dir/bin" -Force
    Copy-Item -Path "$llvm_install_dir/bin/not.exe" -Destination "$install_dir/bin" -Force
}

Copy-Item -Path "$build_dir/tools/testing/e2e/iree-e2e-matmul-test.exe" -Destination "$install_dir/bin" -Force
Copy-Item -Path "$build_dir/tools/xrt_coreutil.dll" -Destination "$install_dir/python_packages/iree_runtime/iree/_runtime_libs" -Force
