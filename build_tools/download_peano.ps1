# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

$ErrorActionPreference = 'Stop'

$this_dir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
$RELEASE = (Get-Content -Path "$this_dir/peano_commit.txt")
pip download llvm_aie==$RELEASE -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
$peano = (Get-ChildItem -Filter llvm*.whl)
$new_name = ($peano.Basename + ".zip")
Rename-Item -Path $peano.Name -NewName $new_name
Expand-Archive $new_name -DestinationPath $PWD.Path
