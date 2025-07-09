#!/bin/bash
#
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

this_dir="$(cd $(dirname $0) && pwd)"
RELEASE=$(cat $this_dir/peano_commit_linux.txt)
pip install llvm_aie==$RELEASE --upgrade --target $PWD -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
