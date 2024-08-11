#!/bin/bash

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

SRC_AIETOOLS_DIR=${AIETOOLS_DIR:-/opt/Xilinx/Vitis/2024.1/aietools}

python setup.py bdist_wheel
export LD_LIBRARY_PATH=$SRC_AIETOOLS_DIR/lib/lnx64.o:$SRC_AIETOOLS_DIR/lnx64/tools/dot/lib
python repair.py
