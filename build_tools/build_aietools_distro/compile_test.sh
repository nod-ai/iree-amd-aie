#!/bin/bash
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script tests whether the copy was successful by attempting to compile the adjacent main.cpp
# using the copied Vitis distro. It also serves the dual purpose of illustrating explicitly how to use
# xchesscc.

set -euo pipefail

AIETOOLS_DIR=${AIETOOLS_DIR:-Vitis/2024.1/aietools}
AIEARCH=aie_ml

export RDI_DATADIR=$AIETOOLS_DIR/data
export me_DIR=${RDI_DATADIR}/$AIEARCH/lib
export TCLLIBPATH=$AIETOOLS_DIR/tps/lnx64/target_$AIEARCH/chessdir/tcltk8.6/LNa64/lib
export PATH=$AIETOOLS_DIR/tps/lnx64/target_$AIEARCH/bin/LNa64bin:$PATH
export LD_LIBRARY_PATH=$AIETOOLS_DIR/lib/lnx64.o:$AIETOOLS_DIR/lnx64/tools/dot/lib
export XILINXD_LICENSE_FILE=${XILINXD_LICENSE_FILE:-$HOME/.Xilinx/aie.lic}

$AIETOOLS_DIR/bin/unwrapped/lnx64.o/xchesscc \
  -p me \
  -C Release_LLVM \
  -d \
  -f \
  +w chesswork \
  -c \
  +s \
  -v \
  -P ${RDI_DATADIR}/$AIEARCH/lib \
  main.cpp
