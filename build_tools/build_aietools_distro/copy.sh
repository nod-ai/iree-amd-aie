#!/bin/bash
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script creates a directory tree structure that emulates the directory structure inside Vitis.
# The script then copies the necessary executables (xchesscc, chess-clang, etc) and headers into the appropriate places
# within that directory structure. Note, one still needs to build_and_repair_wheel.sh after running this script.

set -euo pipefail

THIS_DIR="$(cd $(dirname $0) && pwd)"

SRC_AIETOOLS_DIR=${AIETOOLS_DIR:-/opt/Xilinx/Vitis/2024.1/aietools}
TARGET_AIETOOLS_DIR=$THIS_DIR/Vitis/2024.1/aietools

mkdir -p $TARGET_AIETOOLS_DIR

for AIE_ARCH in aie_ml aie2p; do
  mkdir -p $TARGET_AIETOOLS_DIR/bin/unwrapped/lnx64.o
  mkdir -p $TARGET_AIETOOLS_DIR/bin/unwrapped/lnx64.o/$AIE_ARCH
  mkdir -p $TARGET_AIETOOLS_DIR/data/$AIE_ARCH
  mkdir -p $TARGET_AIETOOLS_DIR/lib/lnx64.o
  mkdir -p $TARGET_AIETOOLS_DIR/tps/lnx64/target_$AIE_ARCH/bin/LNa64bin
  mkdir -p $TARGET_AIETOOLS_DIR/tps/lnx64/target_$AIE_ARCH/chessdir

  cp $SRC_AIETOOLS_DIR/bin/unwrapped/lnx64.o/xchesscc $TARGET_AIETOOLS_DIR/bin/unwrapped/lnx64.o
  cp $SRC_AIETOOLS_DIR/bin/unwrapped/lnx64.o/xca_udm_dbg $TARGET_AIETOOLS_DIR/bin/unwrapped/lnx64.o
  cp $SRC_AIETOOLS_DIR/bin/unwrapped/lnx64.o/$AIE_ARCH/ca_udm_dbg $TARGET_AIETOOLS_DIR/bin/unwrapped/lnx64.o/$AIE_ARCH

  cp -r $SRC_AIETOOLS_DIR/data/$AIE_ARCH/lib $TARGET_AIETOOLS_DIR/data/$AIE_ARCH
  cp -r $SRC_AIETOOLS_DIR/include $TARGET_AIETOOLS_DIR

  for e in bridge chess-backend chess-clang chess-llvm-link chesscc chesspe darts noodle tct_gcpp3; do
    cp $SRC_AIETOOLS_DIR/tps/lnx64/target_$AIE_ARCH/bin/LNa64bin/$e $TARGET_AIETOOLS_DIR/tps/lnx64/target_$AIE_ARCH/bin/LNa64bin
  done

  for d in clangdir tcltk8.6 ychessdir release_version; do
    cp -r $SRC_AIETOOLS_DIR/tps/lnx64/target_$AIE_ARCH/chessdir/$d $TARGET_AIETOOLS_DIR/tps/lnx64/target_$AIE_ARCH/chessdir
  done

  cp $SRC_AIETOOLS_DIR/lib/lnx64.o/libpython3.8.so.1.0 $TARGET_AIETOOLS_DIR/lib/lnx64.o

  sed -i "s/package require -exact Tcl 8.6.11/#package require -exact Tcl 8.6.11/g" $TARGET_AIETOOLS_DIR/tps/lnx64/target_$AIE_ARCH/chessdir/tcltk8.6/LNa64/lib/tcl8.6/init.tcl
done
