#!/bin/bash
#
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

RELEASE=19.0.0.2024092601+562ccea2
pip download llvm_aie==$RELEASE -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
unzip llvm_aie*whl
