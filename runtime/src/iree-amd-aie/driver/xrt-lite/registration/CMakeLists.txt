# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    registration
  HDRS
    "driver_module.h"
  SRCS
    "driver_module.c"
  DEPS
    iree::base
    iree::hal
    iree-amd-aie::driver::xrt-lite
  DEFINES
    "IREE_HAVE_HAL_XRT_LITE_DRIVER_MODULE=1"
  PUBLIC
)
