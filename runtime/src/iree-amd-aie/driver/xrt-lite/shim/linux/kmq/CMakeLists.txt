# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    kmq
  SRCS
    amdxdna_accel.h
    bo.cpp
    bo.h
    device.cpp
    device.h
    fence.cpp
    fence.h
    hwctx.cpp
    hwctx.h
    hwq.cpp
    hwq.h
    pcidev.cpp
    pcidev.h
    pcidrv.cpp
    pcidrv.h
    shared.h
    shim_debug.cpp
    shim_debug.h
  PUBLIC
)
