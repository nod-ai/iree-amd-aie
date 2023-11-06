# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_AMD_AIE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

# TODO: Enable once turnkey.
option(IREE_AMD_AIE_ENABLE_XRT_DRIVER "Builds the XRT HAL driver" OFF)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/runtime/src AMD-AIE)
