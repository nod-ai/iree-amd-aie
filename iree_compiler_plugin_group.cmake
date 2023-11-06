# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_AMD_AIE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/compiler/plugins/target/AMD-AIE target/AMD-AIE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tools/plugins AMD-AIE/tools)
if(ADD_XRT_RUNTIME)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/runtime/plugins/XRT XRT)
endif()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/experimental xrt)


