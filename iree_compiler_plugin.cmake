# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(IREE_AMD_AIE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/compiler/plugins/target/AMD-AIE target/AMD-AIE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/experimental AMD-AIE-experimental)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tools/plugins AMD-AIE/tools)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests/samples AMD-AIE/tests/samples)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests/OPT/failing_tests AMD-AIE/tests/OPT/failing_tests)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests/transform_dialect AMD-AIE/tests/transform_dialect)
