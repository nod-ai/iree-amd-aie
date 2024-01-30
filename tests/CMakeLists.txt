# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT PEANO_PATH)
    message(STATUS "AIE e2e tests will not be run because peano install path was not provided Use -DPEANO_PATH=/path/to/peano/install")
endif()

if(NOT MLIRAIE_PATH)
    message(STATUS "AIE e2e tests will not be run because mlir-aie install path was not provided Use -DMLIRAIE_PATH=/path/to/mlir-aie/install")
endif()

if(NOT VITIS_PATH)
    message(STATUS "AIE e2e tests will not be run because VITIS install path was not provided Use -DVITIS_PATH=/path/to/vitis/install")
endif()

if(NOT IREE_AMD_AIE_ENABLE_XRT_DRIVER)
    message(STATUS "AIE e2e tests will not be run because XRT driver is not enabled")
endif()

if(PEANO_PATH AND MLIRAIE_PATH AND VITIS_PATH AND IREE_AMD_AIE_ENABLE_XRT_DRIVER)
    message(STATUS "IREE_AMDAIE_TESTS::tests/CMakeLists.txt")
    add_subdirectory(matmul)
endif()
