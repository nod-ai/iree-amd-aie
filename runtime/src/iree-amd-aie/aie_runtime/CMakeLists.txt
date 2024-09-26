# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(Utils)

iree_tablegen_library(
  NAME
    AMDAIEEnumsGen
  TD_FILE
    "AMDAIEEnums.td"
  OUTS
    --gen-enum-decls AMDAIEEnums.h.inc
    --gen-enum-defs AMDAIEEnums.cpp.inc
)

iree_cc_library(
  NAME
    AMDAIEEnums
  HDRS
    "AMDAIEEnums.h"
  TEXTUAL_HDRS
    "AMDAIEEnums.cpp.inc"
    "AMDAIEEnums.h.inc"
  SRCS
    "AMDAIEEnums.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRSupport
  PUBLIC
)

# TODO(max): not sure why adding DEPS AMDAIEEnumsGen to AMDAIEEnums doesn't work...
add_dependencies(iree-amd-aie_aie_runtime_AMDAIEEnums
                 iree-amd-aie_aie_runtime_AMDAIEEnumsGen)

include(iree_aie_rt)

iree_cc_library(
  NAME
    iree_aie_runtime_static
  HDRS
    amsel_generator.h
    iree_aie_configure.h
    iree_aie_router.h
    iree_aie_runtime.h
    xaie_hwcfg.h
  SRCS
    amsel_generator.cc
    iree_aie_configure.cc
    iree_aie_router.cc
    iree_aie_runtime.cc
    mlir_aie_legacy.cc
    xaie_hwcfg.c
  INCLUDES
    "${_AIE_RT_BINARY_INCLUDE_DIR}"
    # For <elf.h>
    ${_BOOTGEN_SOURCE_DIR}
  DEPS
    ::AMDAIEEnums
  PUBLIC
)

# this one is private because this lib should wrap/encapsulate aie-rt
target_link_libraries(iree-amd-aie_aie_runtime_iree_aie_runtime_static
                      PRIVATE xaiengine)
# this one is public because it's safe and it's easier than forcing
# consumers (like tests) to link individually
target_link_libraries(iree-amd-aie_aie_runtime_iree_aie_runtime_static
                      PUBLIC LLVMSupport)

if (IREE_BUILD_TESTS)
  add_subdirectory(test)
endif()

