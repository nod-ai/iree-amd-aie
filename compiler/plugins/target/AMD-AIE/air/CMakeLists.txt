# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::target::amd-aie::air")

list(APPEND IREE_COMPILER_TABLEGEN_INCLUDE_DIRS
    "${IREE_MLIR_AIR_SOURCE_DIR}/include")

iree_cc_library(
  NAME
    defs
  INCLUDES
    "${IREE_MLIR_AIR_SOURCE_DIR}/include"
)

###############################################################################
# AIR Dialect
###############################################################################

iree_cc_library(
    NAME
        AIRDialectIR
    SRCS
      ${IREE_MLIR_AIR_SOURCE_DIR}/lib/Dialect/AIR/IR/AIRDialect.cpp
    DEPS
      ::defs
      ::AIRDialectGen
      ::AIRInterfaceGen
      MLIRIR
)

iree_tablegen_library(
  NAME
    AIRDialectGen
  TD_FILE
    "${IREE_MLIR_AIR_SOURCE_DIR}/include/air/Dialect/AIR/AIR.td"
  OUTS
    -gen-op-decls Dialect/AIR/AIR.h.inc
    -gen-op-defs Dialect/AIR/AIR.cpp.inc
    -gen-typedef-decls Dialect/AIR/AIRTypes.h.inc
    -gen-typedef-defs Dialect/AIR/AIRTypes.cpp.inc
    -gen-dialect-decls Dialect/AIR/AIRDialect.h.inc --dialect=air
    -gen-dialect-defs Dialect/AIR/AIRDialect.cpp.inc --dialect=air
    -gen-enum-decls Dialect/AIR/AIREnums.h.inc
    -gen-enum-defs Dialect/AIR/AIREnums.cpp.inc
)

iree_tablegen_library(
  NAME
    AIRInterfaceGen
  TD_FILE
    "${IREE_MLIR_AIR_SOURCE_DIR}/include/air/Dialect/AIR/AIROpBase.td"
  OUTS
    -gen-op-interface-decls Dialect/AIR/AIROpInterfaces.h.inc
    -gen-op-interface-defs Dialect/AIR/AIROpInterfaces.cpp.inc
)
