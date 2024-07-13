# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::target::amd-aie::aievec")

list(APPEND IREE_COMPILER_TABLEGEN_INCLUDE_DIRS
    "${IREE_MLIR_AIE_SOURCE_DIR}/include")

iree_cc_library(
  NAME
    defs
  INCLUDES
    "${IREE_MLIR_AIE_SOURCE_DIR}/include"
)

###############################################################################
# AIEVec Dialect
###############################################################################

iree_tablegen_library(
  NAME
    AIEVecDialectGen
  TD_FILE
    AIEVecOps.td
  OUTS
    -gen-dialect-decls -dialect=aievec AIEVecOpsDialect.h.inc
    -gen-dialect-defs -dialect=aievec AIEVecOpsDialect.cpp.inc
    -gen-typedef-decls -typedefs-dialect=aievec AIEVecOpsTypes.h.inc
    -gen-typedef-defs -typedefs-dialect=aievec AIEVecOpsTypes.cpp.inc
)

iree_tablegen_library(
  NAME
    AIEVecAttrsGen
  TD_FILE
    AIEVecAttributes.td
  OUTS
    -gen-attrdef-decls -attrdefs-dialect=aievec AIEVecAttributes.h.inc
    -gen-attrdef-defs -attrdefs-dialect=aievec AIEVecAttributes.cpp.inc
    -gen-enum-decls AIEVecEnums.h.inc
    -gen-enum-defs AIEVecEnums.cpp.inc
)

iree_tablegen_library(
  NAME
    AIEVecOpsGen
  TD_FILE
    AIEVecOps.td
  OUTS
    -gen-op-decls AIEVecOps.h.inc
    -gen-op-defs AIEVecOps.cpp.inc
)

iree_tablegen_library(
  NAME
    AIEVecXLLVMOpsGen
  TD_FILE
    XLLVMOps.td
  OUTS
    -gen-dialect-decls -dialect=xllvm XLLVMDialect.h.inc
    -gen-dialect-defs -dialect=xllvm XLLVMDialect.cpp.inc
    -gen-op-decls XLLVMOps.h.inc
    -gen-op-defs XLLVMOps.cpp.inc
    -gen-llvmir-conversions XLLVMConversions.inc
)

iree_cc_library(
  NAME
    AIEVecDialectIR
  SRCS
    AIEVecOps.cpp
    AIEVecTypes.cpp
    XLLVMOps.cpp
  DEPS
    ::defs
    ::AIEVecOpsGen
    ::AIEVecDialectGen
    ::AIEVecAttrsGen
    MLIRIR
)

iree_cc_library(
  NAME
    AIEVecConvertToLLVM
  SRCS
    AIEVecToLLVM.cpp
    ConvertVectorToAIEVec.cpp
    CopyRemoval.cpp
    Utils.cpp
    VectorToAIEVecConversions.cpp
    VectorToVectorConversions.cpp
    XLLVMToLLVMIRTranslation.cpp
  DEPS
    MLIREmitCDialect
    ::defs
    ::AIEVecDialectIR
    ::AIEVecXLLVMOpsGen
)

add_subdirectory(test)
