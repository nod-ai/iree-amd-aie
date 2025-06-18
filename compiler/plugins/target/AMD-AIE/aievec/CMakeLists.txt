# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::target::amd-aie::aievec")

###############################################################################
# AIEVec Dialect
###############################################################################


iree_tablegen_library(
  NAME
    AIEVecAttrsGen
  TD_FILE
    AIEVecAttributes.td
  OUTS
    -gen-attrdef-decls -attrdefs-dialect=aievec AIEVecAttributes.h.inc
    -gen-attrdef-defs -attrdefs-dialect=aievec AIEVecAttributes.cpp.inc
    -gen-attrdef-doc AIEVecAttributes.md
    -gen-enum-decls AIEVecEnums.h.inc
    -gen-enum-defs AIEVecEnums.cpp.inc
    -gen-enum-doc AIEVecEnums.md
)

list(APPEND IREE_COMPILER_TABLEGEN_INCLUDE_DIRS
  ${IREE_AMD_AIE_RUNTIME_SOURCE_DIR})
iree_tablegen_library(
  NAME
    AIEVecDialectGen
  TD_FILE
    AIEVecDialect.td
  OUTS
    -gen-dialect-decls -dialect=aievec AIEVecDialect.h.inc
    -gen-dialect-defs -dialect=aievec AIEVecDialect.cpp.inc
    -gen-dialect-doc -dialect=aievec AIEVecDialect.md
)

iree_tablegen_library(
  NAME
    AIEVecOpsGen
  TD_FILE
    AIEVecOps.td
  OUTS
    -gen-op-decls AIEVecOps.h.inc
    -gen-op-defs AIEVecOps.cpp.inc
    -gen-op-doc AIEVecOps.md
)

iree_tablegen_library(
  NAME
    AIEVecXLLVMOpsGen
  TD_FILE
    XLLVMOps.td
  OUTS
    -gen-dialect-decls -dialect=xllvm XLLVMDialect.h.inc
    -gen-dialect-defs -dialect=xllvm XLLVMDialect.cpp.inc
    -gen-dialect-doc -dialect=xllvm XLLVMDialect.md
    -gen-op-decls XLLVMOps.h.inc
    -gen-op-defs XLLVMOps.cpp.inc
    -gen-op-doc XLLVMOps.md
    -gen-llvmir-conversions XLLVMConversions.inc
)

iree_cc_library(
  NAME
    AIEVecDialectIR
  SRCS
    AIEVecOps.cpp
    XLLVMOps.cpp
  DEPS
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
    ::AIEVecOpsGen
    ::AIEVecDialectGen
    ::AIEVecAttrsGen
    ::AIEVecXLLVMOpsGen
    MLIRIR
)

iree_cc_library(
  NAME
    AIEVecConvertToLLVM
  SRCS
    AIEVecToLLVM.cpp
    ConvertVectorToAIEVec.cpp
    VectorToAIEVecConversions.cpp
    VectorToVectorConversions.cpp
    XLLVMToLLVMIRTranslation.cpp
  DEPS
    MLIREmitCDialect
    ::AIEVecDialectIR
    ::AIEVecXLLVMOpsGen
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::HAL::IR::HALDialect
)

add_subdirectory(test)
