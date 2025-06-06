# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::target::amd-aie::IR")
iree_add_all_subdirs()

iree_cc_library(
  NAME
    AMDAIEDialect
  HDRS
    "AMDAIEAttrs.h"
    "AMDAIEDialect.h"
    "AMDAIELogicalObjFifoOpInterface.h"
    "AMDAIEOps.h"
    "AMDAIETypes.h"
    "AMDAIETraits.h"
  TEXTUAL_HDRS
    "AMDAIEAttrs.cpp.inc"
    "AMDAIEAttrs.h.inc"
    "AMDAIEDialect.cpp.inc"
    "AMDAIEDialect.h.inc"
    "AMDAIEEnums.cpp.inc"
    "AMDAIEEnums.h.inc"
    "AMDAIETypes.cpp.inc"
    "AMDAIETypes.h.inc"
    "AMDAIEOps.cpp.inc"
    "AMDAIEOps.h.inc"
    "AMDAIEDmaOpInterface.cpp.inc"
    "AMDAIEDmaOpInterface.h.inc"
    "AMDAIELogicalObjFifoOpInterface.h.inc"
    "AMDAIELogicalObjFifoOpInterface.cpp.inc"
  SRCS
    "AMDAIEAttrs.cpp"
    "AMDAIEDmaOpInterface.cpp"
    "AMDAIEDialect.cpp"
    "AMDAIELogicalObjFifoOpInterface.cpp"
    "AMDAIEOps.cpp"
    "AMDAIETypes.cpp"
  DEPS
    ::AMDAIEDialectGen
    ::AMDAIEDmaOpInterfaceGen
    ::AMDAIELogicalObjFifoOpInterface
    ::AMDAIEOpsGen
    ::AMDAIETypesGen
    ::AMDAIEAttrsGen
    iree-amd-aie::aie_runtime::AMDAIEEnums
    LLVMSupport
    MLIRArithUtils
    MLIRCopyOpInterface
    MLIRDialectUtils
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRViewLikeInterface
  PUBLIC
)

iree_tablegen_library(
  NAME
    AMDAIEDialectGen
  TD_FILE
    "AMDAIEDialect.td"
  OUTS
    --gen-dialect-decls AMDAIEDialect.h.inc
    --gen-dialect-defs AMDAIEDialect.cpp.inc
)

iree_tablegen_library(
  NAME
    AMDAIEAttrsGen
  TD_FILE
    "AMDAIEAttrs.td"
  OUTS
    --gen-attrdef-decls --attrdefs-dialect=amdaie AMDAIEAttrs.h.inc
    --gen-attrdef-defs --attrdefs-dialect=amdaie AMDAIEAttrs.cpp.inc
    --gen-enum-decls AMDAIEEnums.h.inc
    --gen-enum-defs AMDAIEEnums.cpp.inc
)


iree_tablegen_library(
  NAME
    AMDAIETypesGen
  TD_FILE
    "AMDAIETypes.td"
  OUTS
    -gen-typedef-decls AMDAIETypes.h.inc
    -gen-typedef-defs AMDAIETypes.cpp.inc
)

list(APPEND IREE_COMPILER_TABLEGEN_INCLUDE_DIRS
            ${IREE_AMD_AIE_RUNTIME_SOURCE_DIR})
iree_tablegen_library(
  NAME
    AMDAIEOpsGen
  TD_FILE
    "AMDAIEOps.td"
  OUTS
    -gen-op-decls AMDAIEOps.h.inc
    -gen-op-defs AMDAIEOps.cpp.inc
)

iree_tablegen_library(
  NAME
    AMDAIEDmaOpInterfaceGen
  TD_FILE
    "AMDAIEDmaOpInterface.td"
  OUTS
    --gen-op-interface-decls AMDAIEDmaOpInterface.h.inc
    --gen-op-interface-defs AMDAIEDmaOpInterface.cpp.inc
)

iree_tablegen_library(
  NAME
    AMDAIELogicalObjFifoOpInterface
  TD_FILE
    "AMDAIELogicalObjFifoOpInterface.td"
  OUTS
    --gen-op-interface-decls AMDAIELogicalObjFifoOpInterface.h.inc
    --gen-op-interface-defs AMDAIELogicalObjFifoOpInterface.cpp.inc
)
