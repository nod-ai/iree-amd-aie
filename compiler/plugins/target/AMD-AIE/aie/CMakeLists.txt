# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::target::amd-aie::aie")

###############################################################################
# AIE Dialect
###############################################################################

iree_tablegen_library(
  NAME
    AIEAttrsGen
  TD_FILE
    AIEAttrs.td
  OUTS
    -gen-attrdef-decls AIEAttrs.h.inc
    -gen-attrdef-defs AIEAttrs.cpp.inc
    -gen-enum-decls AIEEnums.h.inc
    -gen-enum-defs AIEEnums.cpp.inc
)

iree_tablegen_library(
  NAME
    AIETypesGen
  TD_FILE
    AIETypes.td
  OUTS
    -gen-typedef-decls AIETypes.h.inc
    -gen-typedef-defs AIETypes.cpp.inc
)

iree_tablegen_library(
  NAME
    AIEDialectGen
  TD_FILE
    AIE.td
  OUTS
    -gen-dialect-decls AIEDialect.h.inc
    -gen-dialect-defs AIEDialect.cpp.inc
)

list(APPEND IREE_COMPILER_TABLEGEN_INCLUDE_DIRS
  ${IREE_AMD_AIE_RUNTIME_SOURCE_DIR})
iree_tablegen_library(
  NAME
    AIEOpsGen
  TD_FILE
    AIEOps.td
  OUTS
    -gen-op-decls AIEOps.h.inc
    -gen-op-defs AIEOps.cpp.inc
)

iree_tablegen_library(
  NAME
    AIENormalizeAddressSpacesGen
  TD_FILE
    AIENormalizeAddressSpaces.td
  OUTS
    -gen-rewriters AIENormalizeAddressSpaces.inc
)

iree_cc_library(
  NAME
    AIEDialectIR
  SRCS
    AIEDialect.cpp
  DEPS
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
    ::AIEAttrsGen
    ::AIEDialectGen
    ::AIEOpsGen
    ::AIETypesGen
    # mlir::DataLayout::closest(mlir::Operation*)
    MLIRDataLayoutInterfaces
    # mlir::OffsetSizeAndStrideOpInterface::getOffsets()
    MLIRViewLikeInterface
    # mlir::arith::ConstantOp::getValue()
    MLIRArithDialect
    # mlir::detail::TypeIDResolver<mlir::func::FuncOp, void>::id
    MLIRFuncDialect
    # mlir::detail::TypeIDResolver<mlir::memref::AllocOp, void>::id
    MLIRMemRefDialect
    MLIRIR
    MLIREmitCDialect
)

###############################################################################
# AIEX Dialect
###############################################################################

iree_tablegen_library(
  NAME
    AIEXDialectGen
  TD_FILE
    AIEX.td
  OUTS
    -gen-dialect-decls -dialect=aiex AIEXDialect.h.inc
    -gen-dialect-defs -dialect=aiex AIEXDialect.cpp.inc
)

iree_tablegen_library(
  NAME
    AIEXOpsGen
  TD_FILE
    AIEX.td
  OUTS
    -gen-op-decls AIEX.h.inc
    -gen-op-defs AIEX.cpp.inc
)

iree_cc_library(
  NAME
    AIEXDialectIR
  SRCS
    AIEXDialect.cpp
  DEPS
    ::AIEDialectIR
    ::AIEXOpsGen
    ::AIEXDialectGen
    MLIRIR
)

###############################################################################
# in-tree AIE and AIEX passes
###############################################################################

iree_cc_library(
  NAME
    AIEPasses
  SRCS
    AMDAIEAssignBufferAddressesBasic.cpp
    AMDAIEAssignBufferDescriptorIDs.cpp
    AMDAIECoreToStandard.cpp
    AMDAIECreatePathFindFlows.cpp
    AMDAIEDmaToNpu.cpp
    AMDAIELocalizeLocks.cpp
    AMDAIENormalizeAddressSpaces.cpp
  DEPS
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
    ::AIEDialectIR
    ::AIEXDialectIR
    ::AIENormalizeAddressSpacesGen
    iree::target::amd-aie::Utils::Utils
)

add_subdirectory(test)
