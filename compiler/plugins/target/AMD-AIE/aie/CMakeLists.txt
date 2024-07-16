# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::target::amd-aie::aie")

list(APPEND IREE_COMPILER_TABLEGEN_INCLUDE_DIRS
    "${IREE_MLIR_AIE_SOURCE_DIR}/include")

iree_cc_library(
  NAME
    defs
  INCLUDES
    "${IREE_MLIR_AIE_SOURCE_DIR}/include"
)

###############################################################################
# AIE Dialect
###############################################################################

iree_tablegen_library(
  NAME
    AIEAttrsGen
  TD_FILE
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIE/IR/AIEAttrs.td"
  OUTS
    -gen-attrdef-decls Dialect/AIE/IR/AIEAttrs.h.inc
    -gen-attrdef-defs Dialect/AIE/IR/AIEAttrs.cpp.inc
    -gen-enum-decls Dialect/AIE/IR/AIEEnums.h.inc
    -gen-enum-defs Dialect/AIE/IR/AIEEnums.cpp.inc
)

iree_tablegen_library(
  NAME
    AIEDialectGen
  TD_FILE
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIE/IR/AIE.td"
  OUTS
    -gen-dialect-decls Dialect/AIE/IR/AIEDialect.h.inc
    -gen-dialect-defs Dialect/AIE/IR/AIEDialect.cpp.inc
)

iree_tablegen_library(
  NAME
    AIEInterfacesGen
  TD_FILE
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIE/IR/AIEInterfaces.td"
  OUTS
    -gen-op-interface-decls Dialect/AIE/IR/AIEInterfaces.h.inc
    -gen-op-interface-defs Dialect/AIE/IR/AIEInterfaces.cpp.inc
)

iree_tablegen_library(
  NAME
    AIEOpsGen
  TD_FILE
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIE/IR/AIEOps.td"
  OUTS
    -gen-op-decls Dialect/AIE/IR/AIEOps.h.inc
    -gen-op-defs Dialect/AIE/IR/AIEOps.cpp.inc
)

iree_tablegen_library(
  NAME
    AIETypesGen
  TD_FILE
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIE/IR/AIETypes.td"
  OUTS
    -gen-typedef-decls -typedefs-dialect=AIE Dialect/AIE/IR/AIETypes.h.inc
    -gen-typedef-defs -typedefs-dialect=AIE Dialect/AIE/IR/AIETypes.cpp.inc
)

iree_cc_library(
  NAME
    AIEDialectIR
  SRCS
    ${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIE/IR/AIEDialect.cpp
    ${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIE/IR/AIETargetModel.cpp
  DEPS
    ::defs
    ::AIEAttrsGen
    ::AIEDialectGen
    ::AIEInterfacesGen
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
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIEX/IR/AIEX.td"
  OUTS
    -gen-dialect-decls -dialect=aiex Dialect/AIEX/IR/AIEXDialect.h.inc
    -gen-dialect-defs -dialect=aiex Dialect/AIEX/IR/AIEXDialect.cpp.inc
)

iree_tablegen_library(
  NAME
    AIEXOpsGen
  TD_FILE
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIEX/IR/AIEX.td"
  OUTS
    -gen-op-decls Dialect/AIEX/IR/AIEX.h.inc
    -gen-op-defs Dialect/AIEX/IR/AIEX.cpp.inc
)

iree_cc_library(
  NAME
    AIEXDialectIR
  SRCS
    ${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIEX/IR/AIEXDialect.cpp
  DEPS
    ::defs
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
    AMDAIEAssignLockIDs.cpp
    AMDAIECoreToStandard.cpp
    AMDAIECreatePathFindFlows.cpp
    AMDAIEDmaToNpu.cpp
    AMDAIELocalizeLocks.cpp
    AMDAIENormalizeAddressSpaces.cpp
    AMDAIEObjectFifoStatefulTransform.cpp
    AMDAIEXToStandard.cpp
  DEPS
    ::defs
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
    ::AIEDialectIR
    ::AIEXDialectIR
    iree::target::amd-aie::Utils::Utils
)

add_subdirectory(test)
