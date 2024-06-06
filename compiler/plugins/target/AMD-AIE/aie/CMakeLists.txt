# Copyright 2023 The IREE Authors
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
    MLIRIR
)

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
    AIEXDialectGen
  TD_FILE
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIEX/IR/AIEX.td"
  OUTS
    -gen-dialect-decls -dialect=aiex Dialect/AIEX/IR/AIEXDialect.h.inc
    -gen-dialect-defs -dialect=aiex Dialect/AIEX/IR/AIEXDialect.cpp.inc
)

iree_tablegen_library(
  NAME
    AIEVecDialectGen
  TD_FILE
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIEVec/IR/AIEVecOps.td"
  OUTS
    -gen-dialect-decls -dialect=aievec Dialect/AIEVec/IR/AIEVecOpsDialect.h.inc
    -gen-dialect-defs -dialect=aievec Dialect/AIEVec/IR/AIEVecOpsDialect.cpp.inc
    -gen-typedef-decls -typedefs-dialect=aievec Dialect/AIEVec/IR/AIEVecOpsTypes.h.inc
    -gen-typedef-defs -typedefs-dialect=aievec Dialect/AIEVec/IR/AIEVecOpsTypes.cpp.inc
)

iree_tablegen_library(
  NAME
    AIEVecAttrsGen
  TD_FILE
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIEVec/IR/AIEVecAttributes.td"
  OUTS
    -gen-attrdef-decls -attrdefs-dialect=aievec Dialect/AIEVec/IR/AIEVecAttributes.h.inc
    -gen-attrdef-defs -attrdefs-dialect=aievec Dialect/AIEVec/IR/AIEVecAttributes.cpp.inc
    -gen-enum-decls Dialect/AIEVec/IR/AIEVecEnums.h.inc
    -gen-enum-defs Dialect/AIEVec/IR/AIEVecEnums.cpp.inc
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

iree_tablegen_library(
  NAME
    AIENormalizeAddressSpacesGen
  TD_FILE
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIE/Transforms/AIENormalizeAddressSpaces.td"
  OUTS
    -gen-rewriters Dialect/AIE/Transforms/AIENormalizeAddressSpaces.inc
)

###############################################################################
# AIEX Dialect
###############################################################################

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

iree_tablegen_library(
  NAME
    AIEXOpsGen
  TD_FILE
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIEX/IR/AIEX.td"
  OUTS
    -gen-op-decls Dialect/AIEX/IR/AIEX.h.inc
    -gen-op-defs Dialect/AIEX/IR/AIEX.cpp.inc
)

###############################################################################
# AIEVec Dialect
###############################################################################

iree_cc_library(
  NAME
    AIEVecDialectIR
  SRCS
    ${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIEVec/IR/AIEVecOps.cpp
    ${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIEVec/IR/AIEVecTypes.cpp
  DEPS
    ::defs
    ::AIEVecOpsGen
    ::AIEVecDialectGen
    ::AIEVecAttrsGen
    MLIRIR
)

iree_tablegen_library(
  NAME
    AIEVecOpsGen
  TD_FILE
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIEVec/IR/AIEVecOps.td"
  OUTS
    -gen-op-decls Dialect/AIEVec/IR/AIEVecOps.h.inc
    -gen-op-defs Dialect/AIEVec/IR/AIEVecOps.cpp.inc
)

###############################################################################
# AIE Transform Passes
###############################################################################

iree_tablegen_library(
  NAME
    AIETransformPassesIncGen
  TD_FILE
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIE/Transforms/AIEPasses.td"
  OUTS
    -gen-pass-decls Dialect/AIE/Transforms/AIEPasses.h.inc
)

iree_cc_library(
  NAME
    AIETransformPassHeaders
  HDRS
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIE/Transforms/Passes.h"
    "Passes.h.inc"
  DEPS
    ::AIETransformPassesIncGen
    MLIRPass
  PUBLIC
)

iree_cc_library(
  NAME
    AIETransformPasses
  SRCS
    "AIETransformPasses.cpp"
    "AIEAssignBufferAddressesBasic.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIE/Transforms/AIEAssignLockIDs.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIE/Transforms/AIEAssignBuffers.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIE/Transforms/AIECanonicalizeDevice.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIE/Transforms/AIECoreToStandard.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIE/Transforms/AIECreatePacketFlows.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIE/Transforms/AIECreatePathFindFlows.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIE/Transforms/AIELocalizeLocks.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIE/Transforms/AIEObjectFifoRegisterProcess.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIE/Transforms/AIENormalizeAddressSpaces.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIE/Transforms/AIEPathFinder.cpp"
  DEPS
    ::defs
    ::AIEDialectIR
    ::AIENormalizeAddressSpacesGen
    ::AIETransformPassHeaders
)

###############################################################################
# AIEX Transform Passes
###############################################################################

iree_tablegen_library(
  NAME
    AIEXTransformPassesIncGen
  TD_FILE
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIEX/Transforms/AIEXPasses.td"
  OUTS
    -gen-pass-decls Dialect/AIEX/Transforms/AIEXPasses.h.inc
)

iree_cc_library(
  NAME
    AIEXTransformPassHeaders
  HDRS
    "${IREE_MLIR_AIE_SOURCE_DIR}/include/aie/Dialect/AIEX/Transforms/Passes.h"
    "Passes.h.inc"
  DEPS
    ::AIEXTransformPassesIncGen
    MLIRPass
  PUBLIC
)

iree_cc_library(
  NAME
    AIEXTransformPasses
  SRCS
    "AIEXTransformPasses.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIEX/Transforms/AIECreateBroadcastPacket.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIEX/Transforms/AIEDmaToNpu.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIEX/Transforms/AIELowerMulticast.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIEX/Transforms/AIECreateBroadcastPacket.cpp"
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Dialect/AIEX/Transforms/AIEXToStandard.cpp"
  DEPS
    ::defs
    ::AIEXDialectIR
    ::AIEXTransformPassHeaders
    MLIRToLLVMIRTranslationRegistration
    MLIRFuncAllExtensions
)

###############################################################################
# AIE CDO Generation
###############################################################################

iree_cc_library(
  NAME
    AIETargetCDODirect
  SRCS
    "${IREE_MLIR_AIE_SOURCE_DIR}/lib/Targets/AIETargetCDODirect.cpp"
  DEPS
    ::AIEDialectIR
    iree-amd-aie::runtime::iree_aie_runtime_static
)