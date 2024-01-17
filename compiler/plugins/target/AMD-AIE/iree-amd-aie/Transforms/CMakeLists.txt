# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::target::amd-aie::Transforms")
iree_add_all_subdirs()

iree_tablegen_library(
  NAME
    PassesIncGen
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-decls Passes.h.inc
)

iree_cc_library(
  NAME
    PassHeaders
  HDRS
    "PassDetail.h"
    "Passes.h"
    "Passes.h.inc"
  DEPS
    ::PassesIncGen
    MLIRPass
  PUBLIC
)

iree_cc_library(
  NAME
    Transforms
  HDRS
    "Passes.h"
  SRCS
    "Passes.cpp"
    "AMDAIELowerExecutableTarget.cpp"
    "AMDAIEPadAndBufferize.cpp"
    "AMDAIETileAndFuse.cpp"
    "AMDAIEPackAndTranspose.cpp"
    "BridgeToAIRPass.cpp"
    "Cleanup.cpp"
  DEPS
    ::PassHeaders
    ::PassesIncGen
    MLIRSupport
    iree::compiler::Codegen::Common::TransformDialectInterpreterPass
    iree::compiler::Codegen::Dialect::Codegen::IR::IREECodegenDialect
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Utils
    iree::target::amd-aie::aie::AIETransformPasses
    iree::target::amd-aie::air::AIRConversionPasses
    iree::target::amd-aie::air::AIRTransformPasses
    IREELinalgExtDialect
    IREELinalgExtPasses
    IREELinalgExtTransforms
    IREELinalgTransformDialectPasses
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRPDLDialect
    MLIRPDLInterpDialect
    MLIRPass
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRSCFTransforms
    MLIRSCFUtils
    MLIRTensorDialect
    MLIRTensorTransforms
    MLIRTransforms
    MLIRVectorDialect
    MLIRVectorToLLVM
    MLIRVectorToSCF
    MLIRVectorTransforms
  PUBLIC
)
