# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(test)

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
    "Placeholder.cpp"
    "AMDAIELowerExecutableTarget.cpp"
  DEPS
    ::PassHeaders
    ::PassesIncGen
    MLIRSupport
    iree::compiler::Codegen::Dialect::IREECodegenDialect
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Utils
    IREELinalgExtDialect
    IREELinalgExtPasses
    IREELinalgExtTransforms
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRPDLDialect
    MLIRPDLInterpDialect
    MLIRPass
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRSCFTransforms
    MLIRSCFUtils
    MLIRTensorDialect
    MLIRTensorTransforms
    MLIRVectorDialect
    MLIRVectorToLLVM
    MLIRVectorToSCF
    MLIRVectorTransforms
  PUBLIC
)
