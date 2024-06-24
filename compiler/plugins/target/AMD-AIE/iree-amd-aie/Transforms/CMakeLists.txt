# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

file(GLOB _cpp_files *.cpp)
file(GLOB _h_files *.h)
list(REMOVE_ITEM _h_files "PassDetail.h")

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
    iree::target::amd-aie::IR::AMDAIEDialect
    MLIRPass
  PUBLIC
)

iree_cc_library(
  NAME
    Transforms
  HDRS
    ${_h_files}
  SRCS
    ${_cpp_files}
  DEPS
    ::PassHeaders
    ::PassesIncGen
    iree::target::amd-aie::IR::AMDAIEDialect
    MLIRSupport
    iree::compiler::Codegen::Common::TransformDialectInterpreterPass
    iree::compiler::Codegen::Dialect::Codegen::IR::IREECodegenDialect
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::HAL::IR::HALDialect
    iree::compiler::Dialect::LinalgExt::IR
    iree::compiler::Dialect::LinalgExt::Transforms
    iree::compiler::Utils
    iree::target::amd-aie::aie::AIETransformPasses
    iree::target::amd-aie::aie::AIEXTransformPasses
    iree::target::amd-aie::air::AIRConversionPasses
    iree::target::amd-aie::air::AIRTransformPasses
    IREELinalgTransformDialectPasses
    MLIRFuncDialect
    MLIRFunctionInterfaces
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
