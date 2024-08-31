# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

iree_cc_binary(
  NAME
    iree-aie-translate
  SRCS
    iree-aie-translate.cpp
  DEPS
    iree::target::amd-aie::aie::AIEDialectIR
    iree::target::amd-aie::aievec::AIEVecDialectIR
    iree::target::amd-aie::aievec::AIEVecConvertToLLVM
    ${dialect_libs}
    ${translation_libs}
    LLVMSupport
    MLIRFuncAllExtensions
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTargetLLVMIRExport
    MLIRTargetLLVMIRImport
)
