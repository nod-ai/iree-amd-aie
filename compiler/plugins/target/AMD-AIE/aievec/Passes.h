//===- Passes.h - AIE Vector pipeline entry points --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all AIE vector pipelines.
//
//===----------------------------------------------------------------------===//
#ifndef AIE_DIALECT_AIEVEC_PIPELINES_PASSES_H
#define AIE_DIALECT_AIEVEC_PIPELINES_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::iree_compiler::aievec {

enum class Aie2Fp32Emulation : uint32_t {
  AccuracySafe = 0,
  AccuracyFast = 1,
  AccuracyLow = 2,
};

struct ConvertAIEVecToLLVMOptions {
  Aie2Fp32Emulation aie2Fp32Emulation = Aie2Fp32Emulation::AccuracySafe;
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the "convert-vector-to-aievec" pipeline to the `OpPassManager`. This
/// pipeline takes `Vector` code, transforms it to make it compatible with the
/// selected `AIE` target, lowers it to `AIEVec` dialect, and performs some
/// optimizations based on the target AIE architecture.
void buildConvertVectorToAIEVec(mlir::OpPassManager &pm);

void buildCanonicalizeVectorForAIEVec(mlir::OpPassManager &pm);

void buildLowerVectorToAIEVec(mlir::OpPassManager &pm);

/// Create a pass that removes unnecessary Copy operations.
std::unique_ptr<mlir::Pass> createCopyRemovalPass();

std::unique_ptr<mlir::Pass> createConvertAIEVecToLLVMPass();
void registerConvertAIEVecToLLVMPass();

/// Register the AIEVec dialect and the translation from it to the LLVM dialect
/// in the given registry.
void registerXLLVMDialectTranslation(mlir::DialectRegistry &registry);

}  // namespace mlir::iree_compiler::aievec

#endif  // AIE_DIALECT_AIEVEC_PIPELINES_PASSES_H
