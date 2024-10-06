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

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::iree_compiler::aievec {

/**
 * Append pass(es) for canonicalizing operations in the vector dialect to a form
 * that can be lowered to the AIEVec dialect.
 */
void buildCanonicalizeVectorForAIEVec(mlir::OpPassManager &);

/**
 * A pass containing patterns for canonicalizing operations in the vector
 * dialect to a form that can be lowered to the AIEVec dialect. This pass is
 * named `canonicalize-vector-for-aievec`.
 */
std::unique_ptr<mlir::Pass> createCanonicalizeVectorForAIEVecPass();

/**
 * Expose the pass `canonicalize-vector-for-aievec` to the command line.
 */
void registerCanonicalizeVectorForAIEVecPass();

/**
 * Append pass(es) for lowering operations in the vector dialect to the AIEVec
 * dialect. Vector dialect ops are expected to be in a canonical form
 * before entering this pass pipeline.
 */
void buildLowerVectorToAIEVec(mlir::OpPassManager &pm);

/**
 * A pass containing patterns for lowering operations in the vector dialect to
 * the AIEVec dialect. The pass is currently named
 * `test-lower-vector-to-aievec`.
 */
std::unique_ptr<mlir::Pass> createLowerVectorToAIEVec();

/**
 * Expose the pass `test-lower-vector-to-aievec` to the command line.
 */
void registerLowerVectorToAIEVecPass();

/**
 * This appends a combination of canonicalization and lowering passes to
 * a pass pipline. It takes vector dialect code, transforms it to make it
 * compatible with the AIE, and then lowers it to the AIEVec dialect.
 */
void buildConvertVectorToAIEVec(mlir::OpPassManager &);

/**
 * Lower from the vector dialect to the AIEVec dialect. The pass is called
 * `convert-aievec-to-llvm`.
 */
std::unique_ptr<mlir::Pass> createConvertAIEVecToLLVMPass();

/**
 * Register all pipelines for the AIE Vector dialect.
 */
void registerAIEVecPipelines();

/**
 * Expose the pass `convert-aievec-to-llvm` to the command line.
 */
void registerConvertAIEVecToLLVMPass();

/// Register the AIEVec dialect and the translation from it to the LLVM dialect
/// in the given registry.
void registerXLLVMDialectTranslation(mlir::DialectRegistry &registry);

}  // namespace mlir::iree_compiler::aievec

#endif  // AIE_DIALECT_AIEVEC_PIPELINES_PASSES_H
