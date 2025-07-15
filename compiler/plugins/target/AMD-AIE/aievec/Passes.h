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
 * Append passes for canonicalizing operations in the vector dialect to a form
 * that can be lowered to the AIEVec dialect.
 */
void buildCanonicalizeVectorForAIEVec(mlir::OpPassManager &);

/**
 * A pass containing some patterns for canonicalizing operations in the vector
 * dialect to a form that can be lowered to the AIEVec dialect. This pass is
 * named `canonicalize-vector-for-aievec`. To ensure all required vector dialect
 * canonicalizations take place, PassManagers should use
 * `buildCanonicalizeVectorForAIEVec`.
 */
std::unique_ptr<mlir::Pass> createCanonicalizeVectorForAIEVecPass();

/**
 * Expose the pass `canonicalize-vector-for-aievec` to the command line.
 */
void registerCanonicalizeVectorForAIEVecPass();

/**
 * This pass ensures that reads from AIE tile memory are aligned according to
 * hardware constraints. For example, suppose we have 128 bytes in tile memory,
 * represented in hex as:
 *
 *    0x00 0x01 ... 0x7E 0x7F
 *
 * On AIE-2, the (vector) read instructions from the tile memory into registers
 * must be aligned to 256-bits (32-bytes). So if we want to read 64 bytes
 * starting from 0x00 that is fine, but if we want to read 64 bytes starting
 * from 0x01, then we cannot use a vector read instruction directly. To work
 * around this constraint, we do the following:
 *
 * 1. Perform a wider read, that loads 128 bytes (2x as many as we want)
 *    starting from 0x00 into a larger register. That is, bytes 0x00-0x7F are
 *    loaded, so we have 1 'junk' byte at the beginning and 63 'junk' bytes at
 *    the end.
 *
 * 2. Extract the target bytes 0x01 ... 0x40 from the larger register into a
 *    smaller register in 2 steps, using 2 AIE specific instructions:
 *
 *   a) Extract:
 *      https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_ml_intrinsics/intrinsics/group__intr__gpvectorconv__elem.html
 *
 *   b) Shift:
 *      https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_ml_intrinsics/intrinsics/group__intr__gpvectorop__shift.html
 *
 *   First, we use the extract instruction to split the read 128-bytes into two
 *   halves, 0x00-0x3F and 0x40-0x7F, each in its own 64-byte register. Then, we
 *   use a shift operation to combine the upper 31 bytes from the first half
 *   and the lower 33 bytes from the second half into a new 64-byte register.
 *   This new register contains exactly the 64 bytes we want to read, starting
 *   from 0x01.
 *
 * If we want to read 32 bytes starting from 0x01, we can use a similar
 * approach. The only consideration is that the shift operation requires 64-byte
 * inputs, so the order of the of the shift and extracts is reversed.
 *
 * We do not currently support unaligned reads of vectors which are not 32-bytes
 * or 64-bytes in length.
 *
 * TODO(newling) use this same approach to align writes to unaligned memory.
 *  */

std::unique_ptr<mlir::Pass> createAlignTransferReadsPass();

void registerAlignTransferReadsPass();

/**
 * Append pass(es) for lowering operations in the vector dialect to the AIEVec
 * dialect. Vector dialect ops are expected to be in a canonical form
 * before entering this pass pipeline.
 */
void buildLowerVectorToAIEVec(mlir::OpPassManager &pm);

/**
 * A pass containing patterns for lowering operations in the vector dialect to
 * the AIEVec dialect. The pass is currently named `test-lower-vector-to-aievec`
 */
static std::unique_ptr<mlir::Pass> createLowerVectorToAIEVec();

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
 * Lower from the AIEVec dialect to the LLVM dialect. The pass is called
 * `convert-aievec-to-llvm`.
 * */
std::unique_ptr<mlir::Pass> createConvertAIEVecToLLVMPass();

/**
 * Expose the pass `convert-aievec-to-llvm` to the command line.
 */
void registerConvertAIEVecToLLVMPass();

/// Register the AIEVec dialect and the translation from it to the LLVM dialect
/// in the given registry.
void registerXLLVMDialectTranslation(mlir::DialectRegistry &registry);

}  // namespace mlir::iree_compiler::aievec

#endif  // AIE_DIALECT_AIEVEC_PIPELINES_PASSES_H
