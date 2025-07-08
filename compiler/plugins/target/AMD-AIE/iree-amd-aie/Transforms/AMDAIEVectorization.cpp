// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {

// IREE's GenericVectorization pass has many options, patterns, and vectorizes
// some non-linalg ops (tensor.pad). It has logic to choose vector sizes. This
// AIE-specific pass is a minimal version of GenericVectorization tailored to
// the needs of amd-aie. iree-amd-aie uses linalg.copy ops to move data between
// memory spaces (DDR, memory tile, core). These copy ops should not be
// vectorized to vector transfer_read/transfer_write ops. This 'fork' of
// GenericVectorization will be extended in the future to support more
// AIE-specific vectorization patterns.

class AMDAIEVectorizationPass
    : public impl::AMDAIEVectorizationBase<AMDAIEVectorizationPass> {
 public:
  using AMDAIEVectorizationBase::AMDAIEVectorizationBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert</* tensor::TensorDialect, */ linalg::LinalgDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override;

  static bool hasOperandWithSmallElementType(Operation *op) {
    for (auto operand : op->getOperands()) {
      if (auto type = dyn_cast<ShapedType>(operand.getType())) {
        auto elementType = type.getElementType();
        if (elementType.getIntOrFloatBitWidth() <= 16) {
          return true;
        }
      }
    }
    return false;
  }
};

void AMDAIEVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  IRRewriter rewriter(context);

  // Collect all operations which must be vectorized.
  SmallVector<Operation *> candidates;
  funcOp.walk([&](Operation *op) {
    // Only vectorize linalg ops (for now)
    if (!isa<linalg::LinalgOp>(op)) return WalkResult::advance();

    // iree-amd-aie's current tiling pipelines use linalg.copy ops to move data
    // between memory spaces. These copy ops should not be vectorized to
    // vector.transfer_read/transfer_write ops.
    if (isa<linalg::CopyOp>(op)) return WalkResult::advance();

    // For quantized ops elementwise ops are vectorized to ops that operate on
    // extremely large vectors, e.g., things like arith.addi %60, %63 :
    // vector<1x1x10x10x4x8xi32>. We'd be better off massaging the vectorization
    // such that the ops are split to narrower ops but this is (currently) and
    // edge case so just disable. See
    // https://github.com/nod-ai/iree-amd-aie/issues/594 for more info.
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      if (isElementwise(genericOp)) {
        for (Operation &innerOps : genericOp.getBody()->getOperations()) {
          if (!isa<arith::TruncFOp, arith::TruncIOp, linalg::YieldOp>(
                  innerOps)) {
            return WalkResult::advance();
          }
        }
      }
    }

    // AIE architecture has no vector instructions for 32/64-bit types.
    if (!isa<linalg::FillOp>(op) && !hasOperandWithSmallElementType(op))
      return WalkResult::advance();

    candidates.push_back(op);
    return WalkResult::advance();
  });

  for (Operation *op : candidates) {
    FailureOr<linalg::VectorizationResult> result =
        linalg::vectorize(rewriter, op);
    if (succeeded(result)) {
      rewriter.replaceOp(op, result->replacements);
    }
  }

  RewritePatternSet vectorizationPatterns(funcOp.getContext());

  vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
  vector::populateSinkVectorOpsPatterns(vectorizationPatterns);

  // Including this pattern prevents broadcasting in vector.transfer_read ops
  vector::populateVectorTransferPermutationMapLoweringPatterns(
      vectorizationPatterns);

  (void)applyPatternsGreedily(funcOp, std::move(vectorizationPatterns));
}
}  // namespace

std::unique_ptr<Pass> createAMDAIEVectorizationPass() {
  return std::make_unique<AMDAIEVectorizationPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
