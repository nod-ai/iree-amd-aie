//===-ConvertVectorToAIEVec.cpp - Lower Vector to AIE vector ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// This is the implementation of the lowering pass from standard Vector
// dialect to AIEVec, compatible with the AIE vector architecture.
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace arith;
using namespace vector;
using namespace mlir::iree_compiler::aievec;

#define DEBUG_TYPE "vector-to-aievec-conversion"

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

template <typename OpTy>
struct SetInboundsToReadStoreOpPattern : public RewritePattern {
  SetInboundsToReadStoreOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    OpTy writeOrReadOp = cast<OpTy>(op);

    // TODO: We are currently setting all `vector.transfer_read` and
    // TODO: `vector.transfer_write` as "in bounds". We need to add
    // TODO: an analysis to verify that this is true before doing so.
    if (writeOrReadOp.getInBounds() || writeOrReadOp.getTransferRank() == 0) {
      return failure();
    }

    SmallVector<bool, 4> bools(writeOrReadOp.getTransferRank(), true);
    auto inBoundsAttr = rewriter.getBoolArrayAttr(bools);
    rewriter.modifyOpInPlace(writeOrReadOp, [&]() {
      writeOrReadOp->setAttr(writeOrReadOp.getInBoundsAttrName(), inBoundsAttr);
    });
    return success();
  }
};

using SetInboundsToReadOp = SetInboundsToReadStoreOpPattern<TransferReadOp>;
using SetInboundsToWriteOp = SetInboundsToReadStoreOpPattern<TransferWriteOp>;

//===----------------------------------------------------------------------===//
// Lowering passes
//===----------------------------------------------------------------------===//

struct RedundantLoadStoreOptimizationPass
    : public PassWrapper<RedundantLoadStoreOptimizationPass, OperationPass<>> {
  void runOnOperation() override {
    auto op = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<SetInboundsToReadOp, SetInboundsToWriteOp>(
        patterns.getContext());

    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
    IRRewriter rewriter(&getContext());
    vector::transferOpflowOpt(rewriter, op);
  }
};

static std::unique_ptr<::mlir::Pass>
createRedundantLoadStoreOptimizationPass() {
  return std::make_unique<RedundantLoadStoreOptimizationPass>();
}

//===---------------------------------------------------------------------------
// Pipeline implementations
//===---------------------------------------------------------------------------
void mlir::iree_compiler::aievec::buildConvertVectorToAIEVec(
    OpPassManager &pm) {
  // Pre-conversion passes.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createRedundantLoadStoreOptimizationPass());

  //============================================================================
  // Vector canonicalization for AIEVec: Vector to Vector conversion.
  //============================================================================

  // NOTE: This sub-pipeline ingests arbitrary MLIR Vector code.
  buildCanonicalizeVectorForAIEVec(pm);
  // NOTE: At this stage, all the Vector code in the IR can be mapped
  // NOTE: to AIEVec operations.

  //============================================================================
  // Vector to AIEVec lowering: Vector to AIEVec conversion.
  //============================================================================

  // NOTE: This sub-pipeline ingests MLIR Vector code that can be mapped to
  // NOTE: AIEVec operations.
  buildLowerVectorToAIEVec(pm);
  // NOTE: At this stage, all vector operations are expressed in AIEVec dialect.

  // Post-conversion passes.
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createCanonicalizerPass());
}

void mlir::iree_compiler::aievec::registerAIEVecPipelines() {
  PassPipelineRegistration<>(
      "convert-vector-to-aievec",
      "This pass pipeline takes standard \"Vector\" code and converts it to "
      "\"AIEVec\" code targeting the selected Xilinx AIE2 vector "
      "architecture.",
      buildConvertVectorToAIEVec);
}
