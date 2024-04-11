// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "air/Dialect/AIR/AIRDialect.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-air-dma-to-amdaie-dma"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Transform AIR::DmaMemcpyNdOp into AMDAIE::DmaCpyNdOp ops, which operate on
/// top of logical objectfifos instead of memrefs.
class AIRDmaToAMDAIEDma : public OpRewritePattern<xilinx::air::DmaMemcpyNdOp> {
  using OpRewritePattern<xilinx::air::DmaMemcpyNdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::air::DmaMemcpyNdOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = op.getSrc().getType().cast<MemRefType>();
    auto dstType = op.getDst().getType().cast<MemRefType>();
    rewriter.setInsertionPointAfter(op.getSrc().getDefiningOp());
    auto src = rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        rewriter.getUnknownLoc(), AMDAIELogicalObjectFifoType::get(op.getLoc(), srcType),
        op.getSrc());
    rewriter.setInsertionPointAfter(op.getDst().getDefiningOp());
    auto dst = rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
        rewriter.getUnknownLoc(), AMDAIELogicalObjectFifoType::get(op.getLoc(), dstType),
        op.getDst());

    rewriter.setInsertionPoint(op);
    rewriter.create<AMDAIE::DmaCpyNdOp>(
        op.getLoc(), rewriter.getIndexType(), dst, op.getDstOffsets(),
        op.getDstSizes(), op.getDstStrides(), src, op.getSrcOffsets(),
        op.getSrcSizes(), op.getSrcStrides());
    rewriter.eraseOp(op);
    return success();
  }
};

class AMDAIEDAIRDmaToAMDAIEDmaPass
    : public impl::AMDAIEAIRDmaToAMDAIEDmaBase<AMDAIEDAIRDmaToAMDAIEDmaPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<xilinx::air::airDialect, AMDAIEDialect>();
  }

  AMDAIEDAIRDmaToAMDAIEDmaPass() = default;
  AMDAIEDAIRDmaToAMDAIEDmaPass(const AMDAIEDAIRDmaToAMDAIEDmaPass &pass){};
  void runOnOperation() override;
};

void AMDAIEDAIRDmaToAMDAIEDmaPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<AIRDmaToAMDAIEDma>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAIRDmaAMDAIEDmaPass() {
  return std::make_unique<AMDAIEDAIRDmaToAMDAIEDmaPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
