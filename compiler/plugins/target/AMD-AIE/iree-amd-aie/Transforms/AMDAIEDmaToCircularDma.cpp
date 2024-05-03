// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"

#define DEBUG_TYPE "iree-amdaie-insert-circular-dma"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Convert dma copy operations to circular dma copy operations from L2 to L1
/// and L1 to L2. This is hardcoded for now because of how MLIR-AIE is set up,
/// but can be generalized in the future.
LogicalResult convertDmaToCircularDma(Operation *op) {
  IRRewriter rewriter(op->getContext());
  op->walk([&](AMDAIE::DmaCpyNdOp dmaOp) {
    Attribute sourceMemSpace = dmaOp.getSourceObjectFifo().getMemorySpace();
    Attribute targetMemSpace = dmaOp.getTargetObjectFifo().getMemorySpace();
    if (sourceMemSpace && targetMemSpace) {
      // L2 -> L1 or L1 -> L2 goes to uController instructions in MLIR-AIE.
      rewriter.setInsertionPointAfter(dmaOp);
      rewriter.replaceOpWithNewOp<AMDAIE::CircularDmaCpyNdOp>(
          dmaOp, dmaOp.getTarget(), dmaOp.getTargetMixedOffsets(),
          dmaOp.getTargetMixedSizes(), dmaOp.getTargetMixedStrides(),
          dmaOp.getSource(), dmaOp.getSourceMixedOffsets(),
          dmaOp.getSourceMixedSizes(), dmaOp.getSourceMixedStrides());
    }
  });
  return success();
}

class AMDAIEDmaToCircularDmaPass
    : public impl::AMDAIEDmaToCircularDmaBase<AMDAIEDmaToCircularDmaPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEDmaToCircularDmaPass() = default;
  AMDAIEDmaToCircularDmaPass(const AMDAIEDmaToCircularDmaPass &pass){};
  void runOnOperation() override;
};

void AMDAIEDmaToCircularDmaPass::runOnOperation() {
  if (failed(convertDmaToCircularDma(getOperation()))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEDmaToCircularDmaPass() {
  return std::make_unique<AMDAIEDmaToCircularDmaPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
