// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIELogicalObjFifoSplittingUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-split-logical-objectfifos"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIESplitLogicalObjFifosPass
    : public impl::AMDAIESplitLogicalObjFifosBase<
          AMDAIESplitLogicalObjFifosPass> {
 public:
  using AMDAIESplitLogicalObjFifosBase::AMDAIESplitLogicalObjFifosBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  void runOnOperation() override;
};

void AMDAIESplitLogicalObjFifosPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  // Walk and collect all dma ops between L3 and L2.
  SmallVector<AMDAIE::DmaCpyNdOp> l3L2DmaOps;
  WalkResult res = moduleOp->walk([&](AMDAIE::DmaCpyNdOp op) {
    std::optional<uint8_t> sourceMemSpace = op.getSourceMemorySpaceAsUInt();
    std::optional<uint8_t> targetMemSpace = op.getTargetMemorySpaceAsUInt();
    if (!sourceMemSpace || !targetMemSpace) {
      op.emitOpError() << "expected a source and target memory space";
      return WalkResult::interrupt();
    }
    if ((sourceMemSpace.value() == 1 && targetMemSpace.value() == 0) ||
        (sourceMemSpace.value() == 0 && targetMemSpace.value() == 1)) {
      l3L2DmaOps.push_back(op);
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();

  // Split the dma ops of L3->L2 / L2->L3.
  for (AMDAIE::DmaCpyNdOp dmaOp : l3L2DmaOps) {
    int64_t splitDim = -1;
    int64_t splitFactor = -1;
    int64_t splitDimInL2Dma = -1;
    if (failed(getSplitDimAndFactorFromDma(dmaOp, splitDim, splitFactor,
                                           splitDimInL2Dma))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to get split dimension and factor from " << dmaOp
                 << " \n");
      return signalPassFailure();
    }

    // No need to split with the following conditions.
    if (splitDim < 0 || splitDimInL2Dma < 0 || splitFactor <= 1) continue;

    if (failed(splitDoublyStridedOp(rewriter, dmaOp, splitDim, splitFactor,
                                    splitDimInL2Dma))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to perform splitting of doubly strided op");
      return signalPassFailure();
    }
  }

  // Walk and split input and output objectfifos in L2 memory space.
  res = moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp op) {
    if (op.getMemorySpaceAsUInt() != 1) return WalkResult::skip();
    int64_t splitDim = -1;
    int64_t splitFactor = -1;
    if (failed(getSplitDimAndFactorFromObjFifo(op, splitDim, splitFactor))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to get split dimension and factor from " << op
                 << " \n");
      return WalkResult::interrupt();
    }

    // No need to split with the following conditions.
    if (splitDim < 0 || splitFactor <= 1) return WalkResult::skip();

    if (failed(splitLogicalObjectFifo(rewriter, op, splitDim, splitFactor))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to perform splitting of objectFifo op");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIESplitLogicalObjFifosPass() {
  return std::make_unique<AMDAIESplitLogicalObjFifosPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
