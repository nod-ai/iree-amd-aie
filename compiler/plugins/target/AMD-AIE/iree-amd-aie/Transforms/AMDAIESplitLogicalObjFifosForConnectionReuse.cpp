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

#define DEBUG_TYPE "iree-amdaie-split-logical-objectfifos-for-connection-reuse"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIESplitLogicalObjFifosForConnectionReusePass
    : public impl::AMDAIESplitLogicalObjFifosForConnectionReuseBase<
          AMDAIESplitLogicalObjFifosForConnectionReusePass> {
 public:
  using AMDAIESplitLogicalObjFifosForConnectionReuseBase::
      AMDAIESplitLogicalObjFifosForConnectionReuseBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  void runOnOperation() override;
};

void AMDAIESplitLogicalObjFifosForConnectionReusePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  // SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps =
  //     fetchDmaCpyNdOpsToSplitOrCombine(moduleOp);

  // if (failed(splitLogicalObjectFifos(rewriter, l2ToL1DmaOps, context))) {
  //   LLVM_DEBUG(llvm::dbgs()
  //              << "Failed to perform splitting of logicalobjectfifos");
  //   return signalPassFailure();
  // }
  SmallVector<AMDAIE::DmaCpyNdOp> dmaOps;
  moduleOp->walk([&](AMDAIE::DmaCpyNdOp op) {
    std::optional<uint8_t> sourceMemSpace = op.getSourceMemorySpaceAsUInt();
    std::optional<uint8_t> targetMemSpace = op.getTargetMemorySpaceAsUInt();
    if (sourceMemSpace && sourceMemSpace.value() == 1 && targetMemSpace &&
        targetMemSpace.value() == 0) {
      dmaOps.push_back(op);
    }
    return WalkResult::advance();
  });

  for (AMDAIE::DmaCpyNdOp dmaOp : dmaOps) {
    auto stridedOp =
        cast<AMDAIE::DoublyStridedOpInterface>(dmaOp.getOperation());
    if (failed(splitDoublyStridedOp(rewriter, stridedOp))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to perform splitting of logicalobjectfifos");
      return signalPassFailure();
    }
  }

  SmallVector<AMDAIE::LogicalObjectFifoFromMemrefOp> objFifoOps;
  moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp op) {
    ArrayRef<int64_t> memrefShape = op.getMemrefType().getShape();
    if (op.getMemorySpaceAsUInt() == 1 && memrefShape.size() > 2 &&
        memrefShape[0] == 2 && memrefShape[1] == 2) {
      llvm::outs() << "push objFifo: " << op << "\n";
      objFifoOps.push_back(op);
    }
    return WalkResult::advance();
  });
  for (AMDAIE::LogicalObjectFifoFromMemrefOp op : objFifoOps) {
    if (failed(splitObjFifo(rewriter, op))) {
      return signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIESplitLogicalObjFifosForConnectionReusePass() {
  return std::make_unique<AMDAIESplitLogicalObjFifosForConnectionReusePass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
