// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-amdaie-create-logical-objectfifo-link"

namespace mlir::iree_compiler::AMDAIE {

/// Utility to add explicit link operations to avoid having to do this during
/// conversion to AIEDialect operations. This function only consider L2/MT for
/// links as L1/L3 don't need this linking through AIE objectfifos. Furthermore,
/// it assumes that all users of a logical objectfifo reside within the same
/// block and an error will be emitted if that's not the case.
LogicalResult createLogicalObjectFifoLink(
    RewriterBase &rewriter,
    AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
  Attribute memSpace = logicalObjectFifo.getMemorySpace();
  if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 1) {
    return success();
  }

  // Visit all CopyOp users of this logical objectfifo and add them to
  // either the input or output side of this logical objectfifo. While
  // doing this, keep track of the last user operation for insertion
  // purposes.
  SmallVector<OpFoldResult> ins;
  SmallVector<OpFoldResult> outs;
  CopyOpInterface lastUserOp;
  for (Operation *userOp : logicalObjectFifo->getUsers()) {
    if (auto copyOp = dyn_cast<CopyOpInterface>(userOp)) {
      if (lastUserOp && copyOp->getBlock() != lastUserOp->getBlock()) {
        logicalObjectFifo->emitError(
            "does have copy-like users not residing in the same block");
        return failure();
      }
      auto sourceLogicalObjectFifo =
          dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(
              copyOp.getSource().getDefiningOp());
      if (!sourceLogicalObjectFifo) {
        copyOp->emitError(
            "does not have a `LogicalObjectFifoFromMemrefOp` as source");
        return failure();
      }
      if (!lastUserOp || lastUserOp->isBeforeInBlock(copyOp)) {
        lastUserOp = copyOp;
      }
      if (logicalObjectFifo == sourceLogicalObjectFifo) {
        outs.push_back(copyOp->getResult(0));
      } else {
        ins.push_back(copyOp->getResult(0));
      }
    }
  }

  // Insert the `LogicalObjectFifoLink` after the last user operation.
  if (lastUserOp) {
    rewriter.setInsertionPointAfter(lastUserOp);
    rewriter.create<AMDAIE::LogicalObjectFifoLink>(
        rewriter.getUnknownLoc(),
        getValueOrCreateConstantIndexOp(rewriter, rewriter.getUnknownLoc(),
                                        ins),
        getValueOrCreateConstantIndexOp(rewriter, rewriter.getUnknownLoc(),
                                        outs));
  }
  return success();
}

namespace {

struct AMDAIECreateLogicalObjectFifoLinkPass
    : public impl::AMDAIECreateLogicalObjectFifoLinkBase<
          AMDAIECreateLogicalObjectFifoLinkPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override {
    Operation *parentOp = getOperation();
    IRRewriter rewriter(parentOp->getContext());

    WalkResult res = parentOp->walk(
        [&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
          if (failed(
                  createLogicalObjectFifoLink(rewriter, logicalObjectFifo))) {
            logicalObjectFifo.emitError() << "couldn't create a link operation";
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (res.wasInterrupted()) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIECreateLogicalObjectFifoLinkPass() {
  return std::make_unique<AMDAIECreateLogicalObjectFifoLinkPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
