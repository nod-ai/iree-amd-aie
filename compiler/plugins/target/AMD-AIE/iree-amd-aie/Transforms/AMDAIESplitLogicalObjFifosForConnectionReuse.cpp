// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIELogicalObjFifoSplittingUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
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

  // Walk through CoreOps gathering 3rd input DmaOps (if applicable) which will
  // be used to split L2 objectFifos of elementwise input for connection reuse.
  SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps;
  moduleOp->walk([&](AMDAIE::CoreOp coreOp) {
    SmallVector<Value> inputDmas = coreOp.getInputDmas();
    if (inputDmas.size() != 3) return WalkResult::skip();
    auto dmaCpyNdOp = inputDmas[2].getDefiningOp<AMDAIE::DmaCpyNdOp>();
    if (!dmaCpyNdOp) {
      coreOp->emitOpError() << "failed to get a DmaCpyNdOp from the input";
    }
    l2ToL1DmaOps.push_back(dmaCpyNdOp);
    return WalkResult::advance();
  });

  if (failed(splitLogicalObjectFifoForElementwiseOp(rewriter, l2ToL1DmaOps,
                                                    context))) {
    LLVM_DEBUG(llvm::dbgs()
               << "Failed to perform splitting of logicalobjectfifos");
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIESplitLogicalObjFifosForConnectionReusePass() {
  return std::make_unique<AMDAIESplitLogicalObjFifosForConnectionReusePass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
