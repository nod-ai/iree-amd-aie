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

#define DEBUG_TYPE \
  "iree-amdaie-combine-logical-objectfifos-for-connection-reuse"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIECombineLogicalObjFifosForConnectionReusePass
    : public impl::AMDAIECombineLogicalObjFifosForConnectionReuseBase<
          AMDAIECombineLogicalObjFifosForConnectionReusePass> {
 public:
  using AMDAIECombineLogicalObjFifosForConnectionReuseBase::
      AMDAIECombineLogicalObjFifosForConnectionReuseBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  void runOnOperation() override;
};

void AMDAIECombineLogicalObjFifosForConnectionReusePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  SmallVector<AMDAIE::DmaCpyNdOp> l2ToL1DmaOps =
      fetchDmaCpyNdOpsToSplitOrCombine(moduleOp);

  if (failed(combineLogicalObjectFifos(rewriter, l2ToL1DmaOps, context))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass>
createAMDAIECombineLogicalObjFifosForConnectionReusePass() {
  return std::make_unique<AMDAIECombineLogicalObjFifosForConnectionReusePass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
