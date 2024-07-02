// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-create-reference-to-allocation"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIECreateReferenceToAllocationPass
    : public impl::AMDAIECreateReferenceToAllocationBase<
          AMDAIECreateReferenceToAllocationPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, memref::MemRefDialect>();
  }

  AMDAIECreateReferenceToAllocationPass() = default;
  AMDAIECreateReferenceToAllocationPass(
      const AMDAIECreateReferenceToAllocationPass &pass){};
  void runOnOperation() override;
};

void AMDAIECreateReferenceToAllocationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();
  IRRewriter rewriter(context);

  funcOp->walk([&](memref::AllocOp allocOp) {
    // Only create reference to allocation in local memory (L1).
    Attribute memSpace =
        cast<MemRefType>(allocOp.getResult().getType()).getMemorySpace();
    if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 2)
      return WalkResult::advance();

    rewriter.setInsertionPointAfter(allocOp);
    auto referenceOp = rewriter.create<AMDAIE::ReferenceToOp>(
        allocOp.getLoc(), allocOp.getType(), allocOp);

    Operation *dealloc;
    for (Operation *userOp : allocOp->getUsers()) {
      if (isa<memref::DeallocOp>(userOp)) {
        dealloc = userOp;
      }
    }
    SmallPtrSet<Operation *, 2> exceptions(
        {referenceOp.getOperation(), dealloc});
    rewriter.replaceAllUsesExcept(allocOp.getResult(),
                                  referenceOp->getResult(0), exceptions);
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIECreateReferenceToAllocationPass() {
  return std::make_unique<AMDAIECreateReferenceToAllocationPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
