// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"

#define DEBUG_TYPE "iree-amdaie-flatten-logicalobjectfifo"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEFlattenLogicalObjectFifoPass
    : public impl::AMDAIEFlattenLogicalObjectFifoBase<
          AMDAIEFlattenLogicalObjectFifoPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, memref::MemRefDialect>();
  }

  AMDAIEFlattenLogicalObjectFifoPass() = default;
  AMDAIEFlattenLogicalObjectFifoPass(
      const AMDAIEFlattenLogicalObjectFifoPass &pass){};
  void runOnOperation() override;
};

void AMDAIEFlattenLogicalObjectFifoPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();
  IRRewriter rewriter(context);

  moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp op) {
    MemRefType oldType = op.getMemrefType();
    // If there is memref.cast and generates `offset: ?`, we need to use the
    // type from memref source.
    // TODO (vivian): check again after proper fix in
    // DistributeCoresAndObjectFifosPass.
    auto castOp = dyn_cast<memref::CastOp>(op.getMemref().getDefiningOp());
    if (castOp) {
      oldType = cast<MemRefType>(castOp.getSource().getType());
    }

    // Get linearized size and new type accordingly.
    SmallVector<int64_t> shape = llvm::to_vector(oldType.getShape());
    int64_t linearizedSize = 1;
    for (auto s : shape) {
      linearizedSize *= s;
    }
    MemRefType newType =
        MemRefType::get(linearizedSize, oldType.getElementType(),
                        MemRefLayoutAttrInterface{}, oldType.getMemorySpace());

    rewriter.setInsertionPoint(op);
    auto newLogicalObjectFifo =
        rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
            rewriter.getUnknownLoc(), LogicalObjectFifoType::get(newType),
            op.getMemref(), op.getTiles());
    rewriter.replaceOp(op, newLogicalObjectFifo);

    // Replace the access op and insert `memref.reinterpret_cast` to get to the
    // original local shape as the objectfifo has a single type, while the DMA
    // operations converted into objectfifos can have a different source and
    // target type.
    for (Operation *user : newLogicalObjectFifo->getUsers()) {
      if (auto accessOp = dyn_cast<AMDAIE::LogicalObjectFifoAccessOp>(user)) {
        rewriter.setInsertionPoint(accessOp);
        auto newAccessOp = rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
            rewriter.getUnknownLoc(), newLogicalObjectFifo.getOutput(),
            accessOp.getAccessType());

        llvm::ArrayRef<int64_t> sizes = oldType.getShape();
        auto [strides, baseOffset] = getStridesAndOffset(oldType);
        auto reinterpretOp = rewriter.create<memref::ReinterpretCastOp>(
            rewriter.getUnknownLoc(), oldType, newAccessOp.getOutput(),
            baseOffset, sizes, strides);
        rewriter.replaceAllUsesWith(accessOp, reinterpretOp);
      }
    }

    return WalkResult::advance();
  });

  // Erase old access operations.
  moduleOp->walk([&](AMDAIE::LogicalObjectFifoAccessOp accessOp) {
    if (accessOp->getUses().empty()) {
      rewriter.eraseOp(accessOp);
    }
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFlattenLogicalObjectFifoPass() {
  return std::make_unique<AMDAIEFlattenLogicalObjectFifoPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
