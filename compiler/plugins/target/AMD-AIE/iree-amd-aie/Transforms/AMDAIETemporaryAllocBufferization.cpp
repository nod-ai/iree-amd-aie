// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"

#define DEBUG_TYPE "iree-amdaie-temporary-alloc-bufferization"

namespace mlir::iree_compiler::AMDAIE {

namespace {

LogicalResult bufferizeTemporaryMemrefs(Operation *parentOp) {
  IRRewriter rewriter(parentOp->getContext());
  /// Create a unique BufferOp for each (AllocOp, TileOp, WorkgroupOp) tuple.
  using Key = std::tuple<memref::AllocOp, CoreOp, WorkgroupOp>;
  DenseMap<Key, BufferOp> bufferMap;
  parentOp->walk([&](memref::AllocOp allocOp) {
    for (Operation *user : allocOp->getUsers()) {
      if (CoreOp coreOp = user->getParentOfType<CoreOp>()) {
        TileOp tileOp = coreOp.getTileOp();
        auto workgroupOp = coreOp->getParentOfType<WorkgroupOp>();
        Key key{allocOp, coreOp, workgroupOp};
        if (bufferMap.count(key) == 0) {
          rewriter.setInsertionPointAfter(tileOp);
          auto bufferOp =
              rewriter.create<BufferOp>(allocOp.getLoc(), allocOp.getType(),
                                        tileOp, /* address */ nullptr);
          bufferMap[key] = bufferOp;
        }
      }
    }
  });

  for (auto [key, bufferOp] : bufferMap) {
    memref::AllocOp allocOp = std::get<0>(key);
    CoreOp coreOp = std::get<1>(key);
    WorkgroupOp workgroupOp = std::get<2>(key);
    rewriter.replaceUsesWithIf(allocOp, bufferOp, [&](OpOperand &operand) {
      Operation *owner = operand.getOwner();
      bool inTile = owner->getParentOfType<CoreOp>() == coreOp;
      bool inGroup = owner->getParentOfType<WorkgroupOp>() == workgroupOp;
      bool isNotDealloc = !isa<memref::DeallocOp>(owner);
      return inTile && inGroup && isNotDealloc;
    });
  }

  // Note: we don't erase allocs/deallocs, we leave this for canonicalization.

  return success();
}

class AMDAIETemporaryAllocBufferizationPass
    : public impl::AMDAIETemporaryAllocBufferizationBase<
          AMDAIETemporaryAllocBufferizationPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override {
    if (failed(bufferizeTemporaryMemrefs(getOperation())))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIETemporaryAllocBufferizationPass() {
  return std::make_unique<AMDAIETemporaryAllocBufferizationPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
