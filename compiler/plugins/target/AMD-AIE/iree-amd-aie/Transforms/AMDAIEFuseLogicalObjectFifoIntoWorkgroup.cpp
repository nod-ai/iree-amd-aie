// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"

#define DEBUG_TYPE "iree-amdaie-fuse-logical-objectfifo-into-workgroup"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// This pass fuses `LogicalObjectFifoFromMemref` operations into the
/// workgroups where they are used. This is a utility which can be relied upon
/// by other passes to perform custom transformations with logical objectfifos
/// within a workgroup. This pass handles multiple workgroups and uses outside
/// workgroups by cloning `LogicalObjectFifoFromMemref` into every workgroup
/// where it's used.
class AMDAIEFuseLogicalObjectFifoIntoWorkgroupPass
    : public impl::AMDAIEFuseLogicalObjectFifoIntoWorkgroupBase<
          AMDAIEFuseLogicalObjectFifoIntoWorkgroupPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEFuseLogicalObjectFifoIntoWorkgroupPass() = default;
  AMDAIEFuseLogicalObjectFifoIntoWorkgroupPass(
      const AMDAIEFuseLogicalObjectFifoIntoWorkgroupPass &pass){};
  void runOnOperation() override;
};

void AMDAIEFuseLogicalObjectFifoIntoWorkgroupPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();
  IRRewriter rewriter(context);

  SmallVector<AMDAIE::LogicalObjectFifoFromMemrefOp> logicalObjectFifos;
  moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp op) {
    // The op is already in a workgroup, so it can just stay there and there is
    // nothing to be done further.
    if (isa<AMDAIE::WorkgroupOp>(op->getParentOp())) {
      return WalkResult::advance();
    }
    logicalObjectFifos.push_back(op);
    return WalkResult::advance();
  });

  for (AMDAIE::LogicalObjectFifoFromMemrefOp op : logicalObjectFifos) {
    // Get all workgroup users
    llvm::SmallSetVector<AMDAIE::WorkgroupOp, 4> workgroupUsers;
    for (Operation *userOp : op->getUsers()) {
      if (auto workgroupOp = userOp->getParentOfType<AMDAIE::WorkgroupOp>()) {
        workgroupUsers.insert(workgroupOp);
      }
    }

    // If no users inside a workgroup found, we don't do any rewrite
    if (workgroupUsers.empty()) continue;

    // Clone into every workgroup
    for (auto workgroupOp : workgroupUsers) {
      rewriter.setInsertionPoint(&workgroupOp.getRegion().front(),
                                 workgroupOp.getRegion().front().begin());
      auto clone = rewriter.clone(*(op.getOperation()));
      op->replaceUsesWithIf(clone, [&](OpOperand &use) {
        return use.getOwner()->getParentOfType<AMDAIE::WorkgroupOp>() ==
               workgroupOp;
      });
    }

    if (op->use_empty()) rewriter.eraseOp(op);
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFuseLogicalObjectFifoIntoWorkgroupPass() {
  return std::make_unique<AMDAIEFuseLogicalObjectFifoIntoWorkgroupPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
