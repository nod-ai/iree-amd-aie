// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements localization of logical objectfifos. This pass moves or
// clones `amdaie.logicalobjectfifo.from_memref` operations close to the loops
// chosen for parallel distribution of the workload. This is a utility which can
// be relied upon by other passes to perform transformations with logical
// objectfifos within the local parallel scope, of which there can be multiple,
// as this avoids different parallalel scopes being dependent on each other by
// relying on the same logical objectfifos.
//
//===----------------------------------------------------------------------===//

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-amdaie-localize-logicalobjectfifo"

namespace mlir::iree_compiler::AMDAIE {

namespace {

// Recursively find and return a parent scf.forall op chosen for parallel
// distribution across cores. For now, this looks for thread mapping, but this
// will probably be changed to include block mapping as well in the future.
scf::ForallOp getParentForallWithParallelMapping(Operation *op) {
  scf::ForallOp forallOp = op->getParentOfType<scf::ForallOp>();
  if (!forallOp) return scf::ForallOp();
  if (forallOp.getMapping().has_value() &&
      isa<mlir::gpu::GPUThreadMappingAttr>(
          *forallOp.getMapping()->getValue().begin())) {
    return forallOp;
  }
  return getParentForallWithParallelMapping(forallOp);
}

class AMDAIELocalizeLogicalObjectfifoPass
    : public impl::AMDAIELocalizeLogicalObjectfifoBase<
          AMDAIELocalizeLogicalObjectfifoPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIELocalizeLogicalObjectfifoPass() = default;
  AMDAIELocalizeLogicalObjectfifoPass(
      const AMDAIELocalizeLogicalObjectfifoPass &pass){};
  void runOnOperation() override;
};

void AMDAIELocalizeLogicalObjectfifoPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();
  IRRewriter rewriter(context);

  SmallVector<AMDAIE::LogicalObjectFifoFromMemrefOp> logicalObjectFifos;
  moduleOp->walk([&](AMDAIE::LogicalObjectFifoFromMemrefOp op) {
    // The op is already within the parallel mapped forall, so doesn't need
    // any change.
    if (getParentForallWithParallelMapping(op)) {
      return WalkResult::advance();
    }
    logicalObjectFifos.push_back(op);
    return WalkResult::advance();
  });

  for (AMDAIE::LogicalObjectFifoFromMemrefOp op : logicalObjectFifos) {
    // Get all parallel scf.forall operations containing users of this logical
    // objectFifo.
    llvm::SmallSetVector<scf::ForallOp, 4> localUsers;
    for (Operation *userOp : op->getUsers()) {
      if (scf::ForallOp forallOp = getParentForallWithParallelMapping(userOp)) {
        localUsers.insert(forallOp);
      }
    }

    // If no parallel scf.forall users found, we don't do any rewrite.
    if (localUsers.empty()) continue;

    // Clone into every scf.forall operation's context.
    for (scf::ForallOp forallOp : localUsers) {
      rewriter.setInsertionPoint(forallOp);
      auto clone = rewriter.clone(*(op.getOperation()));
      op->replaceUsesWithIf(clone, [&](OpOperand &use) {
        scf::ForallOp localParentForall =
            getParentForallWithParallelMapping(use.getOwner());
        return localParentForall && localParentForall == forallOp;
      });
    }

    if (op->use_empty()) rewriter.eraseOp(op);
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIELocalizeLogicalObjectFifoPass() {
  return std::make_unique<AMDAIELocalizeLogicalObjectfifoPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
