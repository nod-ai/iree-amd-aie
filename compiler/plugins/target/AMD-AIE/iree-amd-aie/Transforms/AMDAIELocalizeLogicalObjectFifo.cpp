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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-amdaie-localize-logicalobjectfifo"

namespace mlir::iree_compiler::AMDAIE {

namespace {

template <class... T>
scf::ForallOp getMappedForallAncestor(Operation *op) {
  auto forall = op->getParentOfType<scf::ForallOp>();
  while (forall) {
    bool isMapped = [&forall]() {
      auto maybeMapping = forall.getMapping();
      auto hasMapping = maybeMapping.has_value();
      if (!hasMapping) return false;
      auto mapping = maybeMapping->getValue();
      if (mapping.size() == 0) return false;
      auto mappingAttr0 = *mapping.begin();
      if (isa<T...>(mappingAttr0)) return true;
      return false;
    }();
    if (isMapped) return forall;
    forall = forall->getParentOfType<scf::ForallOp>();
  }
  return scf::ForallOp();
}

scf::ForallOp getThreadMappedForallAncestor(Operation *op) {
  return getMappedForallAncestor<mlir::gpu::GPUThreadMappingAttr>(op);
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
    // The op is already within the thread mapped forall, so doesn't need
    // any change.
    if (getThreadMappedForallAncestor(op)) {
      return WalkResult::advance();
    }
    logicalObjectFifos.push_back(op);
    return WalkResult::advance();
  });

  for (AMDAIE::LogicalObjectFifoFromMemrefOp op : logicalObjectFifos) {
    // Get all thread mapped scf.forall operations containing users of this
    // logical objectFifo.
    llvm::SmallSetVector<scf::ForallOp, 4> localUsers;
    for (Operation *userOp : op->getUsers()) {
      if (scf::ForallOp forallOp = getThreadMappedForallAncestor(userOp)) {
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
            getThreadMappedForallAncestor(use.getOwner());
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
