// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the insertion of `amdaie.core` operations in the
// innermost `scf.forall` operations with thread mapping. Each core has a tile
// location which is computed from the for loop's induction variables. This pass
// will probably be updated in the future to work with loops with block mapping
// as well.
//
//===----------------------------------------------------------------------===//

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEOpUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"

#define DEBUG_TYPE "iree-amdaie-insert-cores"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Utility to map the parallel mapping attributes to the corresponding
/// induction variables.
void getAttributeMapping(SmallVector<scf::ForallOp> forallOps,
                         DenseMap<Attribute, Value> &attrMapping) {
  for (auto forallOp : forallOps) {
    if (!forallOp.getMapping().has_value()) continue;
    SmallVector<Attribute> mapping =
        llvm::to_vector(forallOp.getMapping()->getValue());
    auto ivs = forallOp.getInductionVars();
    for (auto &&[attr, iv] : llvm::zip(mapping, ivs)) attrMapping[attr] = iv;
  }
}

/// Insert core ops inside innermost forall ops around computational ops and
/// add synchronization ops along the way to synchronize with surrounding
/// dma ops.
LogicalResult insertCoreOps(mlir::ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  WalkResult res = moduleOp->walk([&](scf::ForallOp forallOp) {
    // Currently, innermost `scf.forall` operations are expected to have thread
    // mapping and are therefore selected for insertion of cores. Advance if no
    // thread mapping.
    SmallVector<Attribute> mapping =
        llvm::to_vector(forallOp.getMapping()->getValue());
    if (!isa<mlir::gpu::GPUThreadMappingAttr>(*mapping.begin()))
      return WalkResult::advance();

    if (!forallOp.isNormalized()) {
      forallOp.emitOpError()
          << "scf.forall operations must be normalized before core "
             "operation insertion";
      return WalkResult::interrupt();
    }
    auto parentOps = getInclusiveParentsOfType<scf::ForallOp>(forallOp);
    DenseMap<Attribute, Value> attrMapping;
    getAttributeMapping(parentOps, attrMapping);
    if (!attrMapping.contains(gpu::threadX(forallOp->getContext())) ||
        !attrMapping.contains(gpu::threadY(forallOp->getContext()))) {
      forallOp.emitOpError() << "no forall with thread mapping found";
      return WalkResult::interrupt();
    }
    Value threadX = attrMapping[gpu::threadX(forallOp->getContext())];
    Value threadY = attrMapping[gpu::threadY(forallOp->getContext())];
    // Create CoreOp at the end of the innermost forall
    rewriter.setInsertionPoint(forallOp.getBody()->getTerminator());
    auto coreOp = rewriter.create<AMDAIE::CoreOp>(rewriter.getUnknownLoc(),
                                                  threadX, threadY);
    Region &region = coreOp.getRegion();
    Block *newBlock = rewriter.createBlock(&region);
    rewriter.setInsertionPointToStart(newBlock);
    auto endOp = rewriter.create<AMDAIE::EndOp>(rewriter.getUnknownLoc());

    // Walk all operations in the workgroup and fill in the CoreOp with
    // computational ops (linalg) and synchronization ops to synchronize
    // with the workgroup DMA ops.
    WalkResult forallRes = forallOp->walk([&](Operation *op) {
      // Skip operations already inside core ops
      if (op->getParentOfType<AMDAIE::CoreOp>()) return WalkResult::advance();
      if (auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(op)) {
        auto sourceMemspace = dmaOp.getSourceObjectFifo().getMemorySpace();
        auto targetMemspace = dmaOp.getTargetObjectFifo().getMemorySpace();
        if (sourceMemspace &&
            dyn_cast<IntegerAttr>(sourceMemspace).getInt() == 2 &&
            targetMemspace &&
            dyn_cast<IntegerAttr>(targetMemspace).getInt() == 2) {
          dmaOp->emitOpError()
              << "dma op with both source and target on L1 is not supported";
          return WalkResult::interrupt();
        } else if (sourceMemspace &&
                   dyn_cast<IntegerAttr>(sourceMemspace).getInt() == 2) {
          // From L1, so insert a logical objectFifo produce op
          rewriter.setInsertionPoint(endOp);
          rewriter.create<AMDAIE::LogicalObjectFifoProduce>(
              rewriter.getUnknownLoc(), SmallVector<Type, 1>{}, dmaOp);
        } else if (targetMemspace &&
                   dyn_cast<IntegerAttr>(targetMemspace).getInt() == 2) {
          // To L1, so insert a logical objectFifo consume op
          rewriter.setInsertionPoint(endOp);
          rewriter.create<AMDAIE::LogicalObjectFifoConsume>(
              rewriter.getUnknownLoc(), SmallVector<Type, 1>{}, dmaOp);
        }
      } else if (isa<linalg::LinalgOp>(op)) {
        Operation *currOp = op;
        while (currOp->getParentOp() != forallOp) {
          currOp = currOp->getParentOp();
        }
        rewriter.setInsertionPoint(endOp);
        rewriter.moveOpBefore(currOp, endOp);
      }
      return WalkResult::advance();
    });
    if (forallRes.wasInterrupted()) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

class AMDAIEInsertCoresPass
    : public impl::AMDAIEInsertCoresBase<AMDAIEInsertCoresPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, AMDAIEDialect>();
  }

  AMDAIEInsertCoresPass() = default;
  AMDAIEInsertCoresPass(const AMDAIEInsertCoresPass &pass){};
  void runOnOperation() override;
};

void AMDAIEInsertCoresPass::runOnOperation() {
  if (failed(insertCoreOps(getOperation()))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEInsertCoresPass() {
  return std::make_unique<AMDAIEInsertCoresPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
