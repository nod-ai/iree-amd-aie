// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEOpUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"

#define DEBUG_TYPE "iree-amdaie-insert-aie-workgroup"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Normalize the loop bounds of `scf.forall` operations within the module.
LogicalResult normalizeModuleLoopBounds(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  WalkResult res = moduleOp->walk([&](scf::ForallOp forallOp) {
    if (failed(normalizeLoopBounds(rewriter, forallOp))) {
      forallOp.emitOpError() << "failed to normalize loop bounds";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

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

/// Insert workgroups around `scf::Forall` ops with thread mapping.
/// NOTE: This can be extended to block mapping as well later.
LogicalResult insertWorkgroupOps(mlir::ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  moduleOp->walk([&](scf::ForallOp forallOp) {
    if (forallOp.getNumResults() != 0) {
      return WalkResult::advance();
    }
    if (!forallOp.getMapping().has_value()) {
      return WalkResult::advance();
    }
    SmallVector<Attribute> threadMapping =
        llvm::to_vector(forallOp.getMapping()->getValue());
    if (llvm::any_of(threadMapping, [](Attribute map) {
          return !llvm::isa<mlir::gpu::GPUThreadMappingAttr>(map);
        })) {
      return WalkResult::advance();
    }
    if (threadMapping.size() != 2) {
      return WalkResult::advance();
    }
    // Create the workgroup region
    rewriter.setInsertionPoint(forallOp);
    auto workgroupOp =
        rewriter.create<AMDAIE::WorkgroupOp>(rewriter.getUnknownLoc());
    rewriter.moveOpBefore(forallOp, workgroupOp.getControlCode());
    return WalkResult::advance();
  });
  return success();
}

/// Insert core ops inside innermost forall ops around computational ops and
/// add synchronization ops along the way to synchronize with surrounding
/// dma ops.
LogicalResult insertCoreOpsInWorkgroup(mlir::ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  WalkResult res = moduleOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    // Find innermost forall loops
    SmallVector<scf::ForallOp> innermostForallLoops =
        getInnermostForallLoops(workgroupOp);
    if (innermostForallLoops.empty()) {
      workgroupOp.emitOpError() << "contains no nested forall op";
      return WalkResult::interrupt();
    }
    if (innermostForallLoops.size() != 1) {
      workgroupOp.emitOpError()
          << "multiple innermost foralls not supported for now";
      return WalkResult::interrupt();
    }
    scf::ForallOp innermostForall = innermostForallLoops[0];
    auto parentOps = getInclusiveParentsOfType<scf::ForallOp>(innermostForall);
    DenseMap<Attribute, Value> attrMapping;
    getAttributeMapping(parentOps, attrMapping);
    if (!attrMapping.contains(gpu::threadX(workgroupOp->getContext())) ||
        !attrMapping.contains(gpu::threadY(workgroupOp->getContext()))) {
      workgroupOp.emitOpError() << "no forall with thread mapping found";
      return WalkResult::interrupt();
    }
    Value threadX = attrMapping[gpu::threadX(workgroupOp->getContext())];
    Value threadY = attrMapping[gpu::threadY(workgroupOp->getContext())];
    // Create CoreOp at the end of the innermost forall
    rewriter.setInsertionPoint(innermostForall.getBody()->getTerminator());
    auto coreOp = rewriter.create<AMDAIE::CoreOp>(rewriter.getUnknownLoc(),
                                                  threadX, threadY);
    Region &region = coreOp.getRegion();
    Block *newBlock = rewriter.createBlock(&region);
    rewriter.setInsertionPointToStart(newBlock);
    auto endOp = rewriter.create<AMDAIE::EndOp>(rewriter.getUnknownLoc());

    // Walk all operations in the workgroup and fill in the CoreOp with
    // computational ops (linalg) and synchronization ops to synchronize
    // with the workgroup DMA ops.
    WalkResult workgroupRes = workgroupOp->walk([&](Operation *op) {
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
          // From L1, so insert a logical objectfifo produce op
          rewriter.setInsertionPoint(endOp);
          rewriter.create<AMDAIE::LogicalObjectFifoProduce>(
              rewriter.getUnknownLoc(), SmallVector<Type, 1>{}, dmaOp);
        } else if (targetMemspace &&
                   dyn_cast<IntegerAttr>(targetMemspace).getInt() == 2) {
          // To L1, so insert a logical objectfifo consume op
          rewriter.setInsertionPoint(endOp);
          rewriter.create<AMDAIE::LogicalObjectFifoConsume>(
              rewriter.getUnknownLoc(), SmallVector<Type, 1>{}, dmaOp);
        }
      } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        rewriter.moveOpBefore(linalgOp, endOp);
      }
      return WalkResult::advance();
    });
    if (workgroupRes.wasInterrupted()) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

class AMDAIEInsertAIEWorkgroupPass
    : public impl::AMDAIEInsertAIEWorkgroupBase<AMDAIEInsertAIEWorkgroupPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, AMDAIEDialect>();
  }

  AMDAIEInsertAIEWorkgroupPass() = default;
  AMDAIEInsertAIEWorkgroupPass(const AMDAIEInsertAIEWorkgroupPass &pass){};
  void runOnOperation() override;
};

void AMDAIEInsertAIEWorkgroupPass::runOnOperation() {
  // Normalize the loop bounds of `scf.forall` operations within the module.
  if (failed(normalizeModuleLoopBounds(getOperation()))) {
    return signalPassFailure();
  }
  if (failed(insertWorkgroupOps(getOperation()))) {
    return signalPassFailure();
  }
  if (failed(insertCoreOpsInWorkgroup(getOperation()))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEInsertAIEWorkgroupPass() {
  return std::make_unique<AMDAIEInsertAIEWorkgroupPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
