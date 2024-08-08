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
#include "mlir/Dialect/Vector/IR/VectorOps.h"

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

    // Find input and output DMAs that need to be added to the core.
    SmallVector<Value> inputDmas;
    SmallVector<Value> outputDmas;
    WalkResult dmaRes = forallOp->walk([&](AMDAIE::DmaCpyNdOp dmaOp) {
      uint8_t sourceMemspace =
          dmaOp.getSourceObjectFifo().getMemorySpaceAsUInt();
      uint8_t targetMemspace =
          dmaOp.getTargetObjectFifo().getMemorySpaceAsUInt();
      if (sourceMemspace == 2 && targetMemspace == 2) {
        dmaOp->emitOpError()
            << "dma op with both source and target on L1 is not supported";
        return WalkResult::interrupt();
      } else if (sourceMemspace == 2) {
        outputDmas.push_back(dmaOp);
      } else if (targetMemspace == 2) {
        inputDmas.push_back(dmaOp);
      }
      return WalkResult::advance();
    });
    if (dmaRes.wasInterrupted()) return WalkResult::interrupt();

    // Create CoreOp at the end of the innermost forall
    rewriter.setInsertionPoint(forallOp.getBody()->getTerminator());
    auto coreOp = rewriter.create<AMDAIE::CoreOp>(
        rewriter.getUnknownLoc(), threadX, threadY, inputDmas, outputDmas);
    Region &region = coreOp.getRegion();
    Block *newBlock = rewriter.createBlock(&region);
    rewriter.setInsertionPointToStart(newBlock);
    auto endOp = rewriter.create<AMDAIE::EndOp>(rewriter.getUnknownLoc());

    // Walk all operations in the workgroup and fill in the CoreOp with
    // computational ops.
    WalkResult forallRes = forallOp->walk([&](Operation *op) {
      // Skip operations already inside core ops
      if (op->getParentOfType<AMDAIE::CoreOp>()) return WalkResult::advance();
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        rewriter.setInsertionPoint(endOp);
        rewriter.moveOpBefore(linalgOp, endOp);
      } else if (isa<vector::ContractionOp>(op)) {
        // Because vector.contract op is surrounded by vectorized loop nest, we
        // need to traverse to the outermost loop to move the entire vectorized
        // computation within amdaie.core op.
        Operation *currOp = op;
        while (currOp->getParentOp() != forallOp) {
          currOp = currOp->getParentOp();
        }
        rewriter.setInsertionPoint(endOp);
        rewriter.moveOpBefore(currOp, endOp);
      } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
        // Fetch name of the ukernel function to look up its declaration in the
        // Symbol table.
        StringRef fnName = callOp.getCallee();
        auto fnDecl = dyn_cast_or_null<func::FuncOp>(
            SymbolTable::lookupSymbolIn(moduleOp, fnName));
        assert(fnDecl && "expected function declaration");
        assert(fnDecl->hasAttr("link_with") &&
               "expected 'link_with' construct for the function declaration");
        // From the declaration of the function, we extract the value of
        // attribute "link_with" and attach it to amdaie.core op.
        // TODO(avarma): What to do when more than one func.call has different
        // ukernel object file linking?
        //               As of now this hasn't turned up yet, so will table this
        //               for now.
        coreOp.setLinkWith(fnDecl->getAttrOfType<StringAttr>("link_with"));
        rewriter.setInsertionPoint(endOp);
        rewriter.moveOpBefore(op, endOp);
      } else if (isa<memref::ExtractStridedMetadataOp>(op)) {
        rewriter.setInsertionPoint(endOp);
        rewriter.moveOpBefore(op, endOp);
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
