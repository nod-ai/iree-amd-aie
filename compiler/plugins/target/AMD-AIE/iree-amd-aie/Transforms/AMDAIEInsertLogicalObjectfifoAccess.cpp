// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-insert-logical-objectfifo-access"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Insert `amdaie.logicalobjectfifo.access` operations which retrieve the
/// memrefs from logical objectfifos and update the computational operations to
/// operate on these local memrefs.
LogicalResult insertLogicalObjectFifoAccess(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  SmallVector<AMDAIE::CoreOp> coreOps;
  moduleOp->walk([&](AMDAIE::CoreOp coreOp) { coreOps.push_back(coreOp); });

  for (AMDAIE::CoreOp coreOp : coreOps) {
    DenseMap<Value, std::tuple<AMDAIE::LogicalObjectFifoFromMemrefOp,
                               AMDAIE::MemoryAccess>>
        memrefToLogicalObjectFifo;
    // First walk to collect consume/produce DMA accesses and map respective
    // memrefs to logical objectifos.
    coreOp->walk([&](Operation *op) {
      // TODO(jornt): can we avoid produce/consume?
      if (auto consumeOp = dyn_cast<AMDAIE::LogicalObjectFifoConsume>(op)) {
        Value targetMemref =
            consumeOp.getDmaCpyNdOp().getTargetObjectFifo().getMemref();
        memrefToLogicalObjectFifo[targetMemref] =
            std::make_pair(consumeOp.getDmaCpyNdOp().getTargetObjectFifo(),
                           AMDAIE::MemoryAccess::Read);
      } else if (auto produceOp =
                     dyn_cast<AMDAIE::LogicalObjectFifoProduce>(op)) {
        Value sourceMemref =
            produceOp.getDmaCpyNdOp().getSourceObjectFifo().getMemref();
        memrefToLogicalObjectFifo[sourceMemref] =
            std::make_pair(produceOp.getDmaCpyNdOp().getSourceObjectFifo(),
                           AMDAIE::MemoryAccess::Write);
      }
    });

    WalkResult res = coreOp->walk([&](Operation *op) {
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        for (auto &&[idx, operand] :
             llvm::enumerate(linalgOp->getOpOperands())) {
          if (memrefToLogicalObjectFifo.contains(operand.get())) {
            rewriter.setInsertionPointToStart(coreOp.getBody());
            std::tuple<AMDAIE::LogicalObjectFifoFromMemrefOp,
                       AMDAIE::MemoryAccess>
                value = memrefToLogicalObjectFifo[operand.get()];
            auto accessOp = rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
                rewriter.getUnknownLoc(), std::get<0>(value),
                std::get<1>(value));
            linalgOp->setOperand(idx, accessOp);
          } else if (auto type =
                         llvm::dyn_cast<MemRefType>(operand.get().getType())) {
            Value memref = operand.get();
            rewriter.setInsertionPoint(coreOp);
            auto logicalObjectFifo =
                rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                    rewriter.getUnknownLoc(), LogicalObjectFifoType::get(type),
                    memref);
            rewriter.setInsertionPointToStart(coreOp.getBody());
            auto accessOp = rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
                rewriter.getUnknownLoc(), logicalObjectFifo,
                AMDAIE::MemoryAccess::None);
            linalgOp->setOperand(idx, accessOp);
          }
        }
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return failure();
  }
  return success();
}

class AMDAIEInsertLogicalObjectfifoAccessPass
    : public impl::AMDAIEInsertLogicalObjectfifoAccessBase<
          AMDAIEInsertLogicalObjectfifoAccessPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEInsertLogicalObjectfifoAccessPass() = default;
  AMDAIEInsertLogicalObjectfifoAccessPass(
      const AMDAIEInsertLogicalObjectfifoAccessPass &pass){};
  void runOnOperation() override;
};

void AMDAIEInsertLogicalObjectfifoAccessPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  // Insert `amdaie.logicalobjectfifo.access` operations which retrieve the
  // memrefs from logical objectfifos and update the computational operations to
  // operate on these local memrefs. These access operations will be used to
  // assign local AIE tiles to local logical objectFifos later.
  if (failed(insertLogicalObjectFifoAccess(moduleOp))) {
    moduleOp.emitOpError()
        << "insertion of `amdaie.logicalobjectfif.access` operations failed";
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEInsertLogicalObjectfifoAccessPass() {
  return std::make_unique<AMDAIEInsertLogicalObjectfifoAccessPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
