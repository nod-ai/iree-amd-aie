// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"

#define DEBUG_TYPE "iree-amdaie-assign-logical-objectfifo-depth"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEAssignLogicalObjectFifoDepthPass
    : public impl::AMDAIEAssignLogicalObjectFifoDepthBase<
          AMDAIEAssignLogicalObjectFifoDepthPass> {
 public:
  AMDAIEAssignLogicalObjectFifoDepthPass() = default;
  AMDAIEAssignLogicalObjectFifoDepthPass(
      const AMDAIEAssignLogicalObjectFifoDepthPass &pass){};
  AMDAIEAssignLogicalObjectFifoDepthPass(
      const AMDAIEAssignLogicalObjectFifoDepthOptions &options)
      : AMDAIEAssignLogicalObjectFifoDepthBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIEAssignLogicalObjectFifoDepthPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp->getContext());
  // Assign buffer depths based on provided options.
  WalkResult res = parentOp->walk(
      [&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
        uint8_t memSpace = logicalObjectFifo.getMemorySpaceAsUInt();
        uint8_t bufferDepth{0};
        if (memSpace == 0) {
          bufferDepth = l3BufferDepth;
        } else if (memSpace == 1) {
          bufferDepth = l2BufferDepth;
        } else if (memSpace == 2) {
          bufferDepth = l1BufferDepth;
        } else {
          return WalkResult::advance();
        }
        MemRefType elementType = logicalObjectFifo.getMemrefType();
        rewriter.setInsertionPoint(logicalObjectFifo);
        rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
            logicalObjectFifo,
            LogicalObjectFifoType::get(elementType, bufferDepth),
            logicalObjectFifo.getMemref(), logicalObjectFifo.getTiles());
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignLogicalObjectFifoDepthPass(
    AMDAIEAssignLogicalObjectFifoDepthOptions options) {
  return std::make_unique<AMDAIEAssignLogicalObjectFifoDepthPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
