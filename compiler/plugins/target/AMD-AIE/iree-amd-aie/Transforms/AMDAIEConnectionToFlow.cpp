// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"

#define DEBUG_TYPE "iree-amdaie-connection-to-flow"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEConnectionToFlowPass
    : public impl::AMDAIEConnectionToFlowBase<AMDAIEConnectionToFlowPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  void runOnOperation() override;
};

void AMDAIEConnectionToFlowPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp->getContext());
  // TODO(jornt): currently, don't delete connections as they are still
  // needed for lowering to AIE dialect dma_bds. This will be changed in the
  // future.
  WalkResult res = parentOp->walk([&](AMDAIE::ConnectionOp connectionOp) {
    rewriter.setInsertionPoint(connectionOp);
    std::optional<AMDAIE::ConnectionType> connectionType =
        connectionOp.getConnectionType();
    bool isPacketFlow = connectionType && connectionType.value() ==
                                              AMDAIE::ConnectionType::Packet;
    auto flowOp = rewriter.create<AMDAIE::FlowOp>(
        rewriter.getUnknownLoc(), connectionOp.getSourceChannels(),
        connectionOp.getTargetChannels(), isPacketFlow, /*packetId*/ nullptr);
    rewriter.replaceOpWithNewOp<AMDAIE::ConnectionOp>(
        connectionOp, connectionOp.getTarget(),
        connectionOp.getTargetChannels(), connectionOp.getSource(),
        connectionOp.getSourceChannels(), connectionOp.getConnectionTypeAttr(),
        flowOp);
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEConnectionToFlowPass() {
  return std::make_unique<AMDAIEConnectionToFlowPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
