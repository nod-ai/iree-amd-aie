// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/aie_runtime/Utils/ChannelGenerator.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"

#define DEBUG_TYPE "iree-amdaie-assign-channels"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Assign channels to `amdaie.connection` ops.
LogicalResult assignChannels(AMDAIE::WorkgroupOp workgroupOp) {
  IRRewriter rewriter(workgroupOp->getContext());
  ChannelGenerator generator;
  SmallVector<AMDAIE::ConnectionOp> connectionOps;
  workgroupOp->walk([&](AMDAIE::ConnectionOp connectionOp) {
    connectionOps.push_back(connectionOp);
  });
  for (AMDAIE::ConnectionOp connectionOp : connectionOps) {
    auto sourceLogicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
            connectionOp.getSource().getDefiningOp());
    if (!sourceLogicalObjFifo) {
      return connectionOp.emitOpError()
             << "expected a `LogicalObjFifoOpInterface` source";
    }
    auto targetLogicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
            connectionOp.getTarget().getDefiningOp());
    if (!targetLogicalObjFifo) {
      return connectionOp.emitOpError()
             << "expected a `LogicalObjFifoOpInterface` target";
    }

    rewriter.setInsertionPoint(connectionOp);
    SmallVector<Value> sourceChannels;
    for (Value tile : sourceLogicalObjFifo.getTiles()) {
      uint8_t channel = generator.getProducerDMAChannel(tile);
      auto channelOp = rewriter.create<AMDAIE::ChannelOp>(
          rewriter.getUnknownLoc(), tile, channel);
      sourceChannels.push_back(channelOp.getResult());
    }
    SmallVector<Value> targetChannels;
    for (Value tile : targetLogicalObjFifo.getTiles()) {
      uint8_t channel = generator.getConsumerDMAChannel(tile);
      auto channelOp = rewriter.create<AMDAIE::ChannelOp>(
          rewriter.getUnknownLoc(), tile, channel);
      targetChannels.push_back(channelOp.getResult());
    }
    rewriter.replaceOpWithNewOp<AMDAIE::ConnectionOp>(
        connectionOp, connectionOp.getTarget(), targetChannels,
        connectionOp.getSource(), sourceChannels,
        connectionOp.getConnectionTypeAttr());
  }
  return success();
}

class AMDAIEAssignChannelsPass
    : public impl::AMDAIEAssignChannelsBase<AMDAIEAssignChannelsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIEAssignChannelsPass::runOnOperation() {
  Operation *parentOp = getOperation();
  SmallVector<AMDAIE::WorkgroupOp> workgroupOps;
  parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    workgroupOps.push_back(workgroupOp);
  });
  for (AMDAIE::WorkgroupOp workgroupOp : workgroupOps) {
    if (failed(assignChannels(workgroupOp))) return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignChannelsPass() {
  return std::make_unique<AMDAIEAssignChannelsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
