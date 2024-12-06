// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"

#define DEBUG_TYPE "iree-amdaie-assign-packet-ids"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEAssignPacketIdsPass
    : public impl::AMDAIEAssignPacketIdsBase<AMDAIEAssignPacketIdsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  void runOnOperation() override;
};

void AMDAIEAssignPacketIdsPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp->getContext());
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(parentOp);
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    parentOp->emitOpError()
        << "has no AMDAIEDevice in the target attribute configuration. This "
           "device-specific information is required to assign packet IDs "
           "within the resource constraints";
    return signalPassFailure();
  }
  AMDAIE::AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(maybeDevice.value());
  auto ui8ty =
      IntegerType::get(rewriter.getContext(), 8, IntegerType::Unsigned);

  // Perform assignment of packet IDs based on the source channels of the flow
  // ops. I.e. `amdaie.flow` ops with the same source channel will get a
  // different packet IDs assigned to accommodate multiple data packets being
  // routed through the same ports.
  DenseMap<AMDAIE::ChannelOp, size_t> channelToPktFlowIndex;
  WalkResult res =
      parentOp->walk([&](AMDAIE::FlowOp flowOp) {
        if (!flowOp.getIsPacketFlow()) return WalkResult::advance();
        SmallVector<Value> sourceChannels = flowOp.getSources();
        if (sourceChannels.size() == 0) {
          flowOp.emitOpError() << "with no source channel is unsupported";
          return WalkResult::interrupt();
        }
        if (sourceChannels.size() > 1) {
          flowOp.emitOpError()
              << "with multiple source channels is unsupported";
          return WalkResult::interrupt();
        }
        auto sourceChannelOp = dyn_cast_if_present<AMDAIE::ChannelOp>(
            sourceChannels[0].getDefiningOp());
        if (!sourceChannelOp) {
          flowOp.emitOpError() << "source should be an `amdaie.channel` op";
          return WalkResult::interrupt();
        }
        size_t pktFlowIndex = channelToPktFlowIndex[sourceChannelOp];
        if (pktFlowIndex > deviceModel.getPacketIdMaxIdx()) {
          flowOp.emitOpError()
              << "ran out of packet IDs to assign for source channel";
          return WalkResult::interrupt();
        }
        IntegerAttr pktIdAttr = IntegerAttr::get(ui8ty, pktFlowIndex);
        rewriter.setInsertionPoint(flowOp);
        rewriter.replaceOpWithNewOp<AMDAIE::FlowOp>(
            flowOp, flowOp.getSources(), flowOp.getTargets(),
            flowOp.getIsPacketFlow(), pktIdAttr);
        channelToPktFlowIndex[sourceChannelOp]++;
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignPacketIdsPass() {
  return std::make_unique<AMDAIEAssignPacketIdsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
