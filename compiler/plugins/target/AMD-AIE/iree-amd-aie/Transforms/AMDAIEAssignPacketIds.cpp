// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "llvm/ADT/STLExtras.h"

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

  // Collect all packet flow operations and categorize them into control and
  // normal flows. Control packet flows will be prioritized for packet ID
  // assignment.
  SmallVector<AMDAIE::FlowOp> ctrlPktFlowOps;
  SmallVector<AMDAIE::FlowOp> dataPktFlowOps;
  WalkResult res = parentOp->walk([&](AMDAIE::FlowOp flowOp) {
    if (!flowOp.getIsPacketFlow()) return WalkResult::advance();
    FailureOr<bool> maybeIsControlFlow = flowOp.isControlFlow();
    if (failed(maybeIsControlFlow)) return WalkResult::interrupt();
    if (*maybeIsControlFlow) {
      ctrlPktFlowOps.push_back(flowOp);
    } else {
      dataPktFlowOps.push_back(flowOp);
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();
  SmallVector<AMDAIE::FlowOp> allPktFlowOps = std::move(ctrlPktFlowOps);
  allPktFlowOps.append(std::make_move_iterator(dataPktFlowOps.begin()),
                       std::make_move_iterator(dataPktFlowOps.end()));

  // Perform assignment of packet IDs based on the source channels of the flow
  // ops. I.e. `amdaie.flow` ops with the same source channel will get a
  // different packet IDs assigned to accommodate multiple data packets being
  // routed through the same ports.
  DenseMap<AMDAIE::ChannelOp, size_t> channelToPktFlowIndex;
  for (AMDAIE::FlowOp flowOp : allPktFlowOps) {
    FailureOr<AMDAIE::ChannelOp> maybeSourceChannelOp =
        flowOp.getSourceChannelOp();
    if (failed(maybeSourceChannelOp)) return signalPassFailure();
    AMDAIE::ChannelOp sourceChannelOp = *maybeSourceChannelOp;
    size_t pktFlowIndex = channelToPktFlowIndex[sourceChannelOp];
    if (pktFlowIndex > deviceModel.getPacketIdMaxIdx()) {
      flowOp.emitOpError()
          << "ran out of packet IDs to assign for source channel";
      return signalPassFailure();
    }
    IntegerAttr pktIdAttr = IntegerAttr::get(ui8ty, pktFlowIndex);
    rewriter.setInsertionPoint(flowOp);
    rewriter.replaceOpWithNewOp<AMDAIE::FlowOp>(
        flowOp, flowOp.getSources(), flowOp.getTargets(),
        flowOp.getIsPacketFlow(), pktIdAttr);
    channelToPktFlowIndex[sourceChannelOp]++;
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignPacketIdsPass() {
  return std::make_unique<AMDAIEAssignPacketIdsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
