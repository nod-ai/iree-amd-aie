// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"

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
  int pktFlowIndex{0};
  WalkResult res = parentOp->walk([&](AMDAIE::FlowOp flowOp) {
    if (pktFlowIndex > deviceModel.getPacketIdMaxIdx()) {
      flowOp.emitOpError() << "ran out of packet IDs to assign";
      return WalkResult::interrupt();
    }
    rewriter.setInsertionPoint(flowOp);
    IntegerAttr pktIdAttr = flowOp.getIsPacketFlow()
                                ? IntegerAttr::get(ui8ty, pktFlowIndex++)
                                : nullptr;
    rewriter.replaceOpWithNewOp<AMDAIE::FlowOp>(
        flowOp, flowOp.getSources(), flowOp.getTargets(),
        flowOp.getIsPacketFlow(), pktIdAttr);
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignPacketIdsPass() {
  return std::make_unique<AMDAIEAssignPacketIdsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
