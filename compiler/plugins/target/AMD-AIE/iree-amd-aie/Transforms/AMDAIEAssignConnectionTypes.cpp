// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"

#define DEBUG_TYPE "iree-amdaie-assign-connection-types"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEAssignConnectionTypesPass
    : public impl::AMDAIEAssignConnectionTypesBase<
          AMDAIEAssignConnectionTypesPass> {
 public:
  AMDAIEAssignConnectionTypesPass(
      const AMDAIEAssignConnectionTypesOptions &options)
      : AMDAIEAssignConnectionTypesBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIEAssignConnectionTypesPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp->getContext());

  WalkResult res = parentOp->walk([&](AMDAIE::ConnectionOp connectionOp) {
    rewriter.setInsertionPoint(connectionOp);

    // Determine the source and target memory spaces of the connection.
    auto sourceLogicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
            connectionOp.getSource().getDefiningOp());
    auto targetLogicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
            connectionOp.getTarget().getDefiningOp());
    if (!sourceLogicalObjFifo || !targetLogicalObjFifo) {
      connectionOp.emitError(
          "source and target of connection must be logical object fifos");
      return WalkResult::interrupt();
    }
    uint8_t sourceMemSpace = sourceLogicalObjFifo.getMemorySpaceAsUInt();
    uint8_t targetMemSpace = targetLogicalObjFifo.getMemorySpaceAsUInt();

    // Default connection type is circuit.
    AMDAIE::ConnectionType connectionType = AMDAIE::ConnectionType::Circuit;
    // Use the memory space to determine if the connetion belongs to the kernel
    // input or output, and set the connection type accordingly.
    if (((sourceMemSpace < targetMemSpace) && enableInputPacketFlow) ||
        ((sourceMemSpace > targetMemSpace) && enableOutputPacketFlow)) {
      connectionType = AMDAIE::ConnectionType::Packet;
    }

    ConnectionTypeAttr connectionTypeAttr =
        ConnectionTypeAttr::get(rewriter.getContext(), connectionType);
    rewriter.replaceOpWithNewOp<AMDAIE::ConnectionOp>(
        connectionOp, connectionOp.getTarget(),
        connectionOp.getTargetChannels(), connectionOp.getSource(),
        connectionOp.getSourceChannels(), connectionTypeAttr,
        /*flow*/ nullptr);
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignConnectionTypesPass(
    AMDAIEAssignConnectionTypesOptions options) {
  return std::make_unique<AMDAIEAssignConnectionTypesPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
