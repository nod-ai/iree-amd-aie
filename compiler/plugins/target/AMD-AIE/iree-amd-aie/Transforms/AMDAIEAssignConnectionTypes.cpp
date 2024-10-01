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
  // Assign connection types based on provided options.
  AMDAIE::ConnectionType connectionType = enablePacketFlow
                                              ? AMDAIE::ConnectionType::Packet
                                              : AMDAIE::ConnectionType::Circuit;
  ConnectionTypeAttr connectionTypeAttr =
      ConnectionTypeAttr::get(rewriter.getContext(), connectionType);
  WalkResult res = parentOp->walk([&](AMDAIE::ConnectionOp connectionOp) {
    rewriter.setInsertionPoint(connectionOp);
    rewriter.replaceOpWithNewOp<AMDAIE::ConnectionOp>(
        connectionOp, connectionOp.getTarget(),
        connectionOp.getTargetChannels(), connectionOp.getSource(),
        connectionOp.getSourceChannels(), connectionTypeAttr);
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
