// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-amdaie-assign-tiles-to-object-fifo"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEAssignTilesToObjectFifoPass : public impl::AMDAIEAssignTilesToObjectFifoBase<AMDAIEAssignTilesToObjectFifoPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  AMDAIEAssignTilesToObjectFifoPass() = default;
  AMDAIEAssignTilesToObjectFifoPass(const AMDAIEAssignTilesToObjectFifoPass &pass){};
  void runOnOperation() override;
};

void AMDAIEAssignTilesToObjectFifoPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp);
  parentOp->walk([&](func::FuncOp funcOp) { /* do something */ });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignTilesToObjectFifoPass() {
  return std::make_unique<AMDAIEAssignTilesToObjectFifoPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
