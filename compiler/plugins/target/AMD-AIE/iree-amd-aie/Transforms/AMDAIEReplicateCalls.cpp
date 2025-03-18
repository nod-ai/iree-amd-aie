// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-amdaie-replicate-calls"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEReplicateCallsPass
    : public impl::AMDAIEReplicateCallsBase<AMDAIEReplicateCallsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, LLVM::LLVMDialect,
                    func::FuncDialect>();
  }
  AMDAIEReplicateCallsPass() = default;
  AMDAIEReplicateCallsPass(const AMDAIEReplicateCallsPass &pass){};

  AMDAIEReplicateCallsPass(const AMDAIEReplicateCallsOptions &opts)
      : AMDAIEReplicateCallsBase(opts) {}

  void runOnOperation() override;
};

void AMDAIEReplicateCallsPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp);
  // parentOp->walk([&](func::FuncOp funcOp) { /* do something */ });

  auto functionsAndCallers = getFunctionsAndTheirCallers(parentOp);
  for (auto [funcOp, callers] : functionsAndCallers) {
    if (funcOp.getNumArguments() == 0) continue;
    if (callers.empty()) continue;

    if (replication > 1) {
      for (auto caller : callers) {
        rewriter.setInsertionPoint(caller);
        scf::ForOp loop = createForOpWithUnrollingDisabled(
            rewriter, caller.getLoc(), 0, replication, 1);
        rewriter.setInsertionPointToStart(loop.getBody());
        rewriter.clone(*caller.getOperation());
        rewriter.eraseOp(caller);
      }
    }

    // Instead of 0 calls, we call into a function that does nothing. We do this
    // because having no calls can result in DCE that removes more than we want
    if (replication == 0) {
      rewriter.setInsertionPoint(funcOp);
      auto funcType = funcOp.getFunctionType();

      std::string newName = funcOp.getName().str() + "_empty";
      // Create a new function with the same type but an empty body.
      auto emptyReplacement =
          rewriter.create<func::FuncOp>(funcOp.getLoc(), newName, funcType);

      emptyReplacement.setSymName(newName);
      emptyReplacement.setPrivate();

      // Add a single block to it and insert a return op.
      auto &entryBlock = *emptyReplacement.addEntryBlock();
      rewriter.setInsertionPointToEnd(&entryBlock);
      rewriter.create<func::ReturnOp>(funcOp.getLoc());

      for (func::CallOp callOp : callers) {
        rewriter.setInsertionPoint(callOp);
        rewriter.replaceOpWithNewOp<func::CallOp>(
            callOp, emptyReplacement.getName(), callOp.getResultTypes(),
            callOp.getOperands());
      }
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEReplicateCallsPass(
    AMDAIEReplicateCallsOptions options) {
  return std::make_unique<AMDAIEReplicateCallsPass>(options);
}
}  // namespace mlir::iree_compiler::AMDAIE
