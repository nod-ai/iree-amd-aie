// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-bufferize-to-allocation"

namespace mlir::iree_compiler::AMDAIE {

namespace {

static LogicalResult applyBufferizeToAllocation(RewriterBase &rewriter,
                                                Operation *op,
                                                Attribute memorySpace) {
  linalg::BufferizeToAllocationOptions options;
  options.memcpyOp =
      linalg::BufferizeToAllocationOptions::MemcpyOp::MaterializeInDestination;
  options.allocOp = linalg::BufferizeToAllocationOptions::AllocOp::MemrefAlloc;
  options.bufferizeDestinationOnly = true;
  options.emitDealloc = true;

  // Bufferize ops.
  Value buffer =
      linalg::bufferizeToAllocation(rewriter, options, op, memorySpace);
  if (!buffer) {
    LLVM_DEBUG(llvm::dbgs() << "----- failed to bufferize operation -----\n");
    return failure();
  }
  return success();
}

class AMDAIEBufferizeToAllocationPass
    : public AMDAIEBufferizeToAllocationBase<AMDAIEBufferizeToAllocationPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, linalg::LinalgDialect>();
  }

  AMDAIEBufferizeToAllocationPass() = default;
  AMDAIEBufferizeToAllocationPass(int64_t memorySpace = 1) {
    this->memorySpace.setValue(memorySpace);
  }
  void runOnOperation() override;
};

void AMDAIEBufferizeToAllocationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp innerModule = variantOp.getInnerModule();

  for (func::FuncOp funcOp : innerModule.getOps<func::FuncOp>()) {
    linalg::LinalgOp linalgOp;
    funcOp->walk([&](linalg::LinalgOp op) {
      if (linalg::isaContractionOpInterface(op)) {
        linalgOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    llvm::outs() << linalgOp << "\n";
    if (!linalgOp) {
      LLVM_DEBUG(llvm::dbgs() << "----- skip, no linalg op -----\n");
      return;
    }

    IRRewriter rewriter(context);

    // Find the producer ops for linalg (matmul) op, and bufferizes them in new
    // allocations
    for (auto operand : linalgOp->getOperands()) {
      auto memorySpaceAttr = rewriter.getI64IntegerAttr(memorySpace);
      rewriter.setInsertionPointAfter(operand.getDefiningOp());
      if (failed(applyBufferizeToAllocation(rewriter, operand.getDefiningOp(),
                                            memorySpaceAttr))) {
        funcOp->emitOpError("failed bufferizing to allocations");
        return signalPassFailure();
      }
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createAMDAIEBufferizeToAllocationPass(int64_t memorySpace) {
  return std::make_unique<AMDAIEBufferizeToAllocationPass>(memorySpace);
}
}  // namespace mlir::iree_compiler::AMDAIE
