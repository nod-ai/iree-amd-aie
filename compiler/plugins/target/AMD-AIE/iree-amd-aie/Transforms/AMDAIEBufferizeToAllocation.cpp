// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
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

// TODO(avarma): This is a temporary workaround until we have PaddingStrategy
// Currently handles a Matmul. Given a Matmul(Lhs, Rhs, Out) and a given
// `bufferizeLevel`, the following is what we bufferize :
//    bufferizeLevel == -1 -> This is for packing op bufferization.
//    bufferizeLevel == 0 -> Lhs, Rhs and Out.
//    bufferizeLevel == 1 -> Out.
//    bufferizeLevel == 2 -> Lhs and Rhs.
static FailureOr<SmallVector<Value>> getOperandsToBufferize(
    int64_t bufferizeLevel, linalg::LinalgOp &linalgOp) {
  if (bufferizeLevel <= 0) {
    return SmallVector<Value>(linalgOp->getOperands());
  } else if (bufferizeLevel == 1) {
    return SmallVector<Value>(linalgOp.getDpsInits());
  } else if (bufferizeLevel == 2) {
    return SmallVector<Value>(linalgOp.getDpsInputs());
  } else {
    return failure();
  }
}

class AMDAIEBufferizeToAllocationPass
    : public AMDAIEBufferizeToAllocationBase<AMDAIEBufferizeToAllocationPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, linalg::LinalgDialect>();
  }

  AMDAIEBufferizeToAllocationPass() = default;
  AMDAIEBufferizeToAllocationPass(int64_t memorySpace = 1,
                                  int64_t bufferizeLevel = 0) {
    this->memorySpace.setValue(memorySpace);
    this->bufferizeLevel.setValue(bufferizeLevel);
  }
  void runOnOperation() override;
};

void AMDAIEBufferizeToAllocationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();
  linalg::LinalgOp linalgOp;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](linalg::LinalgOp op) {
    if (linalg::isaContractionOpInterface(op)) {
      linalgOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!linalgOp) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, no linalg op -----\n");
    return;
  }

  IRRewriter rewriter(context);

  // Find the producer ops for linalg (matmul) op, and bufferizes them in new
  // allocations.
  FailureOr<SmallVector<Value>> bufferizeOperands =
      getOperandsToBufferize(bufferizeLevel, linalgOp);
  if (failed(bufferizeOperands)) {
    linalgOp->emitOpError("could not fetch operands to bufferize");
    return signalPassFailure();
  }

  for (auto operand : *bufferizeOperands) {
    auto memorySpaceAttr = rewriter.getI64IntegerAttr(memorySpace);
    rewriter.setInsertionPointAfter(operand.getDefiningOp());
    if (failed(applyBufferizeToAllocation(rewriter, operand.getDefiningOp(),
                                          memorySpaceAttr))) {
      funcOp->emitOpError("failed bufferizing to allocations");
      return signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createAMDAIEBufferizeToAllocationPass(int64_t memorySpace,
                                      int64_t bufferizeLevel) {
  return std::make_unique<AMDAIEBufferizeToAllocationPass>(memorySpace,
                                                           bufferizeLevel);
}
}  // namespace mlir::iree_compiler::AMDAIE