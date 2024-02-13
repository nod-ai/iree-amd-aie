// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
// This function is to take certain operands from a matmul op and used for
// new allocation creation. Since we have different pipelines (packing/padding)
// and padding strategies. We use buffereizeLevel to indicate what pipeline
// and padding strategy we are targeting.
static FailureOr<SmallVector<Value>> getOperandsToBufferize(
    int64_t bufferizeLevel, linalg::LinalgOp &linalgOp) {
  switch (bufferizeLevel) {
    // Use with packing pipeline, create new allocations for Lhs, Rhs and Out.
    case -1:
      return SmallVector<Value>(linalgOp->getOperands());
    // Use with paddingLevel == 0, create new allocations for Lhs, Rhs and Out.
    case 0:
      return SmallVector<Value>(linalgOp->getOperands());
    // Use with paddingLevel == 1, create new allocation only for Out.
    case 1:
      return SmallVector<Value>(linalgOp.getDpsInits());
    // Use with paddingLevel == 2, create new allocations only for Lhs, Rhs.
    case 2:
      return SmallVector<Value>(linalgOp.getDpsInputs());
    default:
      return failure();
  }
}

/// Utility to create and return AMDAIEMemSpaceAttr with a given integer
/// `memorySpace`.
static AMDAIEMemSpaceAttr getMemorySpaceAttr(RewriterBase &rewriter,
                                             int64_t memorySpace) {
  AMDAIEMemSpace memSpace;
  switch (memorySpace) {
    case 1:
      memSpace = AMDAIEMemSpace::L1;
      break;
    case 2:
      memSpace = AMDAIEMemSpace::L2;
      break;
    default:
      assert(false && "incorrect memory space");
  }
  return AMDAIEMemSpaceAttr::get(rewriter.getContext(), memSpace);
}

class AMDAIEBufferizeToAllocationPass
    : public impl::AMDAIEBufferizeToAllocationBase<
          AMDAIEBufferizeToAllocationPass> {
 public:
  AMDAIEBufferizeToAllocationPass() = default;
  AMDAIEBufferizeToAllocationPass(const AMDAIEBufferizeToAllocationPass &pass) {
  }
  AMDAIEBufferizeToAllocationPass(
      const AMDAIEBufferizeToAllocationOptions &options)
      : AMDAIEBufferizeToAllocationBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, linalg::LinalgDialect>();
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
    AMDAIEMemSpaceAttr memorySpaceAttr =
        getMemorySpaceAttr(rewriter, memorySpace);
    rewriter.setInsertionPointAfter(operand.getDefiningOp());
    if (failed(applyBufferizeToAllocation(rewriter, operand.getDefiningOp(),
                                          memorySpaceAttr))) {
      funcOp->emitOpError("failed bufferizing to allocations");
      return signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEBufferizeToAllocationPass(
    AMDAIEBufferizeToAllocationOptions options) {
  return std::make_unique<AMDAIEBufferizeToAllocationPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
