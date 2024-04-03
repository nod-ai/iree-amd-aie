// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
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

/// Utility to get operands from the defining ops of LinalgOp input operands.
/// For example, we want to promote %pack0 and %pack1 to memory.
/// %pack0 = tensor.pack % arg0
/// %pack1 = tensor.pack % arg1
/// %pack2 = tensor.pack % pack0
/// %pack3 = tensor.pack % pack1
/// %generic = linalg.generic ins(%pack2, %pack3)
static FailureOr<SmallVector<Value>> getOperandsFromDefOp(
    linalg::LinalgOp &linalgOp) {
  SmallVector<Value> operands;
  for (auto input : linalgOp.getDpsInputs()) {
    auto defOp = input.getDefiningOp();
    if (!defOp || !defOp->getOperand(0)) {
      return failure();
    }
    operands.push_back(defOp->getOperand(0));
  }
  return operands;
}

// This function helps to fetch operands of either a LinalgOp or its defining
// ops, based on which operands the caller wants to bufferize via
// `bufferizeOperand` parameter.
static FailureOr<SmallVector<Value>> getOperandsToBufferize(
    BufferizeOperand bufferizeOperand, linalg::LinalgOp &linalgOp) {
  switch (bufferizeOperand) {
    // Create new allocations for Lhs, Rhs and Out.
    case BufferizeOperand::InputOutput:
      return SmallVector<Value>(linalgOp->getOperands());
    // Create new allocation only for Lhs, Rhs.
    case BufferizeOperand::Input:
      return SmallVector<Value>(linalgOp.getDpsInputs());
    // Create new allocations only for Out.
    case BufferizeOperand::Output:
      return SmallVector<Value>(linalgOp.getDpsInits());
    // Create new allocations for operands from the input def ops.
    case BufferizeOperand::DefInput:
      return getOperandsFromDefOp(linalgOp);
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
      memSpace = AMDAIEMemSpace::Shared;
      break;
    case 2:
      memSpace = AMDAIEMemSpace::Local;
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
  FailureOr<SmallVector<Value>> operandsToBufferize =
      getOperandsToBufferize(bufferizeOperand, linalgOp);
  if (failed(operandsToBufferize)) {
    linalgOp->emitOpError("could not fetch operands to bufferize");
    return signalPassFailure();
  }

  for (auto operand : *operandsToBufferize) {
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
