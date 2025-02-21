// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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

/// Utility to fetch input and output operands from the LinalgOp (matmul or
/// elementwise op). For matmul-elementwise special case, since one of the
/// elementwise op's input is the output of the matmul op and has already been
/// promoted, there is no need to promote such operand again.
static SmallVector<Value> getInputOutputOperands(linalg::LinalgOp &linalgOp) {
  SmallVector<Value> operands;
  for (Value operand : linalgOp->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType())) continue;
    if (isElementwise(linalgOp) && isMatmulInDefChain(operand)) continue;
    operands.push_back(operand);
  }
  return operands;
}

/// Utility to fetch input operands from the LinalgOp (matmul or elementwise
/// op). For matmul-elementwise special case, since one of the elementwise op's
/// input is the output of the matmul op and has already been promoted, there is
/// no need to promote such operand again.
static SmallVector<Value> getInputOperands(linalg::LinalgOp &linalgOp) {
  SmallVector<Value> operands;
  for (Value operand : linalgOp.getDpsInputs()) {
    if (!isa<RankedTensorType>(operand.getType())) continue;
    if (isElementwise(linalgOp) && isMatmulInDefChain(operand)) continue;
    operands.push_back(operand);
  }
  return operands;
}

/// Utility to fetch pack operands at a specified depth from the LinalgOp's
/// input operands.
static FailureOr<SmallVector<Value>> getPackOrCopyOperands(
    linalg::LinalgOp linalgOp, uint32_t depthLevel) {
  SmallVector<Value> operands;
  for (auto input : llvm::enumerate(linalgOp.getDpsInputs())) {
    uint32_t currentLevel{0};
    Operation *currentOp = input.value().getDefiningOp();
    while (currentLevel < depthLevel && currentOp != nullptr) {
      if (dyn_cast<linalg::PackOp>(currentOp)) {
        currentLevel++;
        if (currentLevel == depthLevel) break;
      } else if (dyn_cast<linalg::CopyOp>(currentOp)) {
        currentLevel++;
        if (currentLevel == depthLevel) break;
      }
      currentOp = currentOp->getOperand(0).getDefiningOp();
    }
    // The defining op has to be a pack or a copy op, fail otherwise.
    if (!currentOp) {
      return linalgOp.emitOpError()
             << "operand #" << input.index()
             << " only has pack/copy ops to depth " << currentLevel
             << ", but request is for a depth " << depthLevel
             << " pack/copy op.";
    }
    // We only want to fetch the input operand of the pack op.
    operands.push_back(currentOp->getResult(0));
  }
  return operands;
}

// This function helps to fetch operands of either a LinalgOp or its defining
// ops, based on which operands the caller wants to bufferize via
// `bufferizeOperand` parameter.
static FailureOr<SmallVector<Value>> getOperandsToBufferize(
    BufferizeOperand bufferizeOperand, linalg::LinalgOp &linalgOp,
    uint32_t inputDepth) {
  switch (bufferizeOperand) {
    /// Create new allocations for Lhs, Rhs and Out.
    case BufferizeOperand::LinalgInputOutput:
      return getInputOutputOperands(linalgOp);
    /// Create new allocation only for Lhs, Rhs.
    case BufferizeOperand::LinalgInput:
      return getInputOperands(linalgOp);
    /// Create new allocations only for Out.
    case BufferizeOperand::LinalgOutput:
      return SmallVector<Value>(linalgOp.getDpsInits());
    /// Create new allocations for operands from the pack ops.
    case BufferizeOperand::PackOrCopyInput:
      return getPackOrCopyOperands(linalgOp, inputDepth);
    default:
      return failure();
  }
}

/// Utility to create and return AMDAIEMemSpaceAttr with a given integer
/// `memorySpace`.
static AMDAIEMemSpaceAttr getMemorySpaceAttr(RewriterBase &rewriter,
                                             int64_t memorySpace) {
  AMDAIEMemSpace memSpace = AMDAIEMemSpace::None;
  switch (memorySpace) {
    case 0:
      memSpace = AMDAIEMemSpace::Global;
      break;
    case 1:
      memSpace = AMDAIEMemSpace::Shared;
      break;
    case 2:
      memSpace = AMDAIEMemSpace::Local;
      break;
    default:
      assert(false && "incorrect memory space");
      break;
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
  mlir::FunctionOpInterface funcOp = getOperation();
  linalg::LinalgOp linalgOp;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](linalg::LinalgOp op) {
    if (!isElementwise(op) && !linalg::isaContractionOpInterface(op) &&
        !linalg::isaConvolutionOpInterface(op)) {
      return WalkResult::advance();
    }
    if (isa<linalg::FillOp, linalg::CopyOp>(op)) {
      return WalkResult::advance();
    }
    // Use flag `bufferizeElementwise` to indicate whether the target for
    // bufferization is an elementwise op.
    if (bufferizeElementwise && !isElementwise(op)) {
      return WalkResult::advance();
    }
    if (!bufferizeElementwise && isElementwise(op)) {
      return WalkResult::advance();
    }
    linalgOp = op;
    return WalkResult::interrupt();
  });

  if (!linalgOp) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, no linalg op -----\n");
    return;
  }

  IRRewriter rewriter(context);

  // Find the producer ops for linalg (matmul) op, and bufferizes them in new
  // allocations.
  FailureOr<SmallVector<Value>> operandsToBufferize =
      getOperandsToBufferize(bufferizeOperand, linalgOp, inputDepth);
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
