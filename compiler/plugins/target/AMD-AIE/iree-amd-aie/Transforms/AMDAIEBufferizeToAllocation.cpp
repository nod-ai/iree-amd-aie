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
static SmallVector<Value> getInputOutputOperands(
    DestinationStyleOpInterface &dstStyleOp) {
  SmallVector<Value> operands;
  for (Value operand : dstStyleOp->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType())) continue;
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(dstStyleOp.getOperation())) {
      if (isElementwise(linalgOp) && isMatmulInDefChain(operand)) continue;
    }
    operands.push_back(operand);
  }
  return operands;
}

/// Utility to fetch input operands from the LinalgOp (matmul or elementwise
/// op). For matmul-elementwise special case, since one of the elementwise op's
/// input is the output of the matmul op and has already been promoted, there is
/// no need to promote such operand again.
static SmallVector<Value> getInputOperands(
    DestinationStyleOpInterface &dstStyleOp) {
  SmallVector<Value> operands;
  for (Value operand : dstStyleOp.getDpsInputs()) {
    if (!isa<RankedTensorType>(operand.getType())) continue;
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(dstStyleOp.getOperation())) {
      if (isElementwise(linalgOp) && isMatmulInDefChain(operand)) continue;
    }
    operands.push_back(operand);
  }
  return operands;
}

/// Utility to fetch pack operands at a specified depth from the LinalgOp's
/// input operands.
static FailureOr<SmallVector<Value>> getPackOrCopyOperands(
    DestinationStyleOpInterface &dstStyleOp, uint32_t depthLevel) {
  SmallVector<Value> operands;
  for (auto input : llvm::enumerate(dstStyleOp.getDpsInputs())) {
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
      return dstStyleOp.emitOpError()
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
    BufferizeOperand bufferizeOperand, Operation *op, uint32_t inputDepth) {
  auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dstStyleOp) {
    llvm::errs() << "expected a destination style op\n";
    return failure();
  }
  switch (bufferizeOperand) {
    /// Create new allocations for Lhs, Rhs and Out.
    case BufferizeOperand::LinalgInputOutput:
      return getInputOutputOperands(dstStyleOp);
    /// Create new allocation only for Lhs, Rhs.
    case BufferizeOperand::LinalgInput:
      return getInputOperands(dstStyleOp);
    /// Create new allocations only for Out.
    case BufferizeOperand::LinalgOutput:
      return SmallVector<Value>(dstStyleOp.getDpsInits());
    /// Create new allocations for operands from the pack ops.
    case BufferizeOperand::PackOrCopyInput:
      return getPackOrCopyOperands(dstStyleOp, inputDepth);
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
  SmallVector<Operation *> targetOps;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // Skip if the op is not elementwise/reduction/contraction/convolution.
      if (!isElementwise(linalgOp) && !isReductionOp(linalgOp) &&
          !linalg::isaContractionOpInterface(linalgOp) &&
          !linalg::isaConvolutionOpInterface(linalgOp)) {
        return WalkResult::advance();
      }
      // Skip FillOp and CopyOp.
      if (isa<linalg::FillOp, linalg::CopyOp>(linalgOp))
        return WalkResult::advance();
      // Skip if the op's elementwise status doesn't match the bufferization
      // mode.
      if (isElementwise(linalgOp) != bufferizeElementwise)
        return WalkResult::advance();
      // Accept as a target op.
      targetOps.push_back(op);
    } else if (isa<linalg::SoftmaxOp>(op)) {
      // Always accept SoftmaxOp as a target op.
      targetOps.push_back(op);
    }
    return WalkResult::advance();
  });

  if (targetOps.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, no linalg op -----\n");
    return;
  }

  if (targetOps.size() > 1) {
    llvm::errs() << "expected only one target op, found " << targetOps.size()
                 << " target ops\n";
    return signalPassFailure();
  }

  IRRewriter rewriter(context);
  for (Operation *targetOp : targetOps) {
    // Find the producer ops for the target op, and bufferizes them in new
    // allocations.
    FailureOr<SmallVector<Value>> operandsToBufferize =
        getOperandsToBufferize(bufferizeOperand, targetOp, inputDepth);
    if (failed(operandsToBufferize)) {
      targetOp->emitOpError("could not fetch operands to bufferize");
      return signalPassFailure();
    }
    for (auto operand : *operandsToBufferize) {
      AMDAIEMemSpaceAttr memorySpaceAttr =
          getMemorySpaceAttr(rewriter, memorySpace);
      rewriter.setInsertionPointAfter(operand.getDefiningOp());
      if (failed(applyBufferizeToAllocation(rewriter, operand.getDefiningOp(),
                                            memorySpaceAttr))) {
        targetOp->emitOpError("failed bufferizing to allocations");
        return signalPassFailure();
      }
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEBufferizeToAllocationPass(
    AMDAIEBufferizeToAllocationOptions options) {
  return std::make_unique<AMDAIEBufferizeToAllocationPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
