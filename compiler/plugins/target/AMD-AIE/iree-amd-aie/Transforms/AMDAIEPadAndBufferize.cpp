// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-pad-and-bufferize"

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

static LogicalResult formLinalgPaddingOptions(
    IRRewriter &rewriter, linalg::MatmulOp &matmulOp, int64_t paddingLevel,
    linalg::LinalgPaddingOptions &options) {
  SmallVector<Attribute> paddingValues;
  for (auto operand : matmulOp->getOperands()) {
    auto type = dyn_cast<RankedTensorType>(operand.getType());
    if (!type) {
      matmulOp->emitOpError("failed to lower workgroup count region");
      return failure();
    }
    Type elementType = type.getElementType();
    if (isa<IntegerType>(elementType)) {
      paddingValues.push_back(rewriter.getIntegerAttr(elementType, 0));
    } else {
      paddingValues.push_back(rewriter.getFloatAttr(elementType, 0));
    }
  }
  options.paddingValues = paddingValues;
  SmallVector<bool> packPaddings(matmulOp->getNumOperands(), true);
  if (paddingLevel == 2) {
    packPaddings[0] = false;
    packPaddings[1] = false;
  } else if (paddingLevel == 3) {
    packPaddings[2] = false;
  }
  options.packPaddings = packPaddings;

  SmallVector<int64_t> paddingDimensions;
  for (unsigned i = 0, n = matmulOp.getNumLoops(); i < n; i++)
    paddingDimensions.push_back(i);
  options.paddingDimensions = paddingDimensions;

  SmallVector<int64_t> padToMultipleOf(options.paddingDimensions.size(), 1);
  options.padToMultipleOf = padToMultipleOf;
  options.copyBackOp = linalg::LinalgPaddingOptions::CopyBackOp::LinalgCopy;

  return success();
}

static LogicalResult applyPadAndConvertToDPS(
    RewriterBase &rewriter, linalg::MatmulOp linalgTarget,
    linalg::LinalgPaddingOptions &options) {
  linalg::LinalgOp paddedOp;
  SmallVector<Value> replacements;
  SmallVector<tensor::PadOp> newPadOps;
  if (failed(rewriteAsPaddedOp(rewriter, linalgTarget, options, paddedOp,
                               replacements, newPadOps))) {
    LLVM_DEBUG(llvm::dbgs() << "----- failed to pad op -----\n");
    return failure();
  }

  // We need to perform our own replacement here because this API is still
  // used in patterns that "pad and hoist", for which the replacement values
  // need to be different.
  // TODO: clean this up and stop "pad and hoist" behavior more globally now
  // that we have more composable abstractions.
  rewriter.replaceOp(linalgTarget, replacements);
  // We rewrite each operand of the linalgOp (currently, MatmulOp) into DSP.
  for (auto newPadOp : newPadOps) {
    rewriter.setInsertionPointAfter(newPadOp);
    if (failed(linalg::rewriteInDestinationPassingStyle(rewriter, newPadOp))) {
      LLVM_DEBUG(llvm::dbgs() << "----- failed to rewrite in DPS -----\n");
      return failure();
    }
  }

  return success();
}

class AMDAIEPadAndBufferizePass
    : public AMDAIEPadAndBufferizeBase<AMDAIEPadAndBufferizePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, tensor::TensorDialect,
                    linalg::LinalgDialect>();
  }
  AMDAIEPadAndBufferizePass() = default;
  AMDAIEPadAndBufferizePass(int64_t paddingLevel = -1) {
    this->paddingLevel.setValue(paddingLevel);
  }
  AMDAIEPadAndBufferizePass(const AMDAIEPadAndBufferizePass &pass){};
  void runOnOperation() override;
};

void AMDAIEPadAndBufferizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp innerModule = variantOp.getInnerModule();
  for (func::FuncOp funcOp : innerModule.getOps<func::FuncOp>()) {
    linalg::MatmulOp matmulOp;
    funcOp->walk<WalkOrder::PostOrder, ReverseIterator>(
        [&](TilingInterface op) {
          // Find the next matmul op if it does not have loops.
          if (op.getLoopIteratorTypes().empty() || !isa<linalg::MatmulOp>(op))
            return WalkResult::advance();
          matmulOp = cast<linalg::MatmulOp>(op);
          return WalkResult::interrupt();
        });
    if (!matmulOp) {
      LLVM_DEBUG(llvm::dbgs() << "----- skip, no matmul op -----\n");
      return;
    }

    IRRewriter rewriter(context);
    linalg::LinalgPaddingOptions options;
    if (failed(formLinalgPaddingOptions(rewriter, matmulOp, paddingLevel,
                                        options))) {
      return signalPassFailure();
    }
    if (failed(applyPadAndConvertToDPS(rewriter, matmulOp, options))) {
      funcOp->emitOpError("failed to apply pad");
      return signalPassFailure();
    }

    // TODO: This part can be coded better instead of againg walking the FuncOp.
    funcOp->walk<WalkOrder::PostOrder, ReverseIterator>(
        [&](TilingInterface op) {
          // Find the next matmul op if it does not have loops.
          if (op.getLoopIteratorTypes().empty() || !isa<linalg::MatmulOp>(op))
            return WalkResult::advance();
          matmulOp = cast<linalg::MatmulOp>(op);
          return WalkResult::interrupt();
        });
    if (!matmulOp) {
      LLVM_DEBUG(llvm::dbgs() << "----- skip, no matmul op -----\n");
      return;
    }
    auto i64Type = rewriter.getI64Type();
    for (unsigned i = 0, n = matmulOp->getNumOperands(); i < n; i++) {
      // TODO(avarma): This is a temporary workaround until we have
      // PaddingStrategy in Given a Matmul(Lhs, Rhs, Out) and a given
      // `paddingLevel`, the following is what we bufferize :-
      //    paddingLevel == 1 -> Lhs, Rhs and Out.
      //    paddingLevel == 2 -> Out.
      //    paddingLevel == 2 -> Lhs and Rhs.
      if (paddingLevel == 2 && i <= 1) continue;
      if (paddingLevel == 3 && i > 1) break;
      Value operand = matmulOp->getOperand(i);
      auto memorySpace = rewriter.getIntegerAttr(i64Type, 1);
      if (paddingLevel >= 2) memorySpace = rewriter.getIntegerAttr(i64Type, 2);
      rewriter.setInsertionPointAfter(operand.getDefiningOp());
      if (failed(applyBufferizeToAllocation(rewriter, operand.getDefiningOp(),
                                            memorySpace))) {
        funcOp->emitOpError("failed bufferizing to allocations");
        return signalPassFailure();
      }
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createAMDAIEPadAndBufferizePass(int64_t paddingLevel) {
  return std::make_unique<AMDAIEPadAndBufferizePass>(paddingLevel);
}
}  // namespace mlir::iree_compiler::AMDAIE
