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
    SmallVector<Attribute> paddingValues;
    auto i64Type = rewriter.getI64Type();
    for (auto operand : matmulOp->getOperands()) {
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      if (!type) {
        funcOp->emitOpError("failed to lower workgroup count region");
        return signalPassFailure();
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
    options.packPaddings = packPaddings;

    SmallVector<int64_t> paddingDimensions;
    for (unsigned i = 0, n = matmulOp.getNumLoops(); i < n; i++)
      paddingDimensions.push_back(i);
    options.paddingDimensions = paddingDimensions;

    SmallVector<int64_t> padToMultipleOf(options.paddingDimensions.size(), 1);
    options.padToMultipleOf = padToMultipleOf;
    options.copyBackOp = linalg::LinalgPaddingOptions::CopyBackOp::LinalgCopy;

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
    for (auto operand : matmulOp->getOperands()) {
      auto memorySpace = rewriter.getIntegerAttr(i64Type, 1);
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
createAMDAIEPadAndBufferizePass() {
  return std::make_unique<AMDAIEPadAndBufferizePass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
