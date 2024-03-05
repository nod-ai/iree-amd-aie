// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-pad"

namespace mlir::iree_compiler::AMDAIE {

namespace {

static FailureOr<linalg::LinalgPaddingOptions>
getFirstLevelLinalgPaddingOptions(IRRewriter &rewriter,
                                  linalg::MatmulOp &matmulOp) {
  linalg::LinalgPaddingOptions options;
  SmallVector<Attribute> paddingValues;
  for (auto operand : matmulOp->getOperands()) {
    auto type = dyn_cast<RankedTensorType>(operand.getType());
    if (!type) {
      matmulOp->emitOpError("expected ranked tensor type");
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
  options.packPaddings = packPaddings;

  SmallVector<int64_t> paddingDimensions;
  for (unsigned i = 0, n = matmulOp.getNumLoops(); i < n; i++)
    paddingDimensions.push_back(i);
  options.paddingDimensions = paddingDimensions;

  SmallVector<int64_t> padToMultipleOf(options.paddingDimensions.size(), 1);
  options.padToMultipleOf = padToMultipleOf;
  options.copyBackOp = linalg::LinalgPaddingOptions::CopyBackOp::LinalgCopy;

  return options;
}

static FailureOr<linalg::LinalgPaddingOptions>
getSecondLevelLinalgPaddingOptions(IRRewriter &rewriter,
                                   linalg::MatmulOp &matmulOp) {
  linalg::LinalgPaddingOptions options;
  SmallVector<Attribute> paddingValues;
  for (auto operand : matmulOp->getOperands()) {
    auto type = dyn_cast<RankedTensorType>(operand.getType());
    if (!type) {
      matmulOp->emitOpError("expected ranked tensor type");
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
  packPaddings[0] = false;
  packPaddings[1] = false;
  options.packPaddings = packPaddings;

  SmallVector<int64_t> paddingDimensions;
  for (unsigned i = 0, n = matmulOp.getNumLoops(); i < n; i++)
    paddingDimensions.push_back(i);
  options.paddingDimensions = paddingDimensions;

  SmallVector<int64_t> padToMultipleOf(options.paddingDimensions.size(), 1);
  options.padToMultipleOf = padToMultipleOf;
  options.copyBackOp = linalg::LinalgPaddingOptions::CopyBackOp::LinalgCopy;

  return options;
}

static FailureOr<linalg::LinalgPaddingOptions>
getThirdLevelLinalgPaddingOptions(IRRewriter &rewriter,
                                  linalg::MatmulOp &matmulOp) {
  linalg::LinalgPaddingOptions options;
  SmallVector<Attribute> paddingValues;
  for (auto operand : matmulOp->getOperands()) {
    auto type = dyn_cast<RankedTensorType>(operand.getType());
    if (!type) {
      matmulOp->emitOpError("expected ranked tensor type");
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
  packPaddings[2] = false;
  options.packPaddings = packPaddings;

  SmallVector<int64_t> paddingDimensions;
  for (unsigned i = 0, n = matmulOp.getNumLoops(); i < n; i++)
    paddingDimensions.push_back(i);
  options.paddingDimensions = paddingDimensions;

  SmallVector<int64_t> padToMultipleOf(options.paddingDimensions.size(), 1);
  options.padToMultipleOf = padToMultipleOf;
  options.copyBackOp = linalg::LinalgPaddingOptions::CopyBackOp::LinalgCopy;

  return options;
}

static FailureOr<linalg::LinalgPaddingOptions> getLinalgPaddingOptions(
    IRRewriter &rewriter, linalg::MatmulOp &matmulOp, int64_t paddingLevel) {
  if (paddingLevel == 0) {
    return getFirstLevelLinalgPaddingOptions(rewriter, matmulOp);
  }
  if (paddingLevel == 1) {
    return getSecondLevelLinalgPaddingOptions(rewriter, matmulOp);
  }
  if (paddingLevel == 2) {
    return getThirdLevelLinalgPaddingOptions(rewriter, matmulOp);
  }
  return failure();
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

class AMDAIEPadPass : public impl::AMDAIEPadBase<AMDAIEPadPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, tensor::TensorDialect,
                    linalg::LinalgDialect>();
  }

  AMDAIEPadPass() = default;
  AMDAIEPadPass(const AMDAIEPadPass &pass) {}
  AMDAIEPadPass(const AMDAIEPadOptions &options) : AMDAIEPadBase(options) {}

  void runOnOperation() override;
};

void AMDAIEPadPass::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();
  linalg::MatmulOp matmulOp;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](TilingInterface op) {
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
  FailureOr<linalg::LinalgPaddingOptions> options =
      getLinalgPaddingOptions(rewriter, matmulOp, paddingLevel);
  if (failed(options)) {
    funcOp->emitOpError("unknown padding level");
    return signalPassFailure();
  }
  if (failed(applyPadAndConvertToDPS(rewriter, matmulOp, *options))) {
    funcOp->emitOpError("failed to apply pad");
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEPadPass(AMDAIEPadOptions options) {
  return std::make_unique<AMDAIEPadPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
