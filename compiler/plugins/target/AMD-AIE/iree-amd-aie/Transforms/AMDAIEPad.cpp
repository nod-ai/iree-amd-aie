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
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-pad"

namespace mlir::iree_compiler::AMDAIE {

namespace {

static SmallVector<Attribute> getPaddingValues(IRRewriter &rewriter,
                                               linalg::LinalgOp &linalgOp) {
  SmallVector<Attribute> paddingValues;
  for (auto operand : linalgOp->getOperands()) {
    Type elementType;
    if (auto type = dyn_cast<RankedTensorType>(operand.getType())) {
      elementType = type.getElementType();
    } else if (isa<arith::ConstantOp>(operand.getDefiningOp())) {
      elementType = operand.getType();
    }

    if (isa<IntegerType>(elementType)) {
      paddingValues.push_back(rewriter.getIntegerAttr(elementType, 0));
    } else {
      paddingValues.push_back(rewriter.getFloatAttr(elementType, 0));
    }
  }
  return paddingValues;
}

static SmallVector<int64_t> getPaddingDimensions(linalg::LinalgOp &linalgOp) {
  SmallVector<int64_t> paddingDimensions;
  for (unsigned i = 0; i < linalgOp.getNumLoops(); i++) {
    paddingDimensions.push_back(i);
  }
  return paddingDimensions;
}

static FailureOr<linalg::LinalgPaddingOptions>
getFirstLevelLinalgPaddingOptions(IRRewriter &rewriter,
                                  linalg::LinalgOp &linalgOp) {
  SmallVector<Attribute> paddingValues = getPaddingValues(rewriter, linalgOp);
  if (paddingValues.empty()) {
    linalgOp->emitOpError("failed to get padding values");
    return failure();
  }
  linalg::LinalgPaddingOptions options;
  options.paddingValues = paddingValues;

  // In the first level, we pad the input, weight and output operands.
  // For the operations with 2 input and 1 output operands, the packPadding
  // option should be [1, 1, 1]. For `linalg.conv_2d_nhwc_hwcf_q` op, there are
  // 5 operands, and the packPadding options should be [1, 1, 0, 0, 1].
  SmallVector<bool> packPaddings(linalgOp->getNumOperands(), false);
  packPaddings[0] = true;
  packPaddings[1] = true;
  packPaddings.back() = true;
  options.packPaddings = packPaddings;

  options.paddingDimensions = getPaddingDimensions(linalgOp);
  SmallVector<int64_t> padToMultipleOf(options.paddingDimensions.size(), 1);
  options.padToMultipleOf = padToMultipleOf;
  options.copyBackOp = linalg::LinalgPaddingOptions::CopyBackOp::LinalgCopy;
  return options;
}

static FailureOr<linalg::LinalgPaddingOptions>
getSecondLevelLinalgPaddingOptions(IRRewriter &rewriter,
                                   linalg::LinalgOp &linalgOp) {
  SmallVector<Attribute> paddingValues = getPaddingValues(rewriter, linalgOp);
  if (paddingValues.empty()) {
    linalgOp->emitOpError("failed to get padding values");
    return failure();
  }
  linalg::LinalgPaddingOptions options;
  options.paddingValues = paddingValues;

  // In the second level, we only pad the output operand.
  SmallVector<bool> packPaddings(linalgOp->getNumOperands(), false);
  packPaddings.back() = true;
  options.packPaddings = packPaddings;

  options.paddingDimensions = getPaddingDimensions(linalgOp);
  SmallVector<int64_t> padToMultipleOf(options.paddingDimensions.size(), 1);
  options.padToMultipleOf = padToMultipleOf;
  options.copyBackOp = linalg::LinalgPaddingOptions::CopyBackOp::LinalgCopy;
  return options;
}

static FailureOr<linalg::LinalgPaddingOptions>
getThirdLevelLinalgPaddingOptions(IRRewriter &rewriter,
                                  linalg::LinalgOp &linalgOp) {
  SmallVector<Attribute> paddingValues = getPaddingValues(rewriter, linalgOp);
  if (paddingValues.empty()) {
    linalgOp->emitOpError("failed to get padding values");
    return failure();
  }
  linalg::LinalgPaddingOptions options;
  options.paddingValues = paddingValues;

  // In the third level, we only pad the input operands.
  SmallVector<bool> packPaddings(linalgOp->getNumOperands(), false);
  packPaddings[0] = true;
  packPaddings[1] = true;
  options.packPaddings = packPaddings;

  options.paddingDimensions = getPaddingDimensions(linalgOp);
  SmallVector<int64_t> padToMultipleOf(options.paddingDimensions.size(), 1);
  options.padToMultipleOf = padToMultipleOf;
  options.copyBackOp = linalg::LinalgPaddingOptions::CopyBackOp::LinalgCopy;
  return options;
}

static FailureOr<linalg::LinalgPaddingOptions> getLinalgPaddingOptions(
    IRRewriter &rewriter, linalg::LinalgOp &linalgOp, int64_t paddingLevel) {
  if (paddingLevel == 0) {
    return getFirstLevelLinalgPaddingOptions(rewriter, linalgOp);
  }
  if (paddingLevel == 1) {
    return getSecondLevelLinalgPaddingOptions(rewriter, linalgOp);
  }
  if (paddingLevel == 2) {
    return getThirdLevelLinalgPaddingOptions(rewriter, linalgOp);
  }
  return failure();
}

static LogicalResult applyPadAndConvertToDPS(
    RewriterBase &rewriter, linalg::LinalgOp linalgTarget,
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
  mlir::FunctionOpInterface funcOp = getOperation();
  linalg::LinalgOp linalgOp;
  funcOp->walk([&](linalg::LinalgOp op) {
    if (linalg::isaContractionOpInterface(op) ||
        linalg::isaConvolutionOpInterface(op)) {
      linalgOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!linalgOp) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, no matmul op -----\n");
    return;
  }

  IRRewriter rewriter(context);
  FailureOr<linalg::LinalgPaddingOptions> options =
      getLinalgPaddingOptions(rewriter, linalgOp, paddingLevel);
  if (failed(options)) {
    funcOp->emitOpError("unknown padding level");
    return signalPassFailure();
  }
  if (failed(applyPadAndConvertToDPS(rewriter, linalgOp, *options))) {
    funcOp->emitOpError("failed to apply pad");
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEPadPass(AMDAIEPadOptions options) {
  return std::make_unique<AMDAIEPadPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
