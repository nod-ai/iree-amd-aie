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

#define DEBUG_TYPE "iree-amdaie-tensor-pad"

namespace mlir::iree_compiler::AMDAIE {

namespace {

static constexpr StringRef kCopyOpNone = "none";

/// TODO(avarma): For now we are creating a temporary struct options
/// in order to adapt to LinalgPadding options. Once we address upstream
/// methods by having APIs, we can better address this instead using the
/// upstream struct. I could've used the upstream struct here, but I chose
/// not to in order to first adapt to minimal changes required for the C++
/// methods to work.
struct _LinalgPaddingOptions {
  SmallVector<Attribute> paddingValues;
  ArrayAttr paddingDimensions;
  ArrayAttr packPaddings;
  StringAttr copyBackOp;
};

static LogicalResult applyPad(RewriterBase &rewriter,
                              linalg::MatmulOp linalgTarget,
                              _LinalgPaddingOptions _options) {
  SmallVector<Operation *> paddedOps, padOps;

  // Convert the integer packing flags to booleans.
  SmallVector<bool> packPaddings;
  for (int64_t packPadding :
       extractFromIntegerArrayAttr<int64_t>(_options.packPaddings))
    packPaddings.push_back(static_cast<bool>(packPadding));

  // Convert the padding values to attributes.
  SmallVector<Attribute> paddingValues = _options.paddingValues;

  linalg::LinalgOp paddedOp;
  linalg::LinalgPaddingOptions options;
  options.paddingDimensions =
      extractFromIntegerArrayAttr<int64_t>(_options.paddingDimensions);
  SmallVector<int64_t> padToMultipleOf(options.paddingDimensions.size(), 1);
  options.padToMultipleOf = padToMultipleOf;
  options.paddingValues = paddingValues;
  options.packPaddings = packPaddings;
  if (_options.copyBackOp ==
      bufferization::MaterializeInDestinationOp::getOperationName()) {
    options.copyBackOp = linalg::LinalgPaddingOptions::CopyBackOp::
        BufferizationMaterializeInDestination;
  } else if (_options.copyBackOp == linalg::CopyOp::getOperationName()) {
    options.copyBackOp = linalg::LinalgPaddingOptions::CopyBackOp::LinalgCopy;
  } else if (_options.copyBackOp == kCopyOpNone) {
    options.copyBackOp = linalg::LinalgPaddingOptions::CopyBackOp::None;
  } else {
    llvm_unreachable("unsupported copy_back op");
  }

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
  paddedOps.push_back(paddedOp);
  padOps.append(newPadOps.begin(), newPadOps.end());

  return success();
}

class AMDAIETensorPadPass : public AMDAIETensorPadBase<AMDAIETensorPadPass> {
 private:
  AMDAIETensorPadOption option = AMDAIETensorPadOption::ParallelDims;

 public:
  explicit AMDAIETensorPadPass(AMDAIETensorPadOption option) : option(option) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override;
};

void AMDAIETensorPadPass::runOnOperation() {
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
    _LinalgPaddingOptions options;
    SmallVector<Attribute> paddingValues;
    SmallVector<Attribute> packPaddingsVal;
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
      packPaddingsVal.push_back(rewriter.getIntegerAttr(i64Type, 1));
    }
    options.paddingValues = paddingValues;
    options.packPaddings = rewriter.getArrayAttr(packPaddingsVal);

    SmallVector<Attribute> paddingDimensionsVal;
    for (unsigned i = 0, n = matmulOp.getNumLoops(); i < n; i++)
      paddingDimensionsVal.push_back(rewriter.getIntegerAttr(i64Type, i));
    options.paddingDimensions = rewriter.getArrayAttr(paddingDimensionsVal);

    options.copyBackOp = rewriter.getStringAttr("linalg.copy");

    if (failed(applyPad(rewriter, matmulOp, options))) {
      funcOp->emitOpError("failed to apply pad");
      return signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createAMDAIETensorPadPass(AMDAIETensorPadOption option) {
  return std::make_unique<AMDAIETensorPadPass>(option);
}
}  // namespace mlir::iree_compiler::AMDAIE
