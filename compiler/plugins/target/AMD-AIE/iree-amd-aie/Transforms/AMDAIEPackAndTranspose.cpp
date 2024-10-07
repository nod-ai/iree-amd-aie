// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-pack-and-transpose"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// The function `linalg::pack` doesn't create rank-preserving pack ops, see:
///
/// https://github.com/llvm/llvm-project/blob/644899addd8fd789c93e9a0f0727d37eb1b29c55/mlir/lib/Dialect/Linalg/Transforms/Transforms.cpp#L542
///
/// This design cannot easily be undone upstream, because it is thoroughly
/// tested for. See:
///
/// https://github.com/llvm/llvm-project/blob/644899addd8fd789c93e9a0f0727d37eb1b29c55/mlir/test/Dialect/Linalg/transform-op-pack.mlir
///
/// In our use case it is sometimes useful to have identity pack ops,
/// specifically if we don't want to pack any dimensions of an operand, but
/// still want to set up a new tensor for bufferization.
///
/// The following logic inserts identity pack/unpack ops where needed into
/// the pack result.
static FailureOr<linalg::PackResult> ensureAllPacksPresent(
    RewriterBase &rewriter, const linalg::PackResult &initialPackResult) {
  linalg::PackResult packResult = initialPackResult;
  linalg::LinalgOp packedOp = packResult.packedLinalgOp;
  // If `v` is the result of a pack op returned by `linalg::pack`, return it.
  // Else return an empty pack op.
  auto maybeGetPack = [&](Value v) -> tensor::PackOp {
    for (auto packOp : packResult.packOps) {
      if (v == packOp.getResult()) {
        return packOp;
      }
    }
    return tensor::PackOp{};
  };

  SmallVector<tensor::PackOp> newPackOps;
  for (uint32_t i = 0; i < packedOp->getNumOperands(); ++i) {
    Value operand = packedOp->getOperand(i);
    tensor::PackOp packOp = maybeGetPack(operand);
    if (packOp) {
      newPackOps.push_back(packOp);
    } else {
      // Create an identity pack op.
      ShapedType shapedType = cast<ShapedType>(operand.getType());
      rewriter.setInsertionPoint(packedOp);
      Value dest = rewriter.create<tensor::EmptyOp>(
          packedOp.getLoc(), shapedType.getShape(),
          shapedType.getElementType());
      tensor::PackOp newPackOp = rewriter.create<tensor::PackOp>(
          operand.getLoc(), operand, dest, SmallVector<int64_t>{},
          SmallVector<OpFoldResult>{});
      rewriter.replaceAllUsesExcept(operand, newPackOp.getResult(), newPackOp);
      newPackOps.push_back(newPackOp);
    }
  }
  packResult.packOps = newPackOps;

  if (packResult.unPackOps.empty()) {
    if (packedOp->getNumResults() != 1) {
      return packedOp->emitOpError(
          "is expected to have 1 result for the current packing approach.");
    }
    Value result = packedOp->getResult(0);
    rewriter.setInsertionPointAfterValue(result);
    ShapedType shapedType = cast<ShapedType>(result.getType());
    Value dest = rewriter.create<tensor::EmptyOp>(
        packedOp.getLoc(), shapedType.getShape(), shapedType.getElementType());
    rewriter.setInsertionPointAfter(dest.getDefiningOp());
    tensor::UnPackOp unpackOp = rewriter.create<tensor::UnPackOp>(
        packedOp.getLoc(), result, dest, SmallVector<int64_t>{},
        SmallVector<OpFoldResult>{});
    rewriter.replaceAllUsesExcept(result, unpackOp.getResult(), unpackOp);
    packResult.unPackOps = {unpackOp};
  }

  return packResult;
}

static FailureOr<linalg::PackResult> applyPackOnLinalgOp(
    RewriterBase &rewriter, linalg::LinalgOp op,
    SmallVector<OpFoldResult> packedSizes) {
  if (packedSizes.size() != op.getNumLoops()) {
    op->emitOpError(
        "requires number of packed sizes match the number of loops (")
        << packedSizes.size() << " vs " << op.getNumLoops() << ")";
    return failure();
  }

  rewriter.setInsertionPoint(op);
  FailureOr<linalg::PackResult> maybePackResult =
      linalg::pack(rewriter, op, packedSizes);
  if (failed(maybePackResult)) {
    op->emitOpError("failed to pack the operation");
    return failure();
  }

  return ensureAllPacksPresent(rewriter, maybePackResult.value());
}

class AMDAIEPackAndTransposePass
    : public impl::AMDAIEPackAndTransposeBase<AMDAIEPackAndTransposePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
  }

  AMDAIEPackAndTransposePass() = default;
  AMDAIEPackAndTransposePass(const AMDAIEPackAndTransposePass &pass) {}
  AMDAIEPackAndTransposePass(const AMDAIEPackAndTransposeOptions &options)
      : AMDAIEPackAndTransposeBase(options) {}

  void runOnOperation() override;
};

void AMDAIEPackAndTransposePass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  // Find the linalg op for packing, currently only consider contraction ops
  linalg::LinalgOp linalgOp;
  funcOp->walk([&](linalg::LinalgOp op) {
    if (linalg::isaContractionOpInterface(op)) {
      linalgOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!linalgOp) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, no linalg op for packing -----\n");
    return;
  }

  // Step 1. Before packing the operation, we will prefetch the lowering and
  // packing config.
  auto config = getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(linalgOp);
  auto packingConfig = getPackingConfig(linalgOp);

  if (!config || !packingConfig) {
    funcOp->emitOpError("failed to get pack configs");
    return signalPassFailure();
  }

  // Step 2. Pack the operation
  IRRewriter rewriter(context);
  // Extract packing config from the `linalgOp`.
  PackingConfigPackingLevelAttr packCfg =
      packingConfig.getPackingConfigVals(packLevel);
  SmallVector<OpFoldResult> packedSizes =
      getAsIndexOpFoldResult(context, packCfg.getPackedSizes());

  FailureOr<linalg::PackResult> packResult =
      applyPackOnLinalgOp(rewriter, linalgOp, packedSizes);
  if (failed(packResult)) {
    return signalPassFailure();
  }

  // Step 3. Pack Transpose
  SmallVector<tensor::PackOp> packOps = packResult->packOps;
  linalg::LinalgOp packedOp = packResult->packedLinalgOp;
  SmallVector<tensor::UnPackOp> unpackOps = packResult->unPackOps;

  if (packOps.size() != 3 || !packedOp || unpackOps.empty()) {
    funcOp->emitOpError("failed to get correct pack and unpack ops");
    return signalPassFailure();
  }

  auto packIndices = packCfg.getTransposePackIndices();
  auto unpackArr = packCfg.getUnpackEmpty();
  auto innerPermArr = packCfg.getInnerPermArr();
  auto outerPermArr = packCfg.getOuterPermArr();

  for (auto [index, unpackEmpty, innerPerm, outerPerm] :
       llvm::zip(packIndices, unpackArr, innerPermArr, outerPermArr)) {
    tensor::UnPackOp unpackOp;
    if (unpackEmpty) {
      unpackOp = unpackOps.back();
    }

    FailureOr<linalg::PackTransposeResult> packTransResult = packTranspose(
        rewriter, packOps[index], packedOp, unpackOp, outerPerm, innerPerm);
    if (failed(packTransResult)) {
      funcOp->emitOpError("failed to transpose the pack operation ") << index;
      return signalPassFailure();
    }

    // Update packed linalg op
    packedOp = packTransResult->transposedLinalgOp;
  }

  // Step 4. Set the lowering config prefetched earlier in step 1 to the
  // packedOp.
  if (config) {
    setLoweringConfig(packedOp, config);
    setPackingConfig(packedOp, packingConfig);
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEPackAndTransposePass(
    AMDAIEPackAndTransposeOptions options) {
  return std::make_unique<AMDAIEPackAndTransposePass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
