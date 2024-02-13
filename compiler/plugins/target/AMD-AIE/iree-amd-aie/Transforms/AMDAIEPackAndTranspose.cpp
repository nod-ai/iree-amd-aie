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

static FailureOr<linalg::PackResult> applyPackOnLinalgOp(
    RewriterBase &rewriter, linalg::LinalgOp op,
    SmallVector<OpFoldResult> packedSizes) {
  // Fail on mismatched number of pack sizes.
  if (packedSizes.size() != op.getNumLoops()) {
    op->emitOpError(
        "requires number of packed sizes match the number of loops (")
        << packedSizes.size() << " vs " << op.getNumLoops() << ")";
    return failure();
  }

  rewriter.setInsertionPoint(op);
  FailureOr<linalg::PackResult> packResult =
      linalg::pack(rewriter, op, packedSizes);
  if (failed(packResult)) {
    op->emitOpError("failed to pack the operation");
    return failure();
  }
  return packResult;
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
  func::FuncOp funcOp = getOperation();

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
  auto config = getLoweringConfig(linalgOp);
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
