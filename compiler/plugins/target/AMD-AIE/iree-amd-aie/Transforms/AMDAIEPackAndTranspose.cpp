// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-pack-and-transpose"

namespace mlir::iree_compiler::AMDAIE {

namespace {

struct PackConfig {
  // Expected packed sizes for specified iterator dimensions
  SmallVector<OpFoldResult> packedSizes;
  // Indices of pack operations need to be transposed
  SmallVector<int64_t> transposePackIndices;
  // Indicator of if there is a unpack op corresponding to a pack op
  SmallVector<int64_t> unpackEmpty;
  // Attributes for inner dimension permutation
  SmallVector<SmallVector<int64_t>> innerPerm;
  // Attributes for outer dimension permutation
  SmallVector<SmallVector<int64_t>> outerPerm;
};

static FailureOr<PackConfig> getPackConfig(RewriterBase &rewriter,
                                           int packLevel) {
  PackConfig config;
  if (packLevel == 0) {
    // packed size for [M, N, K]
    config.packedSizes = {rewriter.getI64IntegerAttr(8),
                          rewriter.getI64IntegerAttr(16),
                          rewriter.getI64IntegerAttr(16)};
    // Transpose B matrix from [K N n k] to [K N k n]
    config.transposePackIndices = {1};
    // There is no corresponding unpack for the specified pack operation
    // 0 is used when unpack is empty
    config.unpackEmpty = {0};
    config.innerPerm = {{1, 0}};
    config.outerPerm = {{0, 1}};
  } else if (packLevel == 1) {
    // packed size for [M, N, K, m, n, k]
    config.packedSizes = {
        rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(4),
        rewriter.getI64IntegerAttr(8), rewriter.getI64IntegerAttr(8)};
    // Transpose A matrix from [M K m k m0 k0] to [M K k m m0 k0]
    // Transpose B matrix from [K N k n n0 k0] to [K N n k k0 n0]
    // Transpose C matrix from [M N m n m0 n0] to [M N n m m0 n0]
    config.transposePackIndices = {0, 1, 2};
    // Only the third pack operation has a corresponding unpack operation
    config.unpackEmpty = {0, 0, 1};
    config.innerPerm = {{0, 1}, {1, 0}, {0, 1}};
    config.outerPerm = {{0, 1, 3, 2}, {0, 1, 3, 2}, {0, 1, 3, 2}};
  } else {
    return failure();
  }
  return config;
}

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

  // Pack the operation
  IRRewriter rewriter(context);
  FailureOr<PackConfig> packCfg = getPackConfig(rewriter, packLevel);
  if (failed(packCfg)) {
    funcOp->emitOpError("failed to get pack configs");
    return signalPassFailure();
  }

  FailureOr<linalg::PackResult> packResult =
      applyPackOnLinalgOp(rewriter, linalgOp, packCfg->packedSizes);
  if (failed(packResult)) {
    return signalPassFailure();
  }

  // Pack Transpose
  SmallVector<tensor::PackOp> packOps = packResult->packOps;
  linalg::LinalgOp packedOp = packResult->packedLinalgOp;
  SmallVector<tensor::UnPackOp> unpackOps = packResult->unPackOps;

  if (packOps.size() != 3 || !packedOp || unpackOps.empty()) {
    funcOp->emitOpError("failed to get correct pack and unpack ops");
    return signalPassFailure();
  }

  auto packIndices = packCfg->transposePackIndices;
  auto unpackArr = packCfg->unpackEmpty;
  auto innerPermArr = packCfg->innerPerm;
  auto outerPermArr = packCfg->outerPerm;

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
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEPackAndTransposePass(
    AMDAIEPackAndTransposeOptions options) {
  return std::make_unique<AMDAIEPackAndTransposePass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
