// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-pack-and-transpose"

namespace mlir::iree_compiler::AMDAIE {

namespace {

struct PackConfig {
  SmallVector<OpFoldResult> packedSizes;
  SmallVector<int64_t> transposePackIndices;
  SmallVector<int64_t> unpackEmpty;
  SmallVector<SmallVector<int64_t>> innerPerm;
  SmallVector<SmallVector<int64_t>> outerPerm;
};

static FailureOr<PackConfig> getPackConfig(RewriterBase &rewriter,
                                           int packLevel) {
  PackConfig config;
  if (packLevel == 1) {
    config.packedSizes = {rewriter.getI64IntegerAttr(16),
                          rewriter.getI64IntegerAttr(64),
                          rewriter.getI64IntegerAttr(64)};
    config.transposePackIndices = {1};
    config.unpackEmpty = {0};  // 0 is empty
    config.innerPerm = {{1, 0}};
    config.outerPerm = {{0, 1}};
  } else if (packLevel == 2) {
    config.packedSizes = {
        rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(4),
        rewriter.getI64IntegerAttr(8), rewriter.getI64IntegerAttr(8)};
    config.transposePackIndices = {0, 1, 2};
    config.unpackEmpty = {0, 0, 1};  // 0 is empty
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
    : public AMDAIEPackAndTransposeBase<AMDAIEPackAndTransposePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
  }

  AMDAIEPackAndTransposePass() = default;
  AMDAIEPackAndTransposePass(int64_t packLevel = 1) {
    this->packLevel.setValue(packLevel);
  }
  AMDAIEPackAndTransposePass(const AMDAIEPackAndTransposePass &pass){};
  void runOnOperation() override;
};

void AMDAIEPackAndTransposePass::runOnOperation() {
  MLIRContext *context = &getContext();
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp innerModule = variantOp.getInnerModule();

  for (func::FuncOp funcOp : innerModule.getOps<func::FuncOp>()) {
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
      LLVM_DEBUG(llvm::dbgs()
                 << "----- skip, no linalg op for packing -----\n");
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
}

}  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createAMDAIEPackAndTransposePass(int64_t packLevel) {
  return std::make_unique<AMDAIEPackAndTransposePass>(packLevel);
}
}  // namespace mlir::iree_compiler::AMDAIE
