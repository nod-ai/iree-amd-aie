// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "air/Util/Util.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Pattern to rewrite `linalg.copy` to `memref.copy`.
class LinalgCopyToMemRefCopy : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern<linalg::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (copyOp.hasIndexSemantics()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<memref::CopyOp>(
        copyOp, copyOp.getInputs().front(), copyOp.getDpsInits().front());
    return success();
  }
};

/// Pattern to rewrite `affine.apply` with AffineExpr on symbols rather than
/// dims.
class AffineApplyOnSym : public OpRewritePattern<affine::AffineApplyOp> {
  using OpRewritePattern<affine::AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    auto map = applyOp.getAffineMap();
    if (map.getNumDims() != 1) return failure();
    if (map.getNumSymbols()) return failure();
    SmallVector<Value> opers;
    for (auto v : applyOp.getMapOperands()) opers.push_back(v);
    auto newMap = map.replaceDimsAndSymbols(
        getAffineSymbolExpr(0, applyOp->getContext()), {}, 0, 1);
    rewriter.replaceOpWithNewOp<affine::AffineApplyOp>(applyOp, newMap, opers);
    return success();
  }
};

class AMDAIEBridgeToAIRPass
    : public impl::AMDAIEBridgeToAIRBase<AMDAIEBridgeToAIRPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, scf::SCFDialect>();
  }

  AMDAIEBridgeToAIRPass() = default;
  AMDAIEBridgeToAIRPass(const AMDAIEBridgeToAIRPass &pass){};

  void runOnOperation() override;
};

void AMDAIEBridgeToAIRPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<LinalgCopyToMemRefCopy, AffineApplyOnSym>(context);
  xilinx::air::populateBufferMemrefToFuncArgsPattern(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEBridgeToAIRPass() {
  return std::make_unique<AMDAIEBridgeToAIRPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
