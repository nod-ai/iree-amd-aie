// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "air/Dialect/AIR/AIRDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-propagate-data-layout"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Pattern to rewriter scf.forall -> scf.parallel after bufferization.
class LinalgExtPackToAirDmaMemcpyNd : public OpRewritePattern<IREE::LinalgExt::PackOp> {
  using OpRewritePattern<IREE::LinalgExt::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::PackOp packOp,
                                PatternRewriter &rewriter) const override {

  // 1. Filter out NYI cases.
  auto packedMemrefType = packOp.getOutputType();
  if (llvm::any_of(packOp.getStaticInnerTiles(),
                   [](int64_t size) { return ShapedType::isDynamic(size); })) {
    return rewriter.notifyMatchFailure(
        packOp,
        "non-static shape NYI");
  }
  //Location loc = packOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(packOp);

  auto innerDimsPos = packOp.getInnerDimsPos();
  auto destShape = packOp.getOutputType().getShape();

  Value input = packOp.getInput();
  Value output = packOp.getOutput();

  llvm::outs()<<"packedMemrefType: "<<packedMemrefType<<"\n";
  llvm::outs()<<"innerDimsPos: "<<"\n";
  for (int64_t element : innerDimsPos) {
        llvm::outs() << element << " ";
  }
  llvm::outs()<<"destShape: "<<"\n";
  for (int64_t element : destShape) {
        llvm::outs() << element << " ";
  }
  llvm::outs()<<"input: "<<input<<"\n";
  llvm::outs()<<"output: "<<output<<"\n";
  Operation* sourceOp = input.getDefiningOp();
  //Operation* DstOp;
  if(auto allocOp = dyn_cast<memref::AllocOp>(sourceOp)){
    auto [strides, offset] = getStridesAndOffset(allocOp.getType());
    for (auto stride : strides) {
        llvm::outs() << stride << " ";
    }
    llvm::outs()<<"\n";
    llvm::outs()<<"offset: "<<offset<<"\n";
  }
  else if(auto subviewOp = dyn_cast<memref::SubViewOp>(sourceOp)){
    SmallVector<OpFoldResult> strides = subviewOp.getMixedStrides();
    for (auto stride : strides) {
        llvm::outs() << stride << " ";
    }
    llvm::outs()<<"\n";
    SmallVector<OpFoldResult> strides2 = subviewOp.getStrides();
    for (auto stride : strides2) {
        llvm::outs() << stride << " ";
    }
    llvm::outs()<<"\n";
    auto [strides3, offset] = getStridesAndOffset(subviewOp.getSource().getType());
    for (auto stride : strides3) {
        llvm::outs() << stride << " ";
    }
    llvm::outs()<<"offset: "<<offset<<"\n";
    sourceOp = subviewOp.getSource().getDefiningOp();
  }
  sourceOp->dump();



    return failure();
  }
};

class AMDAIEPackToDmaPass
    : public impl::AMDAIEPackToDmaBase<
          AMDAIEPackToDmaPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
  }

  AMDAIEPackToDmaPass() = default;
  AMDAIEPackToDmaPass(const AMDAIEPackToDmaPass &pass){};
  void runOnOperation() override;
};

void AMDAIEPackToDmaPass::runOnOperation() {
    MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<LinalgExtPackToAirDmaMemcpyNd>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEPackToDmaPass() {
  return std::make_unique<AMDAIEPackToDmaPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
