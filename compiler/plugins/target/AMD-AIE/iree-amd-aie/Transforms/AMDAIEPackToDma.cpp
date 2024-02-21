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
#include "mlir/Dialect/Utils/IndexingUtils.h"

#define DEBUG_TYPE "iree-amdaie-propagate-data-layout"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Pattern to rewriter scf.forall -> scf.parallel after bufferization.
class LinalgExtPackToAirDmaMemcpyNd : public OpRewritePattern<IREE::LinalgExt::PackOp> {
  using OpRewritePattern<IREE::LinalgExt::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::PackOp packOp,
                                PatternRewriter &rewriter) const override {

  // 1. Filter out NYI cases.
  llvm::ArrayRef<int64_t> innerTiles = packOp.getStaticInnerTiles();
  auto packedMemrefType = packOp.getOutputType();
  auto permutation = packOp.getOuterDimsPerm();
  if (llvm::any_of(innerTiles,
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
  Operation* DstOp = output.getDefiningOp();
  SmallVector<Value> baseStridesValues;
  SmallVector<Value> srcShapeValues;
  SmallVector<Value> mixedOffsets;
  SmallVector<Value> mixedSizes;
  SmallVector<int64_t> baseStrides;
  SmallVector<int64_t> srcShape;
  int64_t baseOffset;
  if(auto allocOp = dyn_cast<memref::AllocOp>(sourceOp)){
    std::tie(baseStrides, baseOffset) = getStridesAndOffset(allocOp.getType());
    srcShape = SmallVector<int64_t>(allocOp.getType().getShape().begin(),allocOp.getType().getShape().end());
    llvm::outs()<<"\n";
    llvm::outs()<<"offset: "<<baseOffset<<"\n";
  }
  else if(auto subviewOp = dyn_cast<memref::SubViewOp>(sourceOp)){
    SmallVector<OpFoldResult> mixedStrides = subviewOp.getMixedStrides();
    for (auto stride : mixedStrides) {
        llvm::outs() << stride << " ";
    }
    llvm::outs()<<"\n";
    mixedOffsets = getValueOrCreateConstantIndexOp(
        rewriter, packOp.getLoc(), subviewOp.getMixedOffsets());
    mixedSizes = getValueOrCreateConstantIndexOp(
        rewriter, packOp.getLoc(), subviewOp.getMixedSizes());

    for (auto offset : mixedOffsets) {
        llvm::outs() << offset << "x";
    }
    llvm::outs()<<"\n";
    SmallVector<OpFoldResult> strides2 = subviewOp.getStrides();
    for (auto stride : strides2) {
        llvm::outs() << stride << " ";
    }
    llvm::outs()<<"\n";
    std::tie(baseStrides, baseOffset) = getStridesAndOffset(subviewOp.getSource().getType());
    llvm::outs()<<"offset: "<<baseOffset<<"\n";
    sourceOp = subviewOp.getSource().getDefiningOp();
    //srcShape = subviewOp.getSource().getType().getShape();
    srcShape = SmallVector<int64_t>(subviewOp.getType().getShape().begin(),subviewOp.getType().getShape().end());
  }
  // apply tiling

  for(int i=0;i<innerTiles.size();i++){
    srcShape.push_back(innerTiles[i]);
    srcShape[innerDimsPos[i]]/=innerTiles[i];
    baseStrides.push_back(baseStrides[innerDimsPos[i]]);
    baseStrides[innerDimsPos[i]]*=innerTiles[i];
  }
  if(!permutation.empty()){
    auto outerShape = SmallVector<int64_t>(srcShape.begin(),srcShape.begin()+permutation.size());
    auto outerStrides = SmallVector<int64_t>(baseStrides.begin(),baseStrides.begin()+permutation.size());
    applyPermutationToVector(outerStrides, permutation);
    applyPermutationToVector(outerShape, permutation);
    for(int i=0;i<outerShape.size();i++){
      baseStrides[i] = outerStrides[i];
      srcShape[i] = outerShape[i];
    }

  }
    for (auto stride : baseStrides) {
        baseStridesValues.push_back(rewriter.create<arith::ConstantIndexOp>(
            packOp.getLoc(), stride));
    }
    for (auto dim : srcShape) {
        srcShapeValues.push_back(rewriter.create<arith::ConstantIndexOp>(
            packOp.getLoc(), dim));
    }
  SmallVector<Value, 2> emptyVec;
  rewriter.replaceOpWithNewOp<xilinx::air::DmaMemcpyNdOp>(
          packOp, SmallVector<Type, 1>{}, emptyVec, DstOp->getResult(0), emptyVec,
          emptyVec, emptyVec, sourceOp->getResult(0), mixedOffsets, srcShapeValues,
          baseStridesValues);
    return success();
  }
};

class AMDAIEPackToDmaPass
    : public impl::AMDAIEPackToDmaBase<
          AMDAIEPackToDmaPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect, xilinx::air::airDialect>();
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
