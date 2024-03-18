// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-map-forall-to-cores"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// The logic to calculate block_id and local_id is as following
/// %num_blocks_y = ceilDiv(%num_cores_y, %block_size_y)
/// %num_blocks_x = ceilDiv(%num_cores_x, %block_size_x)
/// %block_id_y = %core_id_y / %block_size_y
/// %block_id_x = %core_id_x / %block_size_x
/// %local_id_y = %core_id_y mod %block_size_y
/// %local_id_x = %core_id_x mod %block_size_x
static void CalculateBlockLocalId(MLIRContext *context, IRRewriter &rewriter,
                                  Location loc, SmallVector<Value> &blockSizes,
                                  SmallVector<Value> &numCores,
                                  SmallVector<Value> &coreIds,
                                  SmallVector<Value> &numBlocks,
                                  SmallVector<Value> &blockIds,
                                  SmallVector<Value> &localIds) {
  AffineExpr sym0, sym1;
  bindSymbols(context, sym0, sym1);
  auto ceilDivMap = AffineMap::get(0, 2, sym0.ceilDiv(sym1), context);
  auto floorDivMap = AffineMap::get(0, 2, sym0.floorDiv(sym1), context);
  auto modMap = AffineMap::get(0, 2, {sym0 % sym1}, context);

  for (auto &&[blockSize, numCore, coreId] :
       llvm::zip(blockSizes, numCores, coreIds)) {
    auto numBlock = rewriter.create<affine::AffineApplyOp>(
        loc, ceilDivMap, ValueRange{numCore, blockSize});
    auto blockId = rewriter.create<affine::AffineApplyOp>(
        loc, floorDivMap, ValueRange{coreId, blockSize});
    auto localId = rewriter.create<affine::AffineApplyOp>(
        loc, modMap, ValueRange{coreId, blockSize});
    numBlocks.push_back(numBlock);
    blockIds.push_back(blockId);
    localIds.push_back(localId);
  }
}

/// Map scf.forall to scf.for ops with the modified lower bound and step size.
/// The logic to calculate the new lower bound and step size is as following
/// %new_block_lb = %lb0 + %block_id * %step0
/// %new_local_lb = %lb1 + %local_id * %step1
/// %new_block_step = %num_blocks * %step0
/// %new_local_step = %block_size * %step1
static LogicalResult ForallToForOp(MLIRContext *context, IRRewriter &rewriter,
                                   Location loc, scf::ForallOp forallOp,
                                   SmallVector<Value> &num,
                                   SmallVector<Value> &ids) {
  AffineExpr sym0, sym1, sym2;
  bindSymbols(context, sym0, sym1, sym2);
  auto mulMap = AffineMap::get(0, 2, {sym0 * sym1}, context);
  auto mulAddMap = AffineMap::get(0, 3, {sym0 * sym1 + sym2}, context);

  SmallVector<OpFoldResult> lbs = forallOp.getMixedLowerBound();
  SmallVector<OpFoldResult> ubs = forallOp.getMixedUpperBound();
  SmallVector<OpFoldResult> steps = forallOp.getMixedStep();
  SmallVector<Value> ivs;

  if (lbs.size() != 2 || ubs.size() != 2 || steps.size() != 2 ||
      num.size() != 2 || ids.size() != 2) {
    return failure();
  }

  for (auto &&[lb, ub, step, count, id] :
       llvm::zip(lbs, ubs, steps, num, ids)) {
    Value lbValue = getValueOrCreateConstantIndexOp(rewriter, loc, lb);
    Value ubValue = getValueOrCreateConstantIndexOp(rewriter, loc, ub);
    Value stepValue = getValueOrCreateConstantIndexOp(rewriter, loc, step);

    auto newLbValue = rewriter.create<affine::AffineApplyOp>(
        loc, mulAddMap, ValueRange{id, stepValue, lbValue});
    auto newStepValue = rewriter.create<affine::AffineApplyOp>(
        loc, mulMap, ValueRange{count, stepValue});

    auto loop = rewriter.create<scf::ForOp>(
        loc, newLbValue, ubValue, newStepValue, ValueRange(),
        [](OpBuilder &, Location, Value, ValueRange) {});
    ivs.push_back(loop.getInductionVar());
    rewriter.setInsertionPointToStart(loop.getBody());
    rewriter.create<scf::YieldOp>(loc);
    rewriter.setInsertionPointToStart(loop.getBody());
  }
  rewriter.eraseOp(forallOp.getBody()->getTerminator());
  rewriter.inlineBlockBefore(forallOp.getBody(), &*rewriter.getInsertionPoint(),
                             ivs);
  rewriter.eraseOp(forallOp);
  return success();
}

class AMDAIEMapForallToCoresPass
    : public impl::AMDAIEMapForallToCoresBase<AMDAIEMapForallToCoresPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  AMDAIEMapForallToCoresPass() = default;
  AMDAIEMapForallToCoresPass(const AMDAIEMapForallToCoresPass &pass){};
  AMDAIEMapForallToCoresPass(const AMDAIEMapForallToCoresOptions &options)
      : AMDAIEMapForallToCoresBase(options){};
  void runOnOperation() override;
};

void AMDAIEMapForallToCoresPass::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();
  IRRewriter rewriter(context);

  // First find the top level and inner scf.forall ops.
  // There could be multiple inner forall ops under the top level forall.
  scf::ForallOp topLevelForallOp;
  SmallVector<scf::ForallOp> innerForallOps;
  funcOp->walk([&](scf::ForallOp forallOp) {
    if (forallOp->getParentOfType<scf::ForallOp>()) {
      innerForallOps.push_back(forallOp);
      return WalkResult::advance();
    }
    if (topLevelForallOp) return WalkResult::interrupt();
    topLevelForallOp = forallOp;
    return WalkResult::advance();
  });

  if (!topLevelForallOp || innerForallOps.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "----- skip, there are no nested scf.forall loops  -----\n");
    return;
  }

  // Set a herd of cores for distribution according to {numCoresRow,
  // numCoresCol}. Currently using a scf.forall op to represent a herd.
  rewriter.setInsertionPoint(topLevelForallOp);
  auto loc = topLevelForallOp.getLoc();
  auto createConst = [&](int dim) {
    return rewriter.create<arith::ConstantIndexOp>(loc, dim);
  };

  Value zero = createConst(0);
  Value one = createConst(1);
  Value numCoreY = createConst(numCoresRow);
  Value numCoreX = createConst(numCoresCol);
  SmallVector<OpFoldResult> lb = {zero, zero};
  SmallVector<OpFoldResult> ub = {numCoreY, numCoreX};
  SmallVector<OpFoldResult> step = {one, one};
  auto herdOp = rewriter.create<scf::ForallOp>(loc, lb, ub, step, ValueRange(),
                                               std::nullopt);
  rewriter.setInsertionPointToStart(herdOp.getBody());

  // Calculate block_id and local_id.
  Value blockSizeY = createConst(blockSizeRow);
  Value blockSizeX = createConst(blockSizeCol);
  SmallVector<Value> blockSizes = {blockSizeY, blockSizeX};
  SmallVector<Value> numCores = {numCoreY, numCoreX};
  SmallVector<Value> coreIds = herdOp.getInductionVars();
  SmallVector<Value> numBlocks;
  SmallVector<Value> blockIds;
  SmallVector<Value> localIds;
  CalculateBlockLocalId(context, rewriter, loc, blockSizes, numCores, coreIds,
                        numBlocks, blockIds, localIds);

  // Map scf.forall to scf.for ops with the modified lower bound and step size.
  auto topFor = ForallToForOp(context, rewriter, loc, topLevelForallOp,
                              numBlocks, blockIds);
  if (failed(topFor)) {
    funcOp->emitOpError("failed to map the top level forall");
    return signalPassFailure();
  }

  for (auto innerForallOp : innerForallOps) {
    rewriter.setInsertionPoint(innerForallOp);
    auto innerFor = ForallToForOp(context, rewriter, loc, innerForallOp,
                                  blockSizes, localIds);
    if (failed(innerFor)) {
      funcOp->emitOpError("failed to map the inner forall");
      return signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEMapForallToCoresPass(
    AMDAIEMapForallToCoresOptions options) {
  return std::make_unique<AMDAIEMapForallToCoresPass>(options);
}
}  // namespace mlir::iree_compiler::AMDAIE
