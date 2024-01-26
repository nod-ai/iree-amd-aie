// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-tile-and-fuse"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Starting from `op` walk all operands backwards to find all
/// potentially fusable operations, i.e. operations that implement
/// the `TilingInterface`.
static void collectTiledAndFusedOps(Operation *rootOp,
                                    llvm::SmallDenseSet<Operation *> &result) {
  SmallVector<Operation *> worklist;
  worklist.push_back(rootOp);
  result.insert(rootOp);
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    for (OpOperand &operand : current->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (!producer || !isa<TilingInterface>(producer) ||
          result.count(producer))
        continue;
      worklist.push_back(producer);
      result.insert(producer);
    }
  }
}

LogicalResult applyTileAndFuseUsingSCFFor(RewriterBase &rewriter,
                                          Operation *rootOp,
                                          DominanceInfo &dominanceInfo,
                                          scf::SCFTilingOptions options) {
  llvm::SmallDenseSet<Operation *> origTiledAndFusedOps;
  collectTiledAndFusedOps(rootOp, origTiledAndFusedOps);
  auto isIgnoredUser = [&](Operation *user) {
    return origTiledAndFusedOps.count(user) || isa<tensor::DimOp>(user);
  };
  FailureOr<scf::SCFTilingResult> tilingResult =
      tileUsingSCFForOp(rewriter, cast<TilingInterface>(rootOp), options);
  if (failed(tilingResult)) {
    return failure();
  }
  auto forLoops = llvm::to_vector(llvm::map_range(
      tilingResult->loops, [](Operation *op) { return cast<scf::ForOp>(op); }));
  SmallVector<OpResult> yieldedValuesToOrigValues;
  yieldedValuesToOrigValues.append(rootOp->result_begin(),
                                   rootOp->result_end());
  scf::ForOp outermostLoop = forLoops.front();
  for (auto [index, origVal] : llvm::enumerate(yieldedValuesToOrigValues)) {
    Value replacement = outermostLoop.getResult(index);
    rewriter.replaceUsesWithIf(origVal, replacement, [&](OpOperand &use) {
      return !isIgnoredUser(use.getOwner()) &&
             dominanceInfo.properlyDominates(outermostLoop, use.getOwner());
    });
  }

  return success();
}
LogicalResult applyTileAndFuseUsingSCFForall(RewriterBase &rewriter,
                                             Operation *rootOp,
                                             DominanceInfo &dominanceInfo,
                                             scf::SCFTilingOptions options) {
  llvm::SmallDenseSet<Operation *> origTiledAndFusedOps;
  collectTiledAndFusedOps(rootOp, origTiledAndFusedOps);
  auto isIgnoredUser = [&](Operation *user) {
    return origTiledAndFusedOps.count(user) || isa<tensor::DimOp>(user);
  };

  // 1. Tile the consumer.
  SmallVector<OpResult> yieldedValuesToOrigValues;
  SmallVector<Operation *> tiledOps;
  rewriter.setInsertionPointAfter(rootOp);
  FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCFForallOp(
      rewriter, cast<TilingInterface>(rootOp), options);
  if (failed(tilingResult)) {
    return failure();
  }
  auto forallLoop = cast<scf::ForallOp>(tilingResult->loops[0]);
  yieldedValuesToOrigValues.append(rootOp->result_begin(),
                                   rootOp->result_end());
  // A map from untiled value to scf.forall shared_outs. The shared_outs is used
  // for DPS init operand if they use the same init operands.
  llvm::DenseMap<Value, Value> mapToSharedOuts;
  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(rootOp)) {
    for (auto [init, sharedOuts] :
         llvm::zip_equal(dpsOp.getDpsInits(), forallLoop.getRegionOutArgs())) {
      mapToSharedOuts[init] = sharedOuts;
    }
  }
  tiledOps.append(tilingResult->tiledOps);

  // 2. Tiling each operation results in generation of slices. The source of
  // these slices could be producers that can be fused into the tiled loops by
  // computing the slices of these producers in-place. This results in more
  // slices created for operands of the "fused producer". This open up more
  // opportunities for fusion. Use a worklist to fuse greedily.
  auto addCandidateSlices =
      [&](Operation *fusedOp, std::deque<tensor::ExtractSliceOp> &candidates) {
        for (OpOperand &operand : fusedOp->getOpOperands()) {
          auto sliceOp = operand.get().getDefiningOp<tensor::ExtractSliceOp>();
          if (!sliceOp) continue;
          candidates.push_back(sliceOp);

          auto dpsOp = dyn_cast<DestinationStyleOpInterface>(fusedOp);
          if (!dpsOp) continue;

          if (dpsOp.isDpsInit(&operand) &&
              mapToSharedOuts.contains(sliceOp.getSource())) {
            rewriter.startRootUpdate(sliceOp);
            sliceOp.getSourceMutable().assign(
                mapToSharedOuts[sliceOp.getSource()]);
            rewriter.finalizeRootUpdate(sliceOp);
          }
        }
      };

  std::deque<tensor::ExtractSliceOp> candidates;
  addCandidateSlices(tilingResult->tiledOps.back(), candidates);
  OpBuilder::InsertionGuard g(rewriter);
  while (!candidates.empty()) {
    // Traverse the slices in BFS fashion.
    tensor::ExtractSliceOp candidateSliceOp = candidates.front();
    candidates.pop_front();

    // Materialize the slice of the producer in place.
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedProducer =
        mlir::iree_compiler::AMDAIE::tileAndFuseProducerOfSlice(rewriter, candidateSliceOp, forallLoop);
    if (!fusedProducer) continue;

    // Add more fusion candidates to the worklist.
    for (auto tiledOp : fusedProducer->tiledOps) {
      addCandidateSlices(tiledOp, candidates);
      tiledOps.push_back(tiledOp);
    }
  }

  for (auto [index, origVal] : llvm::enumerate(yieldedValuesToOrigValues)) {
    Value replacement = forallLoop.getResult(index);
    rewriter.replaceUsesWithIf(origVal, replacement, [&](OpOperand &use) {
      return !isIgnoredUser(use.getOwner()) &&
             dominanceInfo.properlyDominates(forallLoop, use.getOwner());
    });
  }

  return success();
}

LogicalResult applyTileAndFuseUsingSCF(RewriterBase &rewriter,
                                       Operation *rootOp,
                                       DominanceInfo &dominanceInfo,
                                       scf::SCFTilingOptions options,
                                       bool useSCFFor, int64_t tilingLevel) {
  // TODO(MaheshRavishankar): Adapt this to use SCFTilingOptions after
  // the upstream changes land.
  if (useSCFFor) {
    return applyTileAndFuseUsingSCFFor(rewriter, rootOp, dominanceInfo,
                                       options);
  } else {
    return applyTileAndFuseUsingSCFForall(rewriter, rootOp, dominanceInfo,
                                          options);
  }
}

/// This pass starts with the last TilingInterface operation, tiles the op and
/// fuses its producers recursively. The `tilingLevel` must be specified. It
/// picks the `tilingLevel`-th list as tiling sizes from lowering_config.
class AMDAIETileAndFusePass
    : public impl::AMDAIETileAndFuseBase<AMDAIETileAndFusePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect>();
  }

  AMDAIETileAndFusePass() = default;
  AMDAIETileAndFusePass(const AMDAIETileAndFusePass &pass) {}
  AMDAIETileAndFusePass(const AMDAIETileAndFuseOptions &options)
      : AMDAIETileAndFuseBase(options) {}

  void runOnOperation() override;
};

void AMDAIETileAndFusePass::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();

  TilingInterface consumerOp;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](TilingInterface op) {
    // Find the next consumer op if it does not have loops OR if it is a
    // linalg.copy op.
    if (op.getLoopIteratorTypes().empty() || isa<linalg::CopyOp>(op))
      return WalkResult::advance();
    consumerOp = op;
    return WalkResult::interrupt();
  });
  if (!consumerOp) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, no consumer op -----\n");
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "consumerOp: " << consumerOp << "\n");
  LLVM_DEBUG(llvm::dbgs() << "tilingLevel: " << tilingLevel << "\n");
  // TODO(avarma): Have a global CONSTANT defining tiling stages and the
  // tiling strategy.
  // If `consumerOp` has its own lowering config, we prefer using it.
  // Otherwise, fallback to find a lowering_config from other operations.
  SmallVector<int64_t> tileSizesVal;
  if (auto loweringConfig = getLoweringConfig(consumerOp)) {
    tileSizesVal = loweringConfig.getTileSizeVals(tilingLevel);
  } else {
    FailureOr<IREE::Codegen::LoweringConfigAttr> maybeLoweringConfig =
        getLoweringConfig(getComputeOps(funcOp));
    if (failed(maybeLoweringConfig)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "can't find lowering_config, skip TileAndFuse");
      return;
    }
    tileSizesVal = maybeLoweringConfig.value().getTileSizeVals(tilingLevel);
  }

  if (llvm::all_of(tileSizesVal, [&](int64_t size) { return size == 0; })) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, all zeros -----\n");
    return;
  }

  SmallVector<OpFoldResult> tileSizes =
      getAsIndexOpFoldResult(context, tileSizesVal);
  auto options = scf::SCFTilingOptions().setTileSizes(tileSizes);
  // When tiling using scf.for we do not need to set any mapping.
  if (tilingLevel != 2) {
    options.setMapping(
        {gpu::GPUBlockMappingAttr::get(context, gpu::MappingId::DimY),
         gpu::GPUBlockMappingAttr::get(context, gpu::MappingId::DimX)});
  }

  IRRewriter rewriter(context);
  DominanceInfo dominanceInfo(funcOp);
  if (failed(applyTileAndFuseUsingSCF(rewriter, consumerOp, dominanceInfo,
                                      options, useSCFFor, tilingLevel))) {
    LLVM_DEBUG(llvm::dbgs() << "----- tile and fuse failed -----\n");
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIETileAndFusePass(
    AMDAIETileAndFuseOptions options) {
  return std::make_unique<AMDAIETileAndFusePass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
