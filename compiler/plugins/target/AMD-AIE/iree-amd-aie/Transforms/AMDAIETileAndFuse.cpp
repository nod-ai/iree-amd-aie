// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/Transforms/Passes.h"
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

/// Clones the operation and updates the destination if the operation
/// implements the `DestinationStyleOpInterface`.
static Operation *cloneOpAndUpdateDestinationArgs(RewriterBase &rewriter,
                                                  Operation *op,
                                                  ValueRange newDestArgs) {
  Operation *clonedOp = rewriter.clone(*op);
  if (newDestArgs.empty()) return clonedOp;
  if (auto destinationStyleOp = dyn_cast<DestinationStyleOpInterface>(clonedOp))
    destinationStyleOp.getDpsInitsMutable().assign(newDestArgs);
  return clonedOp;
}

/// Return the untiled producer whose slice is used in a tiled consumer. The
/// method traverses the tile loop nest (`loops`) if needed, and returns the
/// `iter_args` of the outer most that is encountered. Traversing the iter_args
/// indicates that this is a destination operand of the consumer. If there was
/// no loop traversal needed, the second value of the returned tuple is empty.
static std::tuple<OpResult, std::optional<OpOperand *>>
getUntiledProducerFromSliceSource(OpOperand *source,
                                  ArrayRef<scf::ForallOp> loops) {
  std::optional<OpOperand *> destinationIterArg;
  auto loopIt = loops.rbegin();
  while (auto iterArg = dyn_cast<BlockArgument>(source->get())) {
    scf::ForallOp loop = *loopIt;
    if (iterArg.getOwner()->getParentOp() != loop) break;
    source = loop.getTiedOpOperand(iterArg);
    loopIt++;
  }
  if (loopIt == loops.rend()) destinationIterArg = source;
  return {dyn_cast<OpResult>(source->get()), destinationIterArg};
}

/// The following is a copy of tileAndFuseProducerOfSlice to make it work for
/// scf.forall.
/// Implementation of fusing producer of a single slice by computing the
/// slice of the producer in-place.
std::optional<scf::SCFFuseProducerOfSliceResult> tileAndFuseProducerOfSlice(
    RewriterBase &rewriter, tensor::ExtractSliceOp candidateSliceOp,
    MutableArrayRef<scf::ForallOp> loops) {
  // 1. Get the producer of the source (potentially walking through
  // `iter_args` of nested `scf.for`)
  auto [fusableProducer, destinationInitArg] =
      getUntiledProducerFromSliceSource(&candidateSliceOp.getSourceMutable(),
                                        loops);
  if (!fusableProducer) return std::nullopt;
  unsigned resultNumber = fusableProducer.getResultNumber();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(candidateSliceOp);

  // 2. Clone the fused producer
  // 2a. Compute the destination operands to use for the cloned operation.
  SmallVector<Value> origDestinationTensors, clonedOpDestinationTensors;
  Operation *fusableProducerOp = fusableProducer.getOwner();
  if (isa<DestinationStyleOpInterface>(fusableProducerOp) &&
      failed(tensor::getOrCreateDestinations(
          rewriter, fusableProducerOp->getLoc(), fusableProducerOp,
          origDestinationTensors)))
    return std::nullopt;

  clonedOpDestinationTensors = origDestinationTensors;
  if (destinationInitArg &&
      isa<DestinationStyleOpInterface>(fusableProducerOp)) {
    // 2b. If the producer is also destination style, then to maintain the
    // destination passing style, update the destination of the producer to be
    // the source of the slice.
    clonedOpDestinationTensors[resultNumber] = candidateSliceOp.getSource();
  }
  // 2c. Clone the fused producer.
  Operation *clonedProducerOp = cloneOpAndUpdateDestinationArgs(
      rewriter, fusableProducerOp, clonedOpDestinationTensors);
  // 2d. Update the source of the candidateSlice to be the cloned producer.
  //     Easier to just clone the slice with different source since replacements
  //     and DCE of cloned ops becomes easier
  SmallVector<Value> candidateSliceOpOperands =
      llvm::to_vector(candidateSliceOp->getOperands());
  candidateSliceOpOperands[0] = clonedProducerOp->getResult(resultNumber);
  tensor::ExtractSliceOp clonedCandidateSliceOp =
      mlir::clone(rewriter, candidateSliceOp,
                  candidateSliceOp->getResultTypes(), candidateSliceOpOperands);

  // 3. Generate the tiled implementation of the producer of the source
  FailureOr<TilingResult> tileAndFuseResult =
      tensor::replaceExtractSliceWithTiledProducer(
          rewriter, clonedCandidateSliceOp,
          clonedProducerOp->getResult(resultNumber));
  if (failed(tileAndFuseResult)) return std::nullopt;
  // Note: Do not delete the candidateSliceOp, since its passed in from the
  // caller.
  rewriter.replaceAllUsesWith(candidateSliceOp,
                              tileAndFuseResult->tiledValues[0]);
  rewriter.eraseOp(clonedCandidateSliceOp);
  rewriter.eraseOp(clonedProducerOp);

  if (destinationInitArg &&
      isa<DestinationStyleOpInterface>(fusableProducerOp) && !loops.empty()) {
    loops.front()
        ->getOpOperands()[destinationInitArg.value()->getOperandNumber()]
        .set(origDestinationTensors[resultNumber]);
  }
  return scf::SCFFuseProducerOfSliceResult{fusableProducer,
                                           tileAndFuseResult->tiledValues[0],
                                           tileAndFuseResult->tiledOps};
}

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

/// Tiling of `tensor.pad` operation generates
///
/// ```mlir
/// scf.if {
///   ...
/// } else {
///    tensor.pad
/// }
/// ```
///
/// For IREEs use case we dont need this. So this folds away the `if` condition.
/// Note this is a fairly hacky workaround, but the current pad operation
/// semantics force us down this path.
static FailureOr<tensor::PadOp> foldIfGeneratedFromPadding(
    RewriterBase &rewriter, tensor::PadOp untiledPadOp,
    tensor::PadOp tiledPadOp) {
  auto ifOp = dyn_cast<scf::IfOp>(tiledPadOp->getParentOp());
  if (!ifOp) {
    return failure();
  };
  Block *block = tiledPadOp->getBlock();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, ifOp, /*blockArgs=*/{});
  rewriter.replaceOp(ifOp, results);
  rewriter.eraseOp(terminator);
  return tiledPadOp;
}

/// Method to add new init values to a loop nest. Updates `loops` in-place with
/// new loops that use the `newInitValues`.
/// The outer-loops are updated to yield the new result values of the inner
/// loop. For the innermost loop, the call back `getNewYields` is invoked to get
/// the additional values to yield form the innermost loop.
static void addInitOperandsToLoopNest(
    RewriterBase &rewriter, MutableArrayRef<scf::ForallOp> loops,
    ValueRange newInitValues,
    llvm::function_ref<SmallVector<Value>(RewriterBase &rewriter,
                                          ValueRange newRegionOutArgs)>
        getNewYieldValsFn) {
  SmallVector<scf::ForallOp> newLoops;
  if (loops.empty()) return;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(loops.front());
  for (auto &loop : loops) {
    rewriter.setInsertionPoint(loop);

    // Create a new loop with the new init values for this loop.
    SmallVector<Value> newInits = llvm::to_vector(loop.getOutputs());
    newInits.append(newInitValues.begin(), newInitValues.end());
    auto newLoop = rewriter.create<scf::ForallOp>(
        loop.getLoc(), loop.getMixedLowerBound(), loop.getMixedUpperBound(),
        loop.getMixedStep(), newInits, loop.getMapping(),
        [&](OpBuilder &b, Location loc, ValueRange values) {});

    // Merge the body of the new loop with the body of the old loops.
    SmallVector<Value> sourceBlockArgs;
    // sourceBlockArgs.append(newLoop.getInductionVars());
    for (auto x : newLoop.getInductionVars()) sourceBlockArgs.push_back(x);
    auto newRegionIterArgs = newLoop.getRegionOutArgs();
    sourceBlockArgs.append(
        newRegionIterArgs.begin(),
        std::next(newRegionIterArgs.begin(), loop.getNumResults()));
    rewriter.mergeBlocks(loop.getBody(), newLoop.getBody(), sourceBlockArgs);
    rewriter.replaceOp(loop,
                       newLoop.getResults().take_front(loop.getNumResults()));
    loop = newLoop;
    newInitValues = newLoop.getRegionOutArgs().take_back(newInitValues.size());
  }

  // Update the loop body of the innermost loop to get new yield values.
  scf::ForallOp innerMostLoop = loops.back();
  auto innerMostYieldOp =
      cast<scf::YieldOp>(innerMostLoop.getBody()->getTerminator());
  rewriter.setInsertionPoint(innerMostYieldOp);
  SmallVector<Value> newYieldVals =
      getNewYieldValsFn(rewriter, innerMostLoop.getRegionOutArgs());
  SmallVector<Value> newYieldOperands =
      llvm::to_vector(innerMostYieldOp->getOperands());
  newYieldOperands.append(newYieldVals);
  rewriter.replaceOpWithNewOp<scf::YieldOp>(innerMostYieldOp, newYieldOperands);

  // Make all other loops except the innermost loops yield the values
  // returned by the inner loop.
  for (auto [outerLoop, innerLoop] :
       llvm::zip_equal(loops.drop_back(), loops.drop_front())) {
    auto outerLoopYield =
        cast<scf::YieldOp>(outerLoop.getBody()->getTerminator());
    SmallVector<Value> newYields =
        llvm::to_vector(outerLoopYield.getOperands());
    ValueRange additionalYields =
        innerLoop.getResults().take_back(newInitValues.size());
    newYields.append(additionalYields.begin(), additionalYields.end());
    rewriter.setInsertionPoint(outerLoopYield);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(outerLoopYield, newYields);
  }
}

/// Reconstruct the fused producer from within the tiled-and-fused code.
void yieldReplacementForFusedProducer(
    RewriterBase &rewriter, tensor::ExtractSliceOp sliceOp,
    scf::SCFFuseProducerOfSliceResult fusedProducerInfo,
    MutableArrayRef<scf::ForallOp> loops) {
  if (loops.empty()) return;

  OpResult fusableProducer = fusedProducerInfo.origProducer;
  Value tiledAndFusedProducer = fusedProducerInfo.tiledAndFusedProducer;
  FailureOr<Value> initValue = tensor::getOrCreateDestination(
      rewriter, fusableProducer.getOwner()->getLoc(), fusableProducer);
  if (succeeded(initValue)) {
    auto newYieldValuesFn =
        [&](RewriterBase &innerRewriter,
            ValueRange newRegionOutArgs) -> SmallVector<Value> {
      OpBuilder::InsertionGuard g(innerRewriter);
      if (auto tiledDestStyleOp =
              tiledAndFusedProducer
                  .getDefiningOp<DestinationStyleOpInterface>()) {
        rewriter.setInsertionPoint(tiledDestStyleOp);
        BlockArgument newRegionArg = loops.back().getRegionOutArgs().back();
        auto destSlice = rewriter.create<tensor::ExtractSliceOp>(
            sliceOp.getLoc(), newRegionArg, sliceOp.getMixedOffsets(),
            sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
        unsigned resultNumber = fusableProducer.getResultNumber();
        rewriter.updateRootInPlace(tiledDestStyleOp, [&]() {
          tiledDestStyleOp.getDpsInitsMutable()[resultNumber].set(destSlice);
        });
      }
      Block *block = rewriter.getInsertionPoint()->getBlock();
      rewriter.setInsertionPoint(block->getTerminator());
      Value replacement = rewriter.create<tensor::InsertSliceOp>(
          fusedProducerInfo.origProducer.getLoc(),
          fusedProducerInfo.tiledAndFusedProducer,
          loops.back().getRegionOutArgs().back(), sliceOp.getMixedOffsets(),
          sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
      return {replacement};
    };

    addInitOperandsToLoopNest(rewriter, loops,
                              SmallVector<Value>{initValue.value()},
                              newYieldValuesFn);
  }
}

LogicalResult applyTileAndFuse(RewriterBase &rewriter, Operation *rootOp,
                               DominanceInfo &dominanceInfo,
                               scf::SCFTilingOptions options) {
  llvm::SmallDenseSet<Operation *> origTiledAndFusedOps;
  collectTiledAndFusedOps(rootOp, origTiledAndFusedOps);
  auto isIgnoredUser = [&](Operation *user, scf::ForallOp outerMostTiledLoop) {
    return origTiledAndFusedOps.count(user) || isa<tensor::DimOp>(user);
  };

  // The rest of this method is similar to
  // scf::tileConsumerAndFuseProducerGreedilyUsingSCFForOp, except that this
  // replaces DPS out operand with iter_arg when they use the same init
  // operands.

  // 1. Tile the consumer.
  SmallVector<OpResult> yieldedValuesToOrigValues;
  SmallVector<Operation *> tiledOps;
  rewriter.setInsertionPointAfter(rootOp);
  FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCFForallOp(
      rewriter, cast<TilingInterface>(rootOp), options);
  if (failed(tilingResult)) {
    return failure();
  }
  auto forLoops =
      llvm::to_vector(llvm::map_range(tilingResult->loops, [](Operation *op) {
        return cast<scf::ForallOp>(op);
      }));
  yieldedValuesToOrigValues.append(rootOp->result_begin(),
                                   rootOp->result_end());
  // A map from untiled value to scf.forall iter_arg. The iter_arg is used for
  // DPS init operand if they use the same init operands.
  llvm::DenseMap<Value, Value> mapToIterArg;

  // WAR for `if` ops generating `scf.if` operations.
  if (auto rootPadOp = dyn_cast<tensor::PadOp>(rootOp)) {
    assert(tilingResult->tiledOps.size() == 1 &&
           "expected tiling of `pad` op to return only one operation");
    FailureOr<Operation *> replacementTiledOp = foldIfGeneratedFromPadding(
        rewriter, rootPadOp, cast<tensor::PadOp>(tilingResult->tiledOps[0]));
    if (!failed(replacementTiledOp)) {
      tilingResult->tiledOps[0] = replacementTiledOp.value();
    }
  } else if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(rootOp)) {
    for (auto [init, iterArg] : llvm::zip_equal(
             dpsOp.getDpsInits(),
             cast<scf::ForallOp>(forLoops.back()).getRegionOutArgs())) {
      mapToIterArg[init] = iterArg;
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
              mapToIterArg.contains(sliceOp.getSource())) {
            rewriter.startRootUpdate(sliceOp);
            sliceOp.getSourceMutable().assign(
                mapToIterArg[sliceOp.getSource()]);
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
        tileAndFuseProducerOfSlice(rewriter, candidateSliceOp, forLoops);
    if (!fusedProducer) continue;

    // Check if the fused producer has other uses that require the value
    // to be yielded from within the tiled loop.
    OpResult untiledProducer = fusedProducer->origProducer;
    if (llvm::any_of(untiledProducer.getUsers(), [&](Operation *user) {
          return !isIgnoredUser(user, forLoops.front()) &&
                 !forLoops.front()->isAncestor(user);
          ;
        })) {
      yieldReplacementForFusedProducer(rewriter, candidateSliceOp,
                                       fusedProducer.value(), forLoops);
      yieldedValuesToOrigValues.push_back(untiledProducer);
    }

    // Add more fusion candidates to the worklist.
    for (auto tiledOp : fusedProducer->tiledOps) {
      addCandidateSlices(tiledOp, candidates);
      tiledOps.push_back(tiledOp);
    }
  }

  scf::ForallOp outermostLoop = forLoops.front();
  for (auto [index, origVal] : llvm::enumerate(yieldedValuesToOrigValues)) {
    Value replacement = outermostLoop.getResult(index);
    rewriter.replaceUsesWithIf(origVal, replacement, [&](OpOperand &use) {
      return !isIgnoredUser(use.getOwner(), outermostLoop) &&
             dominanceInfo.properlyDominates(outermostLoop, use.getOwner());
    });
  }

  return success();
}

/// This pass starts with the last TilingInterface operation, tiles the op and
/// fuses its producers recursively. The `tilingLevel` must be specified. It
/// picks the `tilingLevel`-th list as tiling sizes from lowering_config.
class AMDAIETileAndFusePass
    : public AMDAIETileAndFuseBase<AMDAIETileAndFusePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  AMDAIETileAndFusePass() = default;
  AMDAIETileAndFusePass(int64_t tilingLevel = -1) {
    this->tilingLevel.setValue(tilingLevel);
  }
  AMDAIETileAndFusePass(const AMDAIETileAndFusePass &pass){};

  void runOnOperation() override;
};

void AMDAIETileAndFusePass::runOnOperation() {
  llvm::outs() << "BEFORE :-\n" << (*getOperation()) << "\n--------\n";
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  TilingInterface consumerOp;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](TilingInterface op) {
    // Find the next consumer op if it does not have loops.
    if (op.getLoopIteratorTypes().empty()) return WalkResult::advance();
    consumerOp = op;
    return WalkResult::interrupt();
  });
  if (!consumerOp) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, no consumer op -----\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "consumerOp: " << consumerOp << "\n");
  LLVM_DEBUG(llvm::dbgs() << "tilingLevel: " << tilingLevel << "\n");
  // TODO:
  //    1. Add tile level.
  //    2. Currently hardcoding the tile size to make the tile and fuse run for
  //    at
  //       least the first level. So, we can generalize this by adding tile
  //       level, similar to LLVMCPUTileAndFuse.cpp by uncommenting the code
  //       below.
  // // If `consumerOp` has its own lowering config, we prefer using it.
  // Otherwise,
  // // fallback to find a lowering_config from other operations.
  // SmallVector<int64_t> tileSizes;
  // SmallVector<bool> tileScalableFlags;
  // if (auto loweringConfig = getLoweringConfig(consumerOp)) {
  //   tileSizes = loweringConfig.getTileSizeVals(tilingLevel);
  //   tileScalableFlags = loweringConfig.getScalableTileFlagVals(tilingLevel);
  // } else {
  //   FailureOr<IREE::Codegen::LoweringConfigAttr> maybeLoweringConfig =
  //       getLoweringConfig(getComputeOps(funcOp));
  //   if (failed(maybeLoweringConfig)) {
  //     LLVM_DEBUG(llvm::dbgs()
  //                << "can't find lowering_config, skip TileAndFuse");
  //     return;
  //   }
  //   tileSizes = maybeLoweringConfig.value().getTileSizeVals(tilingLevel);
  //   tileScalableFlags =
  //       maybeLoweringConfig.value().getScalableTileFlagVals(tilingLevel);
  // }

  // if (llvm::all_of(tileSizes, [&](int64_t size) { return size == 0; })) {
  //   LLVM_DEBUG(llvm::dbgs() << "----- skip, all zeros -----\n");
  //   return;
  // }

  // scf::SCFTilingOptions options{};
  // setSCFTileSizes(options, consumerOp, std::move(tileSizes),
  //                 std::move(tileScalableFlags));

  SmallVector<OpFoldResult> tileSizes = getAsIndexOpFoldResult(context, {8, 8});
  auto options = scf::SCFTilingOptions().setTileSizes(tileSizes);
  options.setMapping(
      {gpu::GPUBlockMappingAttr::get(context, gpu::MappingId::DimY),
       gpu::GPUBlockMappingAttr::get(context, gpu::MappingId::DimX)});

  IRRewriter rewriter(context);
  DominanceInfo dominanceInfo(funcOp);
  if (failed(applyTileAndFuse(rewriter, consumerOp, dominanceInfo, options))) {
    LLVM_DEBUG(llvm::dbgs() << "----- tile and fuse failed -----\n");
    return signalPassFailure();
  }

  llvm::outs() << "AFTER :-\n" << (*getOperation()) << "\n--------\n";
}

}  // namespace

std::unique_ptr<OperationPass<>> createAMDAIETileAndFusePass(int64_t tilingLevel) {
  return std::make_unique<AMDAIETileAndFusePass>(tilingLevel);
}

}  // namespace mlir::iree_compiler::AMDAIE
