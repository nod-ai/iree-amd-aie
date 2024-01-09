// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/Transforms/Passes.h"
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

/// Lower WorkgroupCountFromSliceOp with all 1's.
static LogicalResult lowerWorkgroupCount(RewriterBase &rewriter,
                                         func::FuncOp entryPointFn) {
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp =
      getEntryPoint(entryPointFn);
  if (failed(exportOp)) {
    return entryPointFn.emitOpError(
        "expected function to be entry point function");
  }
  Block *body = exportOp->getWorkgroupCountBody();
  if (!body) {
    return exportOp->emitOpError("unexpected empty workgroup count region");
  }
  SmallVector<IREE::Flow::DispatchWorkgroupCountFromSliceOp> workgroupCountOps;
  for (Operation &op : *body) {
    if (isa<IREE::Flow::DispatchWorkgroupCountFromSliceOp>(&op)) {
      workgroupCountOps.push_back(
          cast<IREE::Flow::DispatchWorkgroupCountFromSliceOp>(&op));
    }
  }
  if (workgroupCountOps.empty()) {
    // If there are no default handled `flow.dispatch.workgroup_count`
    // operation, do nothing.
    return success();
  }
  if (!llvm::hasSingleElement(workgroupCountOps)) {
    return exportOp->emitOpError(
        "unexpected multiple flow.dispatch.workgroup_count_slice_op "
        "operations in body");
  }
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(workgroupCountOps[0]);
  Location loc = workgroupCountOps[0].getLoc();
  SmallVector<OpFoldResult> results;
  results.resize(workgroupCountOps[0].getNumResults(),
                 rewriter.getIndexAttr(1));
  rewriter.replaceOp(workgroupCountOps[0],
                     getValueOrCreateConstantIndexOp(rewriter, loc, results));
  return success();
}

//===----------------------------------------------------------------------------------===//
// Methods copied over from core to implement tile and fuse with scf.forall.
// TODO(MaheshRavishankar) Replace them with core methods when upstream
// can handle this natively
//===----------------------------------------------------------------------------------===//
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

/// Return the untiled producer whose slice is used in a tiled consumer. If
/// there was no loop traversal needed, the second value of the returned tuple
/// is empty. In case the source op is a linalg.copy it returns both values as
/// empty.
static std::tuple<OpResult, std::optional<OpOperand *>>
getUntiledProducerFromSliceSource(OpOperand *source, scf::ForallOp loop) {
  std::optional<OpOperand *> destinationSharedOuts;
  if (auto sharedOuts = dyn_cast<BlockArgument>(source->get())) {
    if (sharedOuts.getOwner()->getParentOp() == loop) {
      source = loop.getTiedOpOperand(sharedOuts);
      destinationSharedOuts = source;
    }
  } else if (isa<linalg::CopyOp>(source->get().getDefiningOp()))
    return {nullptr, nullptr};
  return {dyn_cast<OpResult>(source->get()), destinationSharedOuts};
}

/// The following is a copy of tileAndFuseProducerOfSlice to make it work for
/// scf.forall.
/// Implementation of fusing producer of a single slice by computing the
/// slice of the producer in-place.
std::optional<scf::SCFFuseProducerOfSliceResult> tileAndFuseProducerOfSlice(
    RewriterBase &rewriter, tensor::ExtractSliceOp candidateSliceOp,
    scf::ForallOp &loop) {
  // 1. Get the producer of the source.
  auto [fusableProducer, destinationInitArg] =
      getUntiledProducerFromSliceSource(&candidateSliceOp.getSourceMutable(),
                                        loop);
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
      isa<DestinationStyleOpInterface>(fusableProducerOp) && loop) {
    loop->getOpOperands()[destinationInitArg.value()->getOperandNumber()].set(
        origDestinationTensors[resultNumber]);
  }
  return scf::SCFFuseProducerOfSliceResult{fusableProducer,
                                           tileAndFuseResult->tiledValues[0],
                                           tileAndFuseResult->tiledOps};
}

//===----------------------------------------------------------------------------------===//
// End methods copied over from core to implement tile and fuse with scf.forall.
//===----------------------------------------------------------------------------------===//

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

LogicalResult applyTileAndFuse(RewriterBase &rewriter, Operation *rootOp,
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
        tileAndFuseProducerOfSlice(rewriter, candidateSliceOp, forallLoop);
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

/// This pass starts with the last TilingInterface operation, tiles the op and
/// fuses its producers recursively. The `tilingLevel` must be specified. It
/// picks the `tilingLevel`-th list as tiling sizes from lowering_config.
class AMDAIETileAndFusePass
    : public AMDAIETileAndFuseBase<AMDAIETileAndFusePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect>();
  }

  AMDAIETileAndFusePass() = default;
  AMDAIETileAndFusePass(int64_t tilingLevel = -1) {
    this->tilingLevel.setValue(tilingLevel);
  }
  AMDAIETileAndFusePass(const AMDAIETileAndFusePass &pass){};

  void runOnOperation() override;
};

void AMDAIETileAndFusePass::runOnOperation() {
  MLIRContext *context = &getContext();
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp innerModule = variantOp.getInnerModule();
  for (func::FuncOp funcOp : innerModule.getOps<func::FuncOp>()) {
    TilingInterface consumerOp;
    funcOp->walk<WalkOrder::PostOrder, ReverseIterator>(
        [&](TilingInterface op) {
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
    // TODO(avarma): Generalize this by adding pulling in tilingConfig based on
    // a given tilinglevel, similar to LLVMCPUTileAndFuse.cpp.
    SmallVector<OpFoldResult> tileSizes;
    // TODO(avarma): Have a global CONSTANT defining tiling stages and the
    // tiling
    //               strategy.
    if (tilingLevel == 1) {
      tileSizes = getAsIndexOpFoldResult(context, {8, 8});
    } else if (tilingLevel == 2) {
      tileSizes = getAsIndexOpFoldResult(context, {4, 4});
    } else {
      assert(false && "unsupported tiling level");
    }
    auto options = scf::SCFTilingOptions().setTileSizes(tileSizes);
    options.setMapping(
        {gpu::GPUBlockMappingAttr::get(context, gpu::MappingId::DimY),
         gpu::GPUBlockMappingAttr::get(context, gpu::MappingId::DimX)});

    IRRewriter rewriter(context);
    DominanceInfo dominanceInfo(funcOp);
    if (failed(
            applyTileAndFuse(rewriter, consumerOp, dominanceInfo, options))) {
      LLVM_DEBUG(llvm::dbgs() << "----- tile and fuse failed -----\n");
      return signalPassFailure();
    }

    if (tilingLevel == 1 && failed(lowerWorkgroupCount(rewriter, funcOp))) {
      funcOp->emitOpError("failed to lower workgroup count region");
      return signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createAMDAIETileAndFusePass(int64_t tilingLevel) {
  return std::make_unique<AMDAIETileAndFusePass>(tilingLevel);
}

}  // namespace mlir::iree_compiler::AMDAIE
