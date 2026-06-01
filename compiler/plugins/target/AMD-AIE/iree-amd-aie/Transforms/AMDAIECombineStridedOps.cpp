// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the transformation that combines doubly strided operations
// in the same block if possible.
//
//===----------------------------------------------------------------------===//

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-combine-strided-ops"

using namespace std::placeholders;

namespace mlir::iree_compiler::AMDAIE {

namespace {

struct CombineStridedOps
    : public OpInterfaceRewritePattern<AMDAIE::DoublyStridedOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(AMDAIE::DoublyStridedOpInterface op,
                                PatternRewriter &rewriter) const override {
    // Get the device model.
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
    std::optional<AMDAIEDevice> device = getConfigAMDAIEDevice(targetAttr);
    if (!device)
      return op.emitOpError()
             << "No AMDAIEDevice found in the target attribute configuration";
    AMDAIE::AMDAIEDeviceModel deviceModel =
        AMDAIE::getDeviceModel(device.value());

    Block *block = op->getBlock();
    if (!block) return failure();

    std::unique_ptr<DmaDimConfig> sourceDmaDimConfig;
    std::unique_ptr<DmaDimConfig> targetDmaDimConfig;

    SmallVector<Operation *> userOpsToBeErased;
    AMDAIE::DoublyStridedOpInterface nextStridedOp;

    if (auto npuDmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(op.getOperation())) {
      LLVM_DEBUG(llvm::dbgs() << "npuDmaOp: " << npuDmaOp << "\n");
      // The wait users of `npuDmaOp` are erased after combining, so each one
      // must reference only `npuDmaOp`'s tokens. Bail on any non-wait user or
      // any multi-token wait (which would also be synchronizing another DMA
      // whose sync we would silently drop). At this pipeline stage
      // (DmaComposition runs before NpuDmaToHalfDmaCpyNd and FoldDmaWaits),
      // every wait is single-token in the standard flow; the check guards
      // hand-authored IR.
      for (Operation *userOp : npuDmaOp->getUsers()) {
        auto waitOp = dyn_cast<AMDAIE::NpuDmaWaitOp>(userOp);
        if (!waitOp || waitOp.getAsyncTokens().size() != 1) return failure();
        userOpsToBeErased.push_back(userOp);
      }
      FailureOr<AMDAIE::NpuDmaCpyNdOp> maybeNextNpuDmaOp =
          findNextDmaOpWithSameConnection(npuDmaOp, block);
      if (failed(maybeNextNpuDmaOp)) return failure();
      AMDAIE::NpuDmaCpyNdOp nextNpuDmaOp = *maybeNextNpuDmaOp;
      // The same hardware connection can be intentionally reused for distinct
      // logical transfers in different phases (for example, draining two
      // different outputs through one route). Combining such ops would keep the
      // first op's source/target operands and silently redirect the later
      // phase.
      if (npuDmaOp.getSource() != nextNpuDmaOp.getSource() ||
          npuDmaOp.getTarget() != nextNpuDmaOp.getTarget()) {
        return failure();
      }
      nextStridedOp =
          cast<AMDAIE::DoublyStridedOpInterface>(nextNpuDmaOp.getOperation());
      if (!nextStridedOp) return failure();

      std::optional<uint8_t> sourceMemspaceInt =
          nextStridedOp.getSourceMemorySpaceAsUInt();
      std::optional<uint8_t> targetMemspaceInt =
          nextStridedOp.getTargetMemorySpaceAsUInt();
      if (!sourceMemspaceInt || !targetMemspaceInt) {
        return rewriter.notifyMatchFailure(
            nextStridedOp, "expected a source and target memory space");
      }
      sourceDmaDimConfig = std::make_unique<DmaDimConfig>(
          deviceModel, sourceMemspaceInt.value());
      targetDmaDimConfig = std::make_unique<DmaDimConfig>(
          deviceModel, targetMemspaceInt.value());
    } else if (auto npuCircularDmaOp =
                   dyn_cast<AMDAIE::NpuCircularDmaCpyNdOp>(op.getOperation())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "npuCircularDmaOp: " << npuCircularDmaOp << "\n");
      FailureOr<AMDAIE::NpuCircularDmaCpyNdOp> maybeNextNpuCircDmaOp =
          findNextDmaOpWithSameConnection(npuCircularDmaOp, block);
      if (failed(maybeNextNpuCircDmaOp)) return failure();
      nextStridedOp = cast<AMDAIE::DoublyStridedOpInterface>(
          maybeNextNpuCircDmaOp->getOperation());
      if (!nextStridedOp) return failure();

      std::optional<uint8_t> sourceMemspaceInt =
          nextStridedOp.getSourceMemorySpaceAsUInt();
      std::optional<uint8_t> targetMemspaceInt =
          nextStridedOp.getTargetMemorySpaceAsUInt();
      if (!sourceMemspaceInt || !targetMemspaceInt) {
        return rewriter.notifyMatchFailure(
            nextStridedOp, "expected a source and target memory space");
      }
      sourceDmaDimConfig = std::make_unique<CircularDmaDimConfig>(
          deviceModel, sourceMemspaceInt.value());
      targetDmaDimConfig = std::make_unique<CircularDmaDimConfig>(
          deviceModel, targetMemspaceInt.value());
    } else {
      return failure();
    }

    MLIRContext *ctx = rewriter.getContext();
    auto dimCountCheck = std::bind(&DmaDimConfig::exceedsNbDims,
                                   std::ref(sourceDmaDimConfig), _1);

    SmallVector<OpFoldResult> sourceOffsetsA = op.getSourceMixedOffsets();
    SmallVector<OpFoldResult> sourceSizesA = op.getSourceMixedSizes();
    SmallVector<OpFoldResult> sourceStridesA = op.getSourceMixedStrides();
    SmallVector<OpFoldResult> sourceOffsetsB =
        nextStridedOp.getSourceMixedOffsets();
    SmallVector<OpFoldResult> sourceSizesB =
        nextStridedOp.getSourceMixedSizes();
    SmallVector<OpFoldResult> sourceStridesB =
        nextStridedOp.getSourceMixedStrides();
    SmallVector<OpFoldResult> newSourceOffsets;
    SmallVector<OpFoldResult> newSourceSizes;
    SmallVector<OpFoldResult> newSourceStrides;
    if (failed(combineAccessPatterns(
            ctx, sourceOffsetsA, sourceSizesA, sourceStridesA, sourceOffsetsB,
            sourceSizesB, sourceStridesB, newSourceOffsets, newSourceSizes,
            newSourceStrides, dimCountCheck))) {
      return failure();
    }

    SmallVector<OpFoldResult> targetOffsetsA = op.getTargetMixedOffsets();
    SmallVector<OpFoldResult> targetSizesA = op.getTargetMixedSizes();
    SmallVector<OpFoldResult> targetStridesA = op.getTargetMixedStrides();
    SmallVector<OpFoldResult> targetOffsetsB =
        nextStridedOp.getTargetMixedOffsets();
    SmallVector<OpFoldResult> targetSizesB =
        nextStridedOp.getTargetMixedSizes();
    SmallVector<OpFoldResult> targetStridesB =
        nextStridedOp.getTargetMixedStrides();
    SmallVector<OpFoldResult> newTargetOffsets;
    SmallVector<OpFoldResult> newTargetSizes;
    SmallVector<OpFoldResult> newTargetStrides;
    if (failed(combineAccessPatterns(
            ctx, targetOffsetsA, targetSizesA, targetStridesA, targetOffsetsB,
            targetSizesB, targetStridesB, newTargetOffsets, newTargetSizes,
            newTargetStrides, dimCountCheck))) {
      return failure();
    }

    rewriter.setInsertionPoint(op);
    FailureOr<AMDAIE::DoublyStridedOpInterface> maybeNewDoublyStridedOp =
        createCombinedDoublyStridedOp(op, nextStridedOp, rewriter,
                                      newTargetOffsets, newTargetSizes,
                                      newTargetStrides, newSourceOffsets,
                                      newSourceSizes, newSourceStrides);
    if (failed(maybeNewDoublyStridedOp)) return failure();
    auto newDoublyStridedOp = *maybeNewDoublyStridedOp;
    rewriter.replaceOp(nextStridedOp, newDoublyStridedOp.getOperation());

    for (Operation *userOp : userOpsToBeErased) rewriter.eraseOp(userOp);
    rewriter.eraseOp(op);
    return success();
  }

  static FailureOr<AMDAIE::DoublyStridedOpInterface>
  createCombinedDoublyStridedOp(AMDAIE::DoublyStridedOpInterface op,
                                AMDAIE::DoublyStridedOpInterface nextStridedOp,
                                PatternRewriter &rewriter,
                                ArrayRef<OpFoldResult> newTargetOffsets,
                                ArrayRef<OpFoldResult> newTargetSizes,
                                ArrayRef<OpFoldResult> newTargetStrides,
                                ArrayRef<OpFoldResult> newSourceOffsets,
                                ArrayRef<OpFoldResult> newSourceSizes,
                                ArrayRef<OpFoldResult> newSourceStrides) {
    // Build the combined npu.dma_cpy_nd op from the first DMA's operands so BD
    // ids and FIFO operands dominate the insertion point. Result-type handling
    // is asymmetric because the combined op is created at the first op's
    // position but takes the second op's place via `replaceOp(nextStridedOp,
    // newOp)`, which requires equal result counts:
    //   - (empty op, token next): combine; the combined op inherits the
    //     second's token, and the trailing wait redirects to it.
    //   - (token op, empty next): bail -- `replaceOp` would mismatch (1 vs 0).
    //   - (token, token same):    combine; keep the shared token type.
    //   - (token, token diff):    bail; no token-union exists.
    if (auto npuDmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(op.getOperation())) {
      SmallVector<Type> resultTypes(op->getResultTypes());
      if (resultTypes.empty()) {
        resultTypes.assign(nextStridedOp->getResultTypes().begin(),
                           nextStridedOp->getResultTypes().end());
      } else if (nextStridedOp->getResultTypes().empty()) {
        // (token op, empty next): the new op would have op's token type but
        // `rewriter.replaceOp(nextStridedOp, newOp)` below requires equal
        // result counts. Preserving op's wait by replacing op's users instead
        // would require a different rewrite strategy; bail.
        return failure();
      } else if (!llvm::equal(resultTypes, nextStridedOp->getResultTypes())) {
        return failure();
      }
      auto newOp = AMDAIE::NpuDmaCpyNdOp::create(
          rewriter, npuDmaOp.getLoc(), TypeRange(resultTypes),
          npuDmaOp.getConnection(), npuDmaOp.getTarget(), newTargetOffsets,
          newTargetSizes, newTargetStrides, npuDmaOp.getTargetBdId(),
          npuDmaOp.getSource(), newSourceOffsets, newSourceSizes,
          newSourceStrides, npuDmaOp.getSourceBdId());
      return cast<AMDAIE::DoublyStridedOpInterface>(newOp.getOperation());
    }

    // Fallback path (e.g. NpuCircularDmaCpyNdOp): the `createDoublyStridedOp`
    // interface method's signature takes mutable `SmallVector<OpFoldResult>&`,
    // so materialize local copies to forward through it.
    SmallVector<OpFoldResult> tgtOffs(newTargetOffsets);
    SmallVector<OpFoldResult> tgtSizes(newTargetSizes);
    SmallVector<OpFoldResult> tgtStrides(newTargetStrides);
    SmallVector<OpFoldResult> srcOffs(newSourceOffsets);
    SmallVector<OpFoldResult> srcSizes(newSourceSizes);
    SmallVector<OpFoldResult> srcStrides(newSourceStrides);
    return op.createDoublyStridedOp(rewriter, tgtOffs, tgtSizes, tgtStrides,
                                    srcOffs, srcSizes, srcStrides);
  }

  template <typename T>
  static FailureOr<T> findNextDmaOpWithSameConnection(T dmaOp, Block *block) {
    // Walk the block (pre-order, recursing into nested regions) looking for
    // the next op of type `T` on the same connection. Policy:
    //   - `amdaie.npu.barrier`: global ordering boundary, bail.
    //   - Same-connection DMA actor of a *different* kind (a
    //     `circular_dma_cpy_nd` between two `dma_cpy_nd`s on the same
    //     connection, or vice versa): bail. The combined op would be hoisted
    //     past it and reorder the controlcode stream.
    //   - Type-T same-connection candidate found in a different block
    //     (nested region): bail; hoisting past nested control flow would
    //     reorder. Empty or unrelated nested regions flow through. This
    //     matters when `SubsumeLoopIntoDMA` leaves an empty loop shell
    //     after hoisting DMAs out -- the combiner needs to keep merging
    //     adjacent hoisted DMAs across that empty loop.
    //   - Same-block same-connection type-T candidate: return.
    Value connection = dmaOp.getConnection();
    T nextDmaOp;
    Block::iterator begin = std::next(dmaOp->getIterator());
    block->walk(begin, block->end(), [&](Operation *other) {
      if (isa<AMDAIE::NpuBarrierOp>(other)) return WalkResult::interrupt();
      auto candidate = dyn_cast<T>(other);
      if (!candidate) {
        // Same-connection DMA actor of a different kind blocks.
        if (auto npuDmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(other)) {
          if (npuDmaOp.getConnection() == connection)
            return WalkResult::interrupt();
        }
        if (auto npuCircDmaOp =
                dyn_cast<AMDAIE::NpuCircularDmaCpyNdOp>(other)) {
          if (npuCircDmaOp.getConnection() == connection)
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (candidate.getConnection() != connection) return WalkResult::advance();
      Block *otherBlock = candidate->getBlock();
      if (!otherBlock) return WalkResult::advance();
      if (otherBlock != block) return WalkResult::interrupt();
      nextDmaOp = candidate;
      return WalkResult::interrupt();
    });
    if (nextDmaOp) return nextDmaOp;
    return failure();
  }
};

class AMDAIECombineStridedOpsPass
    : public impl::AMDAIECombineStridedOpsBase<AMDAIECombineStridedOpsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIECombineStridedOpsPass() = default;
  AMDAIECombineStridedOpsPass(const AMDAIECombineStridedOpsPass &pass){};
  void runOnOperation() override;
};

void AMDAIECombineStridedOpsPass::runOnOperation() {
  Operation *parentOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  populateStridedOpCombinationPattern(patterns);
  if (failed(applyPatternsGreedily(parentOp, std::move(patterns)))) {
    parentOp->emitOpError("failed to combine strided operations");
    return signalPassFailure();
  }
}

}  // namespace

void populateStridedOpCombinationPattern(RewritePatternSet &patterns) {
  patterns.insert<CombineStridedOps>(patterns.getContext());
}

std::unique_ptr<Pass> createAMDAIECombineStridedOpsPass() {
  return std::make_unique<AMDAIECombineStridedOpsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
