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
#include "llvm/ADT/STLExtras.h"
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
      // Fail if any non-wait user operations.
      for (Operation *userOp : npuDmaOp->getUsers()) {
        if (isa<AMDAIE::NpuDmaWaitOp>(userOp)) {
          userOpsToBeErased.push_back(userOp);
        } else {
          return failure();
        }
      }
      FailureOr<AMDAIE::NpuDmaCpyNdOp> maybeNextNpuDmaOp =
          findNextDmaOpWithSameConnection(npuDmaOp, block);
      if (failed(maybeNextNpuDmaOp)) return failure();
      nextStridedOp = cast<AMDAIE::DoublyStridedOpInterface>(
          maybeNextNpuDmaOp->getOperation());
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

    SmallVector<OpFoldResult> sourceOffsetsA = op.getSourceMixedOffsets();
    SmallVector<OpFoldResult> sourceSizesA = op.getSourceMixedSizes();
    SmallVector<OpFoldResult> sourceStridesA = op.getSourceMixedStrides();
    SmallVector<OpFoldResult> sourceOffsetsB =
        nextStridedOp.getSourceMixedOffsets();
    SmallVector<OpFoldResult> sourceSizesB =
        nextStridedOp.getSourceMixedSizes();
    SmallVector<OpFoldResult> sourceStridesB =
        nextStridedOp.getSourceMixedStrides();
    bool areSourcesCombinable = areAccessPatternsCombinable(
        sourceOffsetsA, sourceSizesA, sourceStridesA, sourceOffsetsB,
        sourceSizesB, sourceStridesB,
        std::bind(&DmaDimConfig::exceedsNbDims, std::ref(sourceDmaDimConfig),
                  _1));

    SmallVector<OpFoldResult> targetOffsetsA = op.getTargetMixedOffsets();
    SmallVector<OpFoldResult> targetSizesA = op.getTargetMixedSizes();
    SmallVector<OpFoldResult> targetStridesA = op.getTargetMixedStrides();
    SmallVector<OpFoldResult> targetOffsetsB =
        nextStridedOp.getTargetMixedOffsets();
    SmallVector<OpFoldResult> targetSizesB =
        nextStridedOp.getTargetMixedSizes();
    SmallVector<OpFoldResult> targetStridesB =
        nextStridedOp.getTargetMixedStrides();
    bool areTargetsCombinable = areAccessPatternsCombinable(
        targetOffsetsA, targetSizesA, targetStridesA, targetOffsetsB,
        targetSizesB, targetStridesB,
        std::bind(&DmaDimConfig::exceedsNbDims, std::ref(targetDmaDimConfig),
                  _1));

    LLVM_DEBUG(llvm::dbgs()
               << "areSourcesCombinable: " << areSourcesCombinable << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "areTargetsCombinable: " << areTargetsCombinable << "\n");

    if (areSourcesCombinable && areTargetsCombinable) {
      SmallVector<OpFoldResult> newSourceOffsets;
      SmallVector<OpFoldResult> newSourceSizes;
      SmallVector<OpFoldResult> newSourceStrides;
      if (failed(combineAccessPatterns(
              rewriter, sourceOffsetsA, sourceSizesA, sourceStridesA,
              sourceOffsetsB, sourceSizesB, sourceStridesB, newSourceOffsets,
              newSourceSizes, newSourceStrides,
              std::bind(&DmaDimConfig::exceedsNbDims,
                        std::ref(sourceDmaDimConfig), _1)))) {
        return failure();
      }

      SmallVector<OpFoldResult> newTargetOffsets;
      SmallVector<OpFoldResult> newTargetSizes;
      SmallVector<OpFoldResult> newTargetStrides;
      if (failed(combineAccessPatterns(
              rewriter, targetOffsetsA, targetSizesA, targetStridesA,
              targetOffsetsB, targetSizesB, targetStridesB, newTargetOffsets,
              newTargetSizes, newTargetStrides,
              std::bind(&DmaDimConfig::exceedsNbDims,
                        std::ref(targetDmaDimConfig), _1)))) {
        return failure();
      }

      rewriter.setInsertionPoint(op);
      auto newDoublyStridedOp = nextStridedOp.createDoublyStridedOp(
          rewriter, newTargetOffsets, newTargetSizes, newTargetStrides,
          newSourceOffsets, newSourceSizes, newSourceStrides);
      rewriter.replaceOp(nextStridedOp, newDoublyStridedOp.getOperation());

      for (Operation *userOp : userOpsToBeErased) rewriter.eraseOp(userOp);
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }

  template <typename T>
  static FailureOr<T> findNextDmaOpWithSameConnection(T dmaOp, Block *block) {
    T nextDmaOp;
    Block::iterator begin = std::next(dmaOp->getIterator());
    block->walk(begin, block->end(), [&](T other) {
      if (dmaOp.getConnection() != other.getConnection())
        return WalkResult::advance();
      Block *otherBlock = other->getBlock();
      if (!otherBlock) return WalkResult::advance();
      if (otherBlock != block) return WalkResult::interrupt();
      nextDmaOp = other;
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
