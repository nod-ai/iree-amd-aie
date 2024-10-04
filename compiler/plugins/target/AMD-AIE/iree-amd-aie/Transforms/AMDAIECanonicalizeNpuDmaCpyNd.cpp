// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#define DEBUG_TYPE "iree-amdaie-canonicalize-npu-dma-cpy-nd"

namespace mlir::iree_compiler::AMDAIE {

class AMDAIECanonicalizeNpuDmaCpyNdPass
    : public impl::AMDAIECanonicalizeNpuDmaCpyNdBase<
          AMDAIECanonicalizeNpuDmaCpyNdPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(context);
    Attribute zero = rewriter.getIndexAttr(0);
    Attribute one = rewriter.getIndexAttr(1);

    WalkResult walkResult = moduleOp->walk([&](NpuDmaCpyNdOp dmaOp) {
      SmallVector<OpFoldResult> srcOffsets = dmaOp.getSourceMixedOffsets();
      SmallVector<OpFoldResult> srcSizes = dmaOp.getSourceMixedSizes();
      SmallVector<OpFoldResult> srcStrides = dmaOp.getSourceMixedStrides();

      SmallVector<OpFoldResult> tgtOffsets = dmaOp.getTargetMixedOffsets();
      SmallVector<OpFoldResult> tgtSizes = dmaOp.getTargetMixedSizes();
      SmallVector<OpFoldResult> tgtStrides = dmaOp.getTargetMixedStrides();

      // The first step in canonicalization is padding the offsets/sizes/strides
      // vectors to be of rank `nbDimensions`. If the rank of any of these
      // vectors is greater than `nbDimensions`, then this is impossible.
      bool allValidRanks = srcOffsets.size() <= nbDimensions &&
                           srcSizes.size() <= nbDimensions &&
                           srcStrides.size() <= nbDimensions &&
                           tgtOffsets.size() <= nbDimensions &&
                           tgtSizes.size() <= nbDimensions &&
                           tgtStrides.size() <= nbDimensions;
      if (!allValidRanks) {
        dmaOp.emitOpError()
            << " has offsets/sizes/strides attributes that are "
               "larger than the target canonicalization dimension of "
            << nbDimensions << ".";
        return WalkResult::interrupt();
      }

      // If the source is in L3, then canonicalize the source addressing.
      // 1) Pad to the correct rank
      // 2) Move the zero stride (if any) to the outer-most (slowest) dim.
      if (dmaOp.getSourceMemorySpaceAsUInt() == 0) {
        if (!dmaOp.hasSourceAddressing()) {
          dmaOp.emitOpError()
              << "has source in L3, but does not have source addressing. "
                 "Source addressing is required to canonicalize here.";
          return WalkResult::interrupt();
        }
        srcOffsets = getPrepended(srcOffsets, zero);
        srcSizes = getPrepended(srcSizes, one);
        srcStrides = getPrepended(srcStrides, zero);
        std::optional<uint32_t> maybeSwapIndex =
            verifyAndGetZeroStrideIndex(srcSizes, srcStrides, dmaOp);
        if (!maybeSwapIndex.has_value()) return WalkResult::interrupt();
        uint32_t swapIndex = maybeSwapIndex.value();
        bubble(srcOffsets, swapIndex);
        bubble(srcSizes, swapIndex);
        bubble(srcStrides, swapIndex);
      }

      if (dmaOp.getTargetMemorySpaceAsUInt() == 0) {
        if (!dmaOp.hasTargetAddressing()) {
          dmaOp.emitOpError()
              << "has target in L3, but does not have target addressing. "
                 "Target addressing is required to canonicalize here.";
          return WalkResult::interrupt();
        }
        tgtOffsets = getPrepended(tgtOffsets, zero);
        tgtSizes = getPrepended(tgtSizes, one);
        tgtStrides = getPrepended(tgtStrides, zero);
        std::optional<uint32_t> maybeSwapIndex =
            verifyAndGetZeroStrideIndex(tgtSizes, tgtStrides, dmaOp);
        if (!maybeSwapIndex.has_value()) return WalkResult::interrupt();
        uint32_t swapIndex = maybeSwapIndex.value();
        bubble(tgtOffsets, swapIndex);
        bubble(tgtSizes, swapIndex);
        bubble(tgtStrides, swapIndex);
      }

      rewriter.setInsertionPoint(dmaOp);

      // Replace the npu.dma_cpy_nd with the canonicalized version.
      dmaOp = rewriter.replaceOpWithNewOp<AMDAIE::NpuDmaCpyNdOp>(
          dmaOp, dmaOp.getResultTypes(), dmaOp.getConnection(),
          dmaOp.getTarget(), tgtOffsets, tgtSizes, tgtStrides,
          dmaOp.getTargetBdId(), dmaOp.getSource(), srcOffsets, srcSizes,
          srcStrides, dmaOp.getSourceBdId());

      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) return signalPassFailure();
  }

 private:
  /// Repeat prepend 'def' to 'tail' to make 'tail' have nbDimensions elements.
  SmallVector<OpFoldResult> getPrepended(ArrayRef<OpFoldResult> tail,
                                         Attribute def) {
    assert(tail.size() <= nbDimensions);
    SmallVector<OpFoldResult> res(nbDimensions, def);
    std::copy(tail.begin(), tail.end(),
              res.begin() + nbDimensions - tail.size());
    return res;
  }

  static size_t getLowestIndexMaybeAboveOne(ArrayRef<OpFoldResult> v) {
    for (size_t i = 0; i < v.size(); i++) {
      std::optional<int64_t> maybe = getConstantIntValue(v[i]);
      if (!maybe.has_value() || maybe.value() > 1) return i;
    }
    return v.size();
  }

  static size_t getHighestIndexMaybeZero(ArrayRef<OpFoldResult> v) {
    for (size_t i = v.size(); i > 0; i--) {
      std::optional<int64_t> maybe = getConstantIntValue(v[i - 1]);
      if (!maybe.has_value() || maybe.value() == 0) return i - 1;
    }
    return 0;
  }

  /// Get the highest index where the stride is 0. If this index is greater
  /// than the lowest index where the size is greater than 1, then fail.
  std::optional<uint32_t> verifyAndGetZeroStrideIndex(
      ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
      NpuDmaCpyNdOp dmaOp) {
    assert(strides.size() == sizes.size() && strides.size() == nbDimensions);

    size_t firstNonUnitDim = getLowestIndexMaybeAboveOne(sizes);
    size_t lastZeroStrideDim = getHighestIndexMaybeZero(strides);

    if (firstNonUnitDim < lastZeroStrideDim) {
      // HW limitation.
      dmaOp.emitOpError("might have stride=0 in dimension ")
          << lastZeroStrideDim << ", and size>1 in dimension "
          << firstNonUnitDim << ". As " << firstNonUnitDim << " < "
          << lastZeroStrideDim
          << ", this cannot be supported -- the zero stride cannot be moved "
             "to the outer-most (slowest) dimension, as required by current "
             "AIE architecture.";
      return {};
    }
    return lastZeroStrideDim;
  }

  /// Example, for swapIndex = 2.
  /// Input
  ///                 [0 1 7 13]
  /// is mutated to
  ///                 [7 0 1 13]
  static void bubble(MutableArrayRef<OpFoldResult> arr, size_t swapIndex) {
    if (swapIndex > 0) {
      std::rotate(arr.begin(), arr.begin() + swapIndex,
                  arr.begin() + swapIndex + 1);
    }
  }
};

std::unique_ptr<Pass> createAMDAIECanonicalizeNpuDmaCpyNdPass() {
  return std::make_unique<AMDAIECanonicalizeNpuDmaCpyNdPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
