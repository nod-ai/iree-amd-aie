// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-transfer-strided-access-pattern"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Utility to copy a vector except for certain dim positions.
static SmallVector<OpFoldResult> copyExcludeDims(
    SmallVector<OpFoldResult> origVals, DenseSet<size_t> excludeDims) {
  if (excludeDims.size() == 0) return origVals;
  SmallVector<OpFoldResult> results;
  for (size_t i = 0; i < origVals.size(); i++) {
    if (!excludeDims.contains(i)) {
      results.push_back(origVals[i]);
    }
  }
  return results;
};

/// Utility to check if any dimension from the L3 dma addressing can be combined
/// with the innermost dimension, if so return the position of the dimension.
/// Two dimensions (i and innermost) can be combined if the following conditions
/// are satisfied: 1) stride[i] = innermost_stride * innermost_size;
/// 2) offset[i] = 0.
static FailureOr<size_t> isL3AddressingCombinable(
    SmallVector<OpFoldResult> &dmaOffsets, SmallVector<OpFoldResult> &dmaSizes,
    SmallVector<OpFoldResult> &dmaStrides) {
  // Offsets could be dynamic but sizes and strides should be static.
  std::optional<SmallVector<int64_t>> maybeSizes =
      getConstantIntValues(dmaSizes);
  std::optional<SmallVector<int64_t>> maybeStrides =
      getConstantIntValues(dmaStrides);
  if (!maybeSizes.has_value() || !maybeSizes.has_value()) {
    return failure();
  }
  SmallVector<int64_t> sizeVals = maybeSizes.value();
  SmallVector<int64_t> strideVals = maybeStrides.value();

  // Get the index of the dim that can be potentially combined with the
  // innermost dim. If there is no such dim, return the last index of the
  // vector.
  auto getPos = [&](SmallVector<int64_t> values, int64_t target) {
    size_t i = 0;
    for (; i < values.size() - 1; i++) {
      if (values[i] == target) return i;
    }
    return i;
  };

  int64_t innerDimTotal = strideVals.back() * sizeVals.back();
  size_t dimForCombine = getPos(strideVals, innerDimTotal);
  if (dimForCombine >= (dmaSizes.size() - 1)) return failure();

  std::optional<int64_t> offsetAtPos =
      getConstantIntValue(dmaOffsets[dimForCombine]);
  if (!offsetAtPos.has_value() || offsetAtPos.value() != 0) return failure();
  return dimForCombine;
}

/// Utility to check if L2 dma addressing is linear. Note here the assumption is
/// the dma ops are already canonicalized, so that the L2 addressing should be
/// empty or 1-d vectors.
static bool isL2AddressingLinear(SmallVector<OpFoldResult> &dmaOffsets,
                                 SmallVector<OpFoldResult> &dmaSizes,
                                 SmallVector<OpFoldResult> &dmaStrides) {
  assert(dmaOffsets.size() == dmaSizes.size() &&
         dmaOffsets.size() == dmaStrides.size() &&
         "expected same number of source offsets and sizes");
  if (dmaOffsets.size() == 0) return true;
  if (dmaOffsets.size() != 1) return false;
  if (!isConstantIntValue(dmaOffsets[0], 0)) return false;
  if (!isConstantIntValue(dmaStrides[0], 1)) return false;
  return true;
}

/// Utility to check if all users of the connection op statisfy the conditions
/// for dma access pattern transfer.
static FailureOr<bool> checkConnectionUsers(AMDAIE::ConnectionOp connectionOp) {
  for (Operation *user : connectionOp->getUsers()) {
    // Check if L3 addressing is combinable.
    if (auto dmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(user)) {
      if (dmaOp.hasSourceAddressing() && dmaOp.hasTargetAddressing()) {
        dmaOp.emitOpError()
            << "should not have both source and target addressing";
        return failure();
      }
      if (!dmaOp.hasSourceAddressing() && !dmaOp.hasTargetAddressing()) {
        dmaOp.emitOpError() << "should have either source or target addressing";
        return failure();
      }

      SmallVector<OpFoldResult> dmaOffsets;
      SmallVector<OpFoldResult> dmaSizes;
      SmallVector<OpFoldResult> dmaStrides;
      if (dmaOp.hasSourceAddressing()) {
        dmaOffsets = dmaOp.getSourceMixedOffsets();
        dmaSizes = dmaOp.getSourceMixedSizes();
        dmaStrides = dmaOp.getSourceMixedStrides();
      } else {
        dmaOffsets = dmaOp.getTargetMixedOffsets();
        dmaSizes = dmaOp.getTargetMixedSizes();
        dmaStrides = dmaOp.getTargetMixedStrides();
      }

      if (failed(isL3AddressingCombinable(dmaOffsets, dmaSizes, dmaStrides))) {
        return false;
      }
    }
    // Check if L2 addressing is linear.
    if (auto circularDma = dyn_cast<AMDAIE::NpuCircularDmaCpyNdOp>(user)) {
      // Circular dma op could have both source and target addressing empty.
      if (circularDma.hasSourceAddressing() &&
          circularDma.hasTargetAddressing()) {
        circularDma.emitOpError()
            << "should not have both source and target addressing";
        return failure();
      }

      SmallVector<OpFoldResult> circularOffsets;
      SmallVector<OpFoldResult> circularSizes;
      SmallVector<OpFoldResult> circularStrides;

      if (circularDma.hasSourceAddressing()) {
        circularOffsets = circularDma.getSourceMixedOffsets();
        circularSizes = circularDma.getSourceMixedSizes();
        circularStrides = circularDma.getSourceMixedStrides();
      }
      if (circularDma.hasTargetAddressing()) {
        circularOffsets = circularDma.getTargetMixedOffsets();
        circularSizes = circularDma.getTargetMixedSizes();
        circularStrides = circularDma.getTargetMixedStrides();
      }
      if (!isL2AddressingLinear(circularOffsets, circularSizes,
                                circularStrides)) {
        return false;
      }
    }
  }
  return true;
}

/// Utility to change the addressing of NpuDmaCpyNdOp and NpuCircularDmaCpyNdOp
/// in place. If the source of NpuDmaCpyNdOp is in L3, then the source
/// addressing from NpuDmaCpyNdOp and target addressing from
/// NpuCircularDmaCpyNdOp need to be changed. The other way around.
static LogicalResult createNewAddressing(
    MLIRContext *ctx, SmallVector<OpFoldResult> &dmaOffsets,
    SmallVector<OpFoldResult> &dmaSizes, SmallVector<OpFoldResult> &dmaStrides,
    SmallVector<OpFoldResult> &circularDmaOffsets,
    SmallVector<OpFoldResult> &circularDmaSizes,
    SmallVector<OpFoldResult> &circularDmaStrides) {
  IRRewriter rewriter(ctx);

  // Make copies of L3 original sizes and strides which will be needed later
  // when creating new L2 addressing.
  SmallVector<OpFoldResult> l3OrigSizes = dmaSizes;
  SmallVector<OpFoldResult> l3OrigStrides = dmaStrides;

  FailureOr<size_t> isCombinable =
      isL3AddressingCombinable(dmaOffsets, dmaSizes, dmaStrides);
  if (failed(isCombinable)) {
    return emitError(rewriter.getUnknownLoc())
           << "failed to get dim position for combination";
  }
  size_t dimForCombine = isCombinable.value();

  // Generate L3 side new source offsets/sizes/strides.
  // Example: [[0, 0, 0] [2, 32, 32] [32, 128, 1]] will become
  // [[0, 0] [32, 64] [128, 1]] after the first and the innermost dims are
  // combined.
  DenseSet<size_t> excludeDims = {dimForCombine};
  dmaOffsets = copyExcludeDims(dmaOffsets, excludeDims);
  dmaStrides = copyExcludeDims(dmaStrides, excludeDims);

  std::optional<SmallVector<int64_t>> maybeSizes =
      getConstantIntValues(l3OrigSizes);
  std::optional<SmallVector<int64_t>> maybeStrides =
      getConstantIntValues(l3OrigStrides);
  if (!maybeSizes.has_value() || !maybeSizes.has_value()) {
    return emitError(rewriter.getUnknownLoc())
           << "failed to get original source sizes / strides.";
  }
  SmallVector<int64_t> sizeVals = maybeSizes.value();
  SmallVector<int64_t> strideVals = maybeStrides.value();

  int64_t innerDimTotal = strideVals.back() * sizeVals.back();
  int64_t newInnerSize = sizeVals[dimForCombine] * innerDimTotal;

  size_t lastIndex = l3OrigSizes.size() - 1;
  excludeDims.insert(lastIndex);
  dmaSizes = copyExcludeDims(dmaSizes, excludeDims);
  dmaSizes.push_back(getAsIndexOpFoldResult(ctx, newInnerSize));

  // Generate L2 side new target offsets/sizes/strides.
  SmallVector<OpFoldResult> newCircularOffsets(l3OrigSizes.size(),
                                               rewriter.getIndexAttr(0));
  circularDmaOffsets = newCircularOffsets;

  circularDmaSizes = copyExcludeDims(l3OrigSizes, excludeDims);
  circularDmaSizes.push_back(
      getAsIndexOpFoldResult(ctx, sizeVals[dimForCombine]));
  circularDmaSizes.push_back(getAsIndexOpFoldResult(ctx, innerDimTotal));

  // Function to create new strides for NpuCircularDmaCpyNdOp.
  auto getNewL2Strides = [&](SmallVector<int64_t> values) {
    SmallVector<OpFoldResult> res = {getAsIndexOpFoldResult(ctx, 1)};
    int64_t initial = values.back();
    // Leave out one dimension for insertion afterwards
    for (size_t i = values.size() - 2; i > 0; i--) {
      initial *= values[i];
      res.push_back(getAsIndexOpFoldResult(ctx, initial));
    }
    return llvm::to_vector(llvm::reverse(res));
  };

  circularDmaStrides = getNewL2Strides(sizeVals);
  circularDmaStrides.insert(
      circularDmaStrides.begin() + dimForCombine,
      getAsIndexOpFoldResult(ctx, strideVals[dimForCombine]));
  return success();
}

/// Walk through all users of a connection op and change the dma addressing of
/// NpuDmaCpyNdOp and NpuCircularDmaCpyNdOp at the same time. A connection op
/// can have multiple NpuDmaCpyNdOp users (with different offsets) but only one
/// NpuCircularDmaCpyNdOp user.
static LogicalResult transferDmaAddressing(MLIRContext *ctx,
                                           AMDAIE::ConnectionOp connectionOp) {
  IRRewriter rewriter(ctx);
  OpBuilder::InsertionGuard guard(rewriter);

  FailureOr<AMDAIE::NpuCircularDmaCpyNdOp> maybeNpuDmaUserOp =
      connectionOp.getNpuCircularDmaCpyNdUser();
  if (failed(maybeNpuDmaUserOp)) {
    connectionOp.emitOpError() << "failed to get circular NPU DMA op user";
    return failure();
  }

  AMDAIE::NpuCircularDmaCpyNdOp circularDma = maybeNpuDmaUserOp.value();
  SmallVector<OpFoldResult> srcCircularOffsets =
      circularDma.getSourceMixedOffsets();
  SmallVector<OpFoldResult> srcCircularSizes =
      circularDma.getSourceMixedSizes();
  SmallVector<OpFoldResult> srcCircularStrides =
      circularDma.getSourceMixedStrides();
  SmallVector<OpFoldResult> tgtCircularOffsets =
      circularDma.getTargetMixedOffsets();
  SmallVector<OpFoldResult> tgtCircularSizes =
      circularDma.getTargetMixedSizes();
  SmallVector<OpFoldResult> tgtCircularStrides =
      circularDma.getTargetMixedStrides();

  // Change the source/target addressing of all users from a connection op.
  llvm::SmallVector<Operation *> users(connectionOp->getUsers());
  for (Operation *user : users) {
    if (auto dmaOp = dyn_cast<AMDAIE::NpuDmaCpyNdOp>(user)) {
      SmallVector<OpFoldResult> srcOffsets = dmaOp.getSourceMixedOffsets();
      SmallVector<OpFoldResult> srcSizes = dmaOp.getSourceMixedSizes();
      SmallVector<OpFoldResult> srcStrides = dmaOp.getSourceMixedStrides();
      SmallVector<OpFoldResult> tgtOffsets = dmaOp.getTargetMixedOffsets();
      SmallVector<OpFoldResult> tgtSizes = dmaOp.getTargetMixedSizes();
      SmallVector<OpFoldResult> tgtStrides = dmaOp.getTargetMixedStrides();

      // Generate new L3 source addressing, and L2 target addressing.
      if (dmaOp.getSourceMemorySpaceAsUInt() == 0) {
        if (circularDma.getTargetMemorySpaceAsUInt() != 1) {
          dmaOp.emitOpError() << "has source in L3, but circular dma doesn't "
                                 "have target in L2.";
          return failure();
        }
        if (failed(createNewAddressing(ctx, srcOffsets, srcSizes, srcStrides,
                                       tgtCircularOffsets, tgtCircularSizes,
                                       tgtCircularStrides))) {
          return failure();
        }
      }

      // Generate new L3 target addressing, and L2 source addressing.
      if (dmaOp.getTargetMemorySpaceAsUInt() == 0) {
        if (circularDma.getSourceMemorySpaceAsUInt() != 1) {
          dmaOp.emitOpError() << "has target in L3, but circular dma doesn't "
                                 "have source in L2.";
          return failure();
        }
        if (failed(createNewAddressing(ctx, tgtOffsets, tgtSizes, tgtStrides,
                                       srcCircularOffsets, srcCircularSizes,
                                       srcCircularStrides))) {
          return failure();
        }
      }

      // Replace the npu.dma_cpy_nd with the combined access pattern.
      rewriter.setInsertionPoint(dmaOp);
      dmaOp = rewriter.replaceOpWithNewOp<AMDAIE::NpuDmaCpyNdOp>(
          dmaOp, dmaOp.getConnection(), dmaOp.getTarget(), tgtOffsets, tgtSizes,
          tgtStrides, dmaOp.getTargetBdId(), dmaOp.getSource(), srcOffsets,
          srcSizes, srcStrides, dmaOp.getSourceBdId());
    }
  }

  // Replace the npu.circular_dma_cpy_nd with the new access pattern.
  rewriter.setInsertionPoint(circularDma);
  circularDma = rewriter.replaceOpWithNewOp<AMDAIE::NpuCircularDmaCpyNdOp>(
      circularDma, circularDma.getConnection(), tgtCircularOffsets,
      tgtCircularSizes, tgtCircularStrides, srcCircularOffsets,
      srcCircularSizes, srcCircularStrides);
  return success();
}

class AMDAIETransferStridedAccessPatternPass
    : public impl::AMDAIETransferStridedAccessPatternBase<
          AMDAIETransferStridedAccessPatternPass> {
 public:
  AMDAIETransferStridedAccessPatternPass() = default;
  AMDAIETransferStridedAccessPatternPass(
      const AMDAIETransferStridedAccessPatternPass &pass){};
  void runOnOperation() override;
};

void AMDAIETransferStridedAccessPatternPass::runOnOperation() {
  Operation *parentOp = getOperation();
  MLIRContext *ctx = &getContext();

  // Walk the NpuDmaCpyNdOp ops and get the defining connections between L3 and
  // L2 objectFifos. Then go through all users of each connection op and check
  // if there is optimization opportunity to transfer strided access pattern
  // from L3 to L2 side. Currently, a connection op can have multiple
  // NpuDmaCpyNdOp users but only one NpuCircularDmaCpyNdOp user.
  DenseSet<AMDAIE::ConnectionOp> connectionOps;
  WalkResult walkRes = parentOp->walk([&](NpuDmaCpyNdOp dmaOp) {
    AMDAIE::ConnectionOp connectionOp = dmaOp.getConnectionOp();
    if (!connectionOp) {
      dmaOp.emitOpError() << "no connection op is found";
      return WalkResult::interrupt();
    }
    if (connectionOps.contains(connectionOp)) {
      return WalkResult::advance();
    }

    FailureOr<bool> checkRes = checkConnectionUsers(connectionOp);
    if (failed(checkRes)) {
      return WalkResult::interrupt();
    }
    if (checkRes.value()) {
      connectionOps.insert(connectionOp);
    }
    return WalkResult::advance();
  });
  if (walkRes.wasInterrupted()) return signalPassFailure();

  // Walk through all users of each connection op and change the dma addressing
  // from NpuDmaCpyNdOp and NpuCircularDmaCpyNdOp at the same time.
  for (AMDAIE::ConnectionOp connectionOp : connectionOps) {
    if (failed(transferDmaAddressing(ctx, connectionOp))) {
      return signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIETransferStridedAccessPatternPass() {
  return std::make_unique<AMDAIETransferStridedAccessPatternPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
