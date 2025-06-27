// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIELogicalObjFifoSplittingUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-split-logical-objectfifos"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Utility struct to represent DMA split information.
struct DmaSplitInfo {
  size_t sourceSplitDim{0};
  int64_t newSourceStride{1};
  size_t targetSplitDim{0};
  int64_t newTargetStride{1};
  int64_t splitSize{1};
};

/// Utility struct to represent objectFifo split information.
struct ObjFifoSplitInfo {
  size_t splitDim{0};
  int64_t splitSize{1};
  int64_t splitStride{1};
  int64_t numUniqueConsumerDMAs{1};
};

using DmaObjFifoPairT =
    std::pair<AMDAIE::DmaCpyNdOp, AMDAIE::LogicalObjectFifoFromMemrefOp>;

/// Utility to derive the split stride to be used from a vector of DMA ops by
/// analyzing the offset scales. Will fail if the provided DMA ops don't have a
/// consistent offset scale.
template <CopyOpOperateOn OperateOn>
FailureOr<int64_t> getSplitStride(ArrayRef<AMDAIE::DmaCpyNdOp> dmaOps,
                                  int64_t sizeAfterSplit) {
  int64_t splitStride{-1};
  for (AMDAIE::DmaCpyNdOp dmaOp : dmaOps) {
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> strides;
    if constexpr (OperateOn == CopyOpOperateOn::Source) {
      offsets = dmaOp.getSourceMixedOffsets();
      strides = dmaOp.getSourceMixedStrides();
    } else if constexpr (OperateOn == CopyOpOperateOn::Target) {
      offsets = dmaOp.getTargetMixedOffsets();
      strides = dmaOp.getTargetMixedStrides();
    } else {
      assert(false && "Function can only operate on Source or Target");
    }
    SmallVector<size_t> splitIndices =
        getStrideIndicesWithDynamicOrNonZeroOffset(offsets, strides,
                                                   sizeAfterSplit);
    if (splitIndices.size() > 1)
      return dmaOp.emitError() << "multiple split indices found";
    int64_t step{-1};
    if (splitIndices.empty()) {
      step = 1;
    } else {
      // splitIndices.size() == 1
      size_t splitIdx = splitIndices[0];
      OpFoldResult offset = offsets[splitIdx];

      if (std::optional<int64_t> staticOffset = getConstantIntValue(offset);
          staticOffset.has_value()) {
        if (staticOffset.value() == 0) continue;
        step = 1;
      } else if (auto offsetValue = dyn_cast_if_present<Value>(offset)) {
        if (isa_and_present<affine::AffineApplyOp>(
                offsetValue.getDefiningOp())) {
          auto applyOp =
              cast<affine::AffineApplyOp>(offsetValue.getDefiningOp());
          if (applyOp.getNumOperands() != 1)
            return applyOp.emitError() << "mulptiple operands is not supported";
          AffineMap affineMap = applyOp.getAffineMap();
          RetrieveScaleAndBias retriever;
          if (failed(retriever.visit(affineMap.getResult(0)))) {
            return applyOp.emitError()
                   << "could not retrieve scale and bias from expression: "
                   << *applyOp.getOperation();
          }
          if (!retriever.scale.has_value()) {
            return applyOp.emitError()
                   << "expected a scale for: " << *applyOp.getOperation();
          }
          step = retriever.scale.value();
        } else if (auto blockArg = dyn_cast<BlockArgument>(offsetValue);
                   blockArg && isa<LoopLikeOpInterface>(
                                   blockArg.getOwner()->getParentOp())) {
          step = 1;
        } else {
          return dmaOp.emitOpError()
                 << "has an offset value that is neither an "
                    "induction variable nor an affine expression";
        }
      } else {
        return dmaOp.emitOpError()
               << "has an offset that is neither a constant nor an affine "
                  "expression, which is not supported";
      }
    }
    if (splitStride == -1) {
      splitStride = step;
    } else if (step != splitStride) {
      return dmaOp.emitOpError() << "has an offset step: " << step
                                 << ", which is different from "
                                    "previous offset steps: "
                                 << splitStride;
    }
  }
  // If all offsets are zero (or no split index found).
  if (splitStride == -1) return 1;
  return splitStride;
}

/// Given a list of Copy Ops, fetch the total no. of unique consumer/producer
/// LogicalObjectFifos. This would helps us figure out the split factor for
/// LogicalObjectFifos.
/// And example case which necessitated this feature :-
///      %lhs = LOF_on_L2
///      %a = LOF_on_L1_0
///      %b = LOF_on_L1_1
///      %c = LOF_on_L1_2
///      DMA(%a, %lhs)
///      DMA(%b, %lhs)
///      DMA(%c, %lhs)
///      DMA(%b, %lhs)
///      DMA(%c, %lhs)
///
///    In the above snippet, assume we want to split %lhs, it has 5 DMA ops.
///    But only 3 of them are unique : (%lhs -> %a), (%lhs -> %b) (%lhs -> %c).
///    Therefore this function is going to return 3. Which the caller is going
///    to use as split factor.
template <CopyOpOperateOn OperateOn>
static FailureOr<int64_t> fetchTotalUniqueLogicalObjFifoUsers(
    SmallVector<CopyOpInterface> copyLikeOps) {
  DenseSet<Operation *> uniqueLof;
  for (CopyOpInterface copyOp : copyLikeOps) {
    AMDAIE::LogicalObjectFifoFromMemrefOp lof = nullptr;
    if constexpr (OperateOn == CopyOpOperateOn::Target) {
      lof = dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          copyOp.getTarget().getDefiningOp());
    } else {
      lof = dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
          copyOp.getSource().getDefiningOp());
    }
    if (!lof) {
      return copyOp.emitOpError()
             << "could not retrieve source/target objectFifo";
    }
    uniqueLof.insert(lof);
  }
  return uniqueLof.size();
}

/// Find the logical objectFifo and DMA source/target splitting dimensions for
/// each DMA and objectFifo pair.
///
/// Each pair is handled in the following way:
/// First, compute the objectFifo splitting dimension based on the last non-unit
/// shape dimension and the number of available columns. Afterwards, depending
/// on which logical objectFifo is being split on, find the outermost dimension
/// in either the source or target access pattern that has:
/// - stride == sizeAfterSplit
/// - size != 1
/// This is the splitting dimension to be used on the respective side of the DMA
/// operation. Then, calculate the product size of that side of the DMA
/// operation after the splitting dimension and use it to calculate the
/// splitting dimension on the other side as the first dimension from the back
/// that has product size larger than the other side's product size after
/// splitting because that's the number of elements that should be
/// produced/consumed on the respective sides before splitting.
/// Towards the end fetch the number of unique producers (or consumers) for the
/// objectFifo which will be split. This would form the split factor which would
/// be capped by the total no. of columns OR std::gcd of source/target size.
LogicalResult collectSplittingDims(
    const SmallVector<DmaObjFifoPairT> &dmaObjFifoPairs,
    DenseMap<AMDAIE::DmaCpyNdOp, DmaSplitInfo> &dmaSplitInfoMap,
    DenseMap<AMDAIE::LogicalObjectFifoFromMemrefOp, ObjFifoSplitInfo>
        &objFifoSplitInfoMap,
    int64_t numCols) {
  for (auto [dmaOp, objFifo] : dmaObjFifoPairs) {
    LLVM_DEBUG(llvm::dbgs() << "dmaOp: " << dmaOp << "\n");
    LLVM_DEBUG(llvm::dbgs() << "objFifo: " << objFifo << "\n");
    ArrayRef<int64_t> memrefShape = objFifo.getMemrefType().getShape();
    if (llvm::any_of(memrefShape, [](int64_t size) {
          return ShapedType::isDynamic(size);
        })) {
      return objFifo.emitOpError()
             << "can't find a valid split dimension for dynamic sizes memref";
    }
    auto iter = std::find_if(memrefShape.begin(), memrefShape.end(),
                             [](int64_t size) { return size > 1; });
    size_t objFifoSplitDim = std::distance(memrefShape.begin(), iter);
    // If all dimensions are unit (1), no splitting can be done, so continue to
    // the next pair.
    if (objFifoSplitDim >= memrefShape.size()) continue;
    int64_t splitDimSize = memrefShape[objFifoSplitDim];
    int64_t sizeAfterSplit =
        std::accumulate(memrefShape.begin() + objFifoSplitDim + 1,
                        memrefShape.end(), 1, std::multiplies<>());

    // Get the producers and consumers of the current objectFifoOp.
    SmallVector<AMDAIE::DmaCpyNdOp> producers;
    SmallVector<AMDAIE::DmaCpyNdOp> consumers;
    if (failed(getDmaCpyNdOpProducersAndConsumers(objFifo, producers,
                                                  consumers))) {
      return failure();
    }

    size_t sourceSplitDim{0};
    size_t targetSplitDim{0};
    if (dmaOp.getTargetObjectFifo() == objFifo) {
      std::optional<SmallVector<int64_t>> targetSizes =
          getConstantIntValues(dmaOp.getTargetMixedSizes());
      std::optional<SmallVector<int64_t>> targetStrides =
          getConstantIntValues(dmaOp.getTargetMixedStrides());
      std::optional<SmallVector<int64_t>> sourceSizes =
          getConstantIntValues(dmaOp.getSourceMixedSizes());
      if (!targetSizes.has_value() || !targetStrides.has_value() ||
          !sourceSizes.has_value()) {
        return dmaOp.emitOpError() << "has unsupported dynamic target strides "
                                      "or sizes or source sizes";
      }
      for (auto iter : llvm::enumerate(
               llvm::zip(targetSizes.value(), targetStrides.value()))) {
        int64_t size = std::get<0>(iter.value());
        int64_t stride = std::get<1>(iter.value());
        if (stride == sizeAfterSplit && size != 1) {
          targetSplitDim = iter.index();
          break;
        }
      }
      int64_t targetSizeAfterSplit =
          std::accumulate(targetSizes.value().begin() + targetSplitDim + 1,
                          targetSizes.value().end(), 1, std::multiplies<>());
      SmallVector<int64_t> sourceProductSizes = sourceSizes.value();
      std::partial_sum(sourceProductSizes.rbegin(), sourceProductSizes.rend(),
                       sourceProductSizes.rbegin(), std::multiplies<int64_t>());
      for (int idx = sourceProductSizes.size() - 1; idx > 0; idx--) {
        if (sourceProductSizes[idx] > targetSizeAfterSplit) {
          sourceSplitDim = idx;
          break;
        }
      }
      FailureOr<int64_t> maybeSplitStride =
          getSplitStride<CopyOpOperateOn::Source>(consumers, sizeAfterSplit);
      if (failed(maybeSplitStride)) {
        objFifo.emitOpError()
            << "could not retrieve a split stride from the consumer DMA ops";
      }
      int64_t splitStride = maybeSplitStride.value();
      // Calculate the new source stride to be used for splitting the DMA.
      int64_t newSourceStride =
          splitStride != 1 ? splitDimSize / splitStride : 1;
      FailureOr<int64_t> maybeNumUniqueConsumers =
          fetchTotalUniqueLogicalObjFifoUsers<CopyOpOperateOn::Target>(
              objFifo.getCopyLikeConsumers());
      if (failed(maybeNumUniqueConsumers)) {
        objFifo.emitOpError() << "could not retrieve the total number of "
                                 "unique consumer objFifos";
      }
      int64_t splitFactor = std::gcd(*maybeNumUniqueConsumers, numCols);
      int64_t sourceSize = (*sourceSizes)[sourceSplitDim];
      int64_t targetSize = (*targetSizes)[targetSplitDim];
      if (sourceSize % splitFactor != 0 || targetSize % splitFactor != 0) {
        splitFactor = std::gcd(sourceSize, targetSize);
      }
      LLVM_DEBUG(llvm::dbgs() << "sourceSplitDim: " << sourceSplitDim << "\n");
      LLVM_DEBUG(llvm::dbgs() << "targetSplitDim: " << targetSplitDim << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "newSourceStride: " << newSourceStride << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "objFifoSplitDim: " << objFifoSplitDim << "\n");
      LLVM_DEBUG(llvm::dbgs() << "splitStride: " << splitStride << "\n");
      LLVM_DEBUG(llvm::dbgs() << "splitFactor: " << splitFactor << "\n");
      dmaSplitInfoMap[dmaOp] = {sourceSplitDim, newSourceStride, targetSplitDim,
                                1, splitFactor};
      objFifoSplitInfoMap[objFifo] = {objFifoSplitDim, splitFactor, splitStride,
                                      *maybeNumUniqueConsumers};
    } else if (dmaOp.getSourceObjectFifo() == objFifo) {
      // Find outermost dimension in the access pattern that has stride ==
      // sizeAfterSplit and size != 1.
      std::optional<SmallVector<int64_t>> sourceSizes =
          getConstantIntValues(dmaOp.getSourceMixedSizes());
      std::optional<SmallVector<int64_t>> sourceStrides =
          getConstantIntValues(dmaOp.getSourceMixedStrides());
      std::optional<SmallVector<int64_t>> targetSizes =
          getConstantIntValues(dmaOp.getTargetMixedSizes());
      if (!sourceSizes.has_value() || !sourceStrides.has_value() ||
          !targetSizes.has_value()) {
        return dmaOp.emitOpError() << "has unsupported dynamic source strides "
                                      "or sizes or target sizes";
      }
      for (auto iter : llvm::enumerate(
               llvm::zip(sourceSizes.value(), sourceStrides.value()))) {
        int64_t size = std::get<0>(iter.value());
        int64_t stride = std::get<1>(iter.value());
        if (stride == sizeAfterSplit && size != 1) {
          sourceSplitDim = iter.index();
          break;
        }
      }
      int64_t sourceRemainderSize =
          std::accumulate(sourceSizes.value().begin() + sourceSplitDim + 1,
                          sourceSizes.value().end(), 1, std::multiplies<>());
      SmallVector<int64_t> targetProductSizes = targetSizes.value();
      std::partial_sum(targetProductSizes.rbegin(), targetProductSizes.rend(),
                       targetProductSizes.rbegin(), std::multiplies<int64_t>());
      for (int idx = targetProductSizes.size() - 1; idx > 0; idx--) {
        if (targetProductSizes[idx] > sourceRemainderSize) {
          targetSplitDim = idx;
          break;
        }
      }
      FailureOr<int64_t> maybeSplitStride =
          getSplitStride<CopyOpOperateOn::Target>(producers, sizeAfterSplit);
      if (failed(maybeSplitStride)) {
        objFifo.emitOpError()
            << "could not retrieve a split stride from the consumer DMA ops";
      }
      int64_t splitStride = maybeSplitStride.value();
      // Calculate the new target stride to be used for splitting the DMA.
      int64_t newTargetStride =
          splitStride != 1 ? splitDimSize / splitStride : 1;
      FailureOr<int64_t> maybeNumUniqueProducers =
          fetchTotalUniqueLogicalObjFifoUsers<CopyOpOperateOn::Source>(
              objFifo.getCopyLikeProducers());
      if (failed(maybeNumUniqueProducers)) {
        objFifo.emitOpError() << "could not retrieve the total number of "
                                 "unique producer objFifos";
      }
      int64_t splitFactor = std::gcd(*maybeNumUniqueProducers, numCols);
      int64_t sourceSize = (*sourceSizes)[sourceSplitDim];
      int64_t targetSize = (*targetSizes)[targetSplitDim];
      if (sourceSize % splitFactor != 0 || targetSize % splitFactor != 0) {
        splitFactor = std::gcd(sourceSize, targetSize);
      }
      LLVM_DEBUG(llvm::dbgs() << "sourceSplitDim: " << sourceSplitDim << "\n");
      LLVM_DEBUG(llvm::dbgs() << "targetSplitDim: " << targetSplitDim << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "newTargetStride: " << newTargetStride << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "objFifoSplitDim: " << objFifoSplitDim << "\n");
      LLVM_DEBUG(llvm::dbgs() << "splitStride: " << splitStride << "\n");
      LLVM_DEBUG(llvm::dbgs() << "splitFactor: " << splitFactor << "\n");
      dmaSplitInfoMap[dmaOp] = {sourceSplitDim, 1, targetSplitDim,
                                newTargetStride, splitFactor};
      objFifoSplitInfoMap[objFifo] = {objFifoSplitDim, splitFactor,
                                      splitStride};
    }
  }
  return success();
}

class AMDAIESplitLogicalObjFifosPass
    : public impl::AMDAIESplitLogicalObjFifosBase<
          AMDAIESplitLogicalObjFifosPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIESplitLogicalObjFifosPass() = default;
  AMDAIESplitLogicalObjFifosPass(const AMDAIESplitLogicalObjFifosPass &pass){};
  void runOnOperation() override;
};

void AMDAIESplitLogicalObjFifosPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  // Retrieve the device model.
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(moduleOp);
  std::optional<int64_t> maybeNumColumns = getConfigNumColumns(targetAttr);
  if (!maybeNumColumns) {
    moduleOp.emitOpError() << "has no number of columns specified in the "
                              "target attribute configuration. This "
                              "device-specific information is required to "
                              "correctly split logical objectFifos.";
    return signalPassFailure();
  }
  // Use the maximum number of columns available on the device for the default
  // split factor.
  int64_t numColumns = maybeNumColumns.value();

  // If the number of columns used by all CoreOps is known,use it for
  // determining the split factor instead.
  std::optional<int64_t> mayNumColumnsInUse =
      getNumColumnsUsedByCores(moduleOp);
  if (mayNumColumnsInUse.has_value()) numColumns = mayNumColumnsInUse.value();

  // Walk and collect all dma ops between L3 and L2.
  SmallVector<AMDAIE::DmaCpyNdOp> l3L2DmaOps;
  SmallVector<DmaObjFifoPairT> dmaObjFifoPairs;
  WalkResult res = moduleOp->walk([&](AMDAIE::DmaCpyNdOp op) {
    std::optional<uint8_t> sourceMemSpace = op.getSourceMemorySpaceAsUInt();
    std::optional<uint8_t> targetMemSpace = op.getTargetMemorySpaceAsUInt();
    if (!sourceMemSpace || !targetMemSpace) {
      op.emitOpError() << "expected a source and target memory space";
      return WalkResult::interrupt();
    }
    if (sourceMemSpace.value() == 1 && targetMemSpace.value() == 0) {
      dmaObjFifoPairs.push_back({op, op.getSourceObjectFifo()});
    } else if (sourceMemSpace.value() == 0 && targetMemSpace.value() == 1) {
      dmaObjFifoPairs.push_back({op, op.getTargetObjectFifo()});
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();

  // Collect the split dimensions for all DMA and ojectFifo pairs.
  DenseMap<AMDAIE::DmaCpyNdOp, DmaSplitInfo> dmaSplitInfoMap;
  DenseMap<AMDAIE::LogicalObjectFifoFromMemrefOp, ObjFifoSplitInfo>
      objFifoSplitInfoMap;
  if (failed(collectSplittingDims(dmaObjFifoPairs, dmaSplitInfoMap,
                                  objFifoSplitInfoMap, numColumns))) {
    return signalPassFailure();
  }

  /// Split the DMA and objectFifo ops based on the calcuated splitting
  /// dimensions.
  for (auto &&[dmaOp, dmaSplitInfo] : dmaSplitInfoMap) {
    auto stridedOp =
        cast<AMDAIE::DoublyStridedOpInterface>(dmaOp.getOperation());
    if (failed(splitDoublyStridedOp(
            rewriter, stridedOp, dmaSplitInfo.sourceSplitDim,
            dmaSplitInfo.targetSplitDim, dmaSplitInfo.splitSize,
            dmaSplitInfo.newSourceStride, dmaSplitInfo.newTargetStride))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to perform splitting of the DMA op: " << dmaOp);
      return signalPassFailure();
    }
  }
  for (auto &&[objFifo, splitInfo] : objFifoSplitInfoMap) {
    if (failed(splitLogicalObjectFifo(
            rewriter, objFifo, splitInfo.splitDim, splitInfo.splitSize,
            splitInfo.splitStride, splitInfo.numUniqueConsumerDMAs))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to perform splitting of objectFifo op");
      return signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIESplitLogicalObjFifosPass() {
  return std::make_unique<AMDAIESplitLogicalObjFifosPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
