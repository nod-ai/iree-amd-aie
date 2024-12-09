// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIELogicalObjFifoSplittingUtils.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-split-logical-objectfifos"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Utility struct to represent DMA split information.
struct DmaSplitInfo {
  size_t sourceSplitDim{0};
  size_t targetSplitDim{0};
};

using DmaObjFifoPairT =
    std::pair<AMDAIE::DmaCpyNdOp, AMDAIE::LogicalObjectFifoFromMemrefOp>;

/// Find the logical objectFifo and DMA source/target splitting dimensions for
/// each DMA and objectFifo pair.
///
/// Each pair is handled in the following way:
/// First, compute the objectFifo splitting dimension as the last non-unit shape
/// dimension. Afterwards, depending on which logical objectFifo is being
/// split on, find the outermost dimension in either the source or
/// target access pattern that has:
/// - stride == sizeAfterSplit
/// - size != 1
/// This is the splitting dimension to be used on the respective side of the DMA
/// operation. Then, calculate the product size of that side of the DMA
/// operation after the splitting dimension and use it to calculate the
/// splitting dimension on the other side as the first dimension from the back
/// that has product size larger than the other side's product size after
/// splitting because that's the number of elements that should be
/// produced/consumed on the respective sides before splitting.
LogicalResult collectSplittingDims(
    const SmallVector<DmaObjFifoPairT> &dmaObjFifoPairs,
    DenseMap<AMDAIE::DmaCpyNdOp, DmaSplitInfo> &dmaSplitInfoMap,
    DenseMap<AMDAIE::LogicalObjectFifoFromMemrefOp, size_t>
        &objFifoSplitDimMap) {
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
    int64_t sizeAfterSplit =
        std::accumulate(memrefShape.begin() + objFifoSplitDim + 1,
                        memrefShape.end(), 1, std::multiplies<>());

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
    }
    LLVM_DEBUG(llvm::dbgs() << "sourceSplitDim: " << sourceSplitDim << "\n");
    LLVM_DEBUG(llvm::dbgs() << "targetSplitDim: " << targetSplitDim << "\n");
    LLVM_DEBUG(llvm::dbgs() << "objFifoSplitDim: " << objFifoSplitDim << "\n");
    DmaSplitInfo dmaSplitInfo = {sourceSplitDim, targetSplitDim};
    dmaSplitInfoMap[dmaOp] = std::move(dmaSplitInfo);
    objFifoSplitDimMap[objFifo] = objFifoSplitDim;
  }
  return success();
}

class AMDAIESplitLogicalObjFifosPass
    : public impl::AMDAIESplitLogicalObjFifosBase<
          AMDAIESplitLogicalObjFifosPass> {
 public:
  using AMDAIESplitLogicalObjFifosBase::AMDAIESplitLogicalObjFifosBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  void runOnOperation() override;
};

void AMDAIESplitLogicalObjFifosPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

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
  DenseMap<AMDAIE::LogicalObjectFifoFromMemrefOp, size_t> objFifoSplitDimMap;
  if (failed(collectSplittingDims(dmaObjFifoPairs, dmaSplitInfoMap,
                                  objFifoSplitDimMap))) {
    return signalPassFailure();
  }

  /// Split the DMA and objectFifo ops based on the calcuated splitting
  /// dimensions.
  for (auto &&[dmaOp, dmaSplitInfo] : dmaSplitInfoMap) {
    auto stridedOp =
        cast<AMDAIE::DoublyStridedOpInterface>(dmaOp.getOperation());
    if (failed(splitDoublyStridedOp(rewriter, stridedOp,
                                    dmaSplitInfo.sourceSplitDim,
                                    dmaSplitInfo.targetSplitDim))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to perform splitting of the DMA op: " << dmaOp);
      return signalPassFailure();
    }
  }
  for (auto &&[objFifo, splitDim] : objFifoSplitDimMap) {
    if (failed(splitLogicalObjectFifo(rewriter, objFifo, splitDim))) {
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
