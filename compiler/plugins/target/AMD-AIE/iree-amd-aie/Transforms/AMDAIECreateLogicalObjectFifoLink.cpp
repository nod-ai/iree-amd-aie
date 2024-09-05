// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-amdaie-create-logical-objectfifo-link"

namespace mlir::iree_compiler::AMDAIE {

/// Utility to verify that there are no overlapping access patterns between
/// different strided operations as this is not supported in the MLIR-AIE
/// backend for now.
///
/// Example:
///
/// There are two strided DMA operations on the same logical objectfifo with
/// following access patterns:
///
/// Access pattern 1: (offsets: [0, 0], sizes: [2, 8], strides: [32, 1])
/// Access pattern 2: (offsets: [1, 0, 0], sizes: [1, 2, 8], strides: [32, 32,
/// 1])
///
/// In this case, access pattern 1 access elements in range [0, 63], while
/// access pattern 2 access elements in range [32, 95]. Therefore,
/// these two access patterns overlap by both accessing elements in range [32,
/// 63].
template <CopyOpOperateOn OperateOn>
LogicalResult checkForContiguousAccessPatterns(
    SmallVector<std::pair<AMDAIE::ConnectionOp, int64_t>> &connectionOps) {
  for (auto &&[i, connectionOpAndOffset] : llvm::enumerate(connectionOps)) {
    FailureOr<AMDAIE::NpuCircularDmaCpyNdOp> npuDmaUserOp =
        connectionOpAndOffset.first.getNpuCircularDmaCpyNdUser();
    if (failed(npuDmaUserOp)) return failure();
    auto stridedOp = cast<AMDAIE::DoublyStridedOpInterface>(
        npuDmaUserOp.value().getOperation());
    std::optional<int64_t> extent;
    if constexpr (OperateOn == CopyOpOperateOn::Source) {
      extent = stridedOp.getSourceStaticExtent();
    } else {
      extent = stridedOp.getTargetStaticExtent();
    }
    if (!extent) {
      return stridedOp.emitOpError()
             << "has a non-constant access extent, which is not supported";
    }
    int64_t offset = connectionOpAndOffset.second;
    if (i < (connectionOps.size() - 1)) {
      if (offset + extent.value() != connectionOps[i + 1].second) {
        // TODO(newling) my understanding from the code is that the link
        // operation effectively replaces the cumulative offset of each
        // circular_dma_cpy_nd with the differential offset with
        // the previous circular_dma_cpy_nd in the 'link' list.
        //
        // This however is hardcoded to a zero offset (later in the pass where
        // discardAllNonZeroOffsets is called, offsets are set to zero). This
        // effectively is constraining the link operation to only work with
        // contiguous access patterns.
        //
        // Is this a bug?
        return stridedOp.emitOpError()
               << "has access pattern of which isn't contiguous with next one "
                  "-- not currently supported.";
      }
    }
  }
  return success();
}

/// Utility to add explicit link operations to avoid having to do this during
/// conversion to AIEDialect operations. This function only consider L2/MT for
/// links as L1/L3 don't need this linking through AIE objectFifos. Furthermore,
/// it assumes that all users of a logical objectFifo reside within the same
/// block and an error will be emitted if that's not the case.
LogicalResult createLogicalObjectFifoLink(
    RewriterBase &rewriter,
    AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo,
    SmallVector<AMDAIE::LogicalObjectFifoLink> &newLinkOps) {
  Attribute memSpace = logicalObjectFifo.getMemorySpace();
  if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 1) {
    return success();
  }

  // Visit all DoublyStridedOpInterface users of this logical objectFifo and
  // add them to either the input or output side of this logical objectFifo
  // together with the base offset to be used later for sorting. While doing
  // this, keep track of the last user operation for insertion purposes.
  SmallVector<std::pair<AMDAIE::ConnectionOp, int64_t>> ins;
  SmallVector<std::pair<AMDAIE::ConnectionOp, int64_t>> outs;
  AMDAIE::ConnectionOp lastUserOp;
  for (Operation *userOp : logicalObjectFifo->getUsers()) {
    auto connectionOp = dyn_cast<AMDAIE::ConnectionOp>(userOp);
    if (!connectionOp) {
      return logicalObjectFifo.emitOpError()
             << "found user which is not an `amdaie.connection` op";
    }
    FailureOr<AMDAIE::NpuCircularDmaCpyNdOp> npuDmaUserOp =
        connectionOp.getNpuCircularDmaCpyNdUser();
    if (failed(npuDmaUserOp)) return failure();
    auto stridedOp = cast<AMDAIE::DoublyStridedOpInterface>(
        npuDmaUserOp.value().getOperation());

    if (lastUserOp && connectionOp->getBlock() != lastUserOp->getBlock()) {
      logicalObjectFifo->emitError(
          "has copy-like users not residing in the same block");
      return failure();
    }
    if (!lastUserOp || lastUserOp->isBeforeInBlock(connectionOp)) {
      lastUserOp = connectionOp;
    }

    auto sourceLogicalObjectFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
            connectionOp.getSource().getDefiningOp());
    // The `sourceLogicalObjectFifo` could be either a
    // `LogicalObjectFifoFromMemrefOp` or `LogicalObjectFifoPlaceholderOp`,
    // but currently the linking only works with
    // `LogicalObjectFifoFromMemrefOp` on L2.
    if (sourceLogicalObjectFifo &&
        logicalObjectFifo == sourceLogicalObjectFifo) {
      if (std::optional<int64_t> offset =
              stridedOp.getSourceStaticBaseOffset()) {
        outs.push_back(std::make_pair(connectionOp, offset.value()));
      } else {
        return stridedOp.emitOpError()
               << "non-constant offset found which is not supported";
      }
    } else {
      if (std::optional<int64_t> offset =
              stridedOp.getTargetStaticBaseOffset()) {
        ins.push_back(std::make_pair(connectionOp, offset.value()));
      } else {
        return stridedOp.emitOpError()
               << "non-constant offset found which is not supported";
      }
    }
  }

  // Sort the inputs and outputs on offset as the link operation uses this order
  // to generate correct data buffer sizes.
  auto comparator = [](std::pair<AMDAIE::ConnectionOp, int64_t> a,
                       std::pair<AMDAIE::ConnectionOp, int64_t> b) -> bool {
    return a.second < b.second;
  };

  llvm::sort(ins.begin(), ins.end(), comparator);
  llvm::sort(outs.begin(), outs.end(), comparator);

  // Check that access patterns are not overlapping between consumers
  // respectively producers.
  if (failed(checkForContiguousAccessPatterns<CopyOpOperateOn::Target>(ins))) {
    return failure();
  }
  if (failed(checkForContiguousAccessPatterns<CopyOpOperateOn::Source>(outs))) {
    return failure();
  }

  SmallVector<Value> inResults = llvm::map_to_vector<8>(
      ins, [](std::pair<AMDAIE::ConnectionOp, int64_t> elem) -> Value {
        return cast<Value>(elem.first->getResult(0));
      });
  SmallVector<Value> outResults = llvm::map_to_vector(
      outs, [](std::pair<AMDAIE::ConnectionOp, int64_t> elem) -> Value {
        return cast<Value>(elem.first->getResult(0));
      });

  // Insert the `LogicalObjectFifoLink` after the last user operation.
  if (lastUserOp) {
    rewriter.setInsertionPointAfter(lastUserOp);
    auto linkOp = rewriter.create<AMDAIE::LogicalObjectFifoLink>(
        rewriter.getUnknownLoc(), inResults, outResults);
    newLinkOps.push_back(linkOp);
  }
  return success();
}

LogicalResult discardLinkNonZeroOffsets(RewriterBase &rewriter,
                                        AMDAIE::LogicalObjectFifoLink linkOp) {
  for (Value input : linkOp.getIns()) {
    auto connectionOp =
        dyn_cast_if_present<AMDAIE::ConnectionOp>(input.getDefiningOp());
    if (!connectionOp) {
      return linkOp.emitOpError()
             << "found input which is not an `amdaie.connection` op";
    }
    FailureOr<AMDAIE::NpuCircularDmaCpyNdOp> npuDmaUserOp =
        connectionOp.getNpuCircularDmaCpyNdUser();
    if (failed(npuDmaUserOp)) return failure();
    auto stridedOp = cast<AMDAIE::DoublyStridedOpInterface>(
        npuDmaUserOp.value().getOperation());
    SmallVector<int64_t> shape;
    (void)discardAllNonZeroOffsets<CopyOpOperateOn::Target>(rewriter, stridedOp,
                                                            shape);
  }
  for (Value output : linkOp.getOuts()) {
    auto connectionOp =
        dyn_cast_if_present<AMDAIE::ConnectionOp>(output.getDefiningOp());
    if (!connectionOp) {
      return linkOp.emitOpError()
             << "found input which is not an `amdaie.connection` op";
    }
    FailureOr<AMDAIE::NpuCircularDmaCpyNdOp> npuDmaUserOp =
        connectionOp.getNpuCircularDmaCpyNdUser();
    if (failed(npuDmaUserOp)) return failure();
    auto stridedOp = cast<AMDAIE::DoublyStridedOpInterface>(
        npuDmaUserOp.value().getOperation());
    SmallVector<int64_t> shape;
    (void)discardAllNonZeroOffsets<CopyOpOperateOn::Source>(rewriter, stridedOp,
                                                            shape);
  }
  return success();
}

namespace {

struct AMDAIECreateLogicalObjectFifoLinkPass
    : public impl::AMDAIECreateLogicalObjectFifoLinkBase<
          AMDAIECreateLogicalObjectFifoLinkPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override {
    Operation *parentOp = getOperation();
    IRRewriter rewriter(parentOp->getContext());

    SmallVector<AMDAIE::LogicalObjectFifoLink> newLinkOps;
    WalkResult res = parentOp->walk(
        [&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
          if (failed(createLogicalObjectFifoLink(rewriter, logicalObjectFifo,
                                                 newLinkOps))) {
            logicalObjectFifo.emitError() << "couldn't create a link operation";
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (res.wasInterrupted()) return signalPassFailure();

    // Remove all non-zero offsets.
    for (AMDAIE::LogicalObjectFifoLink linkOp : newLinkOps) {
      if (failed(discardLinkNonZeroOffsets(rewriter, linkOp)))
        return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIECreateLogicalObjectFifoLinkPass() {
  return std::make_unique<AMDAIECreateLogicalObjectFifoLinkPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
