// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-amdaie-create-logical-objectfifo-link"

namespace mlir::iree_compiler::AMDAIE {

std::optional<int64_t> getFlatConstantSourceOffset(
    DoublyStridedOpInterface op) {
  llvm::outs() << "getFlatConstantSourceOffset: " << op << "\n";
  int64_t resOffset = 0;
  for (auto &&[offset, size, stride] :
       llvm::zip(op.getSourceMixedOffsets(), op.getSourceMixedSizes(),
                 op.getSourceMixedStrides())) {
    std::optional<int64_t> constantOffset = getConstantIntValue(offset);
    std::optional<int64_t> constantSize = getConstantIntValue(size);
    std::optional<int64_t> constantStride = getConstantIntValue(stride);
    if (!constantOffset)
      llvm::outs() << "no constant offset: " << offset << "\n";
    if (!constantSize) llvm::outs() << "no constant size: " << size << "\n";
    if (!constantStride)
      llvm::outs() << "no constant stride: " << stride << "\n";

    if (constantOffset && constantOffset.value() == 0) continue;
    if (constantOffset && constantSize && constantStride &&
        constantOffset.value() != 0 && constantSize.value() == 1) {
      resOffset += (constantOffset.value() * constantStride.value());
    } else {
      return std::nullopt;
    }
  }
  return resOffset;
}

std::optional<int64_t> getFlatConstantTargetOffset(
    DoublyStridedOpInterface op) {
  llvm::outs() << "getFlatConstantTargetOffset: " << op << "\n";
  int64_t resOffset = 0;
  for (auto &&[offset, size, stride] :
       llvm::zip(op.getTargetMixedOffsets(), op.getTargetMixedSizes(),
                 op.getTargetMixedStrides())) {
    std::optional<int64_t> constantOffset = getConstantIntValue(offset);
    std::optional<int64_t> constantSize = getConstantIntValue(size);
    std::optional<int64_t> constantStride = getConstantIntValue(stride);
    if (constantOffset && constantOffset.value() == 0) continue;
    if (constantOffset && constantSize && constantStride &&
        constantSize.value() == 1) {
      resOffset += (constantOffset.value() * constantStride.value());
    } else {
      return std::nullopt;
    }
  }
  return resOffset;
}

/// Utility to add explicit link operations to avoid having to do this during
/// conversion to AIEDialect operations. This function only consider L2/MT for
/// links as L1/L3 don't need this linking through AIE objectFifos. Furthermore,
/// it assumes that all users of a logical objectFifo reside within the same
/// block and an error will be emitted if that's not the case.
LogicalResult createLogicalObjectFifoLink(
    RewriterBase &rewriter,
    AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
  Attribute memSpace = logicalObjectFifo.getMemorySpace();
  if (!memSpace || dyn_cast<IntegerAttr>(memSpace).getInt() != 1) {
    return success();
  }

  // Visit all CopyOp users of this logical objectFifo and add them to
  // either the input or output side of this logical objectFifo. While
  // doing this, keep track of the last user operation for insertion
  // purposes.
  SmallVector<std::pair<DoublyStridedOpInterface, int64_t>> ins;
  SmallVector<std::pair<DoublyStridedOpInterface, int64_t>> outs;
  CopyOpInterface lastUserOp;
  for (Operation *userOp : logicalObjectFifo->getUsers()) {
    auto stridedOp = dyn_cast<DoublyStridedOpInterface>(userOp);
    auto copyOp = dyn_cast<CopyOpInterface>(userOp);
    if (stridedOp && copyOp) {
      if (lastUserOp && copyOp->getBlock() != lastUserOp->getBlock()) {
        logicalObjectFifo->emitError(
            "does have copy-like users not residing in the same block");
        return failure();
      }
      auto sourceLogicalObjectFifo =
          dyn_cast<AMDAIE::LogicalObjectFifoFromMemrefOp>(
              copyOp.getSource().getDefiningOp());
      if (!sourceLogicalObjectFifo) {
        copyOp->emitError(
            "does not have a `LogicalObjectFifoFromMemrefOp` as source");
        return failure();
      }
      if (!lastUserOp || lastUserOp->isBeforeInBlock(copyOp)) {
        lastUserOp = copyOp;
      }
      if (logicalObjectFifo == sourceLogicalObjectFifo) {
        if (std::optional<int64_t> offset =
                getFlatConstantSourceOffset(stridedOp)) {
          outs.push_back(std::make_pair(stridedOp, offset.value()));
        } else {
          return stridedOp.emitOpError()
                 << "non-constant offset found which is not supported";
        }
      } else {
        if (std::optional<int64_t> offset =
                getFlatConstantSourceOffset(stridedOp)) {
          ins.push_back(std::make_pair(stridedOp, offset.value()));
        } else {
          return stridedOp.emitOpError()
                 << "non-constant offset found which is not supported";
        }
      }
    }
  }

  // TODO(jornt): Add checks on size equality etc due to objectfifo constraints.

  auto comparator = [](std::pair<DoublyStridedOpInterface, int64_t> a,
                       std::pair<DoublyStridedOpInterface, int64_t> b) -> bool {
    return a.second < b.second;
  };

  llvm::sort(ins.begin(), ins.end(), comparator);
  llvm::sort(outs.begin(), outs.end(), comparator);

  SmallVector<Value> inResults = llvm::map_to_vector(
      ins, [](std::pair<DoublyStridedOpInterface, int64_t> elem) {
        return cast<Value>(elem.first->getResult(0));
      });
  SmallVector<Value> outResults = llvm::map_to_vector(
      outs, [](std::pair<DoublyStridedOpInterface, int64_t> elem) {
        return cast<Value>(elem.first->getResult(0));
      });

  // Insert the `LogicalObjectFifoLink` after the last user operation.
  if (lastUserOp) {
    rewriter.setInsertionPointAfter(lastUserOp);
    rewriter.create<AMDAIE::LogicalObjectFifoLink>(rewriter.getUnknownLoc(),
                                                   inResults, outResults);
  }
  return success();
}

LogicalResult removeAllNonZeroOffsets(RewriterBase &rewriter,
                                      AMDAIE::DoublyStridedOpInterface op) {
  auto copyOp = dyn_cast<CopyOpInterface>(op.getOperation());
  if (!copyOp) return success();
  SmallVector<OpFoldResult> newSourceOffsets;
  SmallVector<OpFoldResult> newSourceSizes;
  SmallVector<OpFoldResult> newSourceStrides;
  SmallVector<OpFoldResult> newTargetOffsets;
  SmallVector<OpFoldResult> newTargetSizes;
  SmallVector<OpFoldResult> newTargetStrides;
  for (auto &&[offset, size, stride] :
       llvm::zip(op.getSourceMixedOffsets(), op.getSourceMixedSizes(),
                 op.getSourceMixedStrides())) {
    std::optional<int64_t> constantOffset = getConstantIntValue(offset);
    if (constantOffset && constantOffset.value() != 0) continue;
    newSourceOffsets.push_back(offset);
    newSourceSizes.push_back(size);
    newSourceStrides.push_back(stride);
  }
  for (auto &&[offset, size, stride] :
       llvm::zip(op.getTargetMixedOffsets(), op.getTargetMixedSizes(),
                 op.getTargetMixedStrides())) {
    std::optional<int64_t> constantOffset = getConstantIntValue(offset);
    if (constantOffset && constantOffset.value() != 0) continue;
    newTargetOffsets.push_back(offset);
    newTargetSizes.push_back(size);
    newTargetStrides.push_back(stride);
  }
  rewriter.setInsertionPointAfter(op);
  auto newDoublyStridedOp = op.createDoublyStridedOp(
      rewriter, newTargetOffsets, newTargetSizes, newTargetStrides,
      newSourceOffsets, newSourceSizes, newSourceStrides);
  rewriter.replaceOp(op, newDoublyStridedOp.getOperation());
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

    WalkResult res = parentOp->walk(
        [&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
          if (failed(
                  createLogicalObjectFifoLink(rewriter, logicalObjectFifo))) {
            logicalObjectFifo.emitError() << "couldn't create a link operation";
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (res.wasInterrupted()) return signalPassFailure();

    // Remove all non-zero offsets as they should be handled through the link by
    // now or errored out.
    // TODO(jornt): rewrite
    res = parentOp->walk([&](AMDAIE::DoublyStridedOpInterface stridedOp) {
      auto copyOp = dyn_cast<CopyOpInterface>(stridedOp.getOperation());
      if (!copyOp) return WalkResult::advance();
      if (failed(removeAllNonZeroOffsets(rewriter, stridedOp))) {
        stridedOp.emitError() << "couldn't remove non-zero offsets";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return signalPassFailure();

    parentOp->walk([&](AMDAIE::DoublyStridedOpInterface stridedOp) {
      auto copyOp = dyn_cast<CopyOpInterface>(stridedOp.getOperation());
      if (!copyOp) return WalkResult::advance();
      (void)foldDmaOpSingleDims(rewriter, stridedOp);
      return WalkResult::advance();
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIECreateLogicalObjectFifoLinkPass() {
  return std::make_unique<AMDAIECreateLogicalObjectFifoLinkPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
