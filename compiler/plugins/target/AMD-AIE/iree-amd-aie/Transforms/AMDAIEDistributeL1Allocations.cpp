// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-amdaie-distribute-l1-allocations"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Try to detect subview(s) that look like they are 'distributing' aka
/// 'privatizing'. That is, subview(s) that take an L1 memref spanning all
/// L1 memories of the AIE array, and slice it along tile specific dimensions.
/// If one or more identical subviews are found, return the MemRefType of
/// the subview(s). Otherwise, return an empty MemRefType.
MemRefType getDistributedType(memref::AllocOp alloc) {
  MemRefType type{};
  for (Operation *allocUser : alloc->getUsers()) {
    if (auto subview = dyn_cast<memref::SubViewOp>(allocUser)) {
      Operation::operand_range offsets = subview.getOffsets();

      // This subview op is contained inside nested scf.for ops. We count how
      // how many of these loop ops are annotated with amdaie.unroll, and are
      // sliced on their induction variable. For distributed L2 memory, we
      // expect this to be exactly 2, and we expect no slicing in other
      // dimensions. It is possible to handle other edge cases, but this is left
      // for future work.
      uint32_t nbNonConstants =
          std::count_if(offsets.begin(), offsets.end(), [](Value v) -> bool {
            return !mlir::matchPattern(v, mlir::m_Constant());
          });
      if (nbNonConstants != 2) return {};
      uint32_t nbDistributionLoops{0};
      SmallVector<Value> inductionVars;

      // The case where the scf.forall hasn't been decomposed yet:
      // Slightly hacky, but we check if there are 2 parents of type
      // scf::ForallOp, and take the more nested of the 2:
      scf::ForallOp parent0, parent1;
      parent0 = subview->getParentOfType<scf::ForallOp>();
      if (parent0) parent1 = parent0->getParentOfType<scf::ForallOp>();
      if (parent1) {
        // All the induction variables are on parent0.
        inductionVars = parent0.getInductionVars();
        for (Value iv : inductionVars) {
          uint64_t sliceCount = std::count(offsets.begin(), offsets.end(), iv);
          if (sliceCount > 1) return {};
          if (sliceCount == 1) {
            ++nbDistributionLoops;
          }
        }
      }

      if (nbDistributionLoops != 2) return {};
      auto nextType = cast<MemRefType>(subview.getResult().getType());
      if (!type) {
        type = nextType;
      } else if (type != nextType) {
        // This is the case where there are 2+ subview ops which look like
        // they should be distributing, but they have different result types.
        // Bail.
        return {};
      }
    }
  }
  return type;
}

/// Distribute local memory accesses through subviews by allocating a single
/// smaller memory. This is needed because cores can't operate on one larger L1
/// memory.
LogicalResult distributeLocalMemory(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  SmallVector<Operation *> toBeErased;

  moduleOp->walk([&](memref::AllocOp oldAlloc) {
    Attribute maybeMemorySpace =
        cast<MemRefType>(oldAlloc.getResult().getType()).getMemorySpace();
    if (!maybeMemorySpace) return WalkResult::advance();
    auto memorySpace = cast<IntegerAttr>(maybeMemorySpace);

    // Only consider local memory (L1).
    if (memorySpace.getInt() != 2) return WalkResult::advance();

    // Don't try and distribute memory if the alloc is inside a scf.for op.
    if (auto scfForOp = oldAlloc->getParentOfType<scf::ForOp>())
      return WalkResult::advance();

    MemRefType memRefType = getDistributedType(oldAlloc);

    // Failed to find a memref.subview that looks like it is distributing.
    // This doesn't mean that we can't distribute (for example there might be
    // no subviews at all), but this requires further work.
    if (!memRefType) return WalkResult::advance();

    ArrayRef<int64_t> newShape = memRefType.getShape();
    Type elementType = memRefType.getElementType();

    rewriter.setInsertionPoint(oldAlloc);
    MemRefType newAllocType = MemRefType::get(
        newShape, elementType, MemRefLayoutAttrInterface{}, memorySpace);
    auto newAlloc = rewriter.create<memref::AllocOp>(rewriter.getUnknownLoc(),
                                                     newAllocType);
    auto newDeallocOp =
        rewriter.create<memref::DeallocOp>(rewriter.getUnknownLoc(), newAlloc);

    newDeallocOp->moveBefore(&newAlloc->getBlock()->back());

    // Replace uses of the old alloc with the new alloc.
    for (Operation *userOp : oldAlloc->getUsers()) {
      LogicalResult switchResult =
          llvm::TypeSwitch<Operation *, LogicalResult>(userOp)
              .Case<memref::SubViewOp>([&](memref::SubViewOp subviewOp) {
                rewriter.replaceAllUsesWith(subviewOp, newAlloc);
                toBeErased.push_back(subviewOp);
                return success();
              })
              .Case<vector::TransferReadOp>(
                  [&](vector::TransferReadOp transferReadOp) {
                    rewriter.setInsertionPoint(transferReadOp);
                    // Since in this function we're basically changing the L1
                    // sizes of the Alloc, for dimensions with size as 1 we need
                    // to set the indices as 0. We need to do this at this step
                    // because there would be loop dependencies on the same and
                    // when we unroll those loops later in this pass we would
                    // have incorrect offset values being formed for those
                    // dimensions.
                    SmallVector<Value> newIndices = transferReadOp.getIndices();
                    Value c0 = rewriter.create<arith::ConstantIndexOp>(
                        transferReadOp.getLoc(), 0);
                    for (unsigned i = 0, n = newShape.size(); i < n; i++) {
                      if (newShape[i] == 1) newIndices[i] = c0;
                    }

                    auto newTransferReadOp =
                        rewriter.create<vector::TransferReadOp>(
                            transferReadOp.getLoc(), transferReadOp.getType(),
                            newAlloc, newIndices,
                            transferReadOp.getPermutationMapAttr(),
                            transferReadOp.getPadding(),
                            transferReadOp.getMask(),
                            transferReadOp.getInBoundsAttr());
                    rewriter.replaceAllUsesWith(transferReadOp,
                                                newTransferReadOp.getResult());
                    toBeErased.push_back(transferReadOp);
                    return success();
                  })
              .Case<vector::TransferWriteOp>(
                  [&](vector::TransferWriteOp transferWriteOp) {
                    rewriter.setInsertionPoint(transferWriteOp);
                    // Since in this function we're basically changing the L1
                    // sizes of the Alloc, for dimensions with size as 1 we need
                    // to set the indices as 0. We need to do this at this step
                    // because there would be loop dependencies on the same and
                    // when we unroll those loops later in this pass we would
                    // have incorrect offset values being formed for those
                    // dimensions.
                    SmallVector<Value> newIndices =
                        transferWriteOp.getIndices();
                    Value c0 = rewriter.create<arith::ConstantIndexOp>(
                        transferWriteOp.getLoc(), 0);

                    for (unsigned i = 0, n = newShape.size(); i < n; i++) {
                      if (newShape[i] == 1) newIndices[i] = c0;
                    }

                    rewriter.create<vector::TransferWriteOp>(
                        transferWriteOp.getLoc(), transferWriteOp.getVector(),
                        newAlloc, newIndices,
                        transferWriteOp.getPermutationMapAttr(),
                        transferWriteOp.getMask(),
                        transferWriteOp.getInBoundsAttr());
                    toBeErased.push_back(transferWriteOp);
                    return success();
                  })
              .Case<memref::ExtractStridedMetadataOp>(
                  [&](memref::ExtractStridedMetadataOp
                          extractStridedMetadataOp) {
                    rewriter.setInsertionPoint(extractStridedMetadataOp);
                    auto newextractStridedMetadataOp =
                        rewriter.create<memref::ExtractStridedMetadataOp>(
                            extractStridedMetadataOp.getLoc(), newAlloc);
                    rewriter.replaceAllUsesWith(
                        extractStridedMetadataOp.getResults(),
                        newextractStridedMetadataOp.getResults());
                    toBeErased.push_back(extractStridedMetadataOp);
                    return success();
                  })
              .Case<memref::DeallocOp>([&](memref::DeallocOp deallocOp) {
                toBeErased.push_back(userOp);
                return success();
              })
              .Case<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                  [&rewriter, &newAlloc, &toBeErased](
                      AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
                    auto type = llvm::cast<MemRefType>(newAlloc.getType());

                    // Collect all DmaCpyNdOps which have 'logicalObjectFifo' as
                    // a source. Currently not handling the case of multiple.
                    SmallVector<AMDAIE::DmaCpyNdOp> dmaOps;
                    for (Operation *objFifoUserOp :
                         logicalObjectFifo->getUsers()) {
                      if (auto dmaOp =
                              dyn_cast<AMDAIE::DmaCpyNdOp>(objFifoUserOp);
                          dmaOp.getSourceObjectFifo() == logicalObjectFifo) {
                        dmaOps.push_back(dmaOp);
                      }
                    }
                    if (dmaOps.size() == 0) return success();
                    if (dmaOps.size() > 1) {
                      logicalObjectFifo->emitOpError(
                          "Case of multiple DMA ops not handled yet (easy "
                          "extension to logic here)");
                      return failure();
                    }
                    AMDAIE::DmaCpyNdOp dmaOp = dmaOps[0];

                    SmallVector<Value> empty;
                    rewriter.setInsertionPoint(logicalObjectFifo);
                    auto source =
                        rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
                            rewriter.getUnknownLoc(),
                            LogicalObjectFifoType::get(type),
                            newAlloc.getResult());
                    rewriter.replaceAllUsesWith(logicalObjectFifo, source);
                    toBeErased.push_back(logicalObjectFifo);
                    rewriter.setInsertionPoint(dmaOp);
                    auto newDmaOp = rewriter.create<AMDAIE::DmaCpyNdOp>(
                        dmaOp.getLoc(), dmaOp.getTarget(),
                        dmaOp.getTargetMixedOffsets(),
                        dmaOp.getTargetMixedSizes(),
                        dmaOp.getTargetMixedStrides(), source,
                        dmaOp.getSourceMixedOffsets(),
                        dmaOp.getSourceMixedSizes(),
                        dmaOp.getSourceMixedStrides());
                    rewriter.replaceAllUsesWith(dmaOp, newDmaOp);
                    // TODO: maybe this should be left to a DCE somewhere,
                    // instead of manually erasing unused ops?
                    toBeErased.push_back(dmaOp);
                    // We have to discard non-zero offsets as subview has
                    // been replaced by a dedicated allocated memref.
                    SmallVector<int64_t> allocShape(type.getShape());
                    (void)discardAllNonZeroOffsets<CopyOpOperateOn::Source>(
                        rewriter,
                        cast<AMDAIE::DoublyStridedOpInterface>(
                            newDmaOp.getOperation()),
                        allocShape);
                    return success();
                  })
              .Default([&](Operation *userOp) {
                userOp->emitOpError(
                    "needs to have logic implemented for handling in "
                    "distributeLocalMemory");
                return failure();
              });

      if (failed(switchResult)) return WalkResult::interrupt();
    }
    toBeErased.push_back(oldAlloc);

    return WalkResult::advance();
  });

  for (Operation *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }

  return success();
}

class AMDAIEDistributeL1AllocationsPass
    : public impl::AMDAIEDistributeL1AllocationsBase<
          AMDAIEDistributeL1AllocationsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEDistributeL1AllocationsPass() = default;
  AMDAIEDistributeL1AllocationsPass(
      const AMDAIEDistributeL1AllocationsPass &pass){};
  void runOnOperation() override;
};

void AMDAIEDistributeL1AllocationsPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  if (failed(distributeLocalMemory(moduleOp))) return signalPassFailure();
}
}  // namespace

std::unique_ptr<Pass> createAMDAIEDistributeL1AllocationsPass() {
  return std::make_unique<AMDAIEDistributeL1AllocationsPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
