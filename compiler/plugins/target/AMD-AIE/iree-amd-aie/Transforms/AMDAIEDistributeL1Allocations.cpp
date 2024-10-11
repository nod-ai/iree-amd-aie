// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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

using namespace mlir;

namespace {

/// Find all induction variables of all `scf.forall` ops that are mapped to
/// gpu thread dimensions (as opposed to gpu block dimensions etc).
FailureOr<DenseSet<Value>> getThreadIndVars(ModuleOp moduleOp) {
  DenseSet<Value> threadIndVars;
  moduleOp.walk([&](scf::ForallOp forallOp) {
    std::optional<ArrayAttr> maybeMapping = forallOp.getMapping();
    if (!maybeMapping) return WalkResult::advance();
    SmallVector<Attribute> mapping = llvm::to_vector(maybeMapping->getValue());
    if (mapping.empty()) return WalkResult::advance();
    if (!isa<gpu::GPUThreadMappingAttr>(*mapping.begin()))
      return WalkResult::advance();
    for (Value indVar : forallOp.getInductionVars()) {
      threadIndVars.insert(indVar);
    }
    return WalkResult::advance();
  });
  return threadIndVars;
}

/// Try to detect subview(s) that look like they're 'distributing' L1 memory.
/// That is: they slice the L1 memory along thread/tile dimensions.
MemRefType getDistributedType(memref::AllocOp alloc,
                              const DenseSet<Value> &indVars) {
  MemRefType type;
  for (Operation *allocUser : alloc->getUsers()) {
    if (auto subview = dyn_cast<memref::SubViewOp>(allocUser)) {
      // Check that all offsets are either constants or thread ids. We assume
      // that if a subview has an offset which is not a constant and not a
      // thread id, it's not 'distributing'.
      Operation::operand_range offsets = subview.getOffsets();
      for (Value offset : offsets) {
        bool isConst = matchPattern(offset, m_Constant());
        bool isIndVar = llvm::is_contained(indVars, offset);
        if (!isConst && !isIndVar) return {};
      }

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

/// Create a copy of `toUpdate` with all values in `toRemove` replaced by
/// `replacement`.
template <typename Container>
SmallVector<Value> substitute(Container toUpdate,
                              const DenseSet<Value> &toRemove,
                              Value replacement) {
  SmallVector<Value> updated(toUpdate.begin(), toUpdate.end());
  for (Value &v : updated) {
    if (toRemove.contains(v)) v = replacement;
  }
  return updated;
}

/// Distribute local memory accesses through subviews by allocating a single,
/// smaller memory. This is ultimately needed because cores can't operate on
/// one shared L1 memory.
LogicalResult distributeLocalMemory(ModuleOp moduleOp) {
  FailureOr<DenseSet<Value>> maybeIndVars = getThreadIndVars(moduleOp);
  if (failed(maybeIndVars)) return failure();
  const DenseSet<Value> &indVars = maybeIndVars.value();
  IRRewriter rewriter(moduleOp.getContext());
  moduleOp->walk([&](memref::AllocOp oldAlloc) {
    // Only consider local memory (L1).
    Attribute maybeMemorySpace = oldAlloc.getType().getMemorySpace();
    if (!maybeMemorySpace) return WalkResult::advance();
    auto memorySpace = cast<IntegerAttr>(maybeMemorySpace);
    if (memorySpace.getInt() != 2) return WalkResult::advance();

    // Don't try and distribute memory if the alloc is inside a scf.for op.
    if (auto scfForOp = oldAlloc->getParentOfType<scf::ForOp>())
      return WalkResult::advance();

    MemRefType memRefType = getDistributedType(oldAlloc, indVars);

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

    const SmallVector<Operation *> users(oldAlloc->user_begin(),
                                         oldAlloc->user_end());

    // Replace uses of the old alloc with the new alloc.
    for (Operation *user : users) {
      LogicalResult switchResult =
          llvm::TypeSwitch<Operation *, LogicalResult>(user)
              .Case<memref::SubViewOp>([&](memref::SubViewOp subviewOp) {
                rewriter.replaceAllUsesWith(subviewOp, newAlloc);
                return success();
              })
              .Case<vector::TransferReadOp>([&](vector::TransferReadOp readOp) {
                rewriter.setInsertionPoint(readOp);
                Value c0 =
                    rewriter.create<arith::ConstantIndexOp>(readOp.getLoc(), 0);
                SmallVector<Value> indices =
                    substitute(readOp.getIndices(), indVars, c0);
                rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
                    readOp, readOp.getType(), newAlloc, indices,
                    readOp.getPermutationMapAttr(), readOp.getPadding(),
                    readOp.getMask(), readOp.getInBoundsAttr());
                return success();
              })
              .Case<vector::TransferWriteOp>(
                  [&](vector::TransferWriteOp writeOp) {
                    rewriter.setInsertionPoint(writeOp);
                    Value c0 = rewriter.create<arith::ConstantIndexOp>(
                        writeOp.getLoc(), 0);
                    SmallVector<Value> indices =
                        substitute(writeOp.getIndices(), indVars, c0);
                    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
                        writeOp, writeOp.getVector(), newAlloc, indices,
                        writeOp.getPermutationMapAttr(), writeOp.getMask(),
                        writeOp.getInBoundsAttr());
                    return success();
                  })
              .Case<memref::ExtractStridedMetadataOp>(
                  [&](memref::ExtractStridedMetadataOp
                          extractStridedMetadataOp) {
                    rewriter
                        .replaceOpWithNewOp<memref::ExtractStridedMetadataOp>(
                            extractStridedMetadataOp, newAlloc);
                    return success();
                  })
              .Case<memref::DeallocOp>([&](memref::DeallocOp deallocOp) {
                rewriter.setInsertionPoint(deallocOp);
                rewriter.create<memref::DeallocOp>(rewriter.getUnknownLoc(),
                                                   newAlloc);
                return success();
              })
              .Default([&](Operation *user) {
                user->emitOpError("needs logic implemented for handling.");
                return failure();
              });

      if (failed(switchResult)) return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

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
