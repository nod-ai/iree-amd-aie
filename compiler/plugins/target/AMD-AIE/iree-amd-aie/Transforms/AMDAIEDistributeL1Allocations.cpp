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

/// Find all `scf.forall` ops that are mapped to gpu thread dimensions (as
/// opposed to gpu block dimensions etc) i.e. the ones which will be mapped to
/// cores.
DenseSet<scf::ForallOp> getCoreForallOps(ModuleOp moduleOp) {
  DenseSet<scf::ForallOp> coreForallOps;
  moduleOp.walk([&](scf::ForallOp forallOp) {
    std::optional<ArrayAttr> maybeMapping = forallOp.getMapping();
    if (!maybeMapping) return WalkResult::advance();
    SmallVector<Attribute> mapping = llvm::to_vector(maybeMapping->getValue());
    if (mapping.empty()) return WalkResult::advance();
    if (!isa<gpu::GPUThreadMappingAttr>(*mapping.begin()))
      return WalkResult::advance();
    coreForallOps.insert(forallOp);
    return WalkResult::advance();
  });
  return coreForallOps;
}

/// For a given alloc, the distributed type is fetched by looking at each of its
/// subview users. For each subview, we check if their users are within the
/// scf.forall(s) which are mapped to GPU thread IDs (i.e. will be mapped to
/// core). We return an empty memref type if :-
///   a. Any subview's user is NOT within the innermost scf.forall.
///   b. The result types of two subviews do not match.
MemRefType getDistributedType(memref::AllocOp alloc,
                              DenseSet<scf::ForallOp> &coreForallOps) {
  MemRefType type;
  for (Operation *allocUser : alloc->getUsers()) {
    if (auto subview = dyn_cast<memref::SubViewOp>(allocUser)) {
      for (Operation *subviewUser : subview->getUsers()) {
        auto parentOp = dyn_cast<scf::ForallOp>(subviewUser->getParentOp());
        if (!parentOp || !coreForallOps.contains(parentOp)) return {};
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
  DenseSet<scf::ForallOp> coreForallOps = getCoreForallOps(moduleOp);
  DenseSet<Value> indVars;
  for (scf::ForallOp forallOp : coreForallOps) {
    for (Value indVar : forallOp.getInductionVars()) indVars.insert(indVar);
  }
  IRRewriter rewriter(moduleOp.getContext());

  auto allocWalkResult = moduleOp->walk([&](memref::AllocOp oldAlloc)
                                            -> WalkResult {
    // Only consider local memory (L1).
    Attribute maybeMemorySpace = oldAlloc.getType().getMemorySpace();
    if (!maybeMemorySpace) return WalkResult::advance();
    auto memorySpace = cast<IntegerAttr>(maybeMemorySpace);
    if (memorySpace.getInt() != 2) return WalkResult::advance();

    // Don't try and distribute memory if the alloc is inside a scf.for op.
    if (auto scfForOp = oldAlloc->getParentOfType<scf::ForOp>())
      return WalkResult::advance();

    MemRefType memRefType = getDistributedType(oldAlloc, coreForallOps);

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
                return user->emitOpError(
                    "needs logic implemented for handling.");
              });

      if (failed(switchResult)) return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (allocWalkResult.wasInterrupted()) return failure();

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
