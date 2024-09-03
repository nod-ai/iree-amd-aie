// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the a transformation pass that performs common
// sub-expression elimination on AMDAIE DMA operations within the same region.
// This eliminates identical circular DMA operations until only one exists. This
// is different from the main upstream  CSE behaviour on ops with the `Pure`
// trait in that `Pure` ops with no users will be eliminated, while DMA ops
// without users are kept around.
//
// The implementation is based on and largely copied from:
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Transforms/CSE.cpp
//
// TODO(jornt): It might be worthwile to try using the main upstream CSE pass,
// however it seems like that would need changes in that pass as I wasn't able
// to accomplish the same result as this pass by only using `MemoryEffect`. Some
// issues I encountered:
// - Making these DMAs `Pure` eliminates all of them if there are no users.
// - A `Write` memory effect won't find the elimination opportunities, seemingly
// due to other unknown operations in between
// (`hasOtherSideEffectingOpInBetween`)
//
//===----------------------------------------------------------------------===//

#include <deque>

#include "iree-amd-aie/IR/AMDAIETraits.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/RecyclingAllocator.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-amdaie-dma-cse"

namespace mlir::iree_compiler::AMDAIE {

using mlir::OpTrait::iree_compiler::AMDAIE::CircularDmaOp;

using namespace mlir;

namespace {
struct SimpleOperationInfo : public llvm::DenseMapInfo<Operation *> {
  static unsigned getHashValue(const Operation *opC) {
    return OperationEquivalence::computeHash(
        const_cast<Operation *>(opC),
        /*hashOperands=*/OperationEquivalence::directHashValue,
        /*hashResults=*/OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
  }
  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    if (lhs == rhs) return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return OperationEquivalence::isEquivalentTo(
        const_cast<Operation *>(lhsC), const_cast<Operation *>(rhsC),
        OperationEquivalence::IgnoreLocations);
  }
};
}  // namespace

namespace {
/// Common sub-expression elimination for AMDAIE DMA ops.
class DmaCSEDriver {
 public:
  DmaCSEDriver(RewriterBase &rewriter, DominanceInfo *domInfo)
      : rewriter(rewriter), domInfo(domInfo) {}

  /// Simplify all operations within the given op.
  void simplify(Operation *op, bool *changed = nullptr);

 private:
  /// Shared implementation of operation elimination and scoped map definitions.
  using AllocatorTy = llvm::RecyclingAllocator<
      llvm::BumpPtrAllocator,
      llvm::ScopedHashTableVal<Operation *, Operation *>>;
  using ScopedMapTy = llvm::ScopedHashTable<Operation *, Operation *,
                                            SimpleOperationInfo, AllocatorTy>;

  /// Represents a single entry in the depth first traversal of a CFG.
  struct CFGStackNode {
    CFGStackNode(ScopedMapTy &knownValues, DominanceInfoNode *node)
        : scope(knownValues), node(node), childIterator(node->begin()) {}

    /// Scope for the known values.
    ScopedMapTy::ScopeTy scope;

    DominanceInfoNode *node;
    DominanceInfoNode::const_iterator childIterator;

    /// If this node has been fully processed yet or not.
    bool processed = false;
  };

  /// Attempt to eliminate a redundant operation. Returns success if the
  /// operation was marked for removal, failure otherwise.
  LogicalResult simplifyOperation(ScopedMapTy &knownValues, Operation *op,
                                  bool hasSSADominance);
  void simplifyBlock(ScopedMapTy &knownValues, Block *bb, bool hasSSADominance);
  void simplifyRegion(ScopedMapTy &knownValues, Region &region);

  void replaceUsesAndDelete(ScopedMapTy &knownValues, Operation *op,
                            Operation *existing, bool hasSSADominance);

  /// A rewriter for modifying the IR.
  RewriterBase &rewriter;

  /// Operations marked as dead and to be erased.
  std::vector<Operation *> opsToErase;
  DominanceInfo *domInfo = nullptr;
};
}  // namespace

void DmaCSEDriver::replaceUsesAndDelete(ScopedMapTy &knownValues, Operation *op,
                                        Operation *existing,
                                        bool hasSSADominance) {
  // If we find one then replace all uses of the current operation with the
  // existing one and mark it for deletion. We can only replace an operand in
  // an operation if it has not been visited yet.
  if (hasSSADominance) {
    // If the region has SSA dominance, then we are guaranteed to have not
    // visited any use of the current operation.
    if (auto *rewriteListener =
            dyn_cast_if_present<RewriterBase::Listener>(rewriter.getListener()))
      rewriteListener->notifyOperationReplaced(op, existing);
    // Replace all uses, but do not remote the operation yet. This does not
    // notify the listener because the original op is not erased.
    rewriter.replaceAllUsesWith(op->getResults(), existing->getResults());
    opsToErase.push_back(op);
  } else {
    // When the region does not have SSA dominance, we need to check if we
    // have visited a use before replacing any use.
    auto wasVisited = [&](OpOperand &operand) {
      return !knownValues.count(operand.getOwner());
    };
    if (auto *rewriteListener =
            dyn_cast_if_present<RewriterBase::Listener>(rewriter.getListener()))
      for (Value v : op->getResults())
        if (all_of(v.getUses(), wasVisited))
          rewriteListener->notifyOperationReplaced(op, existing);

    // Replace all uses, but do not remote the operation yet. This does not
    // notify the listener because the original op is not erased.
    rewriter.replaceUsesWithIf(op->getResults(), existing->getResults(),
                               wasVisited);

    // There may be some remaining uses of the operation.
    if (op->use_empty()) opsToErase.push_back(op);
  }

  // If the existing operation has an unknown location and the current
  // operation doesn't, then set the existing op's location to that of the
  // current op.
  if (isa<UnknownLoc>(existing->getLoc()) && !isa<UnknownLoc>(op->getLoc()))
    existing->setLoc(op->getLoc());
}

/// Attempt to eliminate a redundant operation.
LogicalResult DmaCSEDriver::simplifyOperation(ScopedMapTy &knownValues,
                                              Operation *op,
                                              bool hasSSADominance) {
  // Only simplify circular DMA opertions.
  if (!op->hasTrait<CircularDmaOp>()) {
    return failure();
  }

  // Look for an existing definition for the operation.
  if (auto *existing = knownValues.lookup(op)) {
    replaceUsesAndDelete(knownValues, op, existing, hasSSADominance);
    return success();
  }

  // Otherwise, we add this operation to the known values map.
  knownValues.insert(op, op);
  return failure();
}

void DmaCSEDriver::simplifyBlock(ScopedMapTy &knownValues, Block *bb,
                                 bool hasSSADominance) {
  for (auto &op : *bb) {
    // Most operations don't have regions, so fast path that case.
    if (op.getNumRegions() != 0) {
      // If this operation is isolated above, we can't process nested regions
      // with the given 'knownValues' map. This would cause the insertion of
      // implicit captures in explicit capture only regions.
      if (op.mightHaveTrait<OpTrait::IsIsolatedFromAbove>()) {
        ScopedMapTy nestedKnownValues;
        for (auto &region : op.getRegions())
          simplifyRegion(nestedKnownValues, region);
      } else {
        // Otherwise, process nested regions normally.
        for (auto &region : op.getRegions())
          simplifyRegion(knownValues, region);
      }
    }

    // If the operation is simplified, we don't process any held regions.
    if (succeeded(simplifyOperation(knownValues, &op, hasSSADominance)))
      continue;
  }
}

void DmaCSEDriver::simplifyRegion(ScopedMapTy &knownValues, Region &region) {
  // If the region is empty there is nothing to do.
  if (region.empty()) return;

  bool hasSSADominance = domInfo->hasSSADominance(&region);

  // If the region only contains one block, then simplify it directly.
  if (region.hasOneBlock()) {
    ScopedMapTy::ScopeTy scope(knownValues);
    simplifyBlock(knownValues, &region.front(), hasSSADominance);
    return;
  }

  // If the region does not have dominanceInfo, then skip it.
  // TODO: Regions without SSA dominance should define a different
  // traversal order which is appropriate and can be used here.
  if (!hasSSADominance) return;

  // Note, deque is being used here because there was significant performance
  // gains over vector when the container becomes very large due to the
  // specific access patterns. If/when these performance issues are no
  // longer a problem we can change this to vector. For more information see
  // the llvm mailing list discussion on this:
  // http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20120116/135228.html
  std::deque<std::unique_ptr<CFGStackNode>> stack;

  // Process the nodes of the dom tree for this region.
  stack.emplace_back(std::make_unique<CFGStackNode>(
      knownValues, domInfo->getRootNode(&region)));

  while (!stack.empty()) {
    auto &currentNode = stack.back();

    // Check to see if we need to process this node.
    if (!currentNode->processed) {
      currentNode->processed = true;
      simplifyBlock(knownValues, currentNode->node->getBlock(),
                    hasSSADominance);
    }

    // Otherwise, check to see if we need to process a child node.
    if (currentNode->childIterator != currentNode->node->end()) {
      auto *childNode = *(currentNode->childIterator++);
      stack.emplace_back(
          std::make_unique<CFGStackNode>(knownValues, childNode));
    } else {
      // Finally, if the node and all of its children have been processed
      // then we delete the node.
      stack.pop_back();
    }
  }
}

void DmaCSEDriver::simplify(Operation *op, bool *changed) {
  /// Simplify all regions.
  ScopedMapTy knownValues;
  for (auto &region : op->getRegions()) simplifyRegion(knownValues, region);

  /// Erase any operations that were marked as dead during simplification.
  for (auto *op : opsToErase) rewriter.eraseOp(op);
  if (changed) *changed = !opsToErase.empty();
}

namespace {
struct AMDAIEDmaCSEPass : public impl::AMDAIEDmaCSEBase<AMDAIEDmaCSEPass> {
  void runOnOperation() override;
};
}  // namespace

void AMDAIEDmaCSEPass::runOnOperation() {
  IRRewriter rewriter(&getContext());
  DmaCSEDriver driver(rewriter, &getAnalysis<DominanceInfo>());
  bool changed = false;
  driver.simplify(getOperation(), &changed);

  // If there was no change to the IR, we mark all analyses as preserved.
  if (!changed) return markAllAnalysesPreserved();

  // We currently don't remove region operations, so mark dominance as
  // preserved.
  markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
}

std::unique_ptr<Pass> createAMDAIEDmaCSEPass() {
  return std::make_unique<AMDAIEDmaCSEPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
