// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIETOAIEWORKGROUP_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIETOAIEWORKGROUP_H_

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::AMDAIE {

//===----------------------------------------------------------------------===//
// IRRewriterAndMapper
//===----------------------------------------------------------------------===//

/// A special type of `IRRewriter` that coordinates mapping and looking up IR
/// entities while creating new operations. The IR entities include operations,
/// results, operands and arguments among others.
class IRRewriterAndMapper : public IRRewriter {
 public:
  using IRRewriter::IRRewriter;

  /// Creates a deep copy of the specified operation, remapping operands based
  /// on the class's IR map and mapping this op to the cloned operation.
  Operation *cloneAndMap(Operation &op) {
    return IRRewriter::clone(op, mapper);
  }

  /// Create an operation of the specific op type at the current insertion point
  /// and lookup and replace the operands based on IR map.
  template <typename OpTy, typename... Args>
  OpTy createAndLookup(Location location, Args &&... args) {
    OpTy newOp =
        IRRewriter::create<OpTy>(location, std::forward<Args>(args)...);
    for (unsigned i = 0, e = newOp->getNumOperands(); i != e; ++i) {
      newOp->setOperand(i, mapper.lookupOrDefault(newOp->getOperand(i)));
    }
    return newOp;
  }

  /// Create an operation of the specific op type at the current insertion point
  /// and map it to the provided operation. When creating the new op, the
  /// operands are looked up and replaced with values found in the IR map if
  /// found.
  template <typename OpTy, typename... Args>
  OpTy createAndMap(Location location, Operation *op, Args &&... args) {
    assert(op && "expected non-null op");
    OpTy newOp = createAndLookup<OpTy>(location, std::forward<Args>(args)...);
    mapOperations(op, newOp.getOperation());
    return newOp;
  }

 protected:
  /// Map the 'source' operation, it's results and regions to the 'target'
  /// counterparts.
  void mapOperations(Operation *source, Operation *target) {
    assert(source->getNumResults() == target->getNumResults() &&
           "expected same number of results");
    assert(source->getNumRegions() == target->getNumRegions() &&
           "expected same number of regions");
    mapper.map(source, target);
    for (unsigned i = 0, e = target->getNumResults(); i != e; ++i) {
      mapper.map(source->getResult(i), target->getResult(i));
    }
    for (unsigned i = 0, e = target->getNumRegions(); i != e; ++i) {
      mapRegions(&source->getRegion(i), &target->getRegion(i));
    }
  }

  /// Map the 'source' regions's block arguments to the 'target' region's block
  /// arguments.
  void mapRegions(Region *source, Region *target) {
    assert(source->getNumArguments() == target->getNumArguments() &&
           "expected same number of arguments");
    for (auto &&[sourceArg, targetArg] :
         llvm::zip(source->getArguments(), target->getArguments())) {
      mapper.map(sourceArg, targetArg);
    }
  }

  /// The map to be used for mapping and looking up IR entities.
  IRMapping mapper;
};

/// Utility class to contain and maintain the core operations' as a map from
/// coordinates to the respective core operation on that location. The core map
/// can be accessed through lookup functions and new entries can be added
/// through the 'mapOrMerge' method or by merging with another CoreContext. This
/// is useful for creating a clean context for building nested operations and
/// then merging with the outer context.
class CoreContext {
 public:
  CoreContext(IRRewriterAndMapper &rewriter) : rewriter(rewriter) {}
  CoreContext(IRRewriterAndMapper &&rewriter) = delete;

  // Check whether the coordinate exists in the map.
  bool contains(const std::tuple<int64_t, int64_t> &coordinate) {
    return coreMap.contains(coordinate);
  }

  /// Return the underlying core map.
  DenseMap<std::tuple<int64_t, int64_t>, AMDAIE::CoreOp> &getCoreMap() {
    return coreMap;
  }

  /// Lookup a coordinate in the core map. This asserts that the provided
  /// coordinate exists.
  AMDAIE::CoreOp lookup(const std::tuple<int64_t, int64_t> &coordinate) {
    AMDAIE::CoreOp res = lookupOrNull(coordinate);
    assert(res && "expected 'coordinate' to be found in the map");
    return res;
  }

  /// Return the underlying core map.
  AMDAIE::CoreOp lookupOrNull(const std::tuple<int64_t, int64_t> &coordinate) {
    return contains(coordinate) ? coreMap[coordinate] : AMDAIE::CoreOp(nullptr);
  }

  // Inserts a new mapping from 'coordinate' to 'coreOp' or merges with a
  // potentially existing entry.
  void mapOrMerge(const std::tuple<int64_t, int64_t> &coordinate,
                  AMDAIE::CoreOp coreOp) {
    AMDAIE::CoreOp existingCoreOp = lookupOrNull(coordinate);
    if (!existingCoreOp) {
      coreMap[coordinate] = coreOp;
    } else {
      mergeCoreOps(coreOp, existingCoreOp);
    }
  }

  /// Merge another context with this one.
  void mergeContext(CoreContext &other) {
    for (auto &&[coordinate, coreOp] : other.getCoreMap())
      mapOrMerge(coordinate, coreOp);
  }

 private:
  /// Merge the 'source' core operations in the end of the 'dest' core
  /// operation.
  void mergeCoreOps(AMDAIE::CoreOp source, AMDAIE::CoreOp dest);

  /// The rewriter to be used.
  IRRewriterAndMapper &rewriter;

  /// Map from coordinates, represented as a tuple of integers, to the
  /// respective core operation on that location.
  DenseMap<std::tuple<int64_t, int64_t>, AMDAIE::CoreOp> coreMap;
};

/// Recursive workgroup build function for an operation.
LogicalResult workgroupBuild(IRRewriterAndMapper &rewriter, Operation *op,
                             Block *target, Block *controlCode,
                             CoreContext &contextCoreMap,
                             Block::iterator targetBegin,
                             Block::iterator controlCodeBegin,
                             Block::iterator controlCodeEnd);

/// Recursive workgroup build function for a block with a provided source and
/// end point.
LogicalResult workgroupBuild(IRRewriterAndMapper &rewriter, Block *source,
                             Block *target, Block *controlCode,
                             CoreContext &contextCoreMap,
                             Block::iterator sourceBegin,
                             Block::iterator sourceEnd,
                             Block::iterator targetBegin,
                             Block::iterator controlCodeBegin,
                             Block::iterator controlCodeEnd);

}  // namespace mlir::iree_compiler::AMDAIE

#endif
