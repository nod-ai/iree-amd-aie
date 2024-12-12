// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-fuse-pack-into-loop"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// A utility function specific to this pass which, given a value `operand`,
/// traverses the def-chain till it finds a tensor.extract_slice. The 2 cases
/// where it successfully finds and returns an extract_slice (SLICE) are:
///
/// Case 1)
/// pack -> SLICE -> pack -> pack -> pack -> operand
///                  ^^^^^^^^^^^^^^^^^^^^
///                  any number (>= 0) of trailing packs
///
/// Case 2)
/// pack -> block arg -> SLICE  -> pack -> pack -> pack -> operand
///                                ^^^^^^^^^^^^^^^^^^^^
///                                any number (>= 0) of trailing packs
///
/// Case 2 only matches where `block arg` is for a loop operation.
static FailureOr<tensor::ExtractSliceOp> getTensorExtractSliceDefiningOp(
    Value operand) {
  // roll back through all the packs immediately preceding `operand`.
  while (isa_and_present<tensor::PackOp>(operand.getDefiningOp())) {
    operand = operand.getDefiningOp()->getOperand(0);
  }

  tensor::ExtractSliceOp sliceOp =
      dyn_cast_if_present<tensor::ExtractSliceOp>(operand.getDefiningOp());
  if (!sliceOp) return failure();

  // Case 1 outlined above.
  if (isa_and_present<tensor::PackOp>(sliceOp.getSource().getDefiningOp())) {
    return sliceOp;
  }

  // Case 2 outlined above.
  else if (auto blkArg = dyn_cast<BlockArgument>(sliceOp.getSource())) {
    Operation *parent = blkArg.getOwner()->getParentOp();
    LoopLikeOpInterface loop = dyn_cast<LoopLikeOpInterface>(parent);
    if (!loop) return failure();
    Operation *operandParent = loop.getTiedLoopInit(blkArg)->getOwner();
    if (isa_and_present<tensor::PackOp>(operandParent)) return sliceOp;
  }

  return failure();
}

class AMDAIEFusePackIntoLoopPass
    : public impl::AMDAIEFusePackIntoLoopBase<AMDAIEFusePackIntoLoopPass> {
 public:
  AMDAIEFusePackIntoLoopPass() = default;
  AMDAIEFusePackIntoLoopPass(const AMDAIEFusePackIntoLoopPass &pass) {}
  AMDAIEFusePackIntoLoopPass(const AMDAIEFusePackIntoLoopOptions &options)
      : AMDAIEFusePackIntoLoopBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }
  void runOnOperation() override;
};

void AMDAIEFusePackIntoLoopPass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();
  IRRewriter rewriter(context);

  // Walk through the graph in post order and find the loop.
  Operation *scfLoopOp = nullptr;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](LoopLikeOpInterface op) {
        if (isa<scf::ForOp>(op) && useSCFFor) {
          scfLoopOp = op;
          return WalkResult::interrupt();
        } else if (isa<scf::ForallOp>(op) && !useSCFFor) {
          scfLoopOp = op;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (!scfLoopOp) {
    LLVM_DEBUG(llvm::dbgs()
               << "There is no scf.for/forall loop to fuse with.\n");
    return;
  }

  LoopLikeOpInterface loops = cast<LoopLikeOpInterface>(scfLoopOp);

  // Based on the `fusePackDepth`, we would greedily fuse the producer
  // tensor.pack ops.
  for (unsigned depth = 1; depth <= fusePackDepth; depth++) {
    // Search the last compute op in the loop and its producer slices.
    linalg::GenericOp genericOp;
    scfLoopOp->walk<WalkOrder::PostOrder, ReverseIterator>(
        [&](linalg::LinalgOp op) {
          if (isa<linalg::GenericOp>(op)) {
            genericOp = cast<linalg::GenericOp>(op);
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });

    if (!genericOp) {
      LLVM_DEBUG(llvm::dbgs() << "----- There is no compute op.-----\n");
      return;
    }

    if (targetElementwise && !isElementwise(genericOp)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "----- The target compute op is not elementwise.-----\n");
      return;
    }

    // Materialize each slice of the producer in place.
    for (Value operand : genericOp.getOperands()) {
      FailureOr<tensor::ExtractSliceOp> maybeSliceOp =
          getTensorExtractSliceDefiningOp(operand);

      if (succeeded(maybeSliceOp)) {
        tensor::ExtractSliceOp sliceOp = maybeSliceOp.value();
        std::optional<scf::SCFFuseProducerOfSliceResult> fusedProducer =
            scf::tileAndFuseProducerOfSlice(rewriter, sliceOp,
                                            MutableArrayRef(&loops, 1));
        if (!fusedProducer) {
          funcOp->emitOpError("Failed to fuse pack ops into for loop.");
          return signalPassFailure();
        }
      }

      // Case where operand of generic is a pack op which is in a different
      // block than the generic's block.
      else if (auto parent = dyn_cast_if_present<tensor::PackOp>(
                   operand.getDefiningOp())) {
        Block *genericBlock = genericOp->getBlock();
        if (parent->getBlock() != genericBlock && parent->hasOneUse()) {
          Operation *firstOpInBlock = &genericBlock->front();
          rewriter.moveOpBefore(parent, firstOpInBlock);
        }
      }
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFusePackIntoLoopPass(
    AMDAIEFusePackIntoLoopOptions options) {
  return std::make_unique<AMDAIEFusePackIntoLoopPass>(options);
}
}  // namespace mlir::iree_compiler::AMDAIE
