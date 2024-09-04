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
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-fuse-pack-into-loop"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// A utility function specific to this pass which, given a value, would
/// traverse the def-chain till it either finds a tensor.extract_slice op or a
/// BlockArgument.
static FailureOr<tensor::ExtractSliceOp> getTensorExtractSliceDefiningOp(
    Value operand) {
  while (Operation *defOp = operand.getDefiningOp()) {
    auto sliceOp = dyn_cast_if_present<tensor::ExtractSliceOp>(defOp);
    if (sliceOp) {
      // The producer of sliceOp should be a pack op.
      if (isa_and_present<tensor::PackOp>(
              sliceOp.getSource().getDefiningOp())) {
        return sliceOp;
      }
      if (isa<BlockArgument>(sliceOp.getSource())) {
        auto blkArg = dyn_cast<BlockArgument>(sliceOp.getSource());
        for (Value blkOperand :
             blkArg.getOwner()->getParentOp()->getOperands()) {
          if (isa_and_present<tensor::PackOp>(blkOperand.getDefiningOp())) {
            return sliceOp;
          }
        }
      }
      break;
    }
    // We perform further traversal only if we have tensor.pack op in the
    // def-chain.
    if (!isa<tensor::PackOp>(defOp)) {
      break;
    }
    operand = defOp->getOperand(0);
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

  if (fusePackDepth < 1) {
    funcOp->emitOpError("Invalid depth of pack ops for fusion.");
    return signalPassFailure();
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

    SmallVector<tensor::ExtractSliceOp> sliceOps;
    for (auto [index, operand] : llvm::enumerate(genericOp.getOperands())) {
      FailureOr<tensor::ExtractSliceOp> sliceOp =
          getTensorExtractSliceDefiningOp(operand);
      if (!failed(sliceOp)) {
        sliceOps.push_back(sliceOp.value());
      }
    }

    if (sliceOps.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "----- Pack ops are already fused or no slice "
                                 "ops were found.-----\n");
      return;
    }

    // Materialize each slice of the producer in place.
    for (auto sliceOp : sliceOps) {
      std::optional<scf::SCFFuseProducerOfSliceResult> fusedProducer =
          scf::tileAndFuseProducerOfSlice(rewriter, sliceOp,
                                          MutableArrayRef(&loops, 1));
      if (!fusedProducer) {
        funcOp->emitOpError("Failed to fuse pack ops into for loop.");
        return signalPassFailure();
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
