// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-fuse-pack-into-for"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// A utility function specific to this pass which, given a value, would
/// traverse the def-chain till it either finds a tensor.extract_slice op or a
/// BlockArgument.
static FailureOr<tensor::ExtractSliceOp> getTensorExtractSliceDefiningOp(
    Value operand) {
  while (Operation *defOp = operand.getDefiningOp()) {
    auto sliceOp = dyn_cast_or_null<tensor::ExtractSliceOp>(defOp);
    if (sliceOp) {
      // The producer of sliceOp should be a pack op.
      if (isa_and_nonnull<tensor::PackOp>(
              sliceOp.getSource().getDefiningOp())) {
        return sliceOp;
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

class AMDAIEFusePackIntoForLoopPass
    : public impl::AMDAIEFusePackIntoForLoopBase<
          AMDAIEFusePackIntoForLoopPass> {
 public:
  AMDAIEFusePackIntoForLoopPass() = default;
  AMDAIEFusePackIntoForLoopPass(const AMDAIEFusePackIntoForLoopPass &pass) {}
  AMDAIEFusePackIntoForLoopPass(const AMDAIEFusePackIntoForLoopOptions &options)
      : AMDAIEFusePackIntoForLoopBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }
  void runOnOperation() override;
};

void AMDAIEFusePackIntoForLoopPass::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();
  IRRewriter rewriter(context);

  // Walk through the graph in post order and find the for loop.
  scf::ForOp forOp;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](LoopLikeOpInterface op) {
        if (isa<scf::ForOp>(op)) {
          forOp = cast<scf::ForOp>(op);
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (!forOp) {
    LLVM_DEBUG(llvm::dbgs() << "There is no for loop to fuse with.\n");
    return;
  }

  // Based on the `fusePackDepth`, we would continue to greedily fuse the
  // producer tensor.pack ops.
  for (unsigned depth = 1; depth <= fusePackDepth; depth++) {
    // Search the compute op and its producer slices within the For loop.
    BlockArgument bbArg = forOp.getRegionIterArg(0);
    SmallVector<tensor::ExtractSliceOp> sliceOps;
    for (auto user : bbArg.getUsers()) {
      if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
        for (auto [index, operand] : llvm::enumerate(genericOp.getOperands())) {
          FailureOr<tensor::ExtractSliceOp> sliceOp =
              getTensorExtractSliceDefiningOp(operand);
          if (!failed(sliceOp)) {
            sliceOps.push_back(sliceOp.value());
          }
        }
      }
    }

    if (sliceOps.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "----- Pack ops are already fused or no slice "
                                 "ops were found.-----\n");
      return;
    }

    // Materialize each slice of the producer in place.
    LoopLikeOpInterface loops = cast<LoopLikeOpInterface>(forOp.getOperation());
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

std::unique_ptr<Pass> createAMDAIEFusePackIntoForLoopPass(
    AMDAIEFusePackIntoForLoopOptions options) {
  return std::make_unique<AMDAIEFusePackIntoForLoopPass>(options);
}
}  // namespace mlir::iree_compiler::AMDAIE
