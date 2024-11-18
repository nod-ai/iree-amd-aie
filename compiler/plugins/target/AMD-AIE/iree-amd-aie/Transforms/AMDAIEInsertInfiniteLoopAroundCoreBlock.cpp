// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file inserts infinite looping around the `amdaie.core` blocks. This
// results in the cores running the same program over and over which is useful
// for measuring performance statistics like latency/throughput, averaged over
// a certain number of runs, while excluding core reconfiguration overhead.
//
//===----------------------------------------------------------------------===//

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEOpUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-amdaie-insert-infinite-loop-around-core-block"

namespace mlir::iree_compiler::AMDAIE {

namespace {

LogicalResult addInfiniteWhileLoopAroundCoreBlock(RewriterBase &rewriter,
                                                  AMDAIE::CoreOp coreOp) {
  rewriter.setInsertionPoint(coreOp);
  auto trueAttr = mlir::BoolAttr::get(rewriter.getContext(), true);
  auto whileOp = rewriter.create<scf::WhileOp>(
      rewriter.getUnknownLoc(), TypeRange{}, ValueRange{},
      [&](OpBuilder &beforeBuilder, Location loc, ValueRange operands) {
        auto condition = beforeBuilder.create<arith::ConstantOp>(loc, trueAttr);
        beforeBuilder.create<scf::ConditionOp>(loc, condition, operands);
      },
      nullptr);
  Block *coreBlock = coreOp.getBody();
  rewriter.eraseOp(coreBlock->getTerminator());
  Block *afterBlock = whileOp.getAfterBody();
  rewriter.mergeBlocks(coreBlock, afterBlock);
  rewriter.setInsertionPoint(afterBlock, afterBlock->end());
  rewriter.create<scf::YieldOp>(rewriter.getUnknownLoc(), ValueRange{});
  Block *newBlock = rewriter.createBlock(&coreOp.getRegion());
  rewriter.setInsertionPoint(newBlock, newBlock->begin());
  auto endOp = rewriter.create<AMDAIE::EndOp>(rewriter.getUnknownLoc());
  rewriter.moveOpBefore(whileOp, endOp);
  return success();
}

/// Insert a loop with extent equal to the least common multiple of the depths
/// of the objFifos operated on. This ensures functional correctness after
/// potential further loop unrolling based on the depths of the objFifos.
/// Specifically, this anticipates the unrolling happening inside the
/// `AcquireReleaseToUseLock` pass and is somewhat coupled with it.
LogicalResult addDepthLoopAroundCoreCodeBlock(RewriterBase &rewriter,
                                              AMDAIE::CoreOp coreOp) {
  rewriter.setInsertionPoint(coreOp);
  auto loc = rewriter.getUnknownLoc();
  // Collect the depths of all the objFifos being operated on within the core to
  // compute the minimum loop extent for the loop that should be inserted around
  // the core block.
  llvm::SmallDenseSet<uint8_t> depths;
  WalkResult res = coreOp->walk([&](AMDAIE::LogicalObjectFifoAcquire acqOp) {
    FailureOr<AMDAIE::LogicalObjectFifoFromBuffersOp> maybeLogicalObjFifo =
        getLogicalObjFifoOperatedOn<AMDAIE::LogicalObjectFifoAcquire>(acqOp);
    if (failed(maybeLogicalObjFifo)) {
      coreOp.emitOpError() << "contains acquire op not operating on "
                              "`amdaie.logicalobjectfifo.from_buffers`";
      return WalkResult::interrupt();
    }
    depths.insert(maybeLogicalObjFifo.value().getDepth());
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return coreOp.emitOpError() << "failed to find some depths";
  int minLoopExtent =
      std::accumulate(depths.begin(), depths.end(), 1, std::lcm<int, int>);

  // Create new loop and merge blocks.
  Value lbValue = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value ubValue = rewriter.create<arith::ConstantIndexOp>(loc, minLoopExtent);
  Value stepValue = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto loop = rewriter.create<scf::ForOp>(
      rewriter.getUnknownLoc(), lbValue, ubValue, stepValue, ValueRange(),
      [](OpBuilder &, Location, Value, ValueRange) {});
  Block *coreBlock = coreOp.getBody();
  rewriter.eraseOp(coreBlock->getTerminator());
  Block *afterBlock = loop.getBody();
  rewriter.mergeBlocks(coreBlock, afterBlock);
  rewriter.setInsertionPoint(afterBlock, afterBlock->end());
  rewriter.create<scf::YieldOp>(rewriter.getUnknownLoc(), ValueRange{});
  Block *newBlock = rewriter.createBlock(&coreOp.getRegion());
  rewriter.setInsertionPoint(newBlock, newBlock->begin());
  auto endOp = rewriter.create<AMDAIE::EndOp>(rewriter.getUnknownLoc());
  rewriter.moveOpBefore(loop, endOp);
  return success();
}

class AMDAIEInsertInfiniteLoopAroundCoreBlockPass
    : public impl::AMDAIEInsertInfiniteLoopAroundCoreBlockBase<
          AMDAIEInsertInfiniteLoopAroundCoreBlockPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, AMDAIEDialect, scf::SCFDialect>();
  }
  void runOnOperation() override;
};

void AMDAIEInsertInfiniteLoopAroundCoreBlockPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp->getContext());
  SmallVector<AMDAIE::CoreOp> coreOps;
  parentOp->walk([&](AMDAIE::CoreOp coreOp) { coreOps.push_back(coreOp); });
  for (AMDAIE::CoreOp coreOp : coreOps) {
    if (failed(addDepthLoopAroundCoreCodeBlock(rewriter, coreOp)))
      return signalPassFailure();
    if (failed(addInfiniteWhileLoopAroundCoreBlock(rewriter, coreOp)))
      return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEInsertInfiniteLoopAroundCoreBlockPass() {
  return std::make_unique<AMDAIEInsertInfiniteLoopAroundCoreBlockPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
