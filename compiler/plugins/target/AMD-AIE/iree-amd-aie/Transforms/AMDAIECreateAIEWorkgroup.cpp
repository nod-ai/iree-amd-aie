// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIECreateAIEWorkgroup.h"

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-amdaie-create-aie-workgroup"

namespace mlir::iree_compiler::AMDAIE {

/// Merge the 'source' core operations in the end of the 'dest' core operation.
void CoreContext::mergeCoreOps(AMDAIE::CoreOp source, AMDAIE::CoreOp dest) {
  OpBuilder::InsertionGuard guard(rewriter);
  Block::iterator insertIt = dest.getBody()->getTerminator()->getIterator();
  Block::iterator sourceBegin = source.getBody()->begin();
  Block::iterator sourceEnd = source.getBody()->getTerminator()->getIterator();
  dest.getBody()->getOperations().splice(
      insertIt, source.getBody()->getOperations(), sourceBegin, sourceEnd);
  rewriter.moveOpBefore(dest, source);
  rewriter.replaceOp(source, dest);
}

/// Clone CoreOp and add to or merge with coreContext.
LogicalResult workgroupBuildForCoreOp(
    IRRewriterAndMapper &rewriter, AMDAIE::CoreOp coreOp, Block *target,
    Block *controlCode, CoreContext &coreContext, Block::iterator targetBegin,
    Block::iterator controlCodeBegin, Block::iterator controlCodeEnd) {
  LLVM_DEBUG(llvm::dbgs() << "workgroupBuild [amdaie.core] Start\n");
  OpBuilder::InsertionGuard guard(rewriter);
  int64_t col = getConstantIntValue(coreOp.getTileOp().getCol()).value();
  int64_t row = getConstantIntValue(coreOp.getTileOp().getRow()).value();
  std::tuple<int64_t, int64_t> coordinate = std::make_tuple(col, row);
  auto cloneCoreOp =
      dyn_cast<AMDAIE::CoreOp>(rewriter.cloneAndMap(*coreOp.getOperation()));
  coreContext.mapOrMerge(coordinate, cloneCoreOp);
  LLVM_DEBUG(llvm::dbgs() << "workgroupBuild [amdaie.core] End\n");
  return success();
}

/// CircularDmaCpyNd operations are just cloned and mapped as they run
/// indefinitely and only need to be programmed once.
LogicalResult workgroupBuildForCircularDmaCpyNdOp(
    IRRewriterAndMapper &rewriter, AMDAIE::CircularDmaCpyNdOp dmaOp,
    Block *target, Block *controlCode, CoreContext &coreContext,
    Block::iterator targetBegin, Block::iterator controlCodeBegin,
    Block::iterator controlCodeEnd) {
  LLVM_DEBUG(
      llvm::dbgs() << "workgroupBuild [amdaie.circular_dma_cpy_nd] Start\n");
  rewriter.cloneAndMap(*dmaOp.getOperation());
  LLVM_DEBUG(
      llvm::dbgs() << "workgroupBuild [amdaie.circular_dma_cpy_nd] End\n");
  return success();
}

/// DmaCpyNd operations are converted into CircularDmaCpyNd operations by moving
/// the strided access specifiers to an npu dma instruction, followed by a wait.
LogicalResult workgroupBuildForDmaCpyNdOp(
    IRRewriterAndMapper &rewriter, AMDAIE::DmaCpyNdOp dmaOp, Block *target,
    Block *controlCode, CoreContext &coreContext, Block::iterator targetBegin,
    Block::iterator controlCodeBegin, Block::iterator controlCodeEnd) {
  LLVM_DEBUG(llvm::dbgs() << "workgroupBuild [amdaie.dma_cpy_nd] Start\n");
  Attribute sourceMemSpace = dmaOp.getSourceObjectFifo().getMemorySpace();
  Attribute targetMemSpace = dmaOp.getTargetObjectFifo().getMemorySpace();
  if (sourceMemSpace && targetMemSpace) {
    dmaOp.emitOpError()
        << "neither source nor target of the DmaNdCpy op is on L3";
    return failure();
  }
  Location loc = rewriter.getUnknownLoc();

  SmallVector<OpFoldResult> empty;

  SmallVector<OpFoldResult> circularDmaTargetOffsets;
  SmallVector<OpFoldResult> circularDmaTargetSizes;
  SmallVector<OpFoldResult> circularDmaTargetStrides;
  SmallVector<OpFoldResult> circularDmaSourceOffsets;
  SmallVector<OpFoldResult> circularDmaSourceSizes;
  SmallVector<OpFoldResult> circularDmaSourceStrides;

  SmallVector<OpFoldResult> ipuDmaTargetOffsets = dmaOp.getTargetMixedOffsets();
  SmallVector<OpFoldResult> ipuDmaTargetSizes = dmaOp.getTargetMixedSizes();
  SmallVector<OpFoldResult> ipuDmaTargetStrides = dmaOp.getTargetMixedStrides();
  SmallVector<OpFoldResult> ipuDmaSourceOffsets = dmaOp.getSourceMixedOffsets();
  SmallVector<OpFoldResult> ipuDmaSourceSizes = dmaOp.getSourceMixedSizes();
  SmallVector<OpFoldResult> ipuDmaSourceStrides = dmaOp.getSourceMixedStrides();
  if (!sourceMemSpace) {
    // Check if the source of DmaCpyNd op is from L3 - then source addressing
    // will be controlled by the uController and target addressing will stay in
    // the circular DMA to be part of the AIE configuration.
    circularDmaTargetOffsets = ipuDmaTargetOffsets;
    circularDmaTargetSizes = ipuDmaTargetSizes;
    circularDmaTargetStrides = ipuDmaTargetStrides;

    ipuDmaTargetOffsets = empty;
    ipuDmaTargetSizes = empty;
    ipuDmaTargetStrides = empty;
  } else if (!targetMemSpace) {
    // Check if the target of DmaCpyNd op is from L3 - then target addressing
    // will be controlled by the uController and source addressing will stay in
    // the circular DMA to be part of the AIE configuration.
    circularDmaSourceOffsets = ipuDmaSourceOffsets;
    circularDmaSourceSizes = ipuDmaSourceSizes;
    circularDmaSourceStrides = ipuDmaSourceStrides;

    ipuDmaSourceOffsets = empty;
    ipuDmaSourceSizes = empty;
    ipuDmaSourceStrides = empty;
  }
  // In case of transfers between L1/L2 - the addressing is always controlled by
  // the uController.
  auto newDmaOp = rewriter.createAndMap<AMDAIE::CircularDmaCpyNdOp>(
      rewriter.getUnknownLoc(), dmaOp, dmaOp.getTarget(),
      circularDmaTargetOffsets, circularDmaTargetSizes,
      circularDmaTargetStrides, dmaOp.getSource(), circularDmaSourceOffsets,
      circularDmaSourceSizes, circularDmaSourceStrides);

  IRRewriter::InsertPoint dmaInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(controlCode, controlCodeEnd);
  auto ipuDmaCpy = rewriter.createAndLookup<AMDAIE::NpuDmaCpyNdOp>(
      loc, newDmaOp.getResult(), ipuDmaTargetOffsets, ipuDmaTargetSizes,
      ipuDmaTargetStrides, ipuDmaSourceOffsets, ipuDmaSourceSizes,
      ipuDmaSourceStrides);
  DMAChannelDir direction =
      !sourceMemSpace ? DMAChannelDir::MM2S : DMAChannelDir::S2MM;
  rewriter.createAndLookup<AMDAIE::NpuDmaWaitOp>(
      rewriter.getUnknownLoc(), SmallVector<Type, 1>{}, ipuDmaCpy.getResult(),
      direction);
  rewriter.restoreInsertionPoint(dmaInsertionPoint);
  LLVM_DEBUG(llvm::dbgs() << "workgroupBuild [amdaie.dma_cpy_nd] End\n");
  return success();
}

/// Create a new loop of type `LoopType` from the provided `loopOp`. A
/// `TFunctor` template parameter specifies the Functor to be used for creating
/// the operation.
template <class TFunctor, typename OpTy>
FailureOr<OpTy> createNewLoopOp(IRRewriterAndMapper &rewriter, OpTy loopOp) {
  return rewriter.notifyMatchFailure(loopOp, "unhandled loop type");
}

/// The implementation of `createNewLoopOp` for `scf.for`.
template <class TFunctor, typename OpTy,
          std::enable_if_t<std::is_same<OpTy, scf::ForOp>::value, bool> = true>
FailureOr<OpTy> createNewLoopOp(IRRewriterAndMapper &rewriter,
                                scf::ForOp forOp) {
  return createOp<TFunctor, scf::ForOp>(
      rewriter, forOp.getLoc(), forOp, forOp.getLowerBound(),
      forOp.getUpperBound(), forOp.getStep(), forOp.getInits());
}

/// The implementation of `createNewLoopOp` for `scf.forall`.
template <
    class TFunctor, typename OpTy,
    std::enable_if_t<std::is_same<OpTy, scf::ForallOp>::value, bool> = true>
FailureOr<OpTy> createNewLoopOp(IRRewriterAndMapper &rewriter,
                                scf::ForallOp forallOp) {
  return createOp<TFunctor, scf::ForallOp>(
      rewriter, forallOp.getLoc(), forallOp, forallOp.getMixedLowerBound(),
      forallOp.getMixedUpperBound(), forallOp.getMixedStep(),
      forallOp.getOutputs(), forallOp.getMapping());
}

/// Recursively build operations with a single body. These operations define a
/// `getBody` method to retrieve the inner body block. This function will
/// recursively visit this body block and use it to continue building the
/// workgroup and control code blocks. Afterwards, the following insertions will
/// be done:
///   1. Create a new operation from the provided operation of type `OpTy` and
///   insert it in each nested core around the existing core body.
///   2. Create a new operation from the provided operation of type `OpTy` and
///   insert it in the control code block around the existing control code body.
template <typename OpTy>
LogicalResult workgroupBuildForSingleBody(
    IRRewriterAndMapper &rewriter, OpTy op, Block *target, Block *controlCode,
    CoreContext &coreContext, Block::iterator targetBegin,
    Block::iterator controlCodeBegin, Block::iterator controlCodeEnd) {
  LLVM_DEBUG(llvm::dbgs() << "workgroupBuild [" << OpTy::getOperationName()
                          << "] Start\n");
  FailureOr<OpTy> maybeControlCodeOp =
      createNewLoopOp<CreateAndMapFunctor, OpTy>(rewriter, op);
  if (failed(maybeControlCodeOp)) {
    return op.emitOpError("failed to create a new loop");
  }
  OpTy newControlCodeForOp = maybeControlCodeOp.value();

  // Create a new core map and control code block for visiting the nested ops.
  CoreContext nestedCoreContext(rewriter);
  Block *nestedControlCode = rewriter.createBlock(controlCode->getParent());
  if (failed(workgroupBuild(
          rewriter, op.getBody(), target, nestedControlCode, nestedCoreContext,
          op.getBody()->begin(), std::prev(op.getBody()->end()), target->end(),
          nestedControlCode->begin(), nestedControlCode->end()))) {
    return op.emitOpError() << "failed to add scf.for body to workgroup";
  }

  // Create a new scf.for for every nested core and insert into the core
  // op around all existing ops, except for the terminator.
  for (auto &&[coordinate, coreOp] : nestedCoreContext.getCoreMap()) {
    FailureOr<OpTy> maybeCoreOp =
        createNewLoopOp<CreateAndLookupFunctor, OpTy>(rewriter, op);
    if (failed(maybeCoreOp)) {
      return op.emitOpError("failed to create a new loop");
    }
    auto newCoreOp = maybeCoreOp.value();
    Block::iterator insertIt = newCoreOp.getBody()->begin();
    Block::iterator coreBegin = coreOp.getBody()->begin();
    Block::iterator coreEnd = coreOp.getBody()->getTerminator()->getIterator();
    newCoreOp.getBody()->getOperations().splice(
        insertIt, coreOp.getBody()->getOperations(), coreBegin, coreEnd);
    rewriter.moveOpBefore(newCoreOp, coreOp.getBody()->getTerminator());
  }
  coreContext.mergeContext(nestedCoreContext);

  // Inline the nested control code within the external control code.
  rewriter.inlineBlockBefore(nestedControlCode,
                             newControlCodeForOp.getBody()->getTerminator());
  rewriter.moveOpBefore(newControlCodeForOp, controlCode, controlCodeEnd);
  LLVM_DEBUG(llvm::dbgs() << "workgroupBuild [" << OpTy::getOperationName()
                          << "] End\n");
  return success();
}

/// Skip workgroup operations and just build their bodies.
/// TODO(jornt): Get rid of the insertion of workgroups before this pass.
LogicalResult workgroupBuildForWorkgroupOp(IRRewriterAndMapper &rewriter,
                                           AMDAIE::WorkgroupOp workgroupOp,
                                           Block *target, Block *controlCode,
                                           CoreContext &coreContext,
                                           Block::iterator targetBegin,
                                           Block::iterator controlCodeBegin,
                                           Block::iterator controlCodeEnd) {
  LLVM_DEBUG(llvm::dbgs() << "workgroupBuild [amdaie.workgroup] Start\n");
  if (failed(workgroupBuild(rewriter, workgroupOp.getBody(), target,
                            controlCode, coreContext,
                            workgroupOp.getBody()->begin(),
                            std::prev(workgroupOp.getBody()->end()),
                            target->end(), controlCodeBegin, controlCodeEnd))) {
    return workgroupOp.emitOpError()
           << "failed to add workgroup body to workgroup";
  }
  LLVM_DEBUG(llvm::dbgs() << "workgroupBuild [amdaie.workgroup] End\n");
  return success();
}

/// Recursive workgroup build function for an operation.
LogicalResult workgroupBuild(IRRewriterAndMapper &rewriter, Operation *op,
                             Block *target, Block *controlCode,
                             CoreContext &coreContext,
                             Block::iterator targetBegin,
                             Block::iterator controlCodeBegin,
                             Block::iterator controlCodeEnd) {
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<AMDAIE::CoreOp>([&](auto coreOp) {
        return workgroupBuildForCoreOp(rewriter, coreOp, target, controlCode,
                                       coreContext, targetBegin,
                                       controlCodeBegin, controlCodeEnd);
      })
      .Case<AMDAIE::CircularDmaCpyNdOp>([&](auto dmaOp) {
        return workgroupBuildForCircularDmaCpyNdOp(
            rewriter, dmaOp, target, controlCode, coreContext, targetBegin,
            controlCodeBegin, controlCodeEnd);
      })
      .Case<AMDAIE::DmaCpyNdOp>([&](auto dmaOp) {
        return workgroupBuildForDmaCpyNdOp(rewriter, dmaOp, target, controlCode,
                                           coreContext, targetBegin,
                                           controlCodeBegin, controlCodeEnd);
      })
      .Case<scf::ForallOp>([&](auto forallOp) {
        return workgroupBuildForSingleBody<scf::ForallOp>(
            rewriter, forallOp, target, controlCode, coreContext, targetBegin,
            controlCodeBegin, controlCodeEnd);
      })
      .Case<scf::ForOp>([&](auto forOp) {
        return workgroupBuildForSingleBody<scf::ForOp>(
            rewriter, forOp, target, controlCode, coreContext, targetBegin,
            controlCodeBegin, controlCodeEnd);
      })
      .Case<AMDAIE::WorkgroupOp>([&](auto workgroupOp) {
        return workgroupBuildForWorkgroupOp(
            rewriter, workgroupOp, target, controlCode, coreContext,
            targetBegin, controlCodeBegin, controlCodeEnd);
      })
      .Default([&](Operation *) {
        // All other operations are cloned.
        rewriter.cloneAndMap(*op);
        return success();
      });
  return success();
}

/// Recursive workgroup build function for a block with a provided source and
/// end point.
LogicalResult workgroupBuild(
    IRRewriterAndMapper &rewriter, Block *source, Block *target,
    Block *controlCode, CoreContext &coreContext, Block::iterator sourceBegin,
    Block::iterator sourceEnd, Block::iterator targetBegin,
    Block::iterator controlCodeBegin, Block::iterator controlCodeEnd) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(target, targetBegin);
  for (Block::iterator it = sourceBegin; it != sourceEnd; ++it) {
    OpBuilder::InsertionGuard guard(rewriter);
    if (failed(workgroupBuild(rewriter, &(*it), target, controlCode,
                              coreContext, targetBegin, controlCodeBegin,
                              controlCodeEnd))) {
      return failure();
    }
  }
  return success();
}

namespace {

/// Traverse the function operation and create a single workgroup and control
/// code.
LogicalResult createSingleWorkgroupAndControlCode(func::FuncOp funcOp) {
  IRRewriterAndMapper rewriter(funcOp.getContext());
  Block *funcBlock = &funcOp.getBody().front();
  Block *newBlock = rewriter.createBlock(&funcOp.getRegion());

  // Create the workgroup op to be filled in with AIE DMAs, cores and the
  // control code.
  rewriter.setInsertionPoint(newBlock, newBlock->begin());
  auto workgroupOp =
      rewriter.create<AMDAIE::WorkgroupOp>(rewriter.getUnknownLoc());
  Block *newWorkgroupBlock = rewriter.createBlock(&workgroupOp.getRegion());
  Block *controlCodeBlock = workgroupOp.getControlCode().getBody();
  Block::iterator controlCodeEnd =
      controlCodeBlock->getTerminator()->getIterator();

  // Recursively build the workgroup and control code.
  CoreContext coreContext(rewriter);
  if (failed(workgroupBuild(rewriter, funcBlock, newWorkgroupBlock,
                            controlCodeBlock, coreContext, funcBlock->begin(),
                            std::prev(funcBlock->end()),
                            newWorkgroupBlock->begin(),
                            controlCodeBlock->begin(), controlCodeEnd))) {
    return failure();
  }

  // Inline the workgroup at the start of the FuncOp and erase the previous
  // block's operations.
  rewriter.inlineBlockBefore(newWorkgroupBlock, workgroupOp.getControlCode());
  rewriter.moveOpBefore(funcBlock->getTerminator(), newBlock, newBlock->end());
  for (Operation &op : llvm::make_early_inc_range(llvm::reverse(*funcBlock))) {
    assert(op.use_empty() && "expected 'op' to have no uses");
    rewriter.eraseOp(&op);
  }
  rewriter.inlineBlockBefore(newBlock, funcBlock, funcBlock->begin());
  return success();
}

class AMDAIECreateAIEWorkgroupPass
    : public impl::AMDAIECreateAIEWorkgroupBase<AMDAIECreateAIEWorkgroupPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, linalg::LinalgDialect, scf::SCFDialect>();
  }

  AMDAIECreateAIEWorkgroupPass() = default;
  AMDAIECreateAIEWorkgroupPass(const AMDAIECreateAIEWorkgroupPass &pass){};
  void runOnOperation() override;
};

void AMDAIECreateAIEWorkgroupPass::runOnOperation() {
  if (failed(createSingleWorkgroupAndControlCode(getOperation()))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIECreateAIEWorkgroupPass() {
  return std::make_unique<AMDAIECreateAIEWorkgroupPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
