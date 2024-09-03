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

/// Merge the 'source' and 'dest' core operations into a new `amdaie.core`
/// operation and combine the input and output DMAs.
AMDAIE::CoreOp CoreContext::mergeCoreOps(AMDAIE::CoreOp source,
                                         AMDAIE::CoreOp dest) {
  OpBuilder::InsertionGuard guard(rewriter);
  AMDAIE::TileOp tile = dest.getTileOp();
  SmallVector<Value> sourceInputDmas = source.getInputDmas();
  SmallVector<Value> destInputDmas = dest.getInputDmas();
  llvm::SmallSetVector<Value, 4> inputDmas(destInputDmas.begin(),
                                           destInputDmas.end());
  inputDmas.insert(sourceInputDmas.begin(), sourceInputDmas.end());
  SmallVector<Value> sourceOutputDmas = source.getOutputDmas();
  SmallVector<Value> destOutputDmas = dest.getOutputDmas();
  llvm::SmallSetVector<Value, 4> outputDmas(destOutputDmas.begin(),
                                            destOutputDmas.end());
  outputDmas.insert(sourceOutputDmas.begin(), sourceOutputDmas.end());
  rewriter.setInsertionPoint(source);
  auto newCoreOp = rewriter.create<AMDAIE::CoreOp>(rewriter.getUnknownLoc(),
                                                   tile, inputDmas.takeVector(),
                                                   outputDmas.takeVector());
  Region &region = newCoreOp.getRegion();
  Block *newBlock = rewriter.createBlock(&region);
  rewriter.setInsertionPointToStart(newBlock);
  rewriter.eraseOp(dest.getBody()->getTerminator());
  rewriter.mergeBlocks(dest.getBody(), newBlock);
  rewriter.mergeBlocks(source.getBody(), newBlock);
  rewriter.eraseOp(dest);
  rewriter.eraseOp(source);
  return newCoreOp;
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
LogicalResult WorkgroupBuilder::buildForCircularDmaCpyNdOp(
    AMDAIE::CircularDmaCpyNdOp dmaOp, Block *target, Block *controlCode,
    CoreContext &coreContext, Block::iterator targetBegin,
    Block::iterator controlCodeBegin, Block::iterator controlCodeEnd) {
  LLVM_DEBUG(
      llvm::dbgs() << "workgroupBuild [amdaie.circular_dma_cpy_nd] Start\n");
  OpBuilder::InsertionGuard workgroupGuard(rewriter);
  OpBuilder::InsertionGuard controlCodeGuard(controlCodeRewriter);
  SmallVector<OpFoldResult> empty;
  auto connectionOp = rewriter.createAndMap<AMDAIE::ConnectionOp>(
      rewriter.getUnknownLoc(), dmaOp, dmaOp.getTarget(), dmaOp.getSource());
  controlCodeRewriter.setInsertionPoint(controlCode, controlCodeEnd);
  controlCodeRewriter.createAndLookup<AMDAIE::NpuCircularDmaCpyNdOp>(
      rewriter.getUnknownLoc(), connectionOp.getResult(),
      dmaOp.getTargetMixedOffsets(), dmaOp.getTargetMixedSizes(),
      dmaOp.getTargetMixedStrides(), dmaOp.getSourceMixedOffsets(),
      dmaOp.getSourceMixedSizes(), dmaOp.getSourceMixedStrides());
  LLVM_DEBUG(
      llvm::dbgs() << "workgroupBuild [amdaie.circular_dma_cpy_nd] End\n");
  return success();
}

/// DmaCpyNd operations are converted into CircularDmaCpyNd operations by moving
/// the strided access specifiers to an npu dma instruction, followed by a wait.
LogicalResult WorkgroupBuilder::buildForDmaCpyNdOp(
    AMDAIE::DmaCpyNdOp dmaOp, Block *target, Block *controlCode,
    CoreContext &coreContext, Block::iterator targetBegin,
    Block::iterator controlCodeBegin, Block::iterator controlCodeEnd) {
  LLVM_DEBUG(llvm::dbgs() << "workgroupBuild [amdaie.dma_cpy_nd] Start\n");
  Attribute sourceMemSpace = dmaOp.getSourceObjectFifo().getMemorySpace();
  Attribute targetMemSpace = dmaOp.getTargetObjectFifo().getMemorySpace();
  // Error out if the DmaCpyNd involves transfer between L1/L2 as these are all
  // circular_dma_cpy_nd operations by this stage.
  if (sourceMemSpace && targetMemSpace) {
    dmaOp.emitError()
        << "neither source nor target of the DmaCpyNd op is on L3";
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

  SmallVector<OpFoldResult> npuDmaTargetOffsets = dmaOp.getTargetMixedOffsets();
  SmallVector<OpFoldResult> npuDmaTargetSizes = dmaOp.getTargetMixedSizes();
  SmallVector<OpFoldResult> npuDmaTargetStrides = dmaOp.getTargetMixedStrides();
  SmallVector<OpFoldResult> npuDmaSourceOffsets = dmaOp.getSourceMixedOffsets();
  SmallVector<OpFoldResult> npuDmaSourceSizes = dmaOp.getSourceMixedSizes();
  SmallVector<OpFoldResult> npuDmaSourceStrides = dmaOp.getSourceMixedStrides();
  Value circularDmaTarget, circularDmaSource, npuDmaTarget, npuDmaSource;
  if (!sourceMemSpace) {
    // Check if the source of DmaCpyNd op is from L3 - then source addressing
    // will be controlled by the uController and target addressing will stay in
    // the circular DMA to be part of the AIE configuration.
    auto logicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
            dmaOp.getSource().getDefiningOp());
    if (!logicalObjFifo) {
      return dmaOp.emitOpError()
             << "`amdaie.logicalobjectfifo.from_memref` expected as source";
    }
    auto type = cast<LogicalObjectFifoType>(dmaOp.getSource().getType());
    auto placeholder =
        rewriter.createAndLookup<AMDAIE::LogicalObjectFifoPlaceholderOp>(
            rewriter.getUnknownLoc(), type, logicalObjFifo.getTiles());
    circularDmaSource = placeholder.getResult();
    circularDmaTarget = dmaOp.getTarget();
    circularDmaTargetOffsets = npuDmaTargetOffsets;
    circularDmaTargetSizes = npuDmaTargetSizes;
    circularDmaTargetStrides = npuDmaTargetStrides;

    npuDmaSource = dmaOp.getSource();
    npuDmaTargetOffsets = empty;
    npuDmaTargetSizes = empty;
    npuDmaTargetStrides = empty;
  } else if (!targetMemSpace) {
    // Check if the target of DmaCpyNd op is from L3 - then target addressing
    // will be controlled by the uController and source addressing will stay in
    // the circular DMA to be part of the AIE configuration.
    auto logicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
            dmaOp.getTarget().getDefiningOp());
    if (!logicalObjFifo) {
      return dmaOp.emitOpError()
             << "`amdaie.logicalobjectfifo.from_memref` expected as source";
    }
    auto type = cast<LogicalObjectFifoType>(dmaOp.getTarget().getType());
    auto placeholder =
        rewriter.createAndLookup<AMDAIE::LogicalObjectFifoPlaceholderOp>(
            rewriter.getUnknownLoc(), type, logicalObjFifo.getTiles());
    circularDmaSource = dmaOp.getSource();
    circularDmaTarget = placeholder.getResult();
    circularDmaSourceOffsets = npuDmaSourceOffsets;
    circularDmaSourceSizes = npuDmaSourceSizes;
    circularDmaSourceStrides = npuDmaSourceStrides;

    npuDmaTarget = dmaOp.getTarget();
    npuDmaSourceOffsets = empty;
    npuDmaSourceSizes = empty;
    npuDmaSourceStrides = empty;
  }
  auto connectionOp = rewriter.createAndMap<AMDAIE::ConnectionOp>(
      rewriter.getUnknownLoc(), dmaOp, circularDmaTarget, circularDmaSource);

  IRRewriter::InsertPoint dmaInsertionPoint = rewriter.saveInsertionPoint();
  controlCodeRewriter.setInsertionPoint(controlCode, controlCodeEnd);
  controlCodeRewriter.createAndLookup<AMDAIE::NpuCircularDmaCpyNdOp>(
      rewriter.getUnknownLoc(), connectionOp.getResult(),
      circularDmaTargetOffsets, circularDmaTargetSizes,
      circularDmaTargetStrides, circularDmaSourceOffsets,
      circularDmaSourceSizes, circularDmaSourceStrides);
  auto npuDmaCpy = controlCodeRewriter.createAndLookup<AMDAIE::NpuDmaCpyNdOp>(
      loc, connectionOp.getResult(), npuDmaTarget, npuDmaTargetOffsets,
      npuDmaTargetSizes, npuDmaTargetStrides, /*target_bd_id=*/nullptr,
      npuDmaSource, npuDmaSourceOffsets, npuDmaSourceSizes, npuDmaSourceStrides,
      /*source_bd_id=*/nullptr);
  DMAChannelDir direction =
      !sourceMemSpace ? DMAChannelDir::MM2S : DMAChannelDir::S2MM;
  controlCodeRewriter.createAndLookup<AMDAIE::NpuDmaWaitOp>(
      rewriter.getUnknownLoc(), SmallVector<Type, 1>{}, npuDmaCpy.getResult(),
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
LogicalResult WorkgroupBuilder::buildForSingleBody(
    OpTy op, Block *target, Block *controlCode, CoreContext &coreContext,
    Block::iterator targetBegin, Block::iterator controlCodeBegin,
    Block::iterator controlCodeEnd) {
  LLVM_DEBUG(llvm::dbgs() << "workgroupBuild [" << OpTy::getOperationName()
                          << "] Start\n");
  controlCodeRewriter.setInsertionPoint(controlCode, controlCodeEnd);
  FailureOr<OpTy> maybeControlCodeOp =
      createNewLoopOp<CreateAndMapFunctor, OpTy>(controlCodeRewriter, op);
  if (failed(maybeControlCodeOp)) {
    return op.emitOpError("failed to create a new loop");
  }
  OpTy newControlCodeForOp = maybeControlCodeOp.value();

  // Create a new core map and control code block for visiting the nested ops.
  CoreContext nestedCoreContext(rewriter);
  Block *nestedControlCode = rewriter.createBlock(controlCode->getParent());
  if (failed(build(op.getBody(), target, nestedControlCode, nestedCoreContext,
                   op.getBody()->begin(), std::prev(op.getBody()->end()),
                   target->end(), nestedControlCode->begin(),
                   nestedControlCode->end()))) {
    return op.emitOpError() << "failed to add scf.for body to workgroup";
  }

  // Create a new scf.for for every nested core and insert into the core
  // op around all existing ops, except for the terminator.
  for (auto &&[coordinate, coreOp] : nestedCoreContext.getCoreMap()) {
    FailureOr<OpTy> maybeOp =
        createNewLoopOp<CreateAndLookupFunctor, OpTy>(rewriter, op);
    if (failed(maybeOp)) {
      return op.emitOpError("failed to create a new loop");
    }
    auto newOp = maybeOp.value();
    Block::iterator insertIt = newOp.getBody()->begin();
    Block::iterator coreBegin = coreOp.getBody()->begin();
    Block::iterator coreEnd = coreOp.getBody()->getTerminator()->getIterator();
    newOp.getBody()->getOperations().splice(
        insertIt, coreOp.getBody()->getOperations(), coreBegin, coreEnd);
    rewriter.moveOpBefore(newOp, coreOp.getBody()->getTerminator());
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

/// Recursive workgroup build function for an operation.
LogicalResult WorkgroupBuilder::build(Operation *op, Block *target,
                                      Block *controlCode,
                                      CoreContext &coreContext,
                                      Block::iterator targetBegin,
                                      Block::iterator controlCodeBegin,
                                      Block::iterator controlCodeEnd) {
  OpBuilder::InsertionGuard guard(rewriter);
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<AMDAIE::CoreOp>([&](auto coreOp) {
        return workgroupBuildForCoreOp(rewriter, coreOp, target, controlCode,
                                       coreContext, targetBegin,
                                       controlCodeBegin, controlCodeEnd);
      })
      .Case<AMDAIE::CircularDmaCpyNdOp>([&](auto dmaOp) {
        return buildForCircularDmaCpyNdOp(dmaOp, target, controlCode,
                                          coreContext, targetBegin,
                                          controlCodeBegin, controlCodeEnd);
      })
      .Case<AMDAIE::DmaCpyNdOp>([&](auto dmaOp) {
        return buildForDmaCpyNdOp(dmaOp, target, controlCode, coreContext,
                                  targetBegin, controlCodeBegin,
                                  controlCodeEnd);
      })
      .Case<scf::ForallOp>([&](auto forallOp) {
        return buildForSingleBody<scf::ForallOp>(
            forallOp, target, controlCode, coreContext, targetBegin,
            controlCodeBegin, controlCodeEnd);
      })
      .Case<scf::ForOp>([&](auto forOp) {
        return buildForSingleBody<scf::ForOp>(forOp, target, controlCode,
                                              coreContext, targetBegin,
                                              controlCodeBegin, controlCodeEnd);
      })
      .Case<AMDAIE::WorkgroupOp>([&](auto workgroupOp) {
        return workgroupOp.emitOpError()
               << "not supported in `amdaie.workgroup` creation";
      })
      .Default([&](Operation *) {
        // All other operations are cloned.
        // Case 1: Try to clone in workgroup.
        if (llvm::all_of(op->getOperands(), [&](Value operand) {
              return rewriter.contains(operand);
            })) {
          rewriter.cloneAndMap(*op);
        }
        // Case 2: Try to clone in controlcode.
        if (llvm::all_of(op->getOperands(), [&](Value operand) {
              return controlCodeRewriter.contains(operand);
            })) {
          controlCodeRewriter.setInsertionPoint(controlCode, controlCodeEnd);
          controlCodeRewriter.cloneAndMap(*op);
        }
        // TODO(avarma): Case 3: Try to clone in core.
        return success();
      });
  return success();
}

/// Recursive workgroup build function for a block with a provided source and
/// end point.
LogicalResult WorkgroupBuilder::build(
    Block *source, Block *target, Block *controlCode, CoreContext &coreContext,
    Block::iterator sourceBegin, Block::iterator sourceEnd,
    Block::iterator targetBegin, Block::iterator controlCodeBegin,
    Block::iterator controlCodeEnd) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(target, targetBegin);
  for (Block::iterator it = sourceBegin; it != sourceEnd; ++it) {
    OpBuilder::InsertionGuard guard(rewriter);
    if (failed(build(&(*it), target, controlCode, coreContext, targetBegin,
                     controlCodeBegin, controlCodeEnd))) {
      return failure();
    }
  }
  return success();
}

namespace {

/// Traverse the function operation and create a single workgroup and control
/// code.
LogicalResult createSingleWorkgroupAndControlCode(func::FuncOp funcOp) {
  // Skip processing Ukernel function declarations which will be marked private.
  if (funcOp.isPrivate()) {
    return success();
  }
  IRRewriterAndMapper rewriter(funcOp.getContext());
  IRRewriterAndMapper controlCodeRewriter(funcOp.getContext());
  Block *funcBlock = &funcOp.getBody().front();
  Block *newBlock = rewriter.createBlock(&funcOp.getRegion());

  // Create an idempotent mapping of FuncOp's blockArgument with itself.
  // The reason to do this is -> while building the workgroup of a function
  // we would be cloning an operation to :-
  // 1. Workgroup body.
  // 2. Controlcode.
  // 3. Core op (TODO(avarma): Not implemented yet).
  // Each of these cloning would be dependent on the respective IRMapper.
  // Specifically we would be checking if the IRMapper has the necessary operand
  // values to perform a successful clone.
  for (Value val : funcBlock->getArguments()) {
    rewriter.map(val, val);
    controlCodeRewriter.map(val, val);
  }

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
  WorkgroupBuilder builder(rewriter, controlCodeRewriter);
  if (failed(builder.build(funcBlock, newWorkgroupBlock, controlCodeBlock,
                           coreContext, funcBlock->begin(),
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
