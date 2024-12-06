// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"

#define DEBUG_TYPE "iree-amdaie-access-to-acquire-release"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Some blocks have terminator ops, which must appear as the very last op in
/// the block. If `block` has a terminator, set the insertion point of
/// `rewriter` to just before the terminator, ready to create a new penultimate
/// op in the block. Otherwise, set the insertion point to the very end of the
/// block.
void setInsertionToEnd(IRRewriter &rewriter, Block *block) {
  if (block->back().hasTrait<OpTrait::IsTerminator>()) {
    rewriter.setInsertionPoint(block->getTerminator());
  } else {
    rewriter.setInsertionPointToEnd(block);
  }
}

llvm::MapVector<Value, SmallVector<AMDAIE::LogicalObjectFifoAccessOp>>
getFifosToAccesses(AMDAIE::CoreOp coreOp, AMDAIE::MemoryAccess type) {
  llvm::MapVector<Value, SmallVector<AMDAIE::LogicalObjectFifoAccessOp>>
      accesses;
  coreOp->walk([&](AMDAIE::LogicalObjectFifoAccessOp accessOp) {
    if (accessOp.getAccessType() != type) return WalkResult::advance();
    Value input = accessOp.getInput();
    auto iter = accesses.find(input);
    if (iter == accesses.end()) {
      accesses.insert({input, {accessOp}});
    } else {
      iter->second.push_back(accessOp);
    }
    return WalkResult::advance();
  });
  return accesses;
}

/// Walk all read access operations within the core operations and insert
/// semaphore acquire and release stubs. Acquire operations will be inserted
/// at the location of the access operation, and release operations will be
/// inserted some time before the next read access.
LogicalResult readAccessToAcquireRelease(Operation *parentOp) {
  AMDAIE::MemoryAccess accessType = AMDAIE::MemoryAccess::Read;
  AMDAIE::LogicalObjectFifoPort port = LogicalObjectFifoPort::Consume;

  IRRewriter rewriter(parentOp->getContext());

  SmallVector<AMDAIE::CoreOp> coreOps;
  parentOp->walk([&](AMDAIE::CoreOp coreOp) { coreOps.push_back(coreOp); });

  // Map from the source and target amdaie.logicalobjectfifo values of
  // amdaie.connections to the amdaie.connections themselves.
  DenseMap<Value, AMDAIE::ConnectionOp> logicalObjectFifoToConnection;
  parentOp->walk([&](AMDAIE::ConnectionOp dmaOp) {
    logicalObjectFifoToConnection.insert({dmaOp.getSource(), dmaOp});
    logicalObjectFifoToConnection.insert({dmaOp.getTarget(), dmaOp});
  });

  for (AMDAIE::CoreOp coreOp : coreOps) {
    auto fifosToAccesses = getFifosToAccesses(coreOp, accessType);

    for (auto &&[logicalObjectFifo, accessOps] : fifosToAccesses) {
      for (uint64_t i = 0; i < accessOps.size(); ++i) {
        AMDAIE::LogicalObjectFifoAccessOp accessOp = accessOps[i];

        Value input = accessOp.getInput();
        if (!logicalObjectFifoToConnection.contains(input)) {
          return accessOp.emitOpError()
                 << "does not have a connection in the logicalobjectfifo map";
        }

        // Insert the access op.
        rewriter.setInsertionPoint(accessOp);
        Block *block = accessOp->getBlock();
        auto acquireOp = rewriter.create<AMDAIE::LogicalObjectFifoAcquire>(
            rewriter.getUnknownLoc(),
            llvm::cast<LogicalObjectFifoType>(input.getType()),
            logicalObjectFifoToConnection[input].getResult(), port);
        auto newAccessOp = rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
            rewriter.getUnknownLoc(), acquireOp.getResult(), accessType);
        rewriter.replaceAllUsesWith(accessOp.getResult(),
                                    newAccessOp.getResult());

        // Insert the release op. The location of the release is as close to the
        // following access op as possible, but always in the same block as the
        // access op being released.
        AMDAIE::LogicalObjectFifoAccessOp nextAccessOp;
        if (i + 1 != accessOps.size()) nextAccessOp = accessOps[i + 1];
        Operation *nextAccessOpsAncestor =
            getAncestorInBlock(nextAccessOp, block);
        if (nextAccessOpsAncestor &&
            nextAccessOpsAncestor->getBlock() == block) {
          rewriter.setInsertionPoint(nextAccessOpsAncestor);
        } else {
          setInsertionToEnd(rewriter, block);
        }
        rewriter.create<AMDAIE::LogicalObjectFifoRelease>(
            rewriter.getUnknownLoc(),
            logicalObjectFifoToConnection[input].getResult(), port);
      }
    }
  }
  return success();
}

/// Walk all write access operations within the core operations and insert
/// semaphore operations. Release operations will be inserted at the location of
/// the access operation and acquire operations will be inserted after the
/// preceding access or at the beginning of the block. TODO(newling): update
/// this to ensure that corresponding accesses and releases are in the same
/// block, as in the case of `readAccessToAcquireRelease`.
LogicalResult writeAccessToAcquireRelease(Operation *parentOp) {
  IRRewriter rewriter(parentOp->getContext());

  SmallVector<AMDAIE::CoreOp> coreOps;
  parentOp->walk([&](AMDAIE::CoreOp coreOp) { coreOps.push_back(coreOp); });

  // Map from DMA source/target logical objectFifos to those respective DMA
  // operations.
  DenseMap<Value, AMDAIE::ConnectionOp> logicalObjectFifoToDma;
  parentOp->walk([&](AMDAIE::ConnectionOp dmaOp) {
    logicalObjectFifoToDma[dmaOp.getSource()] = dmaOp;
    logicalObjectFifoToDma[dmaOp.getTarget()] = dmaOp;
  });

  for (AMDAIE::CoreOp coreOp : coreOps) {
    DenseMap<Value, SmallVector<AMDAIE::LogicalObjectFifoAccessOp>>
        logicalObjectFifoToAccesses;
    llvm::MapVector<Value, AMDAIE::LogicalObjectFifoAccessOp>
        logicalObjectFifoLastWriteAccesses;
    WalkResult res = coreOp->walk([&](AMDAIE::LogicalObjectFifoAccessOp
                                          accessOp) {
      // If there is another write op on the same logical objectFifo,
      // release it before the acquire.
      if (logicalObjectFifoLastWriteAccesses.contains(accessOp.getInput())) {
        AMDAIE::LogicalObjectFifoAccessOp prevAccess =
            logicalObjectFifoLastWriteAccesses[accessOp.getInput()];
        if (!logicalObjectFifoToDma.contains(prevAccess.getInput())) {
          prevAccess.emitOpError()
              << "write access not found as source of DMA operation";
          return WalkResult::interrupt();
        }
        rewriter.setInsertionPoint(accessOp);
        rewriter.create<AMDAIE::LogicalObjectFifoRelease>(
            rewriter.getUnknownLoc(),
            logicalObjectFifoToDma[prevAccess.getInput()],
            LogicalObjectFifoPort::Produce);
        // Remove from last access as settled.
        logicalObjectFifoLastWriteAccesses.erase(prevAccess.getInput());
      }
      // Insert acquire for write access at first `Any` access or before the
      // current access op.
      if (accessOp.getAccessType() == AMDAIE::MemoryAccess::Write) {
        if (!logicalObjectFifoToAccesses.contains(accessOp.getInput())) {
          rewriter.setInsertionPoint(accessOp);
        } else {
          AMDAIE::LogicalObjectFifoAccessOp firstAccess =
              logicalObjectFifoToAccesses[accessOp.getInput()][0];
          rewriter.setInsertionPoint(firstAccess);
        }
        if (!logicalObjectFifoToDma.contains(accessOp.getInput())) {
          accessOp.emitOpError()
              << "write access not found as source of DMA operation";
          return WalkResult::interrupt();
        }
        auto acquireOp = rewriter.create<AMDAIE::LogicalObjectFifoAcquire>(
            rewriter.getUnknownLoc(),
            llvm::cast<LogicalObjectFifoType>(accessOp.getInput().getType()),
            logicalObjectFifoToDma[accessOp.getInput()].getResult(),
            LogicalObjectFifoPort::Produce);
        auto newAccessOp = rewriter.create<AMDAIE::LogicalObjectFifoAccessOp>(
            rewriter.getUnknownLoc(), acquireOp.getResult(),
            AMDAIE::MemoryAccess::Write);

        // Update uses of this access operation and the preceding ones.
        rewriter.replaceAllUsesWith(accessOp.getResult(),
                                    newAccessOp.getResult());
        if (logicalObjectFifoToAccesses.contains(accessOp.getInput())) {
          for (AMDAIE::LogicalObjectFifoAccessOp precedingAccessOp :
               logicalObjectFifoToAccesses[accessOp.getInput()]) {
            rewriter.replaceAllUsesWith(precedingAccessOp.getResult(),
                                        newAccessOp.getResult());
          }
        }

        // Insert into last access map
        logicalObjectFifoLastWriteAccesses[accessOp.getInput()] = accessOp;
      }
      // Insert any access operation into first access map.
      if (!logicalObjectFifoToAccesses.contains(accessOp.getInput())) {
        logicalObjectFifoToAccesses[accessOp.getInput()] = {accessOp};
      } else {
        logicalObjectFifoToAccesses[accessOp.getInput()].push_back(accessOp);
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return failure();

    // Insert release for remaining access operations at end of block.
    for (auto &&[value, writeAccessOp] : logicalObjectFifoLastWriteAccesses) {
      Block *parentBlock = writeAccessOp->getBlock();
      if (!parentBlock->back().hasTrait<OpTrait::IsTerminator>()) {
        rewriter.setInsertionPointToEnd(parentBlock);
      } else {
        rewriter.setInsertionPoint(parentBlock->getTerminator());
      }
      if (!logicalObjectFifoToDma.contains(writeAccessOp.getInput())) {
        writeAccessOp.emitOpError()
            << "write access not found as source of DMA operation";
        return failure();
      }
      rewriter.create<AMDAIE::LogicalObjectFifoRelease>(
          rewriter.getUnknownLoc(),
          logicalObjectFifoToDma[writeAccessOp.getInput()],
          LogicalObjectFifoPort::Produce);
    }
  }
  return success();
}

class AMDAIEAccessToAcquireReleasePass
    : public impl::AMDAIEAccessToAcquireReleaseBase<
          AMDAIEAccessToAcquireReleasePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEAccessToAcquireReleasePass() = default;
  AMDAIEAccessToAcquireReleasePass(
      const AMDAIEAccessToAcquireReleasePass &pass){};
  void runOnOperation() override;
};

void AMDAIEAccessToAcquireReleasePass::runOnOperation() {
  Operation *parentOp = getOperation();
  if (failed(readAccessToAcquireRelease(parentOp))) {
    parentOp->emitOpError() << "failed to convert read access operations to "
                               "acquire-release semaphore stubs";
    return signalPassFailure();
  }

  if (failed(writeAccessToAcquireRelease(parentOp))) {
    parentOp->emitOpError() << "failed to convert write access operations to "
                               "acquire-release semaphore stubs";
    return signalPassFailure();
  }
  // Erase old access operations.
  IRRewriter rewriter(parentOp->getContext());
  parentOp->walk([&](AMDAIE::LogicalObjectFifoAccessOp accessOp) {
    if (accessOp->getUses().empty()) {
      rewriter.eraseOp(accessOp);
    }
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAccessToAcquireReleasePass() {
  return std::make_unique<AMDAIEAccessToAcquireReleasePass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
