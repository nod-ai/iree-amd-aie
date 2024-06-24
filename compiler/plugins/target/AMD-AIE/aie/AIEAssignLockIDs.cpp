// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This pass aims to assign lockIDs to AIE.lock operations. The lockID is
// numbered from the most recent AIE.lock within the same tile. If the lockID
// exceeds the number of locks on the tile, the pass generates an error and
// terminates. AIE.lock operations for different tiles are numbered
// independently. If there are existing lock IDs, this pass is idempotent
// and only assigns lock IDs to locks without an ID.

#include "Passes.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "aie-assign-lock-ids"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define GEN_PASS_DECL_AIEASSIGNLOCKIDS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
#undef GEN_PASS_DECL_AIEASSIGNLOCKIDS

#define GEN_PASS_DEF_AIEASSIGNLOCKIDS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
#undef GEN_PASS_DEF_AIEASSIGNLOCKIDS

namespace mlir::iree_compiler::AMDAIE {
struct AIEAssignLockIDsPass
    : ::impl::AIEAssignLockIDsBase<AIEAssignLockIDsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder rewriter = OpBuilder::atBlockEnd(device.getBody());

    // All of the lock ops on a tile, separated into ops which have been
    // assigned to a lock, and ops which have not.
    struct TileLockOps {
      DenseSet<int> assigned;
      SmallVector<LockOp> unassigned;
    };

    DenseMap<TileOp, TileLockOps> tileToLocks;

    // Construct data structure storing locks by tile.
    device.walk<WalkOrder::PreOrder>([&](LockOp lockOp) {
      TileOp tileOp = lockOp.getTileOp();
      if (lockOp.getLockID().has_value()) {
        auto lockID = lockOp.getLockID().value();
        auto iter = tileToLocks.find(tileOp);
        if (iter == tileToLocks.end())
          tileToLocks.insert({tileOp, {{lockID}, /* unassigned = */ {}}});
        else {
          if (iter->second.assigned.find(lockID) !=
              iter->second.assigned.end()) {
            auto diag = lockOp->emitOpError("is assigned to the same lock (")
                        << lockID << ") as another op.";
            diag.attachNote(tileOp.getLoc())
                << "tile has lock ops assigned to same lock.";
            return signalPassFailure();
          }
          iter->second.assigned.insert(lockID);
        }
      } else {
        auto iter = tileToLocks.find(tileOp);
        if (iter == tileToLocks.end())
          tileToLocks.insert({tileOp, {/* assigned = */ {}, {lockOp}}});
        else
          iter->second.unassigned.push_back(lockOp);
      }
    });

    // IR mutation: assign locks to all unassigned lock ops.
    for (auto [tileOp, locks] : tileToLocks) {
      const auto locksPerTile =
          getTargetModel(tileOp).getNumLocks(tileOp.getCol(), tileOp.getRow());
      uint32_t nextID = 0;
      for (auto lockOp : locks.unassigned) {
        while (nextID < locksPerTile &&
               (locks.assigned.find(nextID) != locks.assigned.end())) {
          ++nextID;
        }
        if (nextID == locksPerTile) {
          mlir::InFlightDiagnostic diag =
              lockOp->emitOpError("not allocated a lock.");
          diag.attachNote(tileOp.getLoc()) << "because only " << locksPerTile
                                           << " locks available in this tile.";
          return signalPassFailure();
        }
        lockOp.setLockIDAttr(rewriter.getI32IntegerAttr(nextID));
        ++nextID;
      }
    }
  }
};
std::unique_ptr<OperationPass<DeviceOp>> createAIEAssignLockIDsPass() {
  return std::make_unique<AIEAssignLockIDsPass>();
}

void registerAIEAssignLockIDs() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAIEAssignLockIDsPass();
  });
}
}  // namespace mlir::iree_compiler::AMDAIE
