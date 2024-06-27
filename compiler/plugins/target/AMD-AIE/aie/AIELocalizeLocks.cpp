// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Passes.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "aie-localize-locks"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define GEN_PASS_DECL_AIELOCALIZELOCKS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
#undef GEN_PASS_DECL_AIELOCALIZELOCKS

#define GEN_PASS_DEF_AIELOCALIZELOCKS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
#undef GEN_PASS_DEF_AIELOCALIZELOCKS

namespace mlir::iree_compiler::AMDAIE {
struct AIELocalizeLocksPass
    : ::impl::AIELocalizeLocksBase<AIELocalizeLocksPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
  void runOnOperation() override {
    DeviceOp deviceOp = getOperation();
    const auto &targetModel = getTargetModel(deviceOp);
    for (auto coreOp : deviceOp.getOps<CoreOp>()) {
      // Collect the locks used in this core.
      auto thisTile = dyn_cast<TileOp>(coreOp.getTile().getDefiningOp());
      int col = thisTile.colIndex();
      int row = thisTile.rowIndex();

      // Find the neighboring tiles
      SmallVector<TileOp, 4> accessibleTiles;
      for (auto tile : deviceOp.getOps<TileOp>())
        if (int dstRow = tile.rowIndex();
            targetModel.isLegalMemAffinity(col, row, tile.colIndex(), dstRow))
          accessibleTiles.push_back(tile);

      for (auto tile : accessibleTiles) {
        int dstCol = tile.colIndex();
        int dstRow = tile.rowIndex();
        int cardinalMemOffset = 0;
        int numLocks = targetModel.getNumLocks(dstCol, dstRow);
        for (auto user : tile.getResult().getUsers())
          if (auto lock = dyn_cast<LockOp>(user)) {
            if (targetModel.isMemSouth(col, row, dstCol, dstRow))
              cardinalMemOffset = 0;
            else if (targetModel.isMemWest(col, row, dstCol, dstRow))
              cardinalMemOffset = numLocks;
            else if (targetModel.isMemNorth(col, row, dstCol, dstRow))
              cardinalMemOffset = 2 * numLocks;
            else if (targetModel.isMemEast(col, row, dstCol, dstRow))
              cardinalMemOffset = 3 * numLocks;
            else
              llvm_unreachable("Found illegal lock user!");

            int localLockIndex = cardinalMemOffset + lock.getLockIDValue();

            OpBuilder builder =
                OpBuilder::atBlockBegin(&coreOp.getBody().front());

            Value coreLockIDValue = builder.create<arith::ConstantIndexOp>(
                builder.getUnknownLoc(), localLockIndex);
            lock.getResult().replaceUsesWithIf(
                coreLockIDValue, [&](OpOperand &opOperand) {
                  return opOperand.getOwner()->getParentOp() == coreOp;
                });
          }
      }
    }
  }
};
std::unique_ptr<OperationPass<DeviceOp>> createAIELocalizeLocksPass() {
  return std::make_unique<AIELocalizeLocksPass>();
}

void registerAIELocalizeLocks() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAIELocalizeLocksPass();
  });
}
}  // namespace mlir::iree_compiler::AMDAIE
