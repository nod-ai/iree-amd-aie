// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AIEDialect.h"
#include "Passes.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "amdaie-localize-locks"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace mlir::iree_compiler::AMDAIE {
struct AMDAIELocalizeLocksPass : mlir::OperationPass<DeviceOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AMDAIELocalizeLocksPass)

  AMDAIELocalizeLocksPass() : mlir::OperationPass<DeviceOp>(resolveTypeID()) {}

  llvm::StringRef getArgument() const override {
    return "amdaie-localize-locks";
  }

  llvm::StringRef getName() const override {
    return " AMDAIELocalizeLocksPass";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<AMDAIELocalizeLocksPass>(
        *static_cast<const AMDAIELocalizeLocksPass *>(this));
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    DeviceOp deviceOp = getOperation();
    AMDAIEDeviceModel deviceModel =
        getDeviceModel(static_cast<AMDAIEDevice>(deviceOp.getDevice()));
    for (auto coreOp : deviceOp.getOps<CoreOp>()) {
      // Collect the locks used in this core.
      auto thisTile = dyn_cast<TileOp>(coreOp.getTile().getDefiningOp());
      int col = thisTile.getCol();
      int row = thisTile.getRow();

      // Find the neighboring tiles
      SmallVector<TileOp, 4> accessibleTiles;
      for (auto tile : deviceOp.getOps<TileOp>())
        if (int dstRow = tile.getRow();
            deviceModel.hasLegalMemAffinity(col, row, tile.getCol(), dstRow))
          accessibleTiles.push_back(tile);

      for (auto tile : accessibleTiles) {
        int dstCol = tile.getCol();
        int dstRow = tile.getRow();
        int cardinalMemOffset = 0;
        int numLocks = deviceModel.getNumLocks(dstCol, dstRow);
        for (auto user : tile.getResult().getUsers())
          if (auto lock = dyn_cast<LockOp>(user)) {
            if (deviceModel.hasMemSouth(col, row, dstCol, dstRow))
              cardinalMemOffset = 0;
            else if (deviceModel.hasMemWest(col, row, dstCol, dstRow))
              cardinalMemOffset = numLocks;
            else if (deviceModel.hasMemNorth(col, row, dstCol, dstRow))
              cardinalMemOffset = 2 * numLocks;
            else if (deviceModel.hasMemEast(col, row, dstCol, dstRow))
              cardinalMemOffset = 3 * numLocks;
            else
              llvm_unreachable("Found illegal lock user!");

            int localLockIndex = cardinalMemOffset + lock.getLockID().value();

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

std::unique_ptr<OperationPass<DeviceOp>> createAMDAIELocalizeLocksPass() {
  return std::make_unique<AMDAIELocalizeLocksPass>();
}

void registerAMDAIELocalizeLocks() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAMDAIELocalizeLocksPass();
  });
}
}  // namespace mlir::iree_compiler::AMDAIE
