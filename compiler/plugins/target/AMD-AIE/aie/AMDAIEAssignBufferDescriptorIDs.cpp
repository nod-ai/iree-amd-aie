// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
#include <set>

#include "Passes.h"
#include "AIEDialect.h"
#include "iree-amd-aie/aie_runtime/Utils/ChannelBdIdGenerator.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "amdaie-assign-bd-ids"
// TODO(max): find these in the device model
#define EVEN_BD_ID_START 0
#define ODD_BD_ID_START 24

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace mlir::iree_compiler::AMDAIE {

/// Assign BD ids to DMABDOp's in MemOps.
LogicalResult assignBdIds(DeviceOp deviceOp) {
  AMDAIEDeviceModel deviceModel = mlir::iree_compiler::AMDAIE::getDeviceModel(
      static_cast<AMDAIEDevice>(deviceOp.getDevice()));

  ChannelBdIdGenerator shimChannelBdIdGenerator(
      deviceModel.getChannelToValidBdIds(AMDAIETileType::SHIMNOC));
  ChannelBdIdGenerator memTileChannelBdIdGenerator(
      deviceModel.getChannelToValidBdIds(AMDAIETileType::MEMTILE));

  auto memOps = llvm::to_vector_of<TileElement>(deviceOp.getOps<MemOp>());
  llvm::append_range(memOps, deviceOp.getOps<MemTileDMAOp>());
  llvm::append_range(memOps, deviceOp.getOps<ShimDMAOp>());
  for (TileElement memOp : memOps) {
    int col = memOp.getTileID().col;
    int row = memOp.getTileID().row;

    // BdIdGenerator gen(col, row, deviceModel);
    ChannelBdIdGenerator gen = deviceModel.isMemTile(col, row)
                                   ? memTileChannelBdIdGenerator
                                   : shimChannelBdIdGenerator;

    memOp->walk<WalkOrder::PreOrder>([&](DMABDOp bd) {
      if (bd.getBdId().has_value()) gen.assignBdId(bd.getBdId().value());
    });

    DenseMap<Block *, int> blockChannelMap;
    // Associate with each block the channel index specified by the
    // dma_start
    for (Block &block : memOp.getOperation()->getRegion(0))
      for (auto op : block.getOps<DMAStartOp>()) {
        int chNum = op.getChannelIndex();
        blockChannelMap[&block] = chNum;
        Block *dest = op.getDest();
        while (dest) {
          blockChannelMap[dest] = chNum;
          if (dest->hasNoSuccessors()) break;
          dest = dest->getSuccessors()[0];
          if (blockChannelMap.contains(dest)) dest = nullptr;
        }
      }

    for (Block &block : memOp.getOperation()->getRegion(0)) {
      if (block.getOps<DMABDOp>().empty()) continue;
      assert(blockChannelMap.count(&block));
      DMABDOp bd = (*block.getOps<DMABDOp>().begin());
      if (bd.getBdId().has_value()) {
        assert(gen.isBdIdAssigned(bd.getBdId().value()) &&
               "bdId assigned by user but not found during previous walk");
      } else {
        std::optional<uint32_t> bdId =
            gen.getAndAssignBdId(blockChannelMap[&block]);
        if (!bdId)
          return memOp.emitOpError()
                 << "could not find and assign a valid BD id";
        bd.setBdId(bdId.value());
      }
    }
  }
  for (TileElement memOp : memOps) {
    DenseMap<Block *, int> blockBdIdMap;
    for (Block &block : memOp.getOperation()->getRegion(0)) {
      if (block.getOps<DMABDOp>().empty()) continue;
      DMABDOp bd = *block.getOps<DMABDOp>().begin();
      assert(bd.getBdId().has_value() &&
             "DMABDOp should have bd_id assigned by now");
      blockBdIdMap[&block] = bd.getBdId().value();
    }

    for (Block &block : memOp.getOperation()->getRegion(0)) {
      if (block.getOps<DMABDOp>().empty()) continue;
      DMABDOp bd = *block.getOps<DMABDOp>().begin();
      std::optional<int> nextBdId;
      if (block.getNumSuccessors()) {
        assert(llvm::range_size(block.getSuccessors()) == 1 &&
               "should have only one successor block");
        Block *nextBlock = block.getSuccessor(0);
        if (!blockBdIdMap.contains(nextBlock))
          assert(nextBlock->getOperations().size() == 1 &&
                 // for some reason i can't stick both of ops in a single
                 // isa<...>
                 (isa<EndOp>(nextBlock->getOperations().front()) ||
                  isa<DMAStartOp>(nextBlock->getOperations().front())) &&
                 "bb that's not in blockMap can only have aie.end");
        else
          nextBdId = blockBdIdMap[nextBlock];
        bd.setNextBdId(nextBdId);
      }
    }
  }
  return success();
}

struct AMDAIEAssignBufferDescriptorIDsPass : mlir::OperationPass<DeviceOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      AMDAIEAssignBufferDescriptorIDsPass)

  AMDAIEAssignBufferDescriptorIDsPass()
      : mlir::OperationPass<DeviceOp>(resolveTypeID()) {}

  llvm::StringRef getArgument() const override {
    return "amdaie-assign-bd-ids";
  }

  llvm::StringRef getName() const override {
    return "AMDAIEAssignBufferDescriptorIDsPass";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<AMDAIEAssignBufferDescriptorIDsPass>(
        *static_cast<const AMDAIEAssignBufferDescriptorIDsPass *>(this));
  }

  void runOnOperation() override {
    DeviceOp deviceOp = getOperation();
    if (failed(assignBdIds(deviceOp))) signalPassFailure();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
createAMDAIEAssignBufferDescriptorIDsPass() {
  return std::make_unique<AMDAIEAssignBufferDescriptorIDsPass>();
}

void registerAMDAIEAssignBufferDescriptorIDs() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAMDAIEAssignBufferDescriptorIDsPass();
  });
}
}  // namespace mlir::iree_compiler::AMDAIE
