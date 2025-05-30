// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
#include <set>


#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/Utils/ChannelBdIdGenerator.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

#define DEBUG_TYPE "iree-amdaie-assign-bd-ids"

// TODO(max): find these in the device model
#define EVEN_BD_ID_START 0
#define ODD_BD_ID_START 24

using namespace mlir;
// using namespace xilinx;

namespace mlir::iree_compiler::AMDAIE {

/// Assign BD ids to DMABDOp's in MemOps.
LogicalResult assignBdIds(Operation* deviceOp) {
    // llvm::outs()<<"DB - 1.1\n";
    // llvm::outs().flush();
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(deviceOp);
    std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
    if (!maybeDevice) {
        deviceOp->emitOpError()
            << "has no AMDAIEDevice in the target attribute configuration. This "
            "device-specific information is required to lower control code "
            "ops.";
        return failure();
    }

    AMDAIE::AMDAIEDeviceModel deviceModel =
        AMDAIE::getDeviceModel(maybeDevice.value());

  ChannelBdIdGenerator shimChannelBdIdGenerator(
      deviceModel.getChannelToValidBdIds(AMDAIETileType::SHIMNOC));
  ChannelBdIdGenerator memTileChannelBdIdGenerator(
      deviceModel.getChannelToValidBdIds(AMDAIETileType::MEMTILE));

//   auto memOps = deviceOp->getRegion(0).front().getOps<AMDAIE::DMAStartOp>();
  SmallVector<AMDAIE::DMAStartOp> memOps;
  deviceOp->walk([&](AMDAIE::DMAStartOp dmaStartOp) {
    memOps.push_back(dmaStartOp);
  });
//   llvm::outs()<<"MemOps.size() = "<<memOps.size()<<"\n";
//   llvm::outs().flush();
//   llvm::append_range(memOps, deviceOp->getRegion(0).back().getOps<MemTileDMAOp>());
    // llvm::outs()<<"DB - 1.2\n";
    // llvm::outs().flush();
  for (AMDAIE::DMAStartOp dmaStartOp : memOps) {
    // llvm::outs()<<"DMA start = "<<dmaStartOp<<"\n";
    // llvm::outs().flush();
    auto tile = dmaStartOp.getTile().getDefiningOp<AMDAIE::TileOp>();
    // BdIdGenerator gen(col, row, deviceModel);
    std::optional<int64_t> col = getConstantIntValue(tile.getCol());
    std::optional<int64_t> row = getConstantIntValue(tile.getRow());
    if (!col || !row) {
        return tile->emitOpError()
                    << "expected column and row integer value/constant";
    }
    ChannelBdIdGenerator gen = deviceModel.isMemTile(*col, *row)
                                   ? memTileChannelBdIdGenerator
                                   : shimChannelBdIdGenerator;

    dmaStartOp->walk<WalkOrder::PreOrder>([&](AMDAIE::DMABDOp bd) {
      if (bd.getBdId().has_value()) gen.assignBdId(bd.getBdId().value());
    });

    DenseMap<Block *, int> blockChannelMap;
    // Associate with each block the channel index specified by the
    // dma_start
    // for (Block &block : memOp->getRegion(0))
    //   for (auto op : block.getOps<DMAStartOp>()) {
    int chNum = dmaStartOp.getChannelIndex();
    // blockChannelMap[&block] = chNum;
    // for (Block &block : dmaStartOp->getRegion(0)) {
        // blockChannelMap[dest] = chNum;
        dmaStartOp->walk<WalkOrder::PreOrder>([&](AMDAIE::DMABDOp bd) {
            if (bd.getBdId().has_value()) {
                assert(gen.isBdIdAssigned(bd.getBdId().value()) &&
                    "bdId assigned by user but not found during previous walk");
            } else {
                // llvm::outs()<<"Assigned BD ID to : "<<bd<<"\n";
                // llvm::outs().flush();
                std::optional<uint32_t> bdId = gen.getAndAssignBdId(chNum);
                if (!bdId) {
                    dmaStartOp->emitOpError()
                            << "could not find and assign a valid BD id";
                    return WalkResult::skip();
                }
                bd.setBdId(bdId.value());
            }
            return WalkResult::advance();
        });
    // }
  }

  for (AMDAIE::DMAStartOp dmaStartOp : memOps) {
    // llvm::outs()<<"\tDMA start = "<<dmaStartOp<<"\n";
    // llvm::outs().flush();
    DenseMap<Block *, int> blockBdIdMap;
    for (Block &block : dmaStartOp->getRegion(0)) {
      if (block.getOps<AMDAIE::DMABDOp>().empty()) continue;
      DMABDOp bd = *block.getOps<AMDAIE::DMABDOp>().begin();
      assert(bd.getBdId().has_value() &&
             "DMABDOp should have bd_id assigned by now");
      blockBdIdMap[&block] = bd.getBdId().value();
    }

    for (Block &block : dmaStartOp->getRegion(0)) {
      if (block.getOps<AMDAIE::DMABDOp>().empty()) continue;
      AMDAIE::DMABDOp bd = *block.getOps<AMDAIE::DMABDOp>().begin();
      std::optional<int> nextBdId;
      if (block.getNumSuccessors()) {
        assert(llvm::range_size(block.getSuccessors()) == 1 &&
               "should have only one successor block");
        Block *nextBlock = block.getSuccessor(0);
        if (!blockBdIdMap.contains(nextBlock))
          assert(nextBlock->getOperations().size() == 1 &&
                 // for some reason i can't stick both of ops in a single
                 // isa<...>
                 (isa<AMDAIE::EndOp>(nextBlock->getOperations().front()) ||
                  isa<AMDAIE::DMAStartOp>(nextBlock->getOperations().front())) &&
                 "bb that's not in blockMap can only have aie.end");
        else
          nextBdId = blockBdIdMap[nextBlock];
        bd.setNextBdId(nextBdId);
      }
    }
  }
  return success();
}

namespace {
class AMDAIEAssignBDIDsPass
    : public impl::AMDAIEAssignBDIDsBase<
          AMDAIEAssignBDIDsPass> {
 public:
//   AMDAIEAssignBDIDsPass(const AMDAIEAssignBDIDsOptions &options)
//       : AMDAIEAssignBDIDsBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  void runOnOperation() override;
};

void AMDAIEAssignBDIDsPass::runOnOperation() {
    Operation* workgroupOp = getOperation();
    // llvm::outs()<<"DB - 1\n";
    // llvm::outs().flush();
    if (failed(assignBdIds(workgroupOp))) signalPassFailure();
    // llvm::outs()<<"SUCCESS\n";
    // llvm::outs().flush();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignBDIDsPass() {
  return std::make_unique<AMDAIEAssignBDIDsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
