// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/aie_runtime/Utils/ChannelBdIdGenerator.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/IR/Iterators.h"

#define DEBUG_TYPE "iree-amdaie-dma-bd-chain"

namespace mlir::iree_compiler::AMDAIE {

namespace {

LogicalResult dmaBdChain(AMDAIE::WorkgroupOp workgroupOp) {
  IRRewriter rewriter(workgroupOp->getContext());

  DenseMap<std::pair<uint32_t, uint32_t>, uint32_t> colArgIdxToNextBdId;
  DenseMap<std::pair<uint32_t, uint32_t>, uint32_t> colArgIdxToStartBdId;
  bool useNextBd = false;
  bool enablePacket = false;
  uint32_t argIdx = 0;
  uint32_t col = 0;

  // Walk the operations in the reverse order to build the chain
  AMDAIE::ControlCodeOp controlCodeOp = workgroupOp.getControlCode();
  WalkResult res = controlCodeOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](Operation *op) {
        if (auto npuAddressPatchOp = dyn_cast<AMDAIE::NpuAddressPatchOp>(op)) {
          argIdx = npuAddressPatchOp.getArgIdx();
        } else if (auto writeBdOp = dyn_cast<AMDAIE::NpuWriteBdOp>(op)) {
          enablePacket |= writeBdOp.getEnablePacket();
          if (enablePacket) {
            // packet mode is enabled, do not chain BDs
            return WalkResult::advance();
          }
          col = writeBdOp.getCol();
          if (colArgIdxToNextBdId.contains({col, argIdx})) {
            uint32_t nextBdId = colArgIdxToNextBdId[{col, argIdx}];
            uint32_t currBdId = writeBdOp.getBdId();
            if (nextBdId > currBdId) {
              // chain the current BD to the next BD
              writeBdOp.setNextBd(nextBdId);
              writeBdOp.setUseNextBd(true);
            }
          }
          colArgIdxToNextBdId[{col, argIdx}] = writeBdOp.getBdId();
        }
        return WalkResult::advance();
      });

  if (res.wasInterrupted()) return failure();
  if (enablePacket) return success();

  // Remove NpuPushToQueueOp and NpuDmaWaitOp that are not needed, unless they
  // are at the end of a chain
  std::vector<Operation *> opsToRemove;
  res = controlCodeOp->walk([&](Operation *op) {
    if (auto writeBdOp = dyn_cast<AMDAIE::NpuWriteBdOp>(op)) {
      useNextBd = writeBdOp.getUseNextBd();
    } else if (auto npuAddressPatchOp =
                   dyn_cast<AMDAIE::NpuAddressPatchOp>(op)) {
      argIdx = npuAddressPatchOp.getArgIdx();
      col = npuAddressPatchOp.getCol();
    } else if (auto npuPushToQueueOp = dyn_cast<AMDAIE::NpuPushToQueueOp>(op)) {
      if (useNextBd) {
        if (!colArgIdxToStartBdId.contains({col, argIdx})) {
          // start of a chain
          colArgIdxToStartBdId[{col, argIdx}] = npuPushToQueueOp.getBdId();
        }
        // remove NpuPushToQueueOp and dmaWaitOp
        for (Value result : npuPushToQueueOp->getResults()) {
          for (auto &use : result.getUses()) {
            Operation *userOp = use.getOwner();
            if (auto dmaWaitOp = dyn_cast<AMDAIE::NpuDmaWaitOp>(userOp)) {
              opsToRemove.push_back(dmaWaitOp);
            }
          }
        }
        opsToRemove.push_back(npuPushToQueueOp);
      } else {
        if (colArgIdxToStartBdId.contains({col, argIdx})) {
          // end of a chain
          npuPushToQueueOp.setBdId(colArgIdxToStartBdId[{col, argIdx}]);
          colArgIdxToStartBdId.erase({col, argIdx});
        }
      }
    }
    return WalkResult::advance();
  });

  if (res.wasInterrupted()) return failure();
  for (Operation *op : opsToRemove) {
    rewriter.eraseOp(op);
  }
  return success();
}

class AMDAIEDmaBdChainPass
    : public impl::AMDAIEDmaBdChainBase<AMDAIEDmaBdChainPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEDmaBdChainPass() = default;
  AMDAIEDmaBdChainPass(const AMDAIEDmaBdChainPass &pass){};
  void runOnOperation() override;
};

void AMDAIEDmaBdChainPass::runOnOperation() {
  Operation *parentOp = getOperation();

  WalkResult res = parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    if (failed(dmaBdChain(workgroupOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEDmaBdChainPass() {
  return std::make_unique<AMDAIEDmaBdChainPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
