// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/aie_runtime/Utils/ChannelBdIdGenerator.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/IR/Iterators.h"
#define DEBUG_TYPE "iree-amdaie-dma-bd-chain"

namespace mlir::iree_compiler::AMDAIE {

namespace {

LogicalResult dmaBdChain(AMDAIE::AMDAIEDeviceModel deviceModel,
                         AMDAIE::WorkgroupOp workgroupOp) {
  IRRewriter rewriter(workgroupOp->getContext());

  // TODO(Zhewen): assign BD IDs here, to get rid of tileArgIdxToAssignedBdIdOp
  // BD ID currently assigned to each DMA operation, used to track the lifetime
  DenseMap<std::pair<AMDAIE::TileOp, uint32_t>, SmallVector<AMDAIE::BdIdOp>>
      tileArgIdxToAssignedBdIdOps;
  // TODO(Zhewen): unroll loops here, to get rid of tileArgIdxToDmaCount
  DenseMap<std::pair<AMDAIE::TileOp, uint32_t>, uint32_t> tileArgIdxToDmaCount;

  // Last DMA operation encountered for each tile argument index pair
  // no matter if it is chained or not
  DenseMap<std::pair<AMDAIE::TileOp, uint32_t>, AMDAIE::NpuHalfDmaCpyNdOp>
      tileArgIdxToLastDmaOp;
  // Last DMA operation that has been chained for each tile argument index pair
  DenseMap<std::pair<AMDAIE::TileOp, uint32_t>, AMDAIE::NpuHalfDmaCpyNdOp>
      tileArgIdxToLastChainedDmaOp;
  // Black list of tile argument index pairs that should not be chained
  SmallVector<std::pair<AMDAIE::TileOp, uint32_t>> tileArgIdxsBlackList;

  AMDAIE::ControlCodeOp controlCodeOp = workgroupOp.getControlCode();

  // Move all BdIdOps to the beginning of the control code.
  // This is to avoid dominance issues when chaining BD IDs.
  SmallVector<Operation *> ops;
  WalkResult res = controlCodeOp->walk([&](Operation *op) {
    if (auto bdIdOp = dyn_cast<AMDAIE::BdIdOp>(op)) {
      ops.push_back(op);
    }
    return WalkResult::advance();
  });
  for (Operation *op : llvm::reverse(ops)) {
    op->moveBefore(&controlCodeOp.front());
  }

  // Find `NpuHalfDmaCpyNdOp` operations and chain BD IDs.
  res = controlCodeOp->walk([&](Operation *op) {
    if (auto npuHalfDmaCpyNdOp = dyn_cast<AMDAIE::NpuHalfDmaCpyNdOp>(op)) {
      // not shim, do not chain BDs
      if (npuHalfDmaCpyNdOp.getMemorySpaceAsUInt() != 0) {
        return WalkResult::advance();
      }

      bool chaining = true;
      // packet mode is enabled, do not chain BDs
      std::optional<AMDAIE::ConnectionOp> maybeConnectionOp =
          npuHalfDmaCpyNdOp.getConnectionOp();
      if (!maybeConnectionOp) {
        npuHalfDmaCpyNdOp.emitOpError()
            << "expected to operate on an `amdaie.connection`";
        return WalkResult::interrupt();
      }
      std::optional<AMDAIE::FlowOp> maybeFlowOp =
          maybeConnectionOp->getFlowOp();
      if (!maybeFlowOp) {
        maybeConnectionOp->emitOpError()
            << "expected to operate on an `amdaie.flow`";
        return WalkResult::interrupt();
      }
      bool enablePacket = maybeFlowOp->getIsPacketFlow();
      if (enablePacket) {
        chaining = false;
      }

      // repeat count > 1, do not chain BDs
      int32_t repeatCount = 1;
      uint8_t numIntraAddrDim = deviceModel.getDmaProp<uint8_t>(
          AMDAIE::AMDAIETileType::SHIMNOC, AMDAIE::AMDAIEDmaProp::NumAddrDim);
      uint8_t numAddrDim = numIntraAddrDim + kAMDAIEDmaNbInterDims;
      auto sizes = npuHalfDmaCpyNdOp.getMixedSizes();
      auto strides = npuHalfDmaCpyNdOp.getMixedStrides();
      if (!sizes.empty() && !strides.empty()) {
        int64_t size = getConstantIndexOrAssert(sizes[0]);
        int64_t stride = getConstantIndexOrAssert(strides[0]);
        if (sizes.size() == numAddrDim || stride == 0) {
          repeatCount = size;
        }
      }
      if (repeatCount > 1) {
        chaining = false;
      }

      // get current BD ID and tile
      std::optional<AMDAIE::BdIdOp> maybeBdIdOp = npuHalfDmaCpyNdOp.getBdIdOp();
      if (!maybeBdIdOp) {
        npuHalfDmaCpyNdOp.emitOpError() << "must have a BD ID op";
        return WalkResult::interrupt();
      }
      AMDAIE::BdIdOp bdIdOp = maybeBdIdOp.value();
      AMDAIE::TileOp tileOp =
          dyn_cast_if_present<AMDAIE::TileOp>(bdIdOp.getTile().getDefiningOp());
      if (!tileOp) {
        bdIdOp.emitOpError() << "must operate on an `amdaie.tile`";
        return WalkResult::interrupt();
      }

      // get arg index
      auto logicalObjFifo =
          dyn_cast_if_present<AMDAIE::LogicalObjectFifoFromMemrefOp>(
              npuHalfDmaCpyNdOp.getInput().getDefiningOp());
      if (!logicalObjFifo) {
        npuHalfDmaCpyNdOp.emitOpError()
            << "expected input to be an "
               "`amdaie.logicalobjectfifo.from_memref`";
        return WalkResult::interrupt();
      }
      auto subspanOp =
          dyn_cast_if_present<IREE::HAL::InterfaceBindingSubspanOp>(
              logicalObjFifo.getMemref().getDefiningOp());
      if (!subspanOp) {
        logicalObjFifo.emitOpError()
            << "must operate on an `hal.interface.binding.subspan`";
        return WalkResult::interrupt();
      }
      uint32_t argIdx = subspanOp.getBinding().getZExtValue();
      tileArgIdxToDmaCount[{tileOp, argIdx}]++;
      for (auto &[pair, count] : tileArgIdxToDmaCount) {
        if (pair.first == tileOp &&
            count > tileArgIdxToDmaCount[{tileOp, argIdx}] + 1) {
          if (!llvm::is_contained(tileArgIdxsBlackList, pair)) {
            tileArgIdxsBlackList.push_back(pair);
          }
        }
      }
      // if the BD ID is currently used by another DMA op, stop the chain
      // for that DMA op from further growing, so that BD ID can be released
      for (auto &[pair, bdIdOps] : tileArgIdxToAssignedBdIdOps) {
        if (pair.first == tileOp && llvm::is_contained(bdIdOps, bdIdOp)) {
          if (!llvm::is_contained(tileArgIdxsBlackList, pair)) {
            tileArgIdxsBlackList.push_back(pair);
          }
          break;
        }
      }

      // if not blacklisted and there is a previous DMA op, chain the BD IDs
      chaining &= !llvm::is_contained(tileArgIdxsBlackList,
                                      std::make_pair(tileOp, argIdx)) &&
                  tileArgIdxToLastDmaOp.contains({tileOp, argIdx});
      if (chaining) {
        // update previous NpuHalfDmaCpyNdOp by changing its useNextBd and
        // nextBd
        AMDAIE::NpuHalfDmaCpyNdOp lastDmaOp =
            tileArgIdxToLastDmaOp[{tileOp, argIdx}];
        rewriter.setInsertionPointAfter(lastDmaOp);
        auto chainedDmaOp = rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
            lastDmaOp.getLoc(), lastDmaOp.getResultTypes(),
            lastDmaOp.getConnection(), lastDmaOp.getInput(),
            lastDmaOp.getMixedOffsets(), lastDmaOp.getMixedSizes(),
            lastDmaOp.getMixedStrides(), lastDmaOp.getBdId(),
            lastDmaOp.getChannel(), true, bdIdOp, lastDmaOp.getStartBd());
        rewriter.replaceOp(lastDmaOp, chainedDmaOp.getResults());
        tileArgIdxToLastChainedDmaOp[{tileOp, argIdx}] = chainedDmaOp;
        // update current NpuHalfDmaCpyNdOp by changing its startBd
        rewriter.setInsertionPoint(npuHalfDmaCpyNdOp);
        auto npuHalfDmaCpyNdOpNew = rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
            npuHalfDmaCpyNdOp.getLoc(), npuHalfDmaCpyNdOp.getResultTypes(),
            npuHalfDmaCpyNdOp.getConnection(), npuHalfDmaCpyNdOp.getInput(),
            npuHalfDmaCpyNdOp.getMixedOffsets(),
            npuHalfDmaCpyNdOp.getMixedSizes(),
            npuHalfDmaCpyNdOp.getMixedStrides(), npuHalfDmaCpyNdOp.getBdId(),
            npuHalfDmaCpyNdOp.getChannel(), npuHalfDmaCpyNdOp.getUseNextBd(),
            npuHalfDmaCpyNdOp.getNextBd(), chainedDmaOp.getStartBd());
        rewriter.replaceOp(npuHalfDmaCpyNdOp,
                           npuHalfDmaCpyNdOpNew.getResults());
        npuHalfDmaCpyNdOp = npuHalfDmaCpyNdOpNew;
      }

      // update the BD ID assignment
      if (chaining && tileArgIdxToAssignedBdIdOps.contains({tileOp, argIdx})) {
        tileArgIdxToAssignedBdIdOps[{tileOp, argIdx}].push_back(bdIdOp);
      } else {
        tileArgIdxToAssignedBdIdOps[{tileOp, argIdx}] = {bdIdOp};
      }

      // not chaining, update the black list

      if (tileArgIdxsBlackList.size() > 0) {
        for (auto &[pair, bdIdOps] : tileArgIdxToAssignedBdIdOps) {
          if (pair.first == tileOp && bdIdOps.size() > 1) {
            if (!llvm::is_contained(tileArgIdxsBlackList, pair)) {
              tileArgIdxsBlackList.push_back(pair);
            }
          }
        }
      }

      if (!chaining) {
        auto it =
            std::find(tileArgIdxsBlackList.begin(), tileArgIdxsBlackList.end(),
                      std::make_pair(tileOp, argIdx));
        if (it != tileArgIdxsBlackList.end()) {
          tileArgIdxsBlackList.erase(it);
        }
      }
      // update the last DMA op
      tileArgIdxToLastDmaOp[{tileOp, argIdx}] = npuHalfDmaCpyNdOp;

    } else if (auto npuDmaWaitOp = dyn_cast<AMDAIE::NpuDmaWaitOp>(op)) {
      // Handle the special case where the blacklist is not empty. This could
      // happen when there are multiple DMA operations associated with the same
      // tile but different argIdx before a DMA wait. In such cases, one DMA
      // operation might initially get chained, but a subsequent DMA operation
      // may later report that the chain must be broken to release the BD IDs.
      for (auto &[tileOp, argIdx] : tileArgIdxsBlackList) {
        if (tileArgIdxToLastChainedDmaOp.contains({tileOp, argIdx})) {
          AMDAIE::NpuHalfDmaCpyNdOp lastChainedDmaOp =
              tileArgIdxToLastChainedDmaOp[{tileOp, argIdx}];
          bool useNextBd{false};
          Value nextBd{nullptr};
          rewriter.setInsertionPointAfter(lastChainedDmaOp);
          auto unchainedDmaOp = rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
              lastChainedDmaOp.getLoc(), lastChainedDmaOp.getResultTypes(),
              lastChainedDmaOp.getConnection(), lastChainedDmaOp.getInput(),
              lastChainedDmaOp.getMixedOffsets(),
              lastChainedDmaOp.getMixedSizes(),
              lastChainedDmaOp.getMixedStrides(), lastChainedDmaOp.getBdId(),
              lastChainedDmaOp.getChannel(), useNextBd, nextBd,
              lastChainedDmaOp.getStartBd());
          rewriter.replaceOp(lastChainedDmaOp, unchainedDmaOp.getResults());
          tileArgIdxToLastChainedDmaOp.erase({tileOp, argIdx});

          if (tileArgIdxToLastDmaOp.contains({tileOp, argIdx})) {
            AMDAIE::NpuHalfDmaCpyNdOp lastDmaOp =
                tileArgIdxToLastDmaOp[{tileOp, argIdx}];
            rewriter.setInsertionPoint(lastDmaOp);
            auto lastDmaOpNew = rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
                lastDmaOp.getLoc(), lastDmaOp.getResultTypes(),
                lastDmaOp.getConnection(), lastDmaOp.getInput(),
                lastDmaOp.getMixedOffsets(), lastDmaOp.getMixedSizes(),
                lastDmaOp.getMixedStrides(), lastDmaOp.getBdId(),
                lastDmaOp.getChannel(), lastDmaOp.getUseNextBd(),
                lastDmaOp.getNextBd(), lastDmaOp.getBdId());
            tileArgIdxToAssignedBdIdOps[{tileOp, argIdx}] = {
                lastDmaOp.getBdIdOp().value()};
            rewriter.replaceOp(lastDmaOp, lastDmaOpNew.getResults());
            tileArgIdxToLastDmaOp[{tileOp, argIdx}] = lastDmaOpNew;
          }
        }
      }

      tileArgIdxsBlackList.clear();
    }
    return WalkResult::advance();
  });

  // erase wait op unless it is at the end of a chain
  res = controlCodeOp->walk([&](Operation *op) {
    if (auto npuDmaWaitOp = dyn_cast<AMDAIE::NpuDmaWaitOp>(op)) {
      bool toErase = true;
      for (Value token : npuDmaWaitOp.getAsyncTokens()) {
        auto npuHalfDmaCpyNdOp = dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(
            token.getDefiningOp());
        bool chaining = npuHalfDmaCpyNdOp && npuHalfDmaCpyNdOp.getUseNextBd();
        if (!chaining) {
          toErase = false;
          break;
        }
      }
      if (toErase) {
        rewriter.eraseOp(npuDmaWaitOp);
      }
    }
    return WalkResult::advance();
  });

  if (res.wasInterrupted()) return failure();
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

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(parentOp);
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    parentOp->emitOpError()
        << "has no AMDAIEDevice in the target attribute configuration. This "
           "device-specific information is required to lower control code "
           "ops.";
    return signalPassFailure();
  }
  AMDAIE::AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(maybeDevice.value());

  WalkResult res = parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    if (failed(dmaBdChain(deviceModel, workgroupOp))) {
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
