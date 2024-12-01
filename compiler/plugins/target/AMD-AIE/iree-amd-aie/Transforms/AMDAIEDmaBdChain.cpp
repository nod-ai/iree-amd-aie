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

  // TODO(Zhewen): to get rid of tileArgIdxToAssignedBdIdOps and
  // tileArgIdxToDmaCount, integrate BD ID assignment and (partial) control code
  // loop unrolling into this pass.

  // BD ID that are currenly assigned to DMA operations
  DenseMap<std::pair<AMDAIE::TileOp, uint32_t>, SmallVector<AMDAIE::BdIdOp>>
      tileArgIdxToAssignedBdIdOps;
  // Counter for the number of DMA operations, helping determine the dependency
  DenseMap<std::pair<AMDAIE::TileOp, uint32_t>, uint32_t> tileArgIdxToDmaCount;

  // Last DMA operation encountered, no matter if it is chained or not
  DenseMap<std::pair<AMDAIE::TileOp, uint32_t>, AMDAIE::NpuHalfDmaCpyNdOp>
      tileArgIdxToLastDmaOp;
  // Last DMA operation that has been chained
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
      // not shim, no need to chain, since it will be earsed when lowering to
      // NPU instructions
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

      // If the current DMA operation was previously part of the outer loop in
      // the control code, force all DMA operations in the inner loop to be
      // synchronized, by adding them to the black list.
      tileArgIdxToDmaCount[{tileOp, argIdx}]++;
      for (auto &[pair, count] : tileArgIdxToDmaCount) {
        if (pair.first == tileOp &&
            count > tileArgIdxToDmaCount[{tileOp, argIdx}] + 1) {
          if (!llvm::is_contained(tileArgIdxsBlackList, pair)) {
            tileArgIdxsBlackList.push_back(pair);
          }
        }
      }

      // If the BD ID is currently used by another DMA op, stop the chain
      // for that DMA op from further growing, by adding it to the black list
      for (auto &[pair, bdIdOps] : tileArgIdxToAssignedBdIdOps) {
        if (pair.first == tileOp && llvm::is_contained(bdIdOps, bdIdOp)) {
          if (!llvm::is_contained(tileArgIdxsBlackList, pair)) {
            tileArgIdxsBlackList.push_back(pair);
          }
          break;
        }
      }

      // If the black list is not empty, there will be a synchronization.
      // Make sure all other DMA chains also break at this point to avoid
      // dependency issues.
      if (tileArgIdxsBlackList.size() > 0) {
        for (auto &[pair, bdIdOps] : tileArgIdxToAssignedBdIdOps) {
          if (pair.first == tileOp && bdIdOps.size() > 1) {
            if (!llvm::is_contained(tileArgIdxsBlackList, pair)) {
              tileArgIdxsBlackList.push_back(pair);
            }
          }
        }
      }

      // When current DMA has not been blacklisted and a previous DMA with same
      // argIdx exists, chain them together
      chaining &= !llvm::is_contained(tileArgIdxsBlackList,
                                      std::make_pair(tileOp, argIdx)) &&
                  tileArgIdxToLastDmaOp.contains({tileOp, argIdx});
      if (chaining) {
        // update the previous DMA op by changing its useNextBd and
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
        // update the current DMA op by changing its startBd
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

      // Update BD ID assignment, if it is chaining, safely release the BD IDs
      // since a synchronization will happen
      if (chaining && tileArgIdxToAssignedBdIdOps.contains({tileOp, argIdx})) {
        tileArgIdxToAssignedBdIdOps[{tileOp, argIdx}].push_back(bdIdOp);
      } else {
        tileArgIdxToAssignedBdIdOps[{tileOp, argIdx}] = {bdIdOp};
      }

      // The current DMA op is not chained with the previous DMA op (i.e.
      // synchroizaiton will happen between these two ops), removing from the
      // black list
      if (!chaining) {
        auto it =
            std::find(tileArgIdxsBlackList.begin(), tileArgIdxsBlackList.end(),
                      std::make_pair(tileOp, argIdx));
        if (it != tileArgIdxsBlackList.end()) {
          tileArgIdxsBlackList.erase(it);
        }
      }
      // Update the last encountered DMA op
      tileArgIdxToLastDmaOp[{tileOp, argIdx}] = npuHalfDmaCpyNdOp;

    } else if (auto npuDmaWaitOp = dyn_cast<AMDAIE::NpuDmaWaitOp>(op)) {
      // Handle the special case where there are multiple DMA ops preceding any
      // Wait op. In such a case, some DMA ops may be chained first, before they
      // are put onto the black list. Therefore, go over the black list and
      // unchain the DMA ops when required.

      for (auto &[tileOp, argIdx] : tileArgIdxsBlackList) {
        if (tileArgIdxToLastChainedDmaOp.contains({tileOp, argIdx}) &&
            tileArgIdxToLastDmaOp.contains({tileOp, argIdx})) {
          // break the chain lastChainedDmaOp -> lastDmaOp
          AMDAIE::NpuHalfDmaCpyNdOp lastChainedDmaOp =
              tileArgIdxToLastChainedDmaOp[{tileOp, argIdx}];
          AMDAIE::NpuHalfDmaCpyNdOp lastDmaOp =
              tileArgIdxToLastDmaOp[{tileOp, argIdx}];
          // revert useNextBd and nextBd in lastChainedDmaOp
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
          // revert startBd in lastDmaOp
          auto startBd = lastDmaOp.getBdId();
          rewriter.setInsertionPoint(lastDmaOp);
          unchainedDmaOp = rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
              lastDmaOp.getLoc(), lastDmaOp.getResultTypes(),
              lastDmaOp.getConnection(), lastDmaOp.getInput(),
              lastDmaOp.getMixedOffsets(), lastDmaOp.getMixedSizes(),
              lastDmaOp.getMixedStrides(), lastDmaOp.getBdId(),
              lastDmaOp.getChannel(), lastDmaOp.getUseNextBd(),
              lastDmaOp.getNextBd(), startBd);
          tileArgIdxToAssignedBdIdOps[{tileOp, argIdx}] = {
              lastDmaOp.getBdIdOp().value()};
          rewriter.replaceOp(lastDmaOp, unchainedDmaOp.getResults());
          tileArgIdxToLastDmaOp[{tileOp, argIdx}] = unchainedDmaOp;
        } else {
          npuDmaWaitOp.emitError() << "unhandled situation in DMA BD chaining, "
                                      "please try to disable this pass";
          return WalkResult::interrupt();
        }
      }

      tileArgIdxsBlackList.clear();
    }
    return WalkResult::advance();
  });

  // Only keep DMA Wait Ops if at the end of a chain, erase others
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
