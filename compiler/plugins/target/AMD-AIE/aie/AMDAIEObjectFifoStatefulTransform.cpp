// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
#include <set>

#include "Passes.h"
#include "AIEDialect.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "amdaie-objectFifo-stateful-transform"

using namespace mlir;
using namespace mlir::iree_compiler::AMDAIE;

using xilinx::AIE::AIEObjectFifoType;
using xilinx::AIE::BDDimLayoutArrayArrayAttr;
using xilinx::AIE::BDDimLayoutArrayAttr;
using xilinx::AIE::BDDimLayoutAttr;
using xilinx::AIE::BufferOp;
using xilinx::AIE::CoreOp;
using xilinx::AIE::DeviceOp;
using xilinx::AIE::DMABDOp;
using xilinx::AIE::DMAChannelDirAttr;
using xilinx::AIE::DMAStartOp;
using xilinx::AIE::EndOp;
using xilinx::AIE::ExternalBufferOp;
using xilinx::AIE::FlowOp;
using xilinx::AIE::LockAction;
using xilinx::AIE::LockOp;
using xilinx::AIE::MemOp;
using xilinx::AIE::MemTileDMAOp;
using xilinx::AIE::NextBDOp;
using xilinx::AIE::ObjectFifoAcquireOp;
using xilinx::AIE::ObjectFifoCreateOp;
using xilinx::AIE::ObjectFifoLinkOp;
using xilinx::AIE::ObjectFifoPort;
using xilinx::AIE::ObjectFifoRegisterExternalBuffersOp;
using xilinx::AIE::ObjectFifoReleaseOp;
using xilinx::AIE::ObjectFifoSubviewAccessOp;
using xilinx::AIE::ShimDMAAllocationOp;
using xilinx::AIE::ShimDMAOp;
using xilinx::AIE::TileOp;
using xilinx::AIE::UseLockOp;
using xilinx::AIE::WireBundle;

class LockAnalysis {
  DenseMap<std::pair<Value, int>, int> locksPerTile;

 public:
  LockAnalysis(DeviceOp &device) {
    for (auto lockOp : device.getOps<LockOp>())
      locksPerTile[{lockOp.getTile(), lockOp.getLockIDValue()}] = 1;
  }

  /// Given a tile, returns next usable lockID for that tile.
  int getLockID(TileOp &tileOp) {
    DeviceOp device = tileOp->getParentOfType<DeviceOp>();
    AMDAIEDeviceModel deviceModel =
        getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
    for (int i = 0;
         i < deviceModel.getNumLocks(tileOp.getCol(), tileOp.getRow()); i++) {
      std::pair<Value, int> lockId = {tileOp, i};
      if (int usageCnt = locksPerTile[lockId]; usageCnt == 0) {
        locksPerTile[lockId] = 1;
        return i;
      }
    }
    return -1;
  }
};

class DMAChannelAnalysis {
  DenseMap<Value, int> producerChannelsPerTile;
  DenseMap<Value, int> consumerChannelsPerTile;

 public:
  DMAChannelAnalysis(DeviceOp &device) {
    // go over the channels used for each tile and update the producer/consumer
    // channel maps
    for (auto memOp : device.getOps<MemOp>()) {
      Region &r = memOp.getBody();
      auto tile = memOp.getTile();
      for (auto &bl : r.getBlocks()) {
        for (auto op : bl.getOps<DMAStartOp>()) {
          static_cast<DMAChannelDir>(op.getChannelDir()) == DMAChannelDir::MM2S
              ? getProducerDMAChannel(tile)
              : getConsumerDMAChannel(tile);
        }
      }
    }
  }

  /// Given an AIE tile, returns its next usable producer channel.
  SwitchDMAConnection getProducerDMAChannel(Value tile) {
    return {DMAChannelDir::MM2S, producerChannelsPerTile[tile]++};
  }

  /// Given an AIE tile, returns its next usable consumer channel.
  SwitchDMAConnection getConsumerDMAChannel(Value tile) {
    return {DMAChannelDir::S2MM, consumerChannelsPerTile[tile]++};
  }
};

enum SharedMemoryDirection { LHS = -1, RHS = 1, NONE = 0 };

/// Retrieve ObjectFifoLinkOp of ObjectFifoCreateOp,
/// if it belongs to one.
std::optional<ObjectFifoLinkOp> getOptionalLinkOp(ObjectFifoCreateOp op) {
  auto device = op->getParentOfType<DeviceOp>();
  for (ObjectFifoLinkOp linkOp : device.getOps<ObjectFifoLinkOp>()) {
    for (ObjectFifoCreateOp in : linkOp.getInputObjectFifos())
      if (in == op) return {linkOp};
    for (ObjectFifoCreateOp out : linkOp.getOutputObjectFifos())
      if (out == op) return {linkOp};
  }
  return {};
}

/// Return true if the objectFifo created by createOp requires a DMA to be set
/// up. This is the case if the tiles are not adjacent (no shared memory), if
/// the objectFifo broadcasts to multiple tiles, if one of the consumers or
/// the producer wants to use the multi-dimensional address generation
/// features of the DMA, if the objectFifo is part of a LinkOp.
bool requiresDMAs(ObjectFifoCreateOp createOp,
                  SharedMemoryDirection &shareDirection,
                  std::vector<ObjectFifoCreateOp> &splitBecauseLink) {
  DeviceOp device = createOp->getParentOfType<DeviceOp>();
  auto haveSharedMemory = [&device](TileOp a, TileOp b) {
    AMDAIEDeviceModel deviceModel =
        getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));

    if ((deviceModel.isShimTile(a.getCol(), a.getRow()) &&
         !deviceModel.isShimTile(b.getCol(), b.getRow())) ||
        (!deviceModel.isShimTile(a.getCol(), a.getRow()) &&
         deviceModel.isShimTile(b.getCol(), b.getRow())))
      return NONE;

    if ((deviceModel.isMemTile(a.getCol(), a.getRow()) &&
         !deviceModel.isMemTile(b.getCol(), b.getRow())) ||
        (!deviceModel.isMemTile(a.getCol(), a.getRow()) &&
         deviceModel.isMemTile(b.getCol(), b.getRow())))
      return NONE;

    bool rightShared = deviceModel.hasLegalMemAffinity(a.getCol(), a.getRow(),
                                                       b.getCol(), b.getRow());

    bool leftShared = deviceModel.hasLegalMemAffinity(b.getCol(), b.getRow(),
                                                      a.getCol(), a.getRow());

    if (leftShared)
      return LHS;
    else if (rightShared)
      return RHS;
    else
      return NONE;
  };

  if (createOp.getConsumerTiles().size() == 1 &&
      createOp.getDimensionsToStream().empty())
    // Test for shared memory
    for (auto consumerTile : createOp.getConsumerTiles()) {
      auto consumerTileOp = dyn_cast<TileOp>(consumerTile.getDefiningOp());
      if (!consumerTileOp) continue;
      if (std::count(splitBecauseLink.begin(), splitBecauseLink.end(),
                     createOp))
        shareDirection = haveSharedMemory(createOp.getProducerTileOp(),
                                          createOp.getProducerTileOp());
      else
        shareDirection =
            haveSharedMemory(createOp.getProducerTileOp(), consumerTileOp);
    }

  if (shareDirection == LHS || shareDirection == RHS) {
    // Only test for use of data layout transformations if we are in the shared
    // memory case; otherwise, we will return `true` in any case.
    // Even if just one of the consumers in the list of consumers wants to
    // perform a memory transform, we need to use DMAs.
    for (BDDimLayoutArrayAttr dims :
         createOp.getDimensionsFromStreamPerConsumer())
      if (!dims.empty()) return true;

    // Only test for this objfifo belonging to a LinkOp if we are in the shared
    // memory case; otherwise, we will return `true` in any case.
    if (auto linkOp = getOptionalLinkOp(createOp)) {
      splitBecauseLink.push_back(createOp);
      return true;
    }
  }

  return !(shareDirection == LHS || shareDirection == RHS);
}

/// Find the size of an objectFifo after split based on
/// the maximum number of elements (of the original objectFifo) acquired
/// by a process running on given tile. If no CoreOp exists for this tile
/// return 0.
int findObjectFifoSize(DeviceOp &device, Value tile,
                       ObjectFifoCreateOp objFifo) {
  if (objFifo.size() == 0) return 0;

  AMDAIEDeviceModel deviceModel =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
  // if memTile, size is equal to objFifo size
  TileOp tileOp = tile.getDefiningOp<TileOp>();
  if (deviceModel.isMemTile(tileOp.getCol(), tileOp.getRow()))
    return objFifo.size();

  int maxAcquire = 0;
  for (auto coreOp : make_filter_range(
           device.getOps<CoreOp>(),
           [&tile](auto coreOp) { return coreOp.getTile() == tile; }))
    coreOp.walk([&](ObjectFifoAcquireOp acqOp) {
      if (acqOp.getObjectFifo() == objFifo && acqOp.acqNumber() > maxAcquire)
        maxAcquire = acqOp.acqNumber();
    });

  if (maxAcquire == 1 && objFifo.size() == 1) return 1;

  // +1 because objectFifo size is always 1 bigger than maxAcquire to allow
  // for prefetching: simplest case scenario is at least a ping-pong buffer
  if (maxAcquire > 0) return maxAcquire + 1;

  return objFifo.size();
}

/// Translate ObjectFifoCreateOp to corresponding DMABD, UseLocks, and NextBDs.
/// Work on <MemTileDMAOp> and <MemOp>.
template <typename MemOp>
void createDMA(
    DeviceOp &device, OpBuilder &builder, ObjectFifoCreateOp createOp,
    ObjectFifoCreateOp target, DMAChannelDir channelDir, int channelIndex,
    BDDimLayoutArrayAttr dims, size_t numBlocks, size_t acqNum, size_t relNum,
    int64_t len, int64_t offset,
    const DenseMap<ObjectFifoCreateOp, std::vector<BufferOp>> &buffersPerFifo,
    const DenseMap<ObjectFifoCreateOp, std::vector<LockOp>> &locksPerFifo) {
  Operation *producer = nullptr;
  for (auto memOp : device.getOps<MemOp>()) {
    if (memOp.getTile() == createOp.getProducerTile()) {
      producer = memOp.getOperation();
      break;
    }
  }

  // if none exists, create one
  TileOp objFifoTileOp = target.getProducerTileOp();
  if (!producer) {
    if (device->getNumRegions() != 1)
      llvm::report_fatal_error("expected num regions for device op");
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(device.getBody());
    auto newMemOp =
        builder.create<MemOp>(builder.getUnknownLoc(), objFifoTileOp);
    {
      OpBuilder::InsertionGuard gg(builder);
      builder.setInsertionPointToStart(&newMemOp.getRegion().emplaceBlock());
      builder.create<EndOp>(builder.getUnknownLoc());
    }
    producer = newMemOp.getOperation();
  }

  Block &endBlock = producer->getRegion(0).getBlocks().back();
  assert(!endBlock.getOps<EndOp>().empty() &&
         "expected last block to have aie.end");
  Block *lastDmaBlock = endBlock.getSinglePredecessor(),
        *dmaBlock = builder.createBlock(&endBlock),
        *bdBlock = builder.createBlock(&endBlock);

  // create DMA channel
  {
    OpBuilder::InsertionGuard gg(builder);
    builder.setInsertionPointToStart(dmaBlock);
    builder.create<DMAStartOp>(
        builder.getUnknownLoc(),
        static_cast<xilinx::AIE::DMAChannelDir>(channelDir), channelIndex,
        /*repeatCount*/ 0, bdBlock, &endBlock);
  }
  if (lastDmaBlock) lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

  auto createBdBlockOps = [&](BufferOp buff, Block *succ) {
    LockOp acqLock = channelDir == DMAChannelDir::S2MM
                         ? locksPerFifo.at(target)[0]
                         : locksPerFifo.at(target)[1],
           relLock = channelDir == DMAChannelDir::S2MM
                         ? locksPerFifo.at(target)[1]
                         : locksPerFifo.at(target)[0];
    builder.create<UseLockOp>(builder.getUnknownLoc(), acqLock,
                              LockAction::AcquireGreaterEqual, acqNum);
    if (!dims.getValue().empty())
      builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len, dims);
    else
      builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len);

    builder.create<UseLockOp>(builder.getUnknownLoc(), relLock,
                              LockAction::Release, relNum);
    builder.create<NextBDOp>(builder.getUnknownLoc(), succ);
  };

  // create Bd blocks
  Block *succ = nullptr, *curr = bdBlock;
  numBlocks = std::min(numBlocks, buffersPerFifo.at(target).size());
  for (size_t blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
    if (blockIndex == numBlocks - 1)
      succ = bdBlock;
    else
      succ = builder.createBlock(&endBlock);

    OpBuilder::InsertionGuard gg(builder);
    builder.setInsertionPointToStart(curr);
    createBdBlockOps(buffersPerFifo.at(target)[blockIndex], succ);
    curr = succ;
  }
}

/// Translate ObjectFifoCreateOp on AIE tiles to corresponding DMABD, UseLocks,
/// and NextBDs.
void createAMDAIETileDMA(
    DeviceOp &device, OpBuilder &builder, ObjectFifoCreateOp createOp,
    DMAChannelDir channelDir, int channelIndex, BDDimLayoutArrayAttr dims,
    const DenseMap<ObjectFifoLinkOp, ObjectFifoCreateOp> &objFifoLinks,
    const DenseMap<ObjectFifoCreateOp, std::vector<BufferOp>> &buffersPerFifo,
    const DenseMap<ObjectFifoCreateOp, std::vector<LockOp>> &locksPerFifo) {
  size_t numBlocks = createOp.size();
  if (numBlocks == 0) return;
  // search for the buffers/locks (based on if this objFifo has a link)
  ObjectFifoCreateOp target = createOp;
  if (std::optional<ObjectFifoLinkOp> linkOp = getOptionalLinkOp(createOp);
      linkOp.has_value())
    if (objFifoLinks.contains(linkOp.value()))
      target = objFifoLinks.at(linkOp.value());

  auto fifo = llvm::cast<AIEObjectFifoType>(createOp.getElemType());
  auto elemType = llvm::cast<MemRefType>(fifo.getElementType());
  int64_t len = elemType.getNumElements();
  createDMA<MemOp>(device, builder, createOp, target, channelDir, channelIndex,
                   dims, numBlocks,
                   /*acqNum*/ 1, /*relNum*/ 1, len, /*offset*/ 0,
                   buffersPerFifo, locksPerFifo);
}

/// Translate ObjectFifoCreateOp on Mem tiles to corresponding DMABD, UseLocks,
/// and NextBDs.
void createMemTileDMA(
    DeviceOp &device, OpBuilder &builder, ObjectFifoCreateOp createOp,
    DMAChannelDir channelDir, int channelIndex, BDDimLayoutArrayAttr dims,
    const DenseMap<ObjectFifoLinkOp, ObjectFifoCreateOp> &objFifoLinks,
    const DenseMap<ObjectFifoCreateOp, std::vector<BufferOp>> &buffersPerFifo,
    const DenseMap<ObjectFifoCreateOp, std::vector<LockOp>> &locksPerFifo) {
  size_t numBlocks = createOp.size();
  if (numBlocks == 0) return;

  auto fifo = llvm::cast<AIEObjectFifoType>(createOp.getElemType());
  auto elemType = llvm::cast<MemRefType>(fifo.getElementType());
  int64_t lenOut = elemType.getNumElements();
  size_t acqNum = 1;
  size_t relNum = 1;
  // offset based on order of this op in join/distribute list
  int64_t extraOffset = 0;
  ObjectFifoCreateOp target = createOp;

  auto getExtraOffset = [&acqNum, &relNum, &extraOffset, &target, &createOp](
                            ObjectFifoLinkOp linkOp,
                            const std::vector<ObjectFifoCreateOp> &fifos,
                            size_t size) {
    if (target == createOp) {
      acqNum = size;
      relNum = size;
    } else
      for (auto fifoIn : fifos) {
        auto fifoType = llvm::cast<AIEObjectFifoType>(fifoIn.getElemType());
        auto fifoElemType = llvm::cast<MemRefType>(fifoType.getElementType());
        if (fifoIn.name() == createOp.name()) break;
        extraOffset += fifoElemType.getNumElements();
      }
  };

  // search for the buffers/locks (based on if this objFifo has a link)
  // identify size difference between input and output memrefs
  if (auto linkOp = getOptionalLinkOp(createOp);
      objFifoLinks.contains(*linkOp)) {
    target = objFifoLinks.at(*linkOp);
    if (linkOp->isJoin()) {
      // find offset based on order of this op in join list
      getExtraOffset(*linkOp, linkOp->getInputObjectFifos(),
                     linkOp->getFifoIns().size());
    } else if (linkOp->isDistribute()) {
      // find offset based on order of this op in distribute list
      getExtraOffset(*linkOp, linkOp->getOutputObjectFifos(),
                     linkOp->getFifoOuts().size());

    } else if (target != createOp) {
      auto targetFifo = llvm::cast<AIEObjectFifoType>(target.getElemType());
      auto targetElemType = llvm::cast<MemRefType>(targetFifo.getElementType());
      lenOut = targetElemType.getNumElements();
    }

    // check if current createOp is of smaller size in link
    if (target != createOp) numBlocks = target.size();
  }

  createDMA<MemTileDMAOp>(device, builder, createOp, target, channelDir,
                          channelIndex, dims, numBlocks, acqNum, relNum, lenOut,
                          extraOffset, buffersPerFifo, locksPerFifo);
}

/// Unroll for-loops that contain objectFifo operations.
LogicalResult unrollForLoops(DeviceOp &device,
                             const std::set<TileOp> &objectFifoTiles) {
  for (auto coreOp : device.getOps<CoreOp>()) {
    if (!objectFifoTiles.count(coreOp.getTileOp())) continue;

    WalkResult res = coreOp.walk([&](scf::ForOp forLoop) {
      // look for operations on objectFifos
      // when multiple fifos in same loop, must use the smallest
      // common multiplier as the unroll factor
      bool found = false;
      std::set<int> objFifoSizes;
      Block *body = forLoop.getBody();
      for (auto acqOp : body->getOps<ObjectFifoAcquireOp>()) {
        if (acqOp.getOperation()->getParentOp() == forLoop) {
          found = true;
          ObjectFifoCreateOp op = acqOp.getObjectFifo();
          objFifoSizes.insert(op.size());
        }
      }
      // also counts original loop body
      int unrollFactor = std::accumulate(
          objFifoSizes.begin(), objFifoSizes.end(), 1, std::lcm<int, int>);
      if (found && failed(mlir::loopUnrollByFactor(forLoop, unrollFactor))) {
        forLoop.emitOpError()
            << "could not be unrolled with unrollFactor: " << unrollFactor
            << "\n";
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    if (res.wasInterrupted()) return failure();
  }

  return success();
}

/// Create a UseLockOp based on input parameters. `acc` is an accumulator map
/// that tracks the indices of the next locks to acquire (or release). Uses op
/// to find index of acc for next lockID and updates acc.
void createUseLocks(
    OpBuilder &builder, ObjectFifoCreateOp op, ObjectFifoPort port,
    DenseMap<std::pair<ObjectFifoCreateOp, int>, int> &acc, size_t numLocks,
    LockAction lockAction,
    const DenseMap<ObjectFifoLinkOp, ObjectFifoCreateOp> &objFifoLinks,
    const DenseMap<ObjectFifoCreateOp, std::vector<LockOp>> &locksPerFifo) {
  ObjectFifoCreateOp target = op;
  if (auto linkOp = getOptionalLinkOp(op))
    if (objFifoLinks.contains(*linkOp)) target = objFifoLinks.at(*linkOp);

  if (numLocks == 0) return;
  // search for the correct lock based on the port of the acq/rel
  // operation e.g. acq as consumer is the read lock (second)
  LockOp lock;
  std::vector<LockOp> locks = locksPerFifo.at(target);
  if (lockAction == LockAction::AcquireGreaterEqual)
    lock = port == ObjectFifoPort::Produce ? locks[0] : locks[1];
  else
    lock = port == ObjectFifoPort::Produce ? locks[1] : locks[0];

  builder.create<UseLockOp>(builder.getUnknownLoc(), lock, lockAction,
                            numLocks);
  std::pair<ObjectFifoCreateOp, int> opPort = {op, static_cast<int>(port)};
  acc[opPort] =
      (acc[opPort] + numLocks) % op.size();  // update to next objFifo elem
}

/// Replace (not really - add) ObjectFifoReleaseOp with appropriate UseLockOp.
void replaceReleaseOp(
    ObjectFifoReleaseOp releaseOp, OpBuilder builder,
    DenseMap<std::pair<ObjectFifoCreateOp, int>, int> &relPerFifo,
    const DenseMap<ObjectFifoLinkOp, ObjectFifoCreateOp> &objFifoLinks,
    const DenseMap<ObjectFifoCreateOp, std::vector<LockOp>> &locksPerFifo,
    DenseMap<std::pair<ObjectFifoCreateOp, int>,
             std::vector<ObjectFifoReleaseOp>> &releaseOps) {
  ObjectFifoCreateOp op = releaseOp.getObjectFifo();
  auto core = releaseOp->getParentOfType<CoreOp>();
  if (auto linkOp = getOptionalLinkOp(op))
    if (core.getTile() == *linkOp->getOptionalSharedTile())
      llvm::report_fatal_error(
          "currently cannot access objectFifo used in "
          "ObjectFifoLinkOp");

  auto port = releaseOp.getPort();
  std::pair<ObjectFifoCreateOp, int> opPort = {op, static_cast<int>(port)};
  // update index of next element to release for this objectFifo
  (void)relPerFifo[opPort];
  // release locks
  {
    OpBuilder::InsertionGuard gg(builder);
    builder.setInsertionPointAfter(releaseOp);
    createUseLocks(builder, op, port, relPerFifo, releaseOp.relNumber(),
                   LockAction::Release, objFifoLinks, locksPerFifo);
  }
  // register release op
  if (releaseOps.contains(opPort)) {
    releaseOps[opPort].push_back(releaseOp);
  } else {
    releaseOps[opPort] = {releaseOp};
  }
}

/// Split objectFifos into a consumer end and producer end if needed
void splitFifo(
    DeviceOp device, ObjectFifoCreateOp createOp, OpBuilder builder,
    std::vector<std::pair<ObjectFifoCreateOp, std::vector<ObjectFifoCreateOp>>>
        &splitFifos) {
  ArrayRef<BDDimLayoutArrayAttr> consumerDims =
      createOp.getDimensionsFromStreamPerConsumer();
  int consumerDepth;
  BDDimLayoutArrayAttr emptyDims =
      BDDimLayoutArrayAttr::get(builder.getContext(), {});
  std::string consumerFifoName;

  auto replaceSplitFifo = [&createOp](ObjectFifoCreateOp newOp, TileOp tile) {
    auto original =
        createOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    auto newSymbol =
        newOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    for (auto user : tile->getUsers())
      if (isa<CoreOp>(user) &&
          failed(SymbolTable::replaceAllSymbolUses(original, newSymbol, user)))
        llvm::report_fatal_error("couldn't replace all symbols uses");
  };

  int consumerIndex = 0;
  auto datatype = cast<AIEObjectFifoType>(createOp.getElemType());
  MLIRContext *ctx = builder.getContext();
  std::vector<ObjectFifoCreateOp> splitConsumerFifos;
  for (auto consumerTile : createOp.getConsumerTiles()) {
    auto consumerTileOp = cast<TileOp>(consumerTile.getDefiningOp());
    if (isa<ArrayAttr>(createOp.getElemNumber()))
      // +1 to account for 1st depth (producer)
      consumerDepth = createOp.size(consumerIndex + 1);
    else
      consumerDepth = findObjectFifoSize(device, consumerTileOp, createOp);

    auto consumerObjFifoSize =
        builder.getIntegerAttr(builder.getI32Type(), consumerDepth);
    // rename and replace split objectFifo
    if (createOp.getConsumerTiles().size() > 1)
      consumerFifoName =
          createOp.name().str() + "_" + std::to_string(consumerIndex) + "_cons";
    else
      consumerFifoName = createOp.name().str() + "_cons";

    BDDimLayoutArrayAttr singletonFromStreamDims =
        BDDimLayoutArrayAttr::get(ctx, {consumerDims[consumerIndex]});
    BDDimLayoutArrayArrayAttr fromStreamDims =
        BDDimLayoutArrayArrayAttr::get(ctx, singletonFromStreamDims);

    {
      builder.setInsertionPointAfter(createOp);
      ObjectFifoCreateOp consumerFifo = builder.create<ObjectFifoCreateOp>(
          builder.getUnknownLoc(), consumerFifoName, consumerTile, consumerTile,
          consumerObjFifoSize, datatype, emptyDims, fromStreamDims);
      replaceSplitFifo(consumerFifo, consumerTileOp);

      // record that this objectFifo was split; it will require DMA config
      splitConsumerFifos.push_back(consumerFifo);
    }

    // update the linkOp if the split objFifo was originally its start point
    consumerIndex++;
    auto linkOp = getOptionalLinkOp(createOp);
    if (!linkOp) continue;
    for (ObjectFifoCreateOp fifoIn : linkOp->getInputObjectFifos())
      if (fifoIn.name() == createOp.name() &&
          consumerTile == *linkOp->getOptionalSharedTile() &&
          failed(SymbolTable::replaceAllSymbolUses(
              createOp, StringAttr::get(ctx, consumerFifoName),
              linkOp->getOperation())))
        llvm::report_fatal_error("unable to update all symbol uses");
  }

  assert(!splitConsumerFifos.empty() &&
         "expected some fifos to have been split");
  splitFifos.emplace_back(createOp, splitConsumerFifos);
}

/// Replace (not really - add) ObjectFifoAcquireOp with appropriate UseLockOp.
void replaceObjectAcquireOp(
    ObjectFifoAcquireOp acquireOp, OpBuilder builder,
    DenseMap<std::pair<ObjectFifoCreateOp, int>, int> &acqPerFifo,
    DenseMap<std::pair<ObjectFifoCreateOp, int>,
             std::vector<ObjectFifoReleaseOp>> &releaseOps,
    DenseMap<std::pair<ObjectFifoCreateOp, int>, std::vector<int>>
        &acquiresPerFifo,
    const DenseMap<ObjectFifoLinkOp, ObjectFifoCreateOp> &objFifoLinks,
    const DenseMap<ObjectFifoCreateOp, std::vector<LockOp>> &locksPerFifo,
    const DenseMap<ObjectFifoCreateOp, std::vector<BufferOp>> &buffersPerFifo,
    DenseMap<ObjectFifoAcquireOp, std::vector<BufferOp>> &subviews) {
  ObjectFifoCreateOp op = acquireOp.getObjectFifo();
  auto core = acquireOp->getParentOfType<CoreOp>();
  auto linkOp = getOptionalLinkOp(op);
  if (linkOp && core.getTile() == *linkOp->getOptionalSharedTile())
    llvm::report_fatal_error(
        "currently cannot access objectFifo used in "
        "ObjectFifoLinkOp");

  // index of next element to acquire for this objectFifo
  // useful for keeping track of which
  // indices are acquired
  auto port = acquireOp.getPort();
  std::pair<ObjectFifoCreateOp, int> opPort = {op, static_cast<int>(port)};
  int start = acqPerFifo[opPort];
  Block *acqBlock = acquireOp.getOperation()->getBlock();
  Operation *acqBlockDefOp = acqBlock->getParentOp();
  Block *acqParentOpBlock = acqBlockDefOp->getBlock();

  // check how many elements have been released in between this AcquireOp
  // and the previous one
  int numRel = 0;
  for (auto relOp : releaseOps[opPort]) {
    if (relOp.getObjectFifo() != op) continue;
    Block *relBlock = relOp.getOperation()->getBlock();
    Operation *relBlockDefOp = relBlock->getParentOp();
    Block *relParentOpBlock = relBlockDefOp->getBlock();
    // TODO: operations may not be in the same block: currently only
    // support one block level of difference
    // TODO(max): don't try to simplify these conditionals because the
    // conditions aren't mutually exclusive...

    // if they are already in the same block, check if releaseOp
    // happened before
    if (acqBlock == relBlock) {
      if (!acquireOp->isBeforeInBlock(relOp)) {
        // to ensure that we do not account
        // the ReleaseOps again later,
        // after the subview is created
        releaseOps[opPort].erase(releaseOps[opPort].begin());
        numRel += relOp.relNumber();
      }
    } else if (acqBlock == relParentOpBlock) {
      if (!acquireOp->isBeforeInBlock(relBlockDefOp)) {
        releaseOps[opPort].erase(releaseOps[opPort].begin());
        numRel += relOp.relNumber();
      }
    } else if (relBlock == acqParentOpBlock) {
      if (!acqBlockDefOp->isBeforeInBlock(relOp)) {
        releaseOps[opPort].erase(releaseOps[opPort].begin());
        numRel += relOp.relNumber();
      }
    } else if (acqParentOpBlock == relParentOpBlock) {
      if (!acqBlockDefOp->isBeforeInBlock(relBlockDefOp)) {
        releaseOps[opPort].erase(releaseOps[opPort].begin());
        numRel += relOp.relNumber();
      }
    }
  }

  // track indices of elements to acquire
  std::vector<int> acquiredIndices;
  if (!acquiresPerFifo[opPort].empty()) {
    // take into account what has already been acquired by previous
    // AcquireOp in program order
    acquiredIndices = acquiresPerFifo[opPort];
    // take into account what has been released in-between
    if (static_cast<size_t>(numRel) > acquiredIndices.size())
      llvm::report_fatal_error(
          "cannot release more elements than are "
          "already acquired");
    for (int i = 0; i < numRel; i++)
      acquiredIndices.erase(acquiredIndices.begin());
  }

  // acquire locks
  size_t numLocks = acquireOp.acqNumber();
  size_t alreadyAcq = acquiredIndices.size();
  size_t numCreate = numLocks > alreadyAcq ? numLocks - alreadyAcq : 0;

  {
    OpBuilder::InsertionGuard gg(builder);
    builder.setInsertionPointAfter(acquireOp);
    createUseLocks(builder, op, port, acqPerFifo, numCreate,
                   LockAction::AcquireGreaterEqual, objFifoLinks, locksPerFifo);
  }

  // if objFifo was linked with others, find which objFifos
  // elements to use
  ObjectFifoCreateOp target = op;
  if (linkOp && objFifoLinks.contains(*linkOp))
    target = objFifoLinks.at(*linkOp);

  // create subview: buffers that were already acquired + new acquires
  for (int i = 0; i < numCreate; i++) {
    acquiredIndices.push_back(start);
    start = (start + 1) % op.size();
  }
  std::vector<BufferOp> subviewRefs;
  subviewRefs.reserve(acquiredIndices.size());
  for (auto index : acquiredIndices)
    subviewRefs.push_back(buffersPerFifo.at(target)[index]);

  subviews[acquireOp] = subviewRefs;
  acquiresPerFifo[opPort] = acquiredIndices;
}

/// Create objectFifo buffers and locks. Also populate a list of tiles
/// containing objectFifos for later processing of the acquires/releases (uses
/// of the FIFO).
void createBuffersAndLocks(
    OpBuilder builder, DeviceOp device, ObjectFifoCreateOp createOp,
    std::vector<ObjectFifoCreateOp> &splitBecauseLink,
    std::set<TileOp> &objectFifoTiles, LockAnalysis &lockAnalysis,
    DenseMap<ObjectFifoLinkOp, ObjectFifoCreateOp> &objFifoLinks,
    DenseMap<ObjectFifoCreateOp, std::vector<BufferOp>> &buffersPerFifo,
    DenseMap<ObjectFifoCreateOp, std::vector<LockOp>> &locksPerFifo) {
  // add all tiles that contain an objectFifo to objectFifoTiles for later
  // loop unrolling pass
  objectFifoTiles.insert(createOp.getProducerTileOp());
  for (auto consumerTile : createOp.getConsumerTiles()) {
    auto consumerTileOp = cast<TileOp>(consumerTile.getDefiningOp());
    objectFifoTiles.insert(consumerTileOp);
  }

  // if split, the necessary size for producer fifo might change
  SharedMemoryDirection shareDirection = NONE;
  if (requiresDMAs(createOp, shareDirection, splitBecauseLink)) {
    IntegerAttr elemNumberAttr;
    if (isa<ArrayAttr>(createOp.getElemNumber()))
      elemNumberAttr = builder.getI32IntegerAttr(createOp.size());
    else
      elemNumberAttr = builder.getI32IntegerAttr(
          findObjectFifoSize(device, createOp.getProducerTileOp(), createOp));
    createOp.setElemNumberAttr(elemNumberAttr);
  }

  if (!createOp.size()) return;

  auto fifo = llvm::cast<AIEObjectFifoType>(createOp.getElemType());
  auto elemType = llvm::cast<MemRefType>(fifo.getElementType());

  // if this objectFifo is linked to another, check if the other's elements
  // have already been created (the elements that are created are those of
  // the objFifo with elements of bigger size)
  auto linkOp = getOptionalLinkOp(createOp);
  if (linkOp) {
    auto fifoIn = linkOp->getInputObjectFifos()[0],
         fifoOut = linkOp->getOutputObjectFifos()[0];
    // elements have already been created
    if (objFifoLinks.contains(*linkOp)) return;
    // if join, fifoOut has bigger size
    if (linkOp->isJoin() && createOp.name() != fifoOut.name()) return;
    // if distribute, fifoIn has bigger size
    if (linkOp->isDistribute() && createOp.name() != fifoIn.name()) return;

    auto fifoInType = llvm::cast<AIEObjectFifoType>(
             linkOp->getInputObjectFifos()[0].getElemType()),
         fifoOutType = llvm::cast<AIEObjectFifoType>(
             linkOp->getOutputObjectFifos()[0].getElemType());
    auto elemInType = llvm::cast<MemRefType>(fifoInType.getElementType()),
         elemOutType = llvm::cast<MemRefType>(fifoOutType.getElementType());
    int64_t inSize = elemInType.getNumElements();
    if (int64_t outSize = elemOutType.getNumElements(); inSize >= outSize) {
      if (createOp.name() != fifoIn.name()) return;
    } else if (linkOp->getOutputObjectFifos()[0] != createOp)
      return;
  }

  TileOp creationTile;
  if (shareDirection == NONE || shareDirection == LHS)
    creationTile = createOp.getProducerTileOp();
  else
    creationTile = cast<TileOp>(createOp.getConsumerTiles()[0].getDefiningOp());

  // Reset opbuilder location to after the last tile declaration
  auto tiles =
      createOp->getParentOfType<DeviceOp>().getBody()->getOps<TileOp>();
  assert(!tiles.empty() && "no tiles in device");

  OpBuilder::InsertionGuard gg(builder);
  builder.setInsertionPointAfter(*std::prev(tiles.end(), 1));

  size_t numElem = createOp.size();
  // if shimTile external buffers are collected from input code
  AMDAIEDeviceModel deviceModel =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
  if (!deviceModel.isShimTile(creationTile.getCol(), creationTile.getRow())) {
    std::vector<BufferOp> buffers;
    // create as many locks as there are external buffers
    for (int ofElemIndex = 0; ofElemIndex < numElem; ofElemIndex++) {
      auto buff = builder.create<BufferOp>(
          builder.getUnknownLoc(), elemType, creationTile,
          builder.getStringAttr(createOp.name().str() + "_buff_" +
                                std::to_string(ofElemIndex)),
          /*address*/ nullptr, /*initial_value*/ nullptr,
          /*mem_bank*/ nullptr);
      buffers.push_back(buff);
    }
    buffersPerFifo[createOp] = buffers;
  }

  if (linkOp) {
    if (linkOp->isDistribute())
      numElem *= linkOp->getFifoOuts().size();
    else if (linkOp->isJoin())
      numElem *= linkOp->getFifoIns().size();
    objFifoLinks[*linkOp] = createOp;
  }

  if (deviceModel.isShimTile(creationTile.getCol(), creationTile.getRow()))
    numElem = 0;

  // create corresponding aie2 locks
  int prodLockID = lockAnalysis.getLockID(creationTile);
  if (prodLockID < 0) {
    creationTile->emitOpError("No more locks to allocate!");
    assert(prodLockID >= 0);
  }
  auto prodLock = builder.create<LockOp>(builder.getUnknownLoc(), creationTile,
                                         prodLockID, numElem);
  prodLock.getOperation()->setAttr(
      SymbolTable::getSymbolAttrName(),
      builder.getStringAttr(createOp.name().str() + "_prod_lock"));
  std::vector<LockOp> locks{prodLock};

  int consLockID = lockAnalysis.getLockID(creationTile);
  if (consLockID < 0) {
    creationTile->emitOpError("No more locks to allocate!");
    assert(consLockID >= 0);
  }
  auto consLock = builder.create<LockOp>(builder.getUnknownLoc(), creationTile,
                                         consLockID, 0);
  consLock.getOperation()->setAttr(
      SymbolTable::getSymbolAttrName(),
      builder.getStringAttr(createOp.name().str() + "_cons_lock"));
  locks.push_back(consLock);

  locksPerFifo[createOp] = locks;
}

/// Translate ObjectFifoCreateOp ops into routing primitives (Flows) and DMA
/// primitives (DMABD, DMAStart, Buffer, UseLock).
void createFlowsAndTileDMAs(
    OpBuilder builder, DeviceOp device, ObjectFifoCreateOp producer,
    const std::vector<ObjectFifoCreateOp> &consumers,
    DMAChannelAnalysis &dmaAnalysis,
    const DenseMap<ObjectFifoCreateOp, std::vector<LockOp>> &locksPerFifo,
    const DenseMap<ObjectFifoLinkOp, ObjectFifoCreateOp> &objFifoLinks,
    const DenseMap<ObjectFifoCreateOp, std::vector<BufferOp>> &buffersPerFifo) {
  AMDAIEDeviceModel deviceModel =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
  auto createDMA = [&deviceModel, &device, &builder, &locksPerFifo,
                    &objFifoLinks, &buffersPerFifo](
                       ObjectFifoCreateOp op, DMAChannelDir channelDir,
                       int channelIndex, BDDimLayoutArrayAttr dims) {
    auto producerOp = op.getProducerTileOp();
    if (deviceModel.isShimTile(producerOp.getCol(), producerOp.getRow()))
      return;
    else if (deviceModel.isMemTile(producerOp.getCol(), producerOp.getRow()))
      createMemTileDMA(device, builder, op, channelDir, channelIndex, dims,
                       objFifoLinks, buffersPerFifo, locksPerFifo);
    else
      createAMDAIETileDMA(device, builder, op, channelDir, channelIndex, dims,
                          objFifoLinks, buffersPerFifo, locksPerFifo);
  };
  // create producer tile DMA
  SwitchDMAConnection producerChan =
      dmaAnalysis.getProducerDMAChannel(producer.getProducerTile());
  createDMA(producer, static_cast<DMAChannelDir>(producerChan.direction),
            producerChan.channel, producer.getDimensionsToStreamAttr());
  // generate objectFifo allocation info
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(&device.getBody()->back());
  if (deviceModel.isShimTile(producer.getProducerTileOp().getCol(),
                             producer.getProducerTileOp().getRow()))
    builder.create<ShimDMAAllocationOp>(
        builder.getUnknownLoc(), producer.getName(),
        static_cast<xilinx::AIE::DMAChannelDir>(producerChan.direction),
        producerChan.channel, producer.getProducerTileOp().getCol());

  for (auto consumer : consumers) {
    // create consumer tile DMA
    SwitchDMAConnection consumerChan =
        dmaAnalysis.getConsumerDMAChannel(consumer.getProducerTile());
    BDDimLayoutArrayAttr consumerDims =
        consumer.getDimensionsFromStreamPerConsumer()[0];
    createDMA(consumer, static_cast<DMAChannelDir>(consumerChan.direction),
              consumerChan.channel, consumerDims);
    // generate objectFifo allocation info
    OpBuilder::InsertionGuard gg(builder);
    builder.setInsertionPoint(&device.getBody()->back());
    if (deviceModel.isShimTile(consumer.getProducerTileOp().getCol(),
                               consumer.getProducerTileOp().getRow()))
      builder.create<ShimDMAAllocationOp>(
          builder.getUnknownLoc(), producer.getName(),
          static_cast<xilinx::AIE::DMAChannelDir>(consumerChan.direction),
          consumerChan.channel, consumer.getProducerTileOp().getCol());

    // create flow
    {
      OpBuilder::InsertionGuard ggg(builder);
      builder.setInsertionPointAfter(producer);
      builder.create<FlowOp>(builder.getUnknownLoc(),
                             producer.getProducerTile(), WireBundle::DMA,
                             producerChan.channel, consumer.getProducerTile(),
                             WireBundle::DMA, consumerChan.channel);
    }
  }
}

namespace mlir::iree_compiler::AMDAIE {
struct AMDAIEObjectFifoStatefulTransformPass : mlir::OperationPass<DeviceOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      AMDAIEObjectFifoStatefulTransformPass)

  AMDAIEObjectFifoStatefulTransformPass()
      : mlir::OperationPass<DeviceOp>(resolveTypeID()) {}

  llvm::StringRef getArgument() const override {
    return "amdaie-objectFifo-stateful-transform";
  }

  llvm::StringRef getName() const override {
    return " AMDAIEObjectFifoStatefulTransformPass";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<AMDAIEObjectFifoStatefulTransformPass>(
        *static_cast<const AMDAIEObjectFifoStatefulTransformPass *>(this));
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    LockAnalysis lockAnalysis(device);
    DMAChannelAnalysis dmaAnalysis(device);
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());
    // track cores to check for loops during unrolling
    std::set<TileOp> objectFifoTiles;
    // maps each objFifo to its corresponding buffer
    DenseMap<ObjectFifoCreateOp, std::vector<BufferOp>> buffersPerFifo;
    // maps each objFifo to its corresponding locks
    DenseMap<ObjectFifoCreateOp, std::vector<LockOp>> locksPerFifo;
    // maps each objFifo between non-adjacent tiles to its
    // corresponding consumer objectFifos
    std::vector<std::pair<ObjectFifoCreateOp, std::vector<ObjectFifoCreateOp>>>
        splitFifos;
    // maps each ObjectFifoLinkOp to objFifo whose elements
    // have been created and should be used
    DenseMap<ObjectFifoLinkOp, ObjectFifoCreateOp> objFifoLinks;
    // objfifos which have been split because they are
    // part of a Link, not because they didn't have a shared memory module
    std::vector<ObjectFifoCreateOp> splitBecauseLink;

    llvm::SmallVector<ObjectFifoCreateOp> createFifoOps =
        llvm::to_vector(device.getOps<ObjectFifoCreateOp>());
    for (ObjectFifoCreateOp createOp : createFifoOps) {
      if (auto _shareDirection = NONE;
          !requiresDMAs(createOp, _shareDirection, splitBecauseLink))
        continue;
      splitFifo(device, createOp, builder, splitFifos);
    }

    for (ObjectFifoCreateOp createOp : device.getOps<ObjectFifoCreateOp>())
      createBuffersAndLocks(builder, device, createOp, splitBecauseLink,
                            objectFifoTiles, lockAnalysis, objFifoLinks,
                            buffersPerFifo, locksPerFifo);

    // Only the objectFifos we split above require DMA communication; the others
    // rely on shared memory and share the same buffers.
    for (auto &[producer, consumers] : splitFifos)
      createFlowsAndTileDMAs(builder, device, producer, consumers, dmaAnalysis,
                             locksPerFifo, objFifoLinks, buffersPerFifo);

    if (failed(unrollForLoops(device, objectFifoTiles))) signalPassFailure();

    // Replace ops
    for (auto coreOp : device.getOps<CoreOp>()) {
      // maps each "subview" to its buffer references (subviews
      // are created by AcquireOps)
      DenseMap<ObjectFifoAcquireOp, std::vector<BufferOp>> subviews;
      // maps each objFifo to indices of buffers acquired
      // in latest subview of that objFifo (useful to
      // cascade acquired elements to next AcquireOp)
      DenseMap<std::pair<ObjectFifoCreateOp, int>, std::vector<int>>
          acquiresPerFifo;
      // useful to check which ReleaseOp has taken place before
      // an AcquireOp per objFifo
      DenseMap<std::pair<ObjectFifoCreateOp, int>,
               std::vector<ObjectFifoReleaseOp>>
          releaseOps;
      // maps each objFifo to its next index to acquire within
      // this CoreOp
      DenseMap<std::pair<ObjectFifoCreateOp, int>, int> acqPerFifo;
      // maps each objFifo to its next index to release within
      // this CoreOp
      DenseMap<std::pair<ObjectFifoCreateOp, int>, int> relPerFifo;

      coreOp.walk([&](ObjectFifoReleaseOp releaseOp) {
        replaceReleaseOp(releaseOp, builder, relPerFifo, objFifoLinks,
                         locksPerFifo, releaseOps);
      });

      coreOp.walk([&](ObjectFifoAcquireOp acquireOp) {
        replaceObjectAcquireOp(acquireOp, builder, acqPerFifo, releaseOps,
                               acquiresPerFifo, objFifoLinks, locksPerFifo,
                               buffersPerFifo, subviews);
      });

      // Replace subview.access ops
      coreOp.walk([&](ObjectFifoSubviewAccessOp accessOp) {
        auto acqOp = accessOp.getSubview().getDefiningOp<ObjectFifoAcquireOp>();
        if (ObjectFifoCreateOp op = acqOp.getObjectFifo();
            getOptionalLinkOp(op))
          llvm::report_fatal_error(
              "currently cannot access objectFifo used in "
              "ObjectFifoLinkOp");
        accessOp.getOutput().replaceAllUsesWith(
            subviews[acqOp][accessOp.getIndex()].getBuffer());
      });
    }

    // make global symbols to replace the to be erased ObjectFifoCreateOps
    for (auto createOp : device.getOps<ObjectFifoCreateOp>()) {
      OpBuilder::InsertionGuard gg(builder);
      builder.setInsertionPointToStart(&device.getBodyRegion().front());
      auto symName = createOp.getName();
      createOp->setAttr(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr("__erase_" + symName));
      auto memrefType = llvm::cast<AIEObjectFifoType>(createOp.getElemType())
                            .getElementType();
      builder.create<memref::GlobalOp>(builder.getUnknownLoc(), symName,
                                       builder.getStringAttr("public"),
                                       memrefType, nullptr, false, nullptr);
    }

    // Remove old ops
    SetVector<Operation *> opsToErase;
    device.walk([&](Operation *op) {
      if (isa<ObjectFifoCreateOp, ObjectFifoLinkOp,
              ObjectFifoRegisterExternalBuffersOp, ObjectFifoAcquireOp,
              ObjectFifoSubviewAccessOp, ObjectFifoReleaseOp>(op))
        opsToErase.insert(op);
    });
    topologicalSort(opsToErase);
    IRRewriter rewriter(&getContext());
    for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it)
      (*it)->erase();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
createAMDAIEObjectFifoStatefulTransformPass() {
  return std::make_unique<AMDAIEObjectFifoStatefulTransformPass>();
}

void registerAMDAIEObjectFifoStatefulTransform() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAMDAIEObjectFifoStatefulTransformPass();
  });
}
}  // namespace mlir::iree_compiler::AMDAIE
