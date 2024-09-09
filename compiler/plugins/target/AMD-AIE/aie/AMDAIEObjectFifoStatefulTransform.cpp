// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AIEDialect.h"
#include "Passes.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "amdaie-objectFifo-stateful-transform"

using namespace mlir;
using namespace mlir::iree_compiler::AMDAIE;

using xilinx::AIE::AIEObjectFifoType;
using xilinx::AIE::BDDimLayoutArrayAttr;
using xilinx::AIE::BufferOp;
using xilinx::AIE::CoreOp;
using xilinx::AIE::DeviceOp;
using xilinx::AIE::DMABDOp;
using xilinx::AIE::DMAStartOp;
using xilinx::AIE::EndOp;
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
using xilinx::AIE::ObjectFifoReleaseOp;
using xilinx::AIE::ObjectFifoSubviewAccessOp;
using xilinx::AIE::ShimDMAAllocationOp;
using xilinx::AIE::TileOp;
using xilinx::AIE::UseLockOp;

namespace {

struct LockResources {
  // Reference to the producer and consumer lock ops created for this resource.
  std::pair<LockOp, LockOp> locks;
  // The acquire and release values to be used for producer and consumer locks
  // for this resource.
  std::pair<uint8_t, uint8_t> locksAcqRel;
  LockResources() {}
  LockResources(const std::pair<LockOp, LockOp> &locks,
                const std::pair<uint8_t, uint8_t> &locksAcqRel)
      : locks(locks), locksAcqRel(locksAcqRel) {}
};

struct ObjectFifoEndpointResource {
  // The buffers used for this objectFifo endpoint (multiple: double buffering).
  SmallVector<BufferOp> buffers;
  // The lock resources used for this objectFifo endpoint.
  LockResources lockResources;
  ObjectFifoEndpointResource() {}
  ObjectFifoEndpointResource(const SmallVector<BufferOp> &buffers,
                             LockResources &&lockResources)
      : buffers(buffers), lockResources(std::move(lockResources)) {}
};

struct ObjectFifoResources {
  // Offset on the producer's side of the objectFifo.
  uint32_t producerOffset{0};
  ObjectFifoEndpointResource producerResource;
  // Offset on the consumers' side of the objectFifo.
  uint32_t consumersOffset{0};
  DenseMap<TileOp, ObjectFifoEndpointResource> consumerResources;
  ObjectFifoResources() {}
  ObjectFifoResources(uint32_t producerOffset, uint32_t consumersOffset)
      : producerOffset(producerOffset), consumersOffset(consumersOffset) {}
};

SmallVector<ObjectFifoCreateOp> getInputObjectFifos(ObjectFifoLinkOp &op) {
  SmallVector<ObjectFifoCreateOp> inputObjFifos;
  Operation *parent = op.getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      for (auto sym : op.getFifoIns()) {
        auto name = dyn_cast<FlatSymbolRefAttr>(sym);
        if (auto *st = SymbolTable::lookupSymbolIn(parent, name);
            isa_and_nonnull<ObjectFifoCreateOp>(st))
          inputObjFifos.push_back(dyn_cast<ObjectFifoCreateOp>(st));
      }
    }
  }
  return inputObjFifos;
}

SmallVector<ObjectFifoCreateOp> getOutputObjectFifos(ObjectFifoLinkOp &op) {
  SmallVector<ObjectFifoCreateOp> outputObjFifos;
  Operation *parent = op.getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      for (auto sym : op.getFifoOuts()) {
        auto name = dyn_cast<FlatSymbolRefAttr>(sym);
        if (auto *st = SymbolTable::lookupSymbolIn(parent, name);
            isa_and_nonnull<ObjectFifoCreateOp>(st))
          outputObjFifos.push_back(dyn_cast<ObjectFifoCreateOp>(st));
      }
    }
  }
  return outputObjFifos;
}

int objFifoSize(ObjectFifoCreateOp op, int index = 0) {
  if (llvm::isa<mlir::ArrayAttr>(op.getElemNumber())) {
    return llvm::dyn_cast<mlir::IntegerAttr>(
               llvm::dyn_cast<mlir::ArrayAttr>(op.getElemNumber())[index])
        .getInt();
  } else {
    return llvm::dyn_cast<mlir::IntegerAttr>(op.getElemNumber()).getInt();
  }
}

template <typename T>
ObjectFifoCreateOp getObjectFifo(T op) {
  Operation *parent = op.getOperation();
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      if (auto *st = SymbolTable::lookupSymbolIn(parent, op.getObjFifoName());
          isa_and_nonnull<ObjectFifoCreateOp>(st))
        return dyn_cast<ObjectFifoCreateOp>(st);
    }
  }
  return {};
}

bool isJoin(ObjectFifoLinkOp op) {
  return op.getFifoIns().size() > 1 && op.getFifoOuts().size() == 1;
}

bool isDistribute(ObjectFifoLinkOp op) {
  return op.getFifoOuts().size() > 1 && op.getFifoIns().size() == 1;
}

bool isOneToOne(ObjectFifoLinkOp op) {
  return op.getFifoIns().size() == 1 && op.getFifoOuts().size() == 1;
}

/// Retrieve ObjectFifoLinkOp of ObjectFifoCreateOp,
/// if it belongs to one.
std::optional<ObjectFifoLinkOp> getOptionalLinkOp(ObjectFifoCreateOp op) {
  auto device = op->getParentOfType<DeviceOp>();
  for (ObjectFifoLinkOp linkOp : device.getOps<ObjectFifoLinkOp>()) {
    for (ObjectFifoCreateOp in : getInputObjectFifos(linkOp))
      if (in == op) return {linkOp};
    for (ObjectFifoCreateOp out : getOutputObjectFifos(linkOp))
      if (out == op) return {linkOp};
  }
  return {};
}

}  // namespace

template <typename MemOp>
void createDMA(DeviceOp &device, OpBuilder &builder, TileOp tileOp,
               DMAChannelDir channelDir, int channelIndex,
               BDDimLayoutArrayAttr dims, size_t acqNum, size_t relNum,
               int64_t len, int64_t offset,
               const SmallVector<BufferOp> &bufferOps,
               const std::pair<LockOp, LockOp> &locks) {
  OpBuilder::InsertionGuard g(builder);
  Operation *producer = nullptr;
  for (auto memOp : device.getOps<MemOp>()) {
    if (memOp.getTile() == tileOp.getResult()) {
      producer = memOp.getOperation();
      break;
    }
  }

  // if none exists, create one
  if (!producer) {
    if (device->getNumRegions() != 1)
      llvm::report_fatal_error("expected num regions for device op");
    OpBuilder::InsertionGuard gg(builder);
    builder.setInsertionPointToEnd(device.getBody());
    auto newMemOp = builder.create<MemOp>(builder.getUnknownLoc(), tileOp);
    {
      OpBuilder::InsertionGuard ggg(builder);
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
    builder.create<DMAStartOp>(builder.getUnknownLoc(), channelDir,
                               channelIndex, /*repeatCount*/ 0, bdBlock,
                               &endBlock);
  }
  if (lastDmaBlock) lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

  auto createBdBlockOps = [&](BufferOp buff, Block *succ) {
    LockOp acqLock = locks.first, relLock = locks.second;
    builder.create<UseLockOp>(builder.getUnknownLoc(), acqLock,
                              LockAction::AcquireGreaterEqual, acqNum);
    if (!dims.getValue().empty()) {
      builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len, dims);
    } else {
      builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len);
    }
    builder.create<UseLockOp>(builder.getUnknownLoc(), relLock,
                              LockAction::Release, relNum);
    builder.create<NextBDOp>(builder.getUnknownLoc(), succ);
  };

  // create Bd blocks
  Block *succ = nullptr, *curr = bdBlock;
  for (size_t blockIndex = 0; blockIndex < bufferOps.size(); ++blockIndex) {
    if (blockIndex == bufferOps.size() - 1) {
      succ = bdBlock;
    } else {
      succ = builder.createBlock(&endBlock);
    }

    OpBuilder::InsertionGuard gg(builder);
    builder.setInsertionPointToStart(curr);
    createBdBlockOps(bufferOps[blockIndex], succ);
    curr = succ;
  }
}

template <typename MemOp>
void createTileDMA(DeviceOp &device, OpBuilder &builder, TileOp tileOp,
                   DMAChannelDir channelDir, uint8_t channelIndex, size_t size,
                   BDDimLayoutArrayAttr dims, uint32_t offset,
                   const ObjectFifoEndpointResource &endpointResource) {
  std::pair<LockOp, LockOp> locks = endpointResource.lockResources.locks;
  uint8_t acqNum = endpointResource.lockResources.locksAcqRel.first;
  uint8_t relNum = endpointResource.lockResources.locksAcqRel.second;
  createDMA<MemOp>(device, builder, tileOp, channelDir, channelIndex, dims,
                   acqNum, relNum, size, offset, endpointResource.buffers,
                   locks);
}

LogicalResult createUseLocks(
    OpBuilder &builder, ObjectFifoCreateOp op, ObjectFifoPort port,
    size_t numLocks, LockAction lockAction,
    const ObjectFifoEndpointResource &endpointResource) {
  if (numLocks == 0) return failure();
  LockOp lock;
  if (lockAction == LockAction::AcquireGreaterEqual) {
    lock = endpointResource.lockResources.locks.second;
  } else if (lockAction == LockAction::Release) {
    lock = endpointResource.lockResources.locks.first;
  } else {
    return op.emitOpError() << "unsupported lock action on this resource: "
                            << stringifyEnum(lockAction);
  }
  builder.create<UseLockOp>(builder.getUnknownLoc(), lock, lockAction,
                            numLocks);
  return success();
}

LogicalResult replaceReleaseOp(
    OpBuilder &builder, ObjectFifoReleaseOp releaseOp, TileOp tileOp,
    const DenseMap<ObjectFifoCreateOp, ObjectFifoResources> &resourceMap) {
  OpBuilder::InsertionGuard g(builder);
  ObjectFifoCreateOp op = getObjectFifo(releaseOp);
  auto port = releaseOp.getPort();
  const ObjectFifoEndpointResource &endpointResource =
      port == ObjectFifoPort::Produce
          ? resourceMap.at(op).producerResource
          : resourceMap.at(op).consumerResources.at(tileOp);
  builder.setInsertionPointAfter(releaseOp);
  return createUseLocks(builder, op, port, releaseOp.getSize(),
                        LockAction::Release, endpointResource);
}

LogicalResult replaceObjectAcquireOp(
    OpBuilder &builder, ObjectFifoAcquireOp acquireOp, TileOp tileOp,
    DenseMap<ObjectFifoCreateOp, size_t> &createOpToIndex,
    const DenseMap<ObjectFifoCreateOp, ObjectFifoResources> &resourceMap) {
  OpBuilder::InsertionGuard g(builder);
  ObjectFifoCreateOp op = getObjectFifo(acquireOp);
  if (!createOpToIndex.contains(op)) createOpToIndex[op] = 0;
  auto port = acquireOp.getPort();
  const ObjectFifoEndpointResource &endpointResource =
      port == ObjectFifoPort::Produce
          ? resourceMap.at(op).producerResource
          : resourceMap.at(op).consumerResources.at(tileOp);

  builder.setInsertionPointAfter(acquireOp);
  if (failed(createUseLocks(builder, op, port, acquireOp.getSize(),
                            LockAction::AcquireGreaterEqual,
                            endpointResource))) {
    return failure();
  }

  for (Operation *userOp : acquireOp->getUsers()) {
    auto subviewAccessOp = dyn_cast<ObjectFifoSubviewAccessOp>(userOp);
    if (!subviewAccessOp) {
      return acquireOp.emitOpError()
             << "currently only supports `aie.objectfifo.subview.access` users";
    }
    size_t index = subviewAccessOp.getIndex();
    size_t bufferIndex =
        (createOpToIndex[op] + index) % endpointResource.buffers.size();
    BufferOp bufferOp = endpointResource.buffers[bufferIndex];
    subviewAccessOp.getResult().replaceAllUsesWith(bufferOp.getResult());
  }
  // Increment index to rotate through available buffers objectFifo acquires.
  createOpToIndex[op] += acquireOp.getSize();
  return success();
}

/// Utility to create a vector of buffer ops for an objectFifo.
SmallVector<BufferOp> createBuffers(OpBuilder &builder,
                                    const AMDAIEDeviceModel &deviceModel,
                                    ObjectFifoCreateOp createOp,
                                    size_t numBuffers, TileOp tile,
                                    const std::string &prefix, size_t index) {
  SmallVector<BufferOp> buffers;
  if (deviceModel.isShimTile(tile.getCol(), tile.getRow())) return buffers;
  auto fifoType = cast<AIEObjectFifoType>(createOp.getElemType());
  auto elemType = cast<MemRefType>(fifoType.getElementType());
  for (int ofElemIndex = 0; ofElemIndex < numBuffers; ofElemIndex++) {
    auto buff = builder.create<BufferOp>(
        builder.getUnknownLoc(), elemType, tile,
        builder.getStringAttr(prefix + "_buff_" + std::to_string(index) + "_" +
                              std::to_string(ofElemIndex)),
        /*address*/ nullptr,
        /*mem_bank*/ nullptr);
    buffers.push_back(buff);
  }
  return buffers;
}

std::pair<LockOp, LockOp> createLockPair(OpBuilder &builder,
                                         const AMDAIEDeviceModel &deviceModel,
                                         TileOp tile, int depth,
                                         const std::string &prefix,
                                         size_t index) {
  // TODO(jornt): make this more extensible towards different lock
  // schemes.
  int producerInitValue{depth};
  int consumerInitValue{0};
  // Use no lock value for shim tiles as the shim DMAs don't need to be
  // synchronized. TODO(jornt): we might be able to just not create any locks
  // for shims, see buffers.
  if (deviceModel.isShimTile(tile.getCol(), tile.getRow()))
    producerInitValue = 0;
  LockOp producerLock = builder.create<LockOp>(
      builder.getUnknownLoc(), tile, IntegerAttr{},
      builder.getI8IntegerAttr(producerInitValue),
      builder.getStringAttr(prefix + "_prod_lock_" + std::to_string(index)));
  LockOp consumerLock = builder.create<LockOp>(
      builder.getUnknownLoc(), tile, IntegerAttr{},
      builder.getI8IntegerAttr(consumerInitValue),
      builder.getStringAttr(prefix + "_cons_lock_" + std::to_string(index)));
  return std::make_pair(producerLock, consumerLock);
}

/// Utility to create buffers and locks for the objectFifo producer side.
LogicalResult createProducerBuffersAndLocks(
    OpBuilder &builder, const AMDAIEDeviceModel &deviceModel,
    ObjectFifoCreateOp createOp, size_t index,
    DenseMap<ObjectFifoCreateOp, ObjectFifoResources> &resourceMap) {
  OpBuilder::InsertionGuard g(builder);
  TileOp producerTileOp =
      dyn_cast_if_present<TileOp>(createOp.getProducerTile().getDefiningOp());
  if (!producerTileOp) {
    return createOp.emitOpError() << "expected a producer tile op, but got: "
                                  << createOp.getProducerTile();
  }
  size_t depth = objFifoSize(createOp);
  SmallVector<BufferOp> producerBuffers =
      createBuffers(builder, deviceModel, createOp, depth, producerTileOp,
                    name(createOp).str() + "_prod", index);
  std::pair<LockOp, LockOp> lockPair =
      createLockPair(builder, deviceModel, producerTileOp, depth,
                     name(createOp).str() + "_prod", index);
  // Swap for producers to synchronize with potential consumers on the other
  // side.
  std::swap(lockPair.first, lockPair.second);
  std::pair<uint8_t, uint8_t> lockAcqRel = std::make_pair(1, 1);
  resourceMap[createOp].producerResource = ObjectFifoEndpointResource(
      producerBuffers, LockResources(lockPair, lockAcqRel));
  return success();
}

/// Utility to create buffers and locks for the objectFifo consumer side.
LogicalResult createConsumerBuffersAndLocks(
    OpBuilder &builder, const AMDAIEDeviceModel &deviceModel,
    ObjectFifoCreateOp createOp, size_t external_idx,
    DenseMap<ObjectFifoCreateOp, ObjectFifoResources> &resourceMap) {
  OpBuilder::InsertionGuard g(builder);
  resourceMap[createOp].consumerResources.clear();
  size_t depth = objFifoSize(createOp);
  for (auto &&[idx1, consumerTile] :
       llvm::enumerate(createOp.getConsumerTiles())) {
    size_t idx = external_idx * createOp.getConsumerTiles().size() + idx1;
    TileOp consumerTileOp =
        dyn_cast_if_present<TileOp>(consumerTile.getDefiningOp());
    if (!consumerTileOp) {
      return createOp.emitOpError()
             << "expected a consumer tile op, but got: " << consumerTile;
    }
    SmallVector<BufferOp> consumerBuffers =
        createBuffers(builder, deviceModel, createOp, depth, consumerTileOp,
                      name(createOp).str() + "_cons", idx);
    std::pair<LockOp, LockOp> lockPair =
        createLockPair(builder, deviceModel, consumerTileOp, depth,
                       name(createOp).str() + "_cons", idx);
    std::pair<uint8_t, uint8_t> lockAcqRel = std::make_pair(1, 1);
    resourceMap[createOp].consumerResources[consumerTileOp] =
        ObjectFifoEndpointResource(consumerBuffers,
                                   LockResources(lockPair, lockAcqRel));
  }
  return success();
}

LogicalResult createBuffersAndLocks(
    OpBuilder &builder, DeviceOp device, ObjectFifoLinkOp linkOp,
    DenseMap<ObjectFifoCreateOp, ObjectFifoResources> &resourceMap) {
  OpBuilder::InsertionGuard g(builder);
  AMDAIEDeviceModel deviceModel =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));

  SmallVector<ObjectFifoCreateOp> inputs = getInputObjectFifos(linkOp);
  SmallVector<ObjectFifoCreateOp> outputs = getOutputObjectFifos(linkOp);
  assert(inputs.size() > 0 && "there should be inputs in the link op");
  assert(outputs.size() > 0 && "there should be outputs in the link op");
  uint32_t inputsOffset{0};
  for (ObjectFifoCreateOp input : inputs) {
    resourceMap[input] = ObjectFifoResources(0, inputsOffset);
    auto fifoType = cast<AIEObjectFifoType>(input.getElemType());
    auto fifoElemType = cast<MemRefType>(fifoType.getElementType());
    inputsOffset += fifoElemType.getNumElements();
  }
  uint32_t outputsOffset{0};
  for (ObjectFifoCreateOp output : outputs) {
    resourceMap[output] = ObjectFifoResources(outputsOffset, 0);
    auto fifoType = cast<AIEObjectFifoType>(output.getElemType());
    auto fifoElemType = cast<MemRefType>(fifoType.getElementType());
    outputsOffset += fifoElemType.getNumElements();
  }

  ObjectFifoCreateOp linkCreateOp;
  SmallVector<ObjectFifoCreateOp> linkOtherOps;
  TileOp linkTileOp;
  if (isJoin(linkOp)) {
    assert(outputs.size() == 1 && "single output expected");
    linkCreateOp = outputs[0];
    linkOtherOps = inputs;
    linkTileOp = dyn_cast_if_present<TileOp>(
        linkCreateOp.getProducerTile().getDefiningOp());
  } else if (isDistribute(linkOp)) {
    assert(inputs.size() == 1 && "single input expected");
    linkCreateOp = inputs[0];
    linkOtherOps = outputs;
    linkTileOp = dyn_cast_if_present<TileOp>(
        linkCreateOp.getConsumerTiles()[0].getDefiningOp());
  } else if (isOneToOne(linkOp)) {
    auto inFifoType = cast<AIEObjectFifoType>(inputs[0].getElemType());
    auto inFifoElemType = cast<MemRefType>(inFifoType.getElementType());
    auto outFifoType = cast<AIEObjectFifoType>(outputs[0].getElemType());
    auto outFifoElemType = cast<MemRefType>(outFifoType.getElementType());
    if (inFifoElemType.getNumElements() >= outFifoElemType.getNumElements()) {
      linkCreateOp = inputs[0];
      linkOtherOps = outputs;
      linkTileOp = dyn_cast_if_present<TileOp>(
          linkCreateOp.getConsumerTiles()[0].getDefiningOp());
    } else {
      linkCreateOp = outputs[0];
      linkOtherOps = inputs;
      linkTileOp = dyn_cast_if_present<TileOp>(
          linkCreateOp.getProducerTile().getDefiningOp());
    }
  } else {
    return linkOp.emitOpError()
           << "only join or distribute link supported currently";
  }
  if (!linkTileOp) {
    return linkCreateOp.emitOpError() << "expected a tile op";
  }

  size_t depth = objFifoSize(linkCreateOp);
  if (!depth) return linkCreateOp.emitOpError() << "doesn't have a size";

  // Reset opbuilder location to after the last tile declaration
  auto tiles = device.getBody()->getOps<TileOp>();
  assert(!tiles.empty() && "no tiles in device");
  builder.setInsertionPointAfter(*std::prev(tiles.end(), 1));

  {
    SmallVector<BufferOp> linkBuffers =
        createBuffers(builder, deviceModel, linkCreateOp, depth, linkTileOp,
                      name(linkCreateOp).str() + "_link", 0);
    size_t linkDepth = depth * linkOtherOps.size();
    std::pair<LockOp, LockOp> linkLockPair =
        createLockPair(builder, deviceModel, linkTileOp, linkDepth,
                       name(linkCreateOp).str() + "_link", 0);
    uint8_t inputAcqRelValue = linkDepth / depth / inputs.size();
    std::pair<uint8_t, uint8_t> inputLockAcqRel =
        std::make_pair(inputAcqRelValue, inputAcqRelValue);
    for (ObjectFifoCreateOp input : inputs) {
      resourceMap[input].consumerResources[linkTileOp] =
          ObjectFifoEndpointResource(
              linkBuffers, LockResources(linkLockPair, inputLockAcqRel));
    }
    // Swap locks for outputs to synchronize link inputs and outputs.
    std::swap(linkLockPair.first, linkLockPair.second);
    uint8_t outputAcqRelValue = linkDepth / depth / outputs.size();
    std::pair<uint8_t, uint8_t> outputLockAcqRel =
        std::make_pair(outputAcqRelValue, outputAcqRelValue);
    for (ObjectFifoCreateOp output : outputs) {
      resourceMap[output].producerResource = ObjectFifoEndpointResource(
          linkBuffers, LockResources(linkLockPair, outputLockAcqRel));
    }
  }

  for (auto &&[idx, input] : llvm::enumerate(inputs)) {
    if (failed(createProducerBuffersAndLocks(builder, deviceModel, input, idx,
                                             resourceMap))) {
      return failure();
    }
  }

  for (auto &&[idx, output] : llvm::enumerate(outputs)) {
    if (failed(createConsumerBuffersAndLocks(builder, deviceModel, output, idx,
                                             resourceMap))) {
      return failure();
    }
  }
  return success();
}

LogicalResult createBuffersAndLocksForNonLinkOps(
    OpBuilder &builder, DeviceOp device, ObjectFifoCreateOp createOp,
    DenseMap<ObjectFifoCreateOp, ObjectFifoResources> &resourceMap) {
  // Skip objectFifoCreateOps in links.
  if (getOptionalLinkOp(createOp)) return success();
  OpBuilder::InsertionGuard g(builder);
  AMDAIEDeviceModel deviceModel =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));
  resourceMap[createOp] = ObjectFifoResources(0, 0);
  size_t depth = objFifoSize(createOp);
  if (!depth) return createOp.emitOpError() << "doesn't have a depth size";

  // Reset opbuilder location to after the last tile declaration
  auto tiles = device.getBody()->getOps<TileOp>();
  assert(!tiles.empty() && "no tiles in device");
  builder.setInsertionPointAfter(*std::prev(tiles.end(), 1));
  if (failed(createProducerBuffersAndLocks(builder, deviceModel, createOp, 0,
                                           resourceMap))) {
    return failure();
  }
  if (failed(createConsumerBuffersAndLocks(builder, deviceModel, createOp, 0,
                                           resourceMap))) {
    return failure();
  }
  return success();
}

LogicalResult createTileDMAs(
    OpBuilder &builder, DeviceOp device, ObjectFifoCreateOp createOp,
    DenseMap<ObjectFifoCreateOp, ObjectFifoResources> &resourceMap,
    const DenseMap<StringRef, SmallVector<FlowOp>> &symbolToFlowOps) {
  OpBuilder::InsertionGuard g(builder);
  AMDAIEDeviceModel deviceModel =
      getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));

  auto createDMA = [&deviceModel, &device, &builder](
                       TileOp tileOp, DMAChannelDir channelDir,
                       uint8_t channelIndex, size_t size,
                       BDDimLayoutArrayAttr dims, StringRef name,
                       uint32_t offset,
                       const ObjectFifoEndpointResource &endpointResource) {
    if (deviceModel.isShimTile(tileOp.getCol(), tileOp.getRow())) {
      builder.create<ShimDMAAllocationOp>(builder.getUnknownLoc(), name,
                                          channelDir, channelIndex,
                                          tileOp.getCol());
    } else if (deviceModel.isMemTile(tileOp.getCol(), tileOp.getRow())) {
      createTileDMA<MemTileDMAOp>(device, builder, tileOp, channelDir,
                                  channelIndex, size, dims, offset,
                                  endpointResource);
    } else {
      createTileDMA<MemOp>(device, builder, tileOp, channelDir, channelIndex,
                           size, dims, offset, endpointResource);
    }
  };

  // Collect producer and consumer DMA channels
  if (!symbolToFlowOps.contains(createOp.getSymName())) {
    return createOp.emitOpError()
           << "symbol name not found in symbol to flow ops map";
  }
  SmallVector<FlowOp> flowOps = symbolToFlowOps.at(createOp.getSymName());
  SmallVector<uint8_t> producerChannelsVec = llvm::map_to_vector(
      flowOps, [](FlowOp flowOp) { return flowOp.getSourceChannel(); });
  llvm::SmallSetVector<uint8_t, 1> producerChannels(producerChannelsVec.begin(),
                                                    producerChannelsVec.end());
  if (producerChannels.size() != 1)
    return createOp.emitOpError() << "expected a single producer channel";
  DenseMap<Value, uint8_t> consumerChannelsMap;
  for (FlowOp flowOp : flowOps)
    consumerChannelsMap[flowOp.getDest()] = flowOp.getDestChannel();
  if (consumerChannelsMap.size() != createOp.getConsumerTiles().size()) {
    return createOp.emitOpError() << "expected same number of consumers as the "
                                     "number of objectFifo consumers";
  }

  auto fifo = cast<AIEObjectFifoType>(createOp.getElemType());
  auto elemType = cast<MemRefType>(fifo.getElementType());
  size_t size = elemType.getNumElements();

  // create producer tile DMA
  builder.setInsertionPoint(&device.getBody()->back());
  TileOp producerTileOp =
      dyn_cast_if_present<TileOp>(createOp.getProducerTile().getDefiningOp());
  if (!producerTileOp)
    return createOp.emitOpError() << "expected a producer TileOp";
  const ObjectFifoResources &opResource = resourceMap[createOp];
  const ObjectFifoEndpointResource &producerEndpointResource =
      opResource.producerResource;
  uint32_t producerOffset = opResource.producerOffset;
  createDMA(producerTileOp, DMAChannelDir::MM2S, producerChannels[0], size,
            createOp.getDimensionsToStreamAttr(), createOp.getName(),
            producerOffset, producerEndpointResource);

  assert(opResource.consumerResources.size() ==
             createOp.getConsumerTiles().size() &&
         "same number of consumer resources expected as the number of consumer "
         "tiles on the objectFifo");
  for (auto &&[idx, consumerTile] :
       llvm::enumerate(createOp.getConsumerTiles())) {
    TileOp consumerTileOp =
        dyn_cast_if_present<TileOp>(consumerTile.getDefiningOp());
    if (!consumerTileOp) {
      return createOp.emitOpError()
             << "expected a consumer TileOp, but got: " << consumerTile;
    }
    if (!consumerChannelsMap.contains(consumerTile)) {
      return createOp.emitOpError()
             << "did not find consumer tile (" << consumerTile
             << ") in consumerChannelsMap";
    }
    uint8_t consumerChannel = consumerChannelsMap[consumerTile];

    // create consumer tile DMA
    BDDimLayoutArrayAttr consumerDims =
        createOp.getDimensionsFromStreamPerConsumer()[idx];
    uint32_t consumersOffset = opResource.consumersOffset;
    const ObjectFifoEndpointResource &consumerEndpointResource =
        opResource.consumerResources.at(consumerTileOp);
    createDMA(consumerTileOp, DMAChannelDir::S2MM, consumerChannel, size,
              consumerDims, createOp.getName(), consumersOffset,
              consumerEndpointResource);
  }
  return success();
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
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());

    // Flow ops contain the DMA information, so create a map for easy lookup
    // based on a global symbol.
    DenseMap<StringRef, SmallVector<FlowOp>> symbolToFlowOps;
    device.walk([&](FlowOp op) {
      std::optional<StringRef> symbolAttr = op.getSymbol();
      if (symbolAttr) symbolToFlowOps[symbolAttr.value()].push_back(op);
    });

    DenseMap<ObjectFifoCreateOp, ObjectFifoResources> resourceMap;
    for (ObjectFifoLinkOp linkOp : device.getOps<ObjectFifoLinkOp>()) {
      if (failed(createBuffersAndLocks(builder, device, linkOp, resourceMap))) {
        return signalPassFailure();
      }
    }

    // Handle objectFifos that are not inside a link.
    for (ObjectFifoCreateOp createOp : device.getOps<ObjectFifoCreateOp>()) {
      if (failed(createBuffersAndLocksForNonLinkOps(builder, device, createOp,
                                                    resourceMap))) {
        return signalPassFailure();
      }
    }

    for (ObjectFifoCreateOp createOp : device.getOps<ObjectFifoCreateOp>()) {
      if (failed(createTileDMAs(builder, device, createOp, resourceMap,
                                symbolToFlowOps))) {
        return signalPassFailure();
      }
    }

    // Replace ops
    for (auto coreOp : device.getOps<CoreOp>()) {
      TileOp tileOp =
          dyn_cast_if_present<TileOp>(coreOp.getTile().getDefiningOp());
      if (!tileOp) {
        coreOp.emitOpError()
            << "expected a TileOp, but got: " << coreOp.getTile();
        return signalPassFailure();
      }
      WalkResult res = coreOp.walk([&](ObjectFifoReleaseOp releaseOp) {
        if (failed(replaceReleaseOp(builder, releaseOp, tileOp, resourceMap))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (res.wasInterrupted()) return signalPassFailure();
      // Use a map from objectFifos to indices to rotate through available
      // buffers for double buffering purposes.
      DenseMap<ObjectFifoCreateOp, size_t> createOpToIndex;
      res = coreOp.walk([&](ObjectFifoAcquireOp acquireOp) {
        if (failed(replaceObjectAcquireOp(builder, acquireOp, tileOp,
                                          createOpToIndex, resourceMap))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (res.wasInterrupted()) return signalPassFailure();
    }

    // make global symbols to replace the to be erased ObjectFifoCreateOps
    for (auto createOp : device.getOps<ObjectFifoCreateOp>()) {
      OpBuilder::InsertionGuard gg(builder);
      builder.setInsertionPointToStart(&device.getBodyRegion().front());
      auto symName = createOp.getName();
      createOp->setAttr(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr("__erase_" + symName));
      auto memrefType =
          cast<AIEObjectFifoType>(createOp.getElemType()).getElementType();
      builder.create<memref::GlobalOp>(builder.getUnknownLoc(), symName,
                                       builder.getStringAttr("public"),
                                       memrefType, nullptr, false, nullptr);
    }

    // Remove old ops
    IRRewriter rewriter(&getContext());
    device.walk([&](Operation *op) {
      if (isa<ObjectFifoCreateOp, ObjectFifoLinkOp, ObjectFifoAcquireOp,
              ObjectFifoSubviewAccessOp, ObjectFifoReleaseOp>(op)) {
        op->dropAllUses();
        rewriter.eraseOp(op);
      }
    });
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
