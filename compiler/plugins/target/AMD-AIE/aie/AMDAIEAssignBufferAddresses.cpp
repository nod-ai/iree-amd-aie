// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AIEDialect.h"
#include "Passes.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "llvm/ADT/Twine.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "amdaie-assign-buffer-addresses"

using namespace mlir;
using namespace xilinx::AIE;

namespace mlir::iree_compiler::AMDAIE {

/// Utility to get the maximum memory size of a given tile.
static uint32_t getMaxMemorySize(AMDAIEDeviceModel deviceModel, TileOp tile) {
  if (deviceModel.isMemTile(tile.getCol(), tile.getRow())) {
    return deviceModel.getMemTileSize(tile.getCol(), tile.getRow());
  } else {
    return deviceModel.getLocalMemorySize(tile.getCol(), tile.getRow());
  }
}

//===----------------------------------------------------------------------===//
// BasicAllocation : sequential alloc from largest to smallest
//===----------------------------------------------------------------------===//
static LogicalResult basicAllocation(
    DenseMap<TileOp, SetVector<BufferOp>> &tileToBuffers,
    AMDAIEDeviceModel deviceModel) {
  for (auto &&[tile, buffers] : tileToBuffers) {
    // Leave room at the bottom of the address range for stack.
    int64_t address = 0;
    if (CoreOp core = getCoreOp(tile)) address += core.getStackSize();

    for (BufferOp buffer : buffers) {
      buffer.setAddress(address);
      address += getAllocationSize(buffer);
    }

    uint32_t maxDataMemorySize = getMaxMemorySize(deviceModel, tile);
    if (address > maxDataMemorySize) {
      return tile.emitOpError("allocated buffers exceeded available memory (")
             << address << ">" << maxDataMemorySize << ")\n";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BankAwareAllocation : round-robin each alloc over available banks
//===----------------------------------------------------------------------===//

// Struct to store the start and end address for each bank.
struct BankLimits {
  int64_t startAddr;
  int64_t endAddr;
  BankLimits(int64_t start, int64_t end) : startAddr(start), endAddr(end) {}
};

// Function that sets the address attribute of the given buffer to the given
// start_addr. It also updates the entry in the nextAddrInBanks for the
// corresponding bank.
void setAndUpdateAddressInBank(BufferOp buffer, int64_t start_addr,
                               int64_t end_addr,
                               SmallVector<int64_t> &nextAddrInBanks) {
  buffer.setAddress(start_addr);
  nextAddrInBanks[buffer.getMemBank().value()] = end_addr;
}

// Function that checks whether the given buffer already has a set address
// attribute. If it does, it finds in which bank the buffer is and checks
// whether there is enough space left for it. If there is the function returns
// true and if not, the function emits an error.
FailureOr<bool> checkAndAddBufferWithAddress(
    BufferOp buffer, uint32_t bankSize, SmallVector<int64_t> &nextAddrInBanks,
    const SmallVector<BankLimits> &bankLimits) {
  std::optional<uint32_t> maybeAddr = buffer.getAddress();
  if (!maybeAddr) return false;
  uint32_t addr = maybeAddr.value();
  uint32_t bankIndex = addr / bankSize;

  // If the bank has pre-assigned, check if the bank index matches.
  if (std::optional<uint32_t> memBank = buffer.getMemBank()) {
    assert(memBank.value() == bankIndex &&
           "bank index is not equal to the preset value");
    return true;
  }

  // If the allocator already allocated this address, fail.
  if (addr < nextAddrInBanks[bankIndex])
    return buffer->emitOpError("would override the allocated address");

  // The allocator can accommodate this existing allocation.
  nextAddrInBanks[bankIndex] = addr + getAllocationSize(buffer);
  if (nextAddrInBanks[bankIndex] > bankLimits[bankIndex].endAddr)
    return buffer->emitOpError("would over run the current bank limit");
  buffer.setMemBank(bankIndex);
  return true;
}

// Function that checks whether the given buffer already has a set mem_bank
// attribute. If it does, it checks whether there is enough space left for
// it. If there is, it sets the buffer's address field and if not, the function
// emits an error.
FailureOr<bool> checkAndAddBufferWithMemBank(
    BufferOp buffer, SmallVector<int64_t> &nextAddrInBanks,
    const SmallVector<BankLimits> &bankLimits) {
  std::optional<uint32_t> maybeBank = buffer.getMemBank();
  if (!maybeBank) return false;
  uint32_t memBank = maybeBank.value();

  // If the buffer has preset address, the next available address for the bank
  // will start from there.
  if (std::optional<uint32_t> addr = buffer.getAddress()) {
    if (addr.value() > nextAddrInBanks[memBank]) {
      nextAddrInBanks[memBank] = addr.value();
    }
  }

  int64_t startAddr = nextAddrInBanks[memBank];
  int64_t endAddr = startAddr + getAllocationSize(buffer);
  if (endAddr > bankLimits[memBank].endAddr)
    return buffer->emitOpError("would over run the current bank limit");
  setAndUpdateAddressInBank(buffer, startAddr, endAddr, nextAddrInBanks);
  return true;
}

// Function that given a buffer will iterate over all the memory banks starting
// from the given index to try and find a bank with enough space. If it does,
// it will set the buffer's address and mem_bank attributes and update the
// nextAddrInBanks vector. If it does not find one with enough space, it will
// emit an error. Returns true if the buffer was successfully allocated, false
// otherwise.
bool setBufferAddress(BufferOp buffer, uint32_t numBanks, int &startBankIndex,
                      SmallVector<int64_t> &nextAddrInBanks,
                      const SmallVector<BankLimits> &bankLimits) {
  assert(startBankIndex < numBanks &&
         "Unexpected input value for startBankIndex");
  int bankIndex = startBankIndex;
  bool allocated = false;
  for (int i = 0; i < numBanks; i++) {
    int64_t startAddr = nextAddrInBanks[bankIndex];
    int64_t endAddr = startAddr + getAllocationSize(buffer);
    if (endAddr <= bankLimits[bankIndex].endAddr) {
      buffer.setMemBank(bankIndex);
      setAndUpdateAddressInBank(buffer, startAddr, endAddr, nextAddrInBanks);
      allocated = true;
      bankIndex = (bankIndex + 1) % numBanks;
      startBankIndex = bankIndex;
      break;
    }
    // Move to the next bank
    bankIndex = (bankIndex + 1) % numBanks;
  }
  // If no bank has enough space, emits an error.
  if (!allocated) {
    buffer.emitError("Failed to allocate buffer: ")
        << buffer.name() << " with size: " << getAllocationSize(buffer)
        << " bytes on any of the bank.";
    return false;
  }
  return true;
}

// Function to deallocate attributes of buffers in case of a failure.
void deAllocateBuffers(SmallVector<BufferOp> &buffers) {
  for (BufferOp buffer : buffers) {
    buffer->removeAttr("address");
    buffer->removeAttr("mem_bank");
  }
}

LogicalResult bankAwareAllocation(
    DenseMap<TileOp, SetVector<BufferOp>> &tileToBuffers,
    AMDAIEDeviceModel deviceModel) {
  for (auto &&[tile, buffers] : tileToBuffers) {
    uint32_t maxDataMemorySize = getMaxMemorySize(deviceModel, tile);
    uint32_t numBanks = deviceModel.getNumBanks(tile.getCol(), tile.getRow());
    uint32_t bankSize = maxDataMemorySize / numBanks;

    // Each entry of `nextAddrInBanks` is the next address available for use
    // in that bank, and the index is the bank number.
    SmallVector<int64_t> nextAddrInBanks;
    for (int i = 0; i < numBanks; i++) nextAddrInBanks.push_back(bankSize * i);
    // Leave room at the bottom of the address range for stack.
    if (CoreOp core = tile.getCoreOp()) {
      int64_t stackSize = core.getStackSize();
      if (stackSize > bankSize)
        return tile.emitOpError("stack size: ")
               << stackSize
               << " should not be larger than the bank size: " << bankSize;
      nextAddrInBanks[0] += stackSize;
    }

    // Each entry of `bankLimits` contains pairs of start and end addresses for
    // that bank.
    SmallVector<BankLimits> bankLimits;
    for (int i = 0; i < numBanks; i++) {
      bankLimits.emplace_back(i * bankSize, (i + 1) * bankSize);
    }

    // The buffers with an already specified address will not be overwritten
    // (the available address range of the bank the buffers are in will start
    // AFTER the specified address + buffer size). Buffers with a specified
    // memory bank will be assigned first, after the above.
    SmallVector<BufferOp> preAllocatedBuffers;
    SmallVector<BufferOp> buffersToAlloc;
    for (BufferOp buffer : buffers) {
      FailureOr<bool> has_bank =
          checkAndAddBufferWithMemBank(buffer, nextAddrInBanks, bankLimits);
      FailureOr<bool> has_addr = checkAndAddBufferWithAddress(
          buffer, bankSize, nextAddrInBanks, bankLimits);
      if (failed(has_addr) || failed(has_bank)) return failure();
      if (!has_addr.value() && !has_bank.value())
        buffersToAlloc.push_back(buffer);
      else
        preAllocatedBuffers.push_back(buffer);
    }

    // Note: This is currently disabled to avoid numerical error in ci for
    // depthwise_conv2d op.
    // // Sort by largest allocation size before allocating.
    // std::sort(buffersToAlloc.begin(), buffersToAlloc.end(),
    //           [](BufferOp a, BufferOp b) {
    //             return getAllocationSize(a) > getAllocationSize(b);
    //           });

    // Set addresses for remaining buffers.
    SmallVector<BufferOp> allocatedBuffers;
    int bankIndex = 0;
    for (BufferOp buffer : buffersToAlloc) {
      // If the buffer doesn't fit in any of the bank space, it emits an error
      // and then deallocates all the buffers.
      if (!setBufferAddress(buffer, numBanks, bankIndex, nextAddrInBanks,
                            bankLimits)) {
        deAllocateBuffers(allocatedBuffers);
        return failure();
      } else {
        allocatedBuffers.push_back(buffer);
      }
    }
  }
  return success();
}

struct AMDAIEAssignBufferAddressesPass
    : public impl::AMDAIEAssignBufferAddressesBase<
          AMDAIEAssignBufferAddressesPass> {
  AMDAIEAssignBufferAddressesPass(
      const AMDAIEAssignBufferAddressesOptions &options)
      : AMDAIEAssignBufferAddressesBase(options) {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<xilinx::AIE::AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    int counter = 0;
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      if (!hasName(buffer))
        buffer.setSymName("_anonymous" + std::to_string(counter++));
    });

    DenseMap<TileOp, SetVector<BufferOp>> tileToBuffers;
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      tileToBuffers[getTileOp(*buffer)].insert(buffer);
    });

    AMDAIEDeviceModel deviceModel =
        getDeviceModel(static_cast<AMDAIEDevice>(device.getDevice()));

    // Select buffer allocation scheme.
    MLIRContext *ctx = &getContext();
    switch (allocScheme) {
      case AllocScheme::Sequential:
        if (failed(basicAllocation(tileToBuffers, deviceModel)))
          return signalPassFailure();
        break;
      case AllocScheme::BankAware:
        if (failed(bankAwareAllocation(tileToBuffers, deviceModel)))
          return signalPassFailure();
        break;
      default:
        emitWarning(UnknownLoc::get(ctx))
            << "Buffer assignment scheme is unrecognized. Defaulting to "
               "bank-aware scheme.";
        if (failed(bankAwareAllocation(tileToBuffers, deviceModel))) {
          emitWarning(UnknownLoc::get(ctx))
              << "Bank-aware scheme is failed. Try the basic sequential "
                 "scheme.";
          if (failed(basicAllocation(tileToBuffers, deviceModel)))
            return signalPassFailure();
        }
        break;
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> createAMDAIEAssignBufferAddressesPass(
    AMDAIEAssignBufferAddressesOptions options) {
  return std::make_unique<AMDAIEAssignBufferAddressesPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
