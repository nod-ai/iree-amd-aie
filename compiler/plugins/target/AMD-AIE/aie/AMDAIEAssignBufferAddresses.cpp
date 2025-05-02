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
// BasicAllocation : sequential allocation for all buffers
//===----------------------------------------------------------------------===//
static LogicalResult basicAllocation(TileOp tile, SetVector<BufferOp> buffers,
                                     AMDAIEDeviceModel deviceModel) {
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

// Function that attempts to allocate the buffer in a single bank or a
// contiguous group of banks, starting from the given index and wrapping around
// if needed. It finds the minimal number of contiguous banks required to hold
// the buffer and checks if sufficient space is available. On success, it sets
// the buffer's memory bank attribute and address, and updates
// `nextAddrInBanks`. If no suitable banks are found, it emits an error and
// returns false.
bool setBufferAddress(BufferOp buffer, uint32_t numBanks, uint32_t bankSize,
                      int &startBankIndex,
                      SmallVector<int64_t> &nextAddrInBanks,
                      const SmallVector<BankLimits> &bankLimits) {
  assert(startBankIndex < numBanks &&
         "Unexpected input value for startBankIndex");

  // Early exit if buffer is larger than total available space.
  int64_t bufferSize = getAllocationSize(buffer);
  int64_t totalAvailable = 0;
  for (uint32_t i = 0; i < numBanks; ++i) {
    int64_t available =
        std::max<int64_t>(0, bankLimits[i].endAddr - nextAddrInBanks[i]);
    totalAvailable += available;
    LLVM_DEBUG(llvm::dbgs() << "Bank " << i << ": next=" << nextAddrInBanks[i]
                            << ", limit=" << bankLimits[i].endAddr
                            << ", available=" << available << "\n");
  }
  if (bufferSize > totalAvailable) {
    buffer.emitError(
        "Buffer size exceeds total available memory across all banks (")
        << bufferSize << " > " << totalAvailable << ")";
    return false;
  }

  int minNumBanksNeeded = std::ceil(float(bufferSize) / float(bankSize));

  // For large buffers, try to allocate on minimum number of contiguous banks.
  for (int offset = 0; offset < numBanks; ++offset) {
    int bankStart = (startBankIndex + offset) % numBanks;

    bool canAllocate = true;
    SmallVector<int64_t> tempStartAddr, tempEndAddr;
    int64_t remainingSize = bufferSize;

    for (int i = 0; i < minNumBanksNeeded; ++i) {
      int bankIndex = (bankStart + i) % numBanks;
      int64_t startAddr = nextAddrInBanks[bankIndex];
      int64_t available = bankLimits[bankIndex].endAddr - startAddr;

      // If there is no space available in the current bank, break and try other
      // window of contiguous banks.
      if (available <= 0) {
        canAllocate = false;
        break;
      }

      int64_t allocated = std::min(remainingSize, available);
      int64_t endAddr = startAddr + allocated;

      tempStartAddr.push_back(startAddr);
      tempEndAddr.push_back(endAddr);
      remainingSize -= allocated;
    }

    if (canAllocate && remainingSize == 0) {
      // Set the next available address for all allocated banks.
      for (int i = 0; i < minNumBanksNeeded; ++i) {
        int bankIndex = (bankStart + i) % numBanks;
        nextAddrInBanks[bankIndex] = tempEndAddr[i];
      }
      // Set `mem_bank` attribute, address and update `startBankIndex`.
      buffer.setMemBank(bankStart);
      startBankIndex = (bankStart + minNumBanksNeeded) % numBanks;
      setAndUpdateAddressInBank(buffer, tempStartAddr[0],
                                tempEndAddr[minNumBanksNeeded - 1],
                                nextAddrInBanks);
      return true;
    }
  }

  buffer.emitError("Failed to allocate buffer: ")
      << buffer.name() << " with size: " << bufferSize << " bytes across "
      << numBanks << " banks.";
  return false;
}

// Function to deallocate attributes of buffers in case of a failure.
void deAllocateBuffers(SmallVector<BufferOp> &buffers) {
  for (BufferOp buffer : buffers) {
    buffer->removeAttr("address");
    buffer->removeAttr("mem_bank");
  }
}

LogicalResult bankAwareAllocation(TileOp tile, SetVector<BufferOp> buffers,
                                  AMDAIEDeviceModel deviceModel) {
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

  // Sort by largest allocation size before allocating.
  // Note: The sorting may cause numerical error for depthwise conv2d op.
  std::sort(buffersToAlloc.begin(), buffersToAlloc.end(),
            [](BufferOp a, BufferOp b) {
              return getAllocationSize(a) > getAllocationSize(b);
            });

  // Set addresses for remaining buffers.
  SmallVector<BufferOp> allocatedBuffers;
  int bankIndex = 0;
  for (BufferOp buffer : buffersToAlloc) {
    // If the buffer doesn't fit in any of the bank space, it emits an error
    // and then deallocates all the buffers.
    if (!setBufferAddress(buffer, numBanks, bankSize, bankIndex,
                          nextAddrInBanks, bankLimits)) {
      deAllocateBuffers(allocatedBuffers);
      return failure();
    } else {
      allocatedBuffers.push_back(buffer);
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

    // Select buffer allocation scheme per tile.
    MLIRContext *ctx = &getContext();
    for (auto &&[tile, buffers] : tileToBuffers) {
      switch (allocScheme) {
        case AllocScheme::Sequential:
          if (failed(basicAllocation(tile, buffers, deviceModel)))
            return signalPassFailure();
          break;
        case AllocScheme::BankAware:
          if (failed(bankAwareAllocation(tile, buffers, deviceModel)))
            return signalPassFailure();
          break;
        default:
          if (failed(bankAwareAllocation(tile, buffers, deviceModel))) {
            emitWarning(UnknownLoc::get(ctx))
                << "Bank-aware scheme for buffer address assignment is failed. "
                   "Try the basic sequential scheme.";
            if (failed(basicAllocation(tile, buffers, deviceModel)))
              return signalPassFailure();
          }
          break;
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> createAMDAIEAssignBufferAddressesPass(
    AMDAIEAssignBufferAddressesOptions options) {
  return std::make_unique<AMDAIEAssignBufferAddressesPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
