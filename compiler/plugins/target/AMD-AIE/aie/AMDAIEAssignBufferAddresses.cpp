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

    int maxDataMemorySize;
    if (deviceModel.isMemTile(tile.getCol(), tile.getRow())) {
      maxDataMemorySize =
          deviceModel.getMemTileSize(tile.getCol(), tile.getRow());
    } else {
      maxDataMemorySize =
          deviceModel.getLocalMemorySize(tile.getCol(), tile.getRow());
    }
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
typedef struct BankLimits {
  int64_t startAddr;
  int64_t endAddr;
} BankLimits;

// Function that given a number of banks and their size, computes the start and
// end addresses for each bank and fills in the entry in the bankLimits vector.
void fillBankLimits(int numBanks, int bankSize,
                    std::vector<BankLimits> &bankLimits) {
  for (int i = 0; i < numBanks; i++) {
    int64_t startAddr = bankSize * i;
    int64_t endAddr = bankSize * (i + 1);
    bankLimits.push_back({startAddr, endAddr});
  }
}

// Function that sets the address attribute of the given buffer to the given
// start_addr. It also updates the entry in the nextAddrInBanks for the
// corresponding bank.
void setAndUpdateAddressInBank(BufferOp buffer, int64_t start_addr,
                               int64_t end_addr,
                               std::vector<int64_t> &nextAddrInBanks) {
  buffer.setAddress(start_addr);
  nextAddrInBanks[buffer.getMemBank().value()] = end_addr;
}

// Function that checks whether the given buffer already has a set address
// attribute. If it does, it finds in which bank the buffer is and checks
// whether there is enough space left for it. If there is the function returns
// true and if not, the function emits a warning that the address will be
// overwritten and returns false (which will cause the buffer to be added to
// the list of buffers without addresses, to be completed later on).
FailureOr<bool> checkAndAddBufferWithAddress(
    BufferOp buffer, int numBanks, std::vector<int64_t> &nextAddrInBanks,
    std::vector<BankLimits> &bankLimits) {
  auto addrAttr = buffer->getAttrOfType<IntegerAttr>("address");
  if (!addrAttr) return false;

  int addr = addrAttr.getInt();
  for (int i = 0; i < numBanks; i++) {
    // If the address is not within the bank, continue.
    if (addr < bankLimits[i].startAddr || addr >= bankLimits[i].endAddr)
      continue;

    // If the allocator already allocated this address, fail.
    if (addr < nextAddrInBanks[i])
      return buffer->emitOpError("would override allocated address");

    // The allocator can accommodate this existing allocation.
    nextAddrInBanks[i] = addr + getAllocationSize(buffer);
    buffer.setMemBank(i);
  }
  return true;
}

// Function that checks whether the given buffer already has a set mem_bank
// attribute. If it does, it checks whether there is enough space left for
// it. If there is, it sets the buffer's address field and if not, the function
// emits a warning that the mem_bank will be overwritten and returns false
// (which will cause the buffer to be added to the list of buffers without
// addresses, to be completed later on).
FailureOr<bool> checkAndAddBufferWithMemBank(
    BufferOp buffer, int numBanks, std::vector<int64_t> &nextAddrInBanks,
    std::vector<BankLimits> &bankLimits) {
  auto memBankAttr = buffer->getAttrOfType<IntegerAttr>("mem_bank");
  if (!memBankAttr) return false;

  int mem_bank = memBankAttr.getInt();
  int64_t startAddr = nextAddrInBanks[mem_bank];
  int64_t endAddr = startAddr + getAllocationSize(buffer);
  if (endAddr > bankLimits[mem_bank].endAddr)
    return buffer->emitOpError("would override existing mem_bank");
  setAndUpdateAddressInBank(buffer, startAddr, endAddr, nextAddrInBanks);
  return true;
}

// Function that given a buffer will iterate over all the memory banks starting
// from the given index to try and find a bank with enough space. If it does,
// it will set the buffer's address and mem_bank attributes and update the
// nextAddrInBanks vector. If it does not find one with enough space, it will
// throw an error. Returns true if the buffer was successfully allocated, false
// otherwise.
bool setBufferAddress(BufferOp buffer, int numBanks, int &startBankIndex,
                      std::vector<int64_t> &nextAddrInBanks,
                      std::vector<BankLimits> &bankLimits) {
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
  // If no bank has enough space, throws an error.
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
    int maxDataMemorySize;
    if (deviceModel.isMemTile(tile.getCol(), tile.getRow())) {
      maxDataMemorySize =
          deviceModel.getMemTileSize(tile.getCol(), tile.getRow());
    } else {
      maxDataMemorySize =
          deviceModel.getLocalMemorySize(tile.getCol(), tile.getRow());
    }

    int numBanks = deviceModel.getNumBanks(tile.getCol(), tile.getRow());
    int bankSize = maxDataMemorySize / numBanks;

    // Each entry of `nextAddrInBanks` is the next address available for use
    // in that bank, and the index is the bank number.
    int stackSize = 0;
    std::vector<int64_t> nextAddrInBanks;
    for (int i = 0; i < numBanks; i++) nextAddrInBanks.push_back(bankSize * i);
    // Leave room at the bottom of the address range for stack.
    if (CoreOp core = tile.getCoreOp()) {
      stackSize = core.getStackSize();
      nextAddrInBanks[0] += stackSize;
    }

    // Each entry of `bankLimits` contains pairs of start and end addresses for
    // that bank.
    std::vector<BankLimits> bankLimits;
    fillBankLimits(numBanks, bankSize, bankLimits);

    // If possible, the buffers with an already specified address will not be
    // overwritten (the available address range of the bank the buffers are in
    // will start AFTER the specified address + buffer size). Buffers with a
    // specified mem_bank will be assigned first, after the above.
    SmallVector<BufferOp> preAllocatedBuffers;
    SmallVector<BufferOp> buffersToAlloc;
    for (BufferOp buffer : buffers) {
      FailureOr<bool> has_addr = checkAndAddBufferWithAddress(
          buffer, numBanks, nextAddrInBanks, bankLimits);
      FailureOr<bool> has_bank = checkAndAddBufferWithMemBank(
          buffer, numBanks, nextAddrInBanks, bankLimits);
      if (failed(has_addr) || failed(has_bank)) return failure();
      if (!has_addr.value() && !has_bank.value())
        buffersToAlloc.push_back(buffer);
      else
        preAllocatedBuffers.push_back(buffer);
    }

    // Sort by largest allocation size before allocating.
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
        llvm_unreachable("unrecognized scheme");
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> createAMDAIEAssignBufferAddressesPass(
    AMDAIEAssignBufferAddressesOptions options) {
  return std::make_unique<AMDAIEAssignBufferAddressesPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
