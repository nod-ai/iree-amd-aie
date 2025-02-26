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

static LogicalResult basicAllocation(
    DenseMap<TileOp, SetVector<BufferOp>> &tileToBuffers,
    AMDAIEDeviceModel deviceModel) {
  for (auto &&[tile, buffers] : tileToBuffers) {
    // Leave room at the bottom of the address range for stack
    int64_t address = 0;
    if (CoreOp core = getCoreOp(tile)) address += core.getStackSize();

    for (BufferOp buffer : buffers) {
      buffer.setAddress(address);
      address += getAllocationSize(buffer);
    }

    int maxDataMemorySize;
    if (deviceModel.isMemTile(tile.getCol(), tile.getRow()))
      maxDataMemorySize =
          deviceModel.getMemTileSize(tile.getCol(), tile.getRow());
    else
      maxDataMemorySize =
          deviceModel.getLocalMemorySize(tile.getCol(), tile.getRow());
    if (address > maxDataMemorySize) {
      return tile.emitOpError("allocated buffers exceeded available memory (")
             << address << ">" << maxDataMemorySize << ")\n";
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
        // To be implemented.
        device.emitError("expected bank-aware scheme to be implemented");
        return signalPassFailure();
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
