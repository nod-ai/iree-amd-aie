// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Passes.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "llvm/ADT/Twine.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "amdaie-assign-buffers-basic"

using namespace mlir;
using namespace xilinx::AIE;

namespace mlir::iree_compiler::AMDAIE {
struct AMDAIEAssignBufferAddressesPassBasic : mlir::OperationPass<DeviceOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      AMDAIEAssignBufferAddressesPassBasic)

  AMDAIEAssignBufferAddressesPassBasic()
      : mlir::OperationPass<DeviceOp>(resolveTypeID()) {}

  llvm::StringRef getArgument() const override {
    return "amdaie-assign-buffer-addresses-basic";
  }

  llvm::StringRef getName() const override {
    return "AMDAIEAssignBufferAddressesBasic";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<AMDAIEAssignBufferAddressesPassBasic>(
        *static_cast<const AMDAIEAssignBufferAddressesPassBasic *>(this));
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    int counter = 0;
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      if (!buffer.hasName())
        buffer.setSymName("_anonymous" + std::to_string(counter++));
    });

    DenseMap<TileOp, SetVector<BufferOp>> tileToBuffers;
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      tileToBuffers[buffer.getTileOp()].insert(buffer);
    });

    const auto &targetModel = getTargetModel(device);
    for (auto [tile, buffers] : tileToBuffers) {
      // Leave room at the bottom of the address range for stack
      int64_t address = 0;
      if (auto core = tile.getCoreOp()) address += core.getStackSize();

      for (auto buffer : buffers) {
        buffer.setAddress(address);
        address += buffer.getAllocationSize();
      }

      int maxDataMemorySize;
      if (tile.isMemTile())
        maxDataMemorySize = targetModel.getMemTileSize();
      else
        maxDataMemorySize = targetModel.getLocalMemorySize();
      if (address > maxDataMemorySize) {
        InFlightDiagnostic error =
            tile.emitOpError("allocated buffers exceeded available memory (")
            << address << ">" << maxDataMemorySize << ")\n";
        return signalPassFailure();
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
createAMDAIEAssignBufferAddressesBasicPass() {
  return std::make_unique<AMDAIEAssignBufferAddressesPassBasic>();
}

void registerAMDAIEAssignBufferAddressesBasic() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAMDAIEAssignBufferAddressesBasicPass();
  });
}
}  // namespace mlir::iree_compiler::AMDAIE
