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

#define DEBUG_TYPE "aie-assign-buffers-basic"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace mlir::iree_compiler::AMDAIE {
struct AIEAssignBufferAddressesPassBasic : mlir::OperationPass<DeviceOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      AIEAssignBufferAddressesPassBasic)

  AIEAssignBufferAddressesPassBasic()
      : mlir::OperationPass<DeviceOp>(resolveTypeID()) {}
  AIEAssignBufferAddressesPassBasic(
      const AIEAssignBufferAddressesPassBasic &other)
      : mlir::OperationPass<DeviceOp>(other) {}

  llvm::StringRef getArgument() const override {
    return "aie-assign-buffer-addresses-basic";
  }

  llvm::StringRef getName() const override {
    return "AIEAssignBufferAddressesBasic";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<AIEAssignBufferAddressesPassBasic>(
        *static_cast<const AIEAssignBufferAddressesPassBasic *>(this));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<AIEDialect>();
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
createAIEAssignBufferAddressesBasicPass() {
  return std::make_unique<AIEAssignBufferAddressesPassBasic>();
}

void registerAIEAssignBufferAddressesBasic() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAIEAssignBufferAddressesBasicPass();
  });
}
}  // namespace mlir::iree_compiler::AMDAIE
