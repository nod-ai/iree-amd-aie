// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "amdaie-normalize-address-spaces"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

// prevent collision with aie which doesn't put this in an anon namespace
namespace {

Type memRefToDefaultAddressSpace(Type t) {
  if (auto memRefType = llvm::dyn_cast<MemRefType>(t);
      memRefType && memRefType.getMemorySpace() != nullptr)
    return MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                           memRefType.getLayout(), nullptr /* Address Space */);
  return t;
}

#include "aie/Dialect/AIE/Transforms/AIENormalizeAddressSpaces.inc"

}  // namespace

namespace mlir::iree_compiler::AMDAIE {
struct AMDAIENormalizeAddressSpacesPass : mlir::OperationPass<DeviceOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AMDAIENormalizeAddressSpacesPass)

  AMDAIENormalizeAddressSpacesPass()
      : mlir::OperationPass<DeviceOp>(resolveTypeID()) {}

  llvm::StringRef getArgument() const override {
    return "amdaie-normalize-address-spaces";
  }

  llvm::StringRef getName() const override {
    return "AMDAIENormalizeAddressSpacesPass";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<AMDAIENormalizeAddressSpacesPass>(
        *static_cast<const AMDAIENormalizeAddressSpacesPass *>(this));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();

    TypeConverter converter;
    converter.addConversion([&](Type type) -> std::optional<Type> {
      return memRefToDefaultAddressSpace(type);
    });

    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<memref::GlobalOp>([](memref::GlobalOp op) {
      return op.getType().getMemorySpace() == nullptr;
    });

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });

    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();

    // Convert any output types to have the default address space
    device.walk([&](Operation *op) {
      for (Value r : op->getResults())
        r.setType(memRefToDefaultAddressSpace(r.getType()));
    });
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
createAMDAIENormalizeAddressSpacesPass() {
  return std::make_unique<AMDAIENormalizeAddressSpacesPass>();
}

void registerAMDAIENormalizeAddressSpaces() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAMDAIENormalizeAddressSpacesPass();
  });
}

}  // namespace mlir::iree_compiler::AMDAIE