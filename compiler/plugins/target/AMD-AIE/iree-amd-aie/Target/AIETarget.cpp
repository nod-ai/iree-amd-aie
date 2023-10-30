// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Target/AIETarget.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"

namespace mlir::iree_compiler::AMDAIE {

class AIETargetBackend final : public IREE::HAL::TargetBackend {
 public:
  std::string name() const override { return "amd-aie"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<xilinx::air::airDialect>();
  }

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    // Indicates that the runtime HAL driver operates only in the legacy
    // synchronous mode.
    configItems.emplace_back(b.getStringAttr("legacy_sync"), b.getUnitAttr());

    configItems.emplace_back(b.getStringAttr("executable_targets"),
                             getExecutableTargets(context));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpPassManager &passManager) override {
    passManager.addPass(createPlaceholderPass());
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    return variantOp.emitError() << "AIE serialization NYI";
  }

 private:
  ArrayAttr getExecutableTargets(MLIRContext *context) const {
    SmallVector<Attribute> targetAttrs;
    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    targetAttrs.push_back(getExecutableTarget(context));
    return ArrayAttr::get(context, targetAttrs);
  }

  IREE::HAL::ExecutableTargetAttr getExecutableTarget(
      MLIRContext *context) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;
    // Add some configurations to the `hal.executable.target` attribute.
    auto addConfig = [&](StringRef name, Attribute value) {
      configItems.emplace_back(StringAttr::get(context, name), value);
    };
    // Set target arch
    addConfig("target_arch", StringAttr::get(context, "chip-tbd"));
    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("amd-aie"), b.getStringAttr("elf"),
        configAttr);
  }
};

std::shared_ptr<IREE::HAL::TargetBackend> createTarget() {
  return std::make_shared<AIETargetBackend>();
}

}  // namespace mlir::iree_compiler::AMDAIE
