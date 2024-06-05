// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Target/AIETargetDirect.h"

#include <fstream>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Support/FileUtilities.h"
#include "runtime/plugins/AMD-AIE/iree-amd-aie/schemas/xrt_executable_def_builder.h"

#define DEBUG_TYPE "aie-target"

namespace mlir::iree_compiler::AMDAIE {

static llvm::cl::opt<std::string> clEnableAMDAIEUkernels(
    "iree-amdaie-enable-ukernels",
    llvm::cl::desc("Enables microkernels in the amdaie backend. May be "
                   "`none`, `all`, or a comma-separated list of specific "
                   "unprefixed microkernels to enable, e.g. `matmul`."),
    llvm::cl::init("none"));

class AIETargetDirectDevice final : public IREE::HAL::TargetDevice {
 public:
  AIETargetDirectDevice(const AMDAIEDirectOptions &options) : options(options) {
    (void)this->options;
  }

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context,
      const IREE::HAL::TargetRegistry &targetRegistry) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    auto configAttr = b.getDictionaryAttr(configItems);
    configItems.emplace_back(b.getStringAttr("legacy_sync"), b.getUnitAttr());

    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("amd-aie-direct")
        ->getDefaultExecutableTargets(context, "amd-aie-direct", configAttr,
                                      executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context,
                                            b.getStringAttr("amd-aie-direct"),
                                            configAttr, executableTargetAttrs);
  }

 private:
  AMDAIEDirectOptions options;
};

class AIETargetDirectBackend final : public IREE::HAL::TargetBackend {
 public:
  explicit AIETargetDirectBackend(const AMDAIEDirectOptions &options)
      : options(options) {}

  std::string getLegacyDefaultDeviceID() const override {
    return "amd-aie-direct";
  }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    executableTargetAttrs.push_back(getExecutableTarget(context));
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
    // Set microkernel enabling flag.
    addConfig("ukernels", StringAttr::get(context, clEnableAMDAIEUkernels));
    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("amd-aie-direct"),
        b.getStringAttr("amdaie-xclbin-fb"), configAttr);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::iree_compiler::AMDAIE::AMDAIEDialect,
                    mlir::iree_compiler::IREE::Codegen::IREECodegenDialect,
                    IREE::LinalgExt::IREELinalgExtDialect,
                    transform::TransformDialect, xilinx::AIE::AIEDialect,
                    xilinx::AIEX::AIEXDialect, xilinx::air::airDialect,
                    xilinx::airrt::AIRRtDialect>();
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetAttr,
                                    OpPassManager &passManager) override {
    buildAMDAIETransformDirectPassPipeline(passManager);
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override;

  const AMDAIEDirectOptions &getOptions() const { return options; }

 private:
  AMDAIEDirectOptions options;
};

LogicalResult AIETargetDirectBackend::serializeExecutable(
    const SerializationOptions &serOptions,
    IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder) {
  ModuleOp moduleOp = variantOp.getInnerModule();
  moduleOp.dump();
  return failure();
}

std::shared_ptr<IREE::HAL::TargetDevice> createTargetDirect(
    const AMDAIEDirectOptions &options) {
  return std::make_shared<AIETargetDirectDevice>(options);
}

std::shared_ptr<IREE::HAL::TargetBackend> createBackendDirect(
    const AMDAIEDirectOptions &options) {
  return std::make_shared<AIETargetDirectBackend>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
