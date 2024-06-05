// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Passes.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Passes.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Target/AIETarget.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"

namespace mlir::iree_compiler {
namespace {

struct AMDAIESession
    : public PluginSession<AMDAIESession, AMDAIE::AMDAIEOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    AMDAIE::registerAMDAIEPasses();
    AMDAIE::registerAIETransformPasses();
    AMDAIE::registerAIEXTransformPasses();
    AMDAIE::registerAIRConversionPasses();
    AMDAIE::registerAIRTransformPasses();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<AMDAIE::AMDAIEDialect, xilinx::AIE::AIEDialect,
                    xilinx::AIEX::AIEXDialect, xilinx::air::airDialect>();
  }

  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) override {
    // #hal.device.target<"amd-aie", ...
    // #hal.executable.target<"amd-aie", ...
    targets.add("amd-aie", [=]() { return AMDAIE::createTarget(options); });
  }

  void populateHALTargetBackends(
      IREE::HAL::TargetBackendList &targets) override {
    targets.add("amd-aie", [=]() { return AMDAIE::createBackend(options); });
  }
};

}  // namespace
}  // namespace mlir::iree_compiler

IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::AMDAIE::AMDAIEOptions);

extern "C" bool iree_register_compiler_plugin_amd_aie(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::AMDAIESession>("amd_aie");
  return true;
}
