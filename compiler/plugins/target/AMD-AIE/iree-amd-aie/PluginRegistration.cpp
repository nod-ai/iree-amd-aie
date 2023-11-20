// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Passes.h"
#include "iree-amd-aie/Target/AIETarget.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"

namespace mlir::iree_compiler {
namespace {

struct AMDAIEOptions {
  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("AMD AIE Options");
    // binder.opt<bool>(
    //     "amd-aie-sample-flag", sampleFlag,
    //     llvm::cl::cat(category),
    //     llvm::cl::desc("Sample flag"));
  }
};

struct AMDAIESession
    : public PluginSession<AMDAIESession, AMDAIEOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    AMDAIE::registerAMDAIEPasses();
    AMDAIE::registerAIRConversionPasses();
    AMDAIE::registerAIRTransformPasses();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<xilinx::air::airDialect>();
  }

  void populateHALTargetBackends(
      IREE::HAL::TargetBackendList &targets) override {
    // #hal.device.target<"amd-aie", ...
    // #hal.executable.target<"amd-aie", ...
    targets.add("amd-aie", [&]() {
      return AMDAIE::createTarget();
      // return std::make_shared<CUDATargetBackend>(options);
    });
  }
};

}  // namespace
}  // namespace mlir::iree_compiler

IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::AMDAIEOptions);

extern "C" bool iree_register_compiler_plugin_amd_aie(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::AMDAIESession>("amd_aie");
  return true;
}
