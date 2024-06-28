// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Passes.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Target/AIETarget.h"
#include "iree-amd-aie/Target/AIETargetDirect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"

namespace mlir::iree_compiler {

namespace AMDAIE {
extern void registerAMDAIEAssignBufferAddressesBasic();
extern void registerAMDAIEAssignBufferDescriptorIDs();
extern void registerAMDAIEAssignLockIDs();
extern void registerAMDAIECoreToStandard();
extern void registerAMDAIELocalizeLocks();
extern void registerAMDAIENormalizeAddressSpaces();
extern void registerAMDAIEObjectFifoStatefulTransform();
extern void registerAMDAIERoutePathfinderFlows();
extern void registerAMDAIEDmaToNpu();
extern void registerAMDAIEXToStandardPass();
}  // namespace AMDAIE

namespace {

struct AMDAIESession
    : public PluginSession<AMDAIESession, AMDAIE::AMDAIEOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    AMDAIE::registerAMDAIEPasses();
    AMDAIE::registerAMDAIEAssignBufferAddressesBasic();
    AMDAIE::registerAMDAIEAssignBufferDescriptorIDs();
    AMDAIE::registerAMDAIEAssignLockIDs();
    AMDAIE::registerAMDAIECoreToStandard();
    AMDAIE::registerAMDAIELocalizeLocks();
    AMDAIE::registerAMDAIENormalizeAddressSpaces();
    AMDAIE::registerAMDAIEObjectFifoStatefulTransform();
    AMDAIE::registerAMDAIERoutePathfinderFlows();
    AMDAIE::registerAMDAIEDmaToNpu();
    AMDAIE::registerAMDAIEXToStandardPass();
    AMDAIE::registerAIRConversionPasses();
    AMDAIE::registerAIRTransformPasses();

    xilinx::AIE::registerAIEAssignBufferAddresses();
    xilinx::AIE::registerAIEAssignBufferDescriptorIDs();
    xilinx::AIE::registerAIEAssignLockIDs();
    xilinx::AIE::registerAIECoreToStandard();
    xilinx::AIE::registerAIELocalizeLocks();
    xilinx::AIE::registerAIENormalizeAddressSpaces();
    xilinx::AIE::registerAIEObjectFifoStatefulTransform();
    xilinx::AIE::registerAIERoutePathfinderFlows();
    xilinx::AIEX::registerAIEDmaToNpu();
    xilinx::AIEX::registerAIEXToStandardPass();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<AMDAIE::AMDAIEDialect, xilinx::AIE::AIEDialect,
                    xilinx::AIEX::AIEXDialect, xilinx::air::airDialect>();
  }

  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) override {
    // #hal.device.target<"amd-aie", ...
    // #hal.executable.target<"amd-aie", ...
    targets.add("amd-aie", [=]() { return AMDAIE::createTarget(options); });
    targets.add("amd-aie-direct",
                [=]() { return AMDAIE::createTargetDirect(options); });
  }

  void populateHALTargetBackends(
      IREE::HAL::TargetBackendList &targets) override {
    targets.add("amd-aie", [=]() { return AMDAIE::createBackend(options); });
    targets.add("amd-aie-direct",
                [=]() { return AMDAIE::createBackendDirect(options); });
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
