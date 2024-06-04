// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Utils/LinkingUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-link-executable"

namespace mlir::iree_compiler::AMDAIE {

namespace {

struct AMDAIELinkExecutablesPass
    : public impl::AMDAIELinkExecutablesBase<AMDAIELinkExecutablesPass> {
  AMDAIELinkExecutablesPass() = default;
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());

    auto sourceExecutableOps =
        llvm::to_vector<8>(moduleOp.getOps<IREE::HAL::ExecutableOp>());
    if (sourceExecutableOps.size() <= 1) return;

    // Guess a module name, if needed, to make the output files readable.
    std::string moduleName = guessModuleName(moduleOp, "amdaie_module");

    // Create our new "linked" hal.executable.
    std::string linkedExecutableName =
        llvm::formatv("{0}_linked_{1}", moduleName, "amd_aie");
    auto linkedExecutableOp = moduleBuilder.create<IREE::HAL::ExecutableOp>(
        moduleOp.getLoc(), linkedExecutableName);
    linkedExecutableOp.setVisibility(
        sourceExecutableOps.front().getVisibility());
    auto executableBuilder =
        OpBuilder::atBlockBegin(&linkedExecutableOp.getBlock());

    // Gather all unique executable targets - we may have multiple.
    // However, in AIE currently we have only one unique target but we still
    // do this for the sake of future proofing and ease of implementation
    // as we can use the upstream gatherExecutableTargets utility this way.
    SetVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs =
        gatherExecutableTargets(sourceExecutableOps);
    for (auto [index, attr] : llvm::enumerate(executableTargetAttrs)) {
      // Add our hal.executable.variant with an empty module.
      std::string linkedVariantName =
          executableTargetAttrs.size() == 1
              ? attr.getSymbolNameFragment()
              : llvm::formatv("{0}_{1}", attr.getSymbolNameFragment(), index);
      auto linkedTargetOp =
          executableBuilder.create<IREE::HAL::ExecutableVariantOp>(
              moduleOp.getLoc(), linkedVariantName, attr);
      auto targetBuilder = OpBuilder::atBlockBegin(&linkedTargetOp.getBlock());
      targetBuilder.create<mlir::ModuleOp>(moduleOp.getLoc());

      auto mergeModuleFn = [](mlir::ModuleOp sourceInnerModule,
                              mlir::ModuleOp linkedInnerModule,
                              DenseMap<StringRef, Operation *> &symbolMap) {
        return mergeModuleInto(sourceInnerModule, linkedInnerModule, symbolMap);
      };

      // Try linking together all executables in moduleOp.
      if (failed(linkExecutablesInto(moduleOp, sourceExecutableOps,
                                     linkedExecutableOp, linkedTargetOp,
                                     mergeModuleFn))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createAMDAIELinkExecutablesPass() {
  return std::make_unique<AMDAIELinkExecutablesPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
