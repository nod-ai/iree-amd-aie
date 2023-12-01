// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TARGET_AIETARGET_H_
#define IREE_AMD_AIE_TARGET_AIETARGET_H_

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"

namespace mlir::iree_compiler::AMDAIE {

struct AMDAIEOptions {
  // Path to Peano installation directory.
  std::string peanoInstallDir;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("AMD AIE Options");

    binder.opt<std::string>(
        "iree-amd-aie-peano-install-dir", peanoInstallDir,
        llvm::cl::cat(category),
        llvm::cl::desc("Path to Peano installation directory"));
  }
};

// Creates the default AIE target.
std::shared_ptr<IREE::HAL::TargetBackend> createTarget(
    const AMDAIEOptions &options);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TARGET_AIETARGET_H_
