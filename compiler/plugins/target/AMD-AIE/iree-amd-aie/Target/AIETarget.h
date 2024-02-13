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
  // Path to MLIR-AIE installation directory.
  // TODO(MaheshRavishankar): Remove this dependency.
  std::string mlirAieInstallDir;

  // Path to Peano installation directory.
  std::string peanoInstallDir;

  // Path to Vitis installation directory.
  std::string vitisInstallDir;

  // Dump to stdout system commands used during compilation
  bool showInvokedCommands;

  // Use the legacy chess compiler.
  bool useChess;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("AMD AIE Options");

    binder.opt<std::string>(
        "iree-amd-aie-mlir-aie-install-dir", mlirAieInstallDir,
        llvm::cl::cat(category),
        llvm::cl::desc("Path to MLIR-AIE installation directory"));

    binder.opt<std::string>(
        "iree-amd-aie-peano-install-dir", peanoInstallDir,
        llvm::cl::cat(category),
        llvm::cl::desc("Path to Peano installation directory"));

    binder.opt<bool>(
        "iree-amd-aie-show-invoked-commands", showInvokedCommands,
        llvm::cl::cat(category),
        llvm::cl::desc("Show commands invoked during binary generation"));

    binder.opt<std::string>(
        "iree-amd-aie-vitis-install-dir", vitisInstallDir,
        llvm::cl::cat(category),
        llvm::cl::desc("Path to aietools in Vitis installation"));

    binder.opt<bool>("iree-amd-aie-enable-chess", useChess,
                     llvm::cl::cat(category),
                     llvm::cl::desc("Use the legacy chess compiler"));
  }
};

// Creates the default AIE target.
std::shared_ptr<IREE::HAL::TargetBackend> createTarget(
    const AMDAIEOptions &options);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TARGET_AIETARGET_H_
