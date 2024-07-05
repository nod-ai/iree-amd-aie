// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TARGET_AIETARGET_H_
#define IREE_AMD_AIE_TARGET_AIETARGET_H_

#include <string>

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler::AMDAIE {

struct AMDAIEOptions {
  // Path to MLIR-AIE installation directory.
  // TODO(MaheshRavishankar): Remove this dependency.
  std::string mlirAieInstallDir;

  std::string amdAieInstallDir;

  // Path to Peano installation directory.
  std::string peanoInstallDir;

  // Path to Vitis installation directory.
  std::string vitisInstallDir;

  // Dump system commands used during compilation
  bool showInvokedCommands{false};

  // Use the legacy chess compiler.
  bool useChess{false};

  // Print IR after all MLIR passes run in aie2xclbin (to stderr).
  bool aie2xclbinPrintIrAfterAll{false};

  // Print IR before all MLIR passes run in aie2xclbin (to stderr).
  bool aie2xclbinPrintIrBeforeAll{false};

  // Print IR at module scope in MLIR passes in aie2xclbin.
  bool aie2xclbinPrintIrModuleScope{false};

  // Print MLIR timing summary for the MLIR passes in aie2xclbin.
  bool aie2xclbinTiming{false};

 public:
  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("AMD AIE Options");

    binder.opt<std::string>(
        "iree-amd-aie-mlir-aie-install-dir", mlirAieInstallDir,
        llvm::cl::cat(category),
        llvm::cl::desc("Path to MLIR-AIE installation directory"));

    binder.opt<std::string>(
        "iree-amd-aie-install-dir", amdAieInstallDir, llvm::cl::cat(category),
        llvm::cl::desc("Path to AMDAIE installation directory (typically the "
                       "IREE install directory)"));

    binder.opt<std::string>(
        "iree-amd-aie-peano-install-dir", peanoInstallDir,
        llvm::cl::cat(category),
        llvm::cl::desc("Path to Peano installation directory"));

    binder.opt<bool>(
        "aie2xclbin-print-ir-after-all", aie2xclbinPrintIrAfterAll,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "If true, print the IR after all MLIR passes run in aie2xclbin"));

    binder.opt<bool>(
        "aie2xclbin-print-ir-before-all", aie2xclbinPrintIrBeforeAll,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "If true, print the IR before all MLIR passes run in aie2xclbin"));

    binder.opt<bool>(
        "aie2xclbin-print-ir-module-scope", aie2xclbinPrintIrModuleScope,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "If true, when printing the IR do so at the module scope"));

    binder.opt<bool>(
        "aie2xclbin-timing", aie2xclbinTiming, llvm::cl::cat(category),
        llvm::cl::desc("If true, print MLIR timing summary for the MLIR passes "
                       "in aie2xclbin"));

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
std::shared_ptr<IREE::HAL::TargetDevice> createTarget(
    const AMDAIEOptions &options);

// Creates the default AIE backend.
std::shared_ptr<IREE::HAL::TargetBackend> createBackend(
    const AMDAIEOptions &options);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TARGET_AIETARGET_H_
