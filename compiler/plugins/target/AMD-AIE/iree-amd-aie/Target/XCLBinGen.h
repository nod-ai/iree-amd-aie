// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#pragma once

namespace xilinx {

struct XCLBinGenConfig {
  std::string TargetArch;
  std::string PeanoDir;
  std::string InstallDir;
  std::string AIEToolsDir;
  std::string TempDir;
  bool Verbose;
  std::string HostArch;
  std::string XCLBinKernelName;
  std::string XCLBinKernelID;
  std::string XCLBinInstanceName;
  bool UseChess = false;
  bool DisableThreading = false;
  bool PrintIRAfterAll = false;
  bool PrintIRBeforeAll = false;
  bool PrintIRModuleScope = false;
  bool Timing = false;
};

mlir::LogicalResult aie2xclbin(mlir::MLIRContext *ctx, mlir::ModuleOp moduleOp,
                               XCLBinGenConfig &TK, mlir::StringRef OutputNPU,
                               mlir::StringRef OutputXCLBin,
                               mlir::StringRef InputXCLBin = "");

}  // namespace xilinx
