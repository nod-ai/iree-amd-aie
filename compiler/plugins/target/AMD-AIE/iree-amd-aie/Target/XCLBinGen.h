// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <string>

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

mlir::LogicalResult aie2xclbin(
    mlir::MLIRContext *ctx, mlir::ModuleOp moduleOp,
    const std::string &outputNPU, const std::string &outputXCLBin,
    bool printIRBeforeAll, bool printIRAfterAll, bool printIRModuleScope,
    bool timing, const std::string &tempDir, bool useChess, bool verbose,
    const std::optional<std::string> &vitisDir, const std::string &targetArch,
    const std::string &peanoDir, const std::string &xclBinKernelID,
    const std::string &xclBinKernelName, const std::string &xclBinInstanceName,
    const std::string &amdAIEInstallDir,
    const std::optional<std::string> &InputXCLBin,
    const std::optional<std::string> &ukernel);
