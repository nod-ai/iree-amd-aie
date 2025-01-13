// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <string>

#include "AIETarget.h"
#include "aie/AIEDialect.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::AMDAIE {
mlir::LogicalResult aie2xclbin(
    mlir::MLIRContext *ctx, xilinx::AIE::DeviceOp, const std::string &outputNPU,
    const std::string &artifactPath, bool printIRBeforeAll,
    bool printIRAfterAll, bool printIRModuleScope, bool timing,
    const std::string &tempDir, bool useChess, bool verbose,
    const std::optional<std::string> &vitisDir, const std::string &targetArch,
    const std::string &npuVersion, const std::string &peanoDir,
    const mlir::iree_compiler::AMDAIE::AMDAIEOptions::DeviceHAL deviceHal,
    const std::string &xclBinKernelID, const std::string &xclBinKernelName,
    const std::string &xclBinInstanceName, const std::string &amdAIEInstallDir,
    const std::optional<std::string> &InputXCLBin,
    const std::optional<std::string> &ukernel,
    const std::string &additionalPeanoOptFlags);

mlir::LogicalResult emitNpuInstructions(xilinx::AIE::DeviceOp deviceOp,
                                        const std::string &outputNPU);

namespace detail {
FailureOr<std::vector<std::string>> makePeanoOptArgs(
    const std::string &filenameIrIn, const std::string &filenameIrOut,
    const std::string &additionalPeanoOptFlags);
}
}  // namespace mlir::iree_compiler::AMDAIE
