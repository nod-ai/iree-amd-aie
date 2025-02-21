// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <string>

#include "AIETarget.h"
#include "aie/AIEDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::AMDAIE {
mlir::LogicalResult aie2xclbin(
    mlir::MLIRContext *ctx, xilinx::AIE::DeviceOp,
    const std::optional<std::string> &outputNPU, bool emitCtrlPkt,
    const std::string &artifactPath, bool printIRBeforeAll,
    bool printIRAfterAll, bool printIRModuleScope, bool timing,
    const std::string &tempDir, bool useChess, bool useChessForUKernel,
    bool verbose, const std::optional<std::string> &vitisDir,
    const std::string &targetArch, const std::string &npuVersion,
    const std::string &peanoDir,
    const mlir::iree_compiler::AMDAIE::AMDAIEOptions::DeviceHAL deviceHal,
    const std::string &xclBinKernelID, const std::string &xclBinKernelName,
    const std::string &xclBinInstanceName, const std::string &amdAIEInstallDir,
    const std::optional<std::string> &InputXCLBin,
    const std::optional<std::string> &ukernel,
    const std::string &additionalPeanoOptFlags,
    const IREE::HAL::ExecutableTargetAttr &targetAttr);

mlir::LogicalResult emitDenseArrayAttrToFile(Operation *op, StringRef attrName,
                                             StringRef fileName);

namespace detail {
FailureOr<std::vector<std::string>> flagStringToVector(
    const std::string &flags);
FailureOr<std::vector<std::string>> makePeanoOptArgs(
    const std::vector<std::string> &additionalPeanoOptFlags);
}  // namespace detail
}  // namespace mlir::iree_compiler::AMDAIE
