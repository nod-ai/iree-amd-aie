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
    const std::string &additionalPeanoOptFlags);

mlir::LogicalResult emitNpuInstructions(xilinx::AIE::DeviceOp deviceOp,
                                        const std::string &outputNPU);

namespace detail {

FailureOr<std::vector<std::string>> flagStringToVector(
    const std::string &flags);

FailureOr<std::vector<std::string>> makePeanoOptArgs(
    const std::vector<std::string> &additionalPeanoOptFlags);

/// An exception-free version of std::stoi, using C++17's std::from_chars.
std::optional<int> safeStoi(std::string_view intString);

/// Get upper-bounds on the maximum stack sizes for the different cores (col,
/// row) by parsing a string of the form:
///
/// ```
/// Stack Sizes:
///      Size     Functions
///        32     some_func
///       512     core_1_3
///        64     some_other_func
///       288     core_3_5
/// ```
///
/// \return A map from (col, row) to an upper bound on maximum stack size for
///         that core. If the analysis of the string fails, a failure is
///         returned.
FailureOr<llvm::DenseMap<std::pair<uint32_t, uint32_t>, uint32_t>>
getUpperBoundStackSizes(const std::string &);

}  // namespace detail
}  // namespace mlir::iree_compiler::AMDAIE
