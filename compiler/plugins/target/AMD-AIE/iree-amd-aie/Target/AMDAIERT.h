// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TARGET_AMDAIERT_H_
#define IREE_AMD_AIE_TARGET_AMDAIERT_H_

#include <filesystem>

#include "aie/AIEDialect.h"

namespace mlir::iree_compiler::AMDAIE {

FailureOr<uint64_t> getProgramSize(
    const std::filesystem::path &elfPath, const AMDAIEDeviceModel &deviceModel,
    function_ref<InFlightDiagnostic()> emitError);

/// Load ELF files for all cores within the device operation.
LogicalResult addAllAieElfs(const AMDAIEDeviceModel &deviceModel,
                            xilinx::AIE::DeviceOp device,
                            const std::filesystem::path &workDirPath,
                            bool aieSim);

/// Update core control registers to enable all cores within the device
/// operation.
LogicalResult addAllCoreEnable(const AMDAIEDeviceModel &deviceModel,
                               xilinx::AIE::DeviceOp &device);

/// Utility function to reset all cores, initialize hardware locks,
/// and configure all switchboxes.
LogicalResult addInitConfig(const AMDAIEDeviceModel &deviceModel,
                            xilinx::AIE::DeviceOp &device);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TARGET_AMDAIERT_H_
