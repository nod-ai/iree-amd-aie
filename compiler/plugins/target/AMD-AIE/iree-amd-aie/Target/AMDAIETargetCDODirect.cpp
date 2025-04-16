// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "AMDAIERT.h"
#include "AMDAIETargets.h"
#include "aie/AIEDialect.h"
#include "iree-amd-aie/aie_runtime/iree_aie_configure.h"

#define DEBUG_TYPE "aie-generate-cdo"

using namespace mlir;

using xilinx::AIE::CoreOp;
using xilinx::AIE::DeviceOp;

using Path = std::filesystem::path;

namespace mlir::iree_compiler::AMDAIE {
LogicalResult generateCDOBinariesSeparately(
    const AMDAIEDeviceModel &deviceModel, const Path &workDirPath,
    DeviceOp &device, bool aieSim, bool enableCtrlPkt) {
  if (enableCtrlPkt) {
    // When control packets are enabled, only the switch configuration
    // binary is needed and all other binaries are skipped
    if (failed(generateCDOBinary(workDirPath / "aie_cdo_switches.bin",
                                 [&deviceModel, &device] {
                                   return addSwitchConfig(deviceModel, device);
                                 })))
      return failure();

  } else {
    if (failed(generateCDOBinary(
            workDirPath / "aie_cdo_elfs.bin",
            [&deviceModel, &device, &workDirPath, &aieSim] {
              return addAllAieElfs(deviceModel, device, workDirPath, aieSim);
            })))
      return failure();

    if (failed(generateCDOBinary(workDirPath / "aie_cdo_init.bin",
                                 [&deviceModel, &device] {
                                   return addInitConfig(deviceModel, device);
                                 })))
      return failure();

    if (failed(generateCDOBinary(workDirPath / "aie_cdo_switches.bin",
                                 [&deviceModel, &device] {
                                   return addSwitchConfig(deviceModel, device);
                                 })))
      return failure();

    if (failed(generateCDOBinary(workDirPath / "aie_cdo_enable.bin",
                                 [&deviceModel, &device] {
                                   return addAllCoreEnable(deviceModel, device);
                                 })))
      return failure();
  }
  return success();
}

LogicalResult AIETranslateToCDODirect(xilinx::AIE::DeviceOp device,
                                      llvm::StringRef workDirPath,
                                      bool enableCtrlPkt, bool bigEndian,
                                      bool cdoDebug, bool aieSim) {
  AMDAIEDeviceModel deviceModel = getDeviceModel(device.getDevice());
  byte_ordering endianness =
      bigEndian ? byte_ordering::Big_Endian : byte_ordering::Little_Endian;
  DEBUG_WITH_TYPE("aie-cdo-driver-debug", cdoDebug = true);
  initializeCDOGenerator(endianness, cdoDebug);
  return generateCDOBinariesSeparately(deviceModel, Path(workDirPath.str()),
                                       device, aieSim, enableCtrlPkt);
}
}  // namespace mlir::iree_compiler::AMDAIE
