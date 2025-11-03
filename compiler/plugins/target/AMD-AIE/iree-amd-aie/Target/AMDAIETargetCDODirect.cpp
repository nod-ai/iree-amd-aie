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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

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
    SmallString<128> aieCdoSwitchesPath(workDirPath.string());
    llvm::sys::path::append(aieCdoSwitchesPath, "aie_cdo_switches.bin");
    std::string filePath =
        llvm::sys::path::convert_to_slash(aieCdoSwitchesPath);
    // When control packets are enabled, only the switch configuration
    // binary is needed and all other binaries are skipped
    if (failed(generateCDOBinary(Path{filePath}, [&deviceModel, &device] {
          return addSwitchConfig(deviceModel, device);
        })))
      return failure();

  } else {
    SmallString<128> aieCdoElfsPath(workDirPath.string());
    llvm::sys::path::append(aieCdoElfsPath, "aie_cdo_elfs.bin");
    std::string filePath = llvm::sys::path::convert_to_slash(aieCdoElfsPath);
    if (failed(generateCDOBinary(
            Path{filePath}, [&deviceModel, &device, &workDirPath, &aieSim] {
              return addAllAieElfs(deviceModel, device, workDirPath, aieSim);
            })))
      return failure();

    SmallString<128> aieCdoInitPath(workDirPath.string());
    llvm::sys::path::append(aieCdoInitPath, "aie_cdo_init.bin");
    filePath = llvm::sys::path::convert_to_slash(aieCdoInitPath);
    if (failed(generateCDOBinary(Path{filePath}, [&deviceModel, &device] {
          return addInitConfig(deviceModel, device);
        })))
      return failure();

    SmallString<128> aieCdoSwitchesPath(workDirPath.string());
    llvm::sys::path::append(aieCdoSwitchesPath, "aie_cdo_switches.bin");
    filePath = llvm::sys::path::convert_to_slash(aieCdoSwitchesPath);
    if (failed(generateCDOBinary(Path{filePath}, [&deviceModel, &device] {
          return addSwitchConfig(deviceModel, device);
        })))
      return failure();

    SmallString<128> aieCdoEnablePath(workDirPath.string());
    llvm::sys::path::append(aieCdoEnablePath, "aie_cdo_enable.bin");
    filePath = llvm::sys::path::convert_to_slash(aieCdoEnablePath);
    if (failed(generateCDOBinary(Path{filePath}, [&deviceModel, &device] {
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
