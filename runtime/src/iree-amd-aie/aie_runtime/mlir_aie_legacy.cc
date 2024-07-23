// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#include "iree_aie_runtime.h"
#include "llvm/ADT/DenseSet.h"

namespace MLIRAIELegacy {
using mlir::iree_compiler::AMDAIE::StrmSwPortType;
namespace VC1902TargetModel {
llvm::SmallDenseSet<unsigned, 16> nocColumns = {2,  3,  6,  7,  10, 11, 18, 19,
                                                26, 27, 34, 35, 42, 43, 46, 47};

int columns() { return 50; }

int rows() { return 9; /* One Shim row and 8 CORE rows. */ }

bool isShimNOCTile(int col, int row) {
  return row == 0 && nocColumns.contains(col);
}

bool isShimPLTile(int col, int row) {
  return row == 0 && !nocColumns.contains(col);
}

bool isShimNOCorPLTile(int col, int row) {
  return isShimNOCTile(col, row) || isShimPLTile(col, row);
}

uint32_t getNumDestSwitchBoxConnections(int col, int row,
                                        StrmSwPortType bundle) {
  if (isShimNOCTile(col, row) || isShimPLTile(col, row)) switch (bundle) {
      case StrmSwPortType::FIFO:
        return 2;
      case StrmSwPortType::NORTH:
        return 6;
      case StrmSwPortType::WEST: {
        if (col == 0) return 0;
        return 4;
      }
      case StrmSwPortType::SOUTH:
        return 6;
      case StrmSwPortType::EAST: {
        if (col == columns() - 1) return 0;
        return 4;
      }
      case StrmSwPortType::CTRL:
        return isShimNOCTile(col, row) ? 1 : 0;
      default:
        return 0;
    }

  switch (bundle) {
    case StrmSwPortType::CORE:
    case StrmSwPortType::DMA:
    case StrmSwPortType::FIFO:
      return 2;
    case StrmSwPortType::NORTH: {
      if (row == rows() - 1) return 0;
      return 6;
    }
    case StrmSwPortType::WEST: {
      if (col == 0) return 0;
      return 4;
    }
    case StrmSwPortType::SOUTH:
      return 4;
    case StrmSwPortType::EAST: {
      if (col == columns() - 1) return 0;
      return 4;
    }
    case StrmSwPortType::CTRL:
      return 1;
    default:
      return 0;
  }
}

uint32_t getNumSourceSwitchBoxConnections(int col, int row,
                                          StrmSwPortType bundle) {
  if (isShimNOCTile(col, row) || isShimPLTile(col, row)) switch (bundle) {
      case StrmSwPortType::FIFO:
        return 2;
      case StrmSwPortType::NORTH:
        return 4;
      case StrmSwPortType::WEST: {
        if (col == 0) return 0;
        return 4;
      }
      case StrmSwPortType::SOUTH:
        return 8;
      case StrmSwPortType::EAST: {
        if (col == columns() - 1) return 0;
        return 4;
      }
      case StrmSwPortType::TRACE:
        return 1;
      case StrmSwPortType::CTRL:
        return isShimNOCTile(col, row) ? 1 : 0;
      default:
        return 0;
    }

  switch (bundle) {
    case StrmSwPortType::CORE:
    case StrmSwPortType::DMA:
    case StrmSwPortType::FIFO:
      return 2;
    case StrmSwPortType::NORTH: {
      if (row == rows() - 1) return 0;
      return 4;
    }
    case StrmSwPortType::WEST: {
      if (col == 0) return 0;
      return 4;
    }
    case StrmSwPortType::SOUTH:
      return 6;
    case StrmSwPortType::EAST: {
      if (col == columns() - 1) return 0;
      return 4;
    }
    case StrmSwPortType::TRACE:
      return 2;
    case StrmSwPortType::CTRL:
      return 1;
    default:
      return 0;
  }
}
uint32_t getNumDestShimMuxConnections(int col, int row, StrmSwPortType bundle) {
  if (isShimNOCorPLTile(col, row)) switch (bundle) {
      case StrmSwPortType::DMA:
        return 2;
      case mlir::iree_compiler::AMDAIE::StrmSwPortType::NOC:
        return 4;
      case StrmSwPortType::SOUTH:
        return 8;  // Connection to the south port of the stream switch
      default:
        return 0;
    }
  return 0;
}
uint32_t getNumSourceShimMuxConnections(int col, int row,
                                        StrmSwPortType bundle) {
  if (isShimNOCorPLTile(col, row)) switch (bundle) {
      case StrmSwPortType::DMA:
        return 2;
      case StrmSwPortType::NOC:
        return 4;
      case StrmSwPortType::SOUTH:
        return 6;  // Connection to the south port of the stream switch
      default:
        return 0;
    }
  return 0;
}

bool isCoreTile(int col, int row) { return row > 0; }
bool isMemTile(int col, int row) { return false; }

bool isLegalTileConnection(int col, int row, StrmSwPortType srcBundle,
                           int srcChan, StrmSwPortType dstBundle, int dstChan) {
  // Check Channel Id within the range
  if (srcChan >= int(getNumSourceSwitchBoxConnections(col, row, srcBundle)))
    return false;
  if (dstChan >= int(getNumDestSwitchBoxConnections(col, row, dstBundle)))
    return false;

  // Memtile
  if (isMemTile(col, row)) {
    return false;
  }
  // Shimtile
  else if (isShimNOCorPLTile(col, row)) {
    if (srcBundle == StrmSwPortType::TRACE)
      return dstBundle == StrmSwPortType::SOUTH;
    else
      return true;
  }
  // Coretile
  else if (isCoreTile(col, row)) {
    if (srcBundle == StrmSwPortType::TRACE)
      return dstBundle == StrmSwPortType::SOUTH;
    else
      return true;
  }
  return false;
}
}  // namespace VC1902TargetModel

namespace VE2802TargetModel {
llvm::SmallDenseSet<unsigned, 16> nocColumns = {2,  3,  6,  7,  14, 15,
                                                22, 23, 30, 31, 34, 35};

bool isShimNOCTile(int col, int row) {
  return row == 0 && nocColumns.contains(col);
}

bool isShimPLTile(int col, int row) {
  return row == 0 && !nocColumns.contains(col);
}

bool isShimNOCorPLTile(int col, int row) {
  return isShimNOCTile(col, row) || isShimPLTile(col, row);
}

int columns() { return 38; }

int rows() { return 11; /* One Shim row, 2 memtile rows, and 8 Core rows. */ }

bool isCoreTile(int col, int row) { return row > 2; }

bool isMemTile(int col, int row) { return row == 1 || row == 2; }

uint32_t getNumDestShimMuxConnections(int col, int row, StrmSwPortType bundle) {
  if (isShimNOCorPLTile(col, row)) switch (bundle) {
      case StrmSwPortType::DMA:
        return 2;
      case StrmSwPortType::NOC:
        return 4;
      case StrmSwPortType::SOUTH:
        return 8;  // Connection to the south port of the stream switch
      default:
        return 0;
    }

  return 0;
}

uint32_t getNumSourceShimMuxConnections(int col, int row,
                                        StrmSwPortType bundle) {
  if (isShimNOCorPLTile(col, row)) switch (bundle) {
      case StrmSwPortType::DMA:
        return 2;
      case StrmSwPortType::NOC:
        return 4;
      case StrmSwPortType::SOUTH:
        return 6;  // Connection to the south port of the stream switch
      default:
        return 0;
    }

  return 0;
}

uint32_t getNumDestSwitchBoxConnections(int col, int row,
                                        StrmSwPortType bundle) {
  if (isMemTile(col, row)) switch (bundle) {
      case StrmSwPortType::DMA:
      case StrmSwPortType::NORTH:
        return 6;
      case StrmSwPortType::SOUTH:
        return 4;
      case StrmSwPortType::CTRL:
        return 1;
      default:
        return 0;
    }

  if (isShimNOCTile(col, row) || isShimPLTile(col, row)) switch (bundle) {
      case StrmSwPortType::FIFO:
        return 1;
      case StrmSwPortType::NORTH:
        return 6;
      case StrmSwPortType::WEST: {
        if (col == 0) return 0;
        return 4;
      }
      case StrmSwPortType::SOUTH:
        return 6;
      case StrmSwPortType::EAST: {
        if (col == columns() - 1) return 0;
        return 4;
      }
      case StrmSwPortType::CTRL:
        return isShimNOCTile(col, row) ? 1 : 0;
      default:
        return 0;
    }

  switch (bundle) {
    case StrmSwPortType::CORE:
      return 1;
    case StrmSwPortType::DMA:
      return 2;
    case StrmSwPortType::FIFO:
      return 1;
    case StrmSwPortType::NORTH: {
      if (row == rows() - 1) return 0;
      return 6;
    }
    case StrmSwPortType::WEST: {
      if (col == 0) return 0;
      return 4;
    }
    case StrmSwPortType::SOUTH:
      return 4;
    case StrmSwPortType::EAST: {
      if (col == columns() - 1) return 0;
      return 4;
    }
    case StrmSwPortType::CTRL:
      return 1;
    default:
      return 0;
  }
}

uint32_t getNumSourceSwitchBoxConnections(int col, int row,
                                          StrmSwPortType bundle) {
  if (isMemTile(col, row)) switch (bundle) {
      case StrmSwPortType::DMA:
        return 6;
      case StrmSwPortType::NORTH:
        return 4;
      case StrmSwPortType::SOUTH:
        return 6;
      case StrmSwPortType::TRACE:
      case StrmSwPortType::CTRL:
        return 1;
      default:
        return 0;
    }

  if (isShimNOCTile(col, row) || isShimPLTile(col, row)) switch (bundle) {
      case StrmSwPortType::FIFO:
        return 1;
      case StrmSwPortType::NORTH:
        return 4;
      case StrmSwPortType::WEST: {
        if (col == 0) return 0;
        return 4;
      }
      case StrmSwPortType::SOUTH:
        return 8;
      case StrmSwPortType::EAST: {
        if (col == columns() - 1) return 0;
        return 4;
      }
      case StrmSwPortType::TRACE:
        return 1;
      case StrmSwPortType::CTRL:
        return isShimNOCTile(col, row) ? 1 : 0;
      default:
        return 0;
    }

  // compute/core tile
  switch (bundle) {
    case StrmSwPortType::CORE:
      return 1;
    case StrmSwPortType::DMA:
      return 2;
    case StrmSwPortType::FIFO:
      return 1;
    case StrmSwPortType::NORTH: {
      if (row == rows() - 1) return 0;
      return 4;
    }
    case StrmSwPortType::WEST: {
      if (col == 0) return 0;
      return 4;
    }
    case StrmSwPortType::SOUTH:
      return 6;
    case StrmSwPortType::EAST: {
      if (col == columns() - 1) return 0;
      return 4;
    }
    case StrmSwPortType::TRACE:
      // Port 0: core trace. Port 1: memory trace.
      return 2;
    case StrmSwPortType::CTRL:
      return 1;
    default:
      return 0;
  }
}

bool isLegalTileConnection(int col, int row, StrmSwPortType srcBundle,
                           int srcChan, StrmSwPortType dstBundle, int dstChan) {
  // Check Channel Id within the range
  if (srcChan >= int(getNumSourceSwitchBoxConnections(col, row, srcBundle)))
    return false;
  if (dstChan >= int(getNumDestSwitchBoxConnections(col, row, dstBundle)))
    return false;

  // Lambda function to check if a bundle is in a list
  auto isBundleInList = [](StrmSwPortType bundle,
                           std::initializer_list<StrmSwPortType> bundles) {
    return std::find(bundles.begin(), bundles.end(), bundle) != bundles.end();
  };

  // Memtile
  if (isMemTile(col, row)) {
    if (srcBundle == StrmSwPortType::DMA) {
      if (dstBundle == StrmSwPortType::DMA) return srcChan == dstChan;
      if (isBundleInList(dstBundle,
                         {StrmSwPortType::CTRL, StrmSwPortType::SOUTH,
                          StrmSwPortType::NORTH}))
        return true;
    }
    if (srcBundle == StrmSwPortType::CTRL) {
      if (dstBundle == StrmSwPortType::DMA) return dstChan == 5;
      if (isBundleInList(dstBundle,
                         {StrmSwPortType::SOUTH, StrmSwPortType::NORTH}))
        return true;
    }
    if (isBundleInList(srcBundle,
                       {StrmSwPortType::SOUTH, StrmSwPortType::NORTH})) {
      if (isBundleInList(dstBundle,
                         {StrmSwPortType::DMA, StrmSwPortType::CTRL}))
        return true;
      if (isBundleInList(dstBundle,
                         {StrmSwPortType::SOUTH, StrmSwPortType::NORTH}))
        return srcChan == dstChan;
    }
    if (srcBundle == StrmSwPortType::TRACE) {
      if (dstBundle == StrmSwPortType::DMA) return dstChan == 5;
      if (dstBundle == StrmSwPortType::SOUTH) return true;
    }
  }
  // Shimtile
  else if (isShimNOCorPLTile(col, row)) {
    if (srcBundle == StrmSwPortType::CTRL)
      return dstBundle != StrmSwPortType::CTRL;
    if (isBundleInList(srcBundle,
                       {StrmSwPortType::FIFO, StrmSwPortType::SOUTH}))
      return isBundleInList(
          dstBundle,
          {StrmSwPortType::CTRL, StrmSwPortType::FIFO, StrmSwPortType::SOUTH,
           StrmSwPortType::WEST, StrmSwPortType::NORTH, StrmSwPortType::EAST});
    if (isBundleInList(srcBundle, {StrmSwPortType::WEST, StrmSwPortType::NORTH,
                                   StrmSwPortType::EAST}))
      return (srcBundle == dstBundle)
                 ? (srcChan == dstChan)
                 : isBundleInList(
                       dstBundle,
                       {StrmSwPortType::CTRL, StrmSwPortType::FIFO,
                        StrmSwPortType::SOUTH, StrmSwPortType::WEST,
                        StrmSwPortType::NORTH, StrmSwPortType::EAST});
    if (srcBundle == StrmSwPortType::TRACE) {
      if (isBundleInList(dstBundle,
                         {StrmSwPortType::FIFO, StrmSwPortType::SOUTH}))
        return true;
      if (isBundleInList(dstBundle,
                         {StrmSwPortType::WEST, StrmSwPortType::EAST}))
        return dstChan == 0;
    }
  }
  // Coretile
  else if (isCoreTile(col, row)) {
    if (isBundleInList(srcBundle,
                       {StrmSwPortType::DMA, StrmSwPortType::FIFO,
                        StrmSwPortType::SOUTH, StrmSwPortType::WEST,
                        StrmSwPortType::NORTH, StrmSwPortType::EAST}))
      if (isBundleInList(dstBundle,
                         {StrmSwPortType::CORE, StrmSwPortType::DMA,
                          StrmSwPortType::CTRL, StrmSwPortType::FIFO,
                          StrmSwPortType::SOUTH, StrmSwPortType::WEST,
                          StrmSwPortType::NORTH, StrmSwPortType::EAST}))
        return (srcBundle == dstBundle) ? (srcChan == dstChan) : true;
    if (srcBundle == StrmSwPortType::CORE)
      return dstBundle != StrmSwPortType::CORE;
    if (srcBundle == StrmSwPortType::CTRL)
      return dstBundle != StrmSwPortType::CTRL &&
             dstBundle != StrmSwPortType::DMA;
    if (srcBundle == StrmSwPortType::TRACE) {
      if (dstBundle == StrmSwPortType::DMA) return dstChan == 0;
      if (isBundleInList(dstBundle,
                         {StrmSwPortType::FIFO, StrmSwPortType::SOUTH}))
        return true;
    }
  }
  return false;
}

}  // namespace VE2802TargetModel

using mlir::iree_compiler::AMDAIE::AMDAIEDevice;
using mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel;
using mlir::iree_compiler::AMDAIE::AMDAIETileType;
bool isShimNOCTile(int col, int row, const AMDAIEDeviceModel &deviceModel) {
  if (deviceModel.device == AMDAIEDevice::xcvc1902)
    return MLIRAIELegacy::VC1902TargetModel::isShimNOCTile(col, row);
  if (deviceModel.device == AMDAIEDevice::xcve2802)
    return MLIRAIELegacy::VE2802TargetModel::isShimNOCTile(col, row);
  assert(isNPUDevice(deviceModel.device) && "expected NPU device");
  return deviceModel.getTileType(col, row) == AMDAIETileType::SHIMNOC;
}

bool isShimNOCorPLTile(int col, int row, const AMDAIEDeviceModel &deviceModel) {
  if (deviceModel.device == AMDAIEDevice::xcvc1902)
    return MLIRAIELegacy::VC1902TargetModel::isShimNOCorPLTile(col, row);
  if (deviceModel.device == AMDAIEDevice::xcve2802)
    return MLIRAIELegacy::VE2802TargetModel::isShimNOCorPLTile(col, row);
  assert(isNPUDevice(deviceModel.device) && "expected NPU device");
  return deviceModel.isShimNOCTile(col, row) ||
         deviceModel.isShimPLTile(col, row);
}

bool isShimPLTile(int col, int row, const AMDAIEDeviceModel &deviceModel) {
  if (deviceModel.device == AMDAIEDevice::xcvc1902)
    return MLIRAIELegacy::VC1902TargetModel::isShimPLTile(col, row);
  if (deviceModel.device == AMDAIEDevice::xcve2802)
    return MLIRAIELegacy::VE2802TargetModel::isShimPLTile(col, row);
  assert(isNPUDevice(deviceModel.device) && "expected NPU device");
  return deviceModel.getTileType(col, row) == AMDAIETileType::SHIMPL;
}

uint32_t getNumShimMuxConnections(int col, int row, StrmSwPortType bundle,
                                  const AMDAIEDeviceModel &deviceModel) {
  if (isShimNOCorPLTile(col, row, deviceModel)) switch (bundle) {
      case StrmSwPortType::DMA:
        return 2;
      case StrmSwPortType::NOC:
        return 4;
      case StrmSwPortType::SOUTH:
        return 6;  // Connection to the south port of the stream switch
      default:
        return 0;
    }
  return 0;
}

uint32_t getNumSourceShimMuxConnections(int col, int row, StrmSwPortType bundle,
                                        const AMDAIEDeviceModel &deviceModel) {
  if (deviceModel.device == AMDAIEDevice::xcvc1902)
    return MLIRAIELegacy::VC1902TargetModel::getNumSourceShimMuxConnections(
        col, row, bundle);
  if (deviceModel.device == AMDAIEDevice::xcve2802)
    return MLIRAIELegacy::VE2802TargetModel::getNumSourceShimMuxConnections(
        col, row, bundle);
  assert(isNPUDevice(deviceModel.device) && "expected NPU device");
  return getNumShimMuxConnections(col, row, bundle, deviceModel);
}

uint32_t getNumDestShimMuxConnections(int col, int row, StrmSwPortType bundle,
                                      const AMDAIEDeviceModel &deviceModel) {
  if (deviceModel.device == AMDAIEDevice::xcvc1902)
    return MLIRAIELegacy::VC1902TargetModel::getNumDestShimMuxConnections(
        col, row, bundle);
  if (deviceModel.device == AMDAIEDevice::xcve2802)
    return MLIRAIELegacy::VE2802TargetModel::getNumDestShimMuxConnections(
        col, row, bundle);
  assert(isNPUDevice(deviceModel.device) && "expected NPU device");
  return getNumShimMuxConnections(col, row, bundle, deviceModel);
}

uint32_t getNumSourceSwitchBoxConnections(
    int col, int row, StrmSwPortType bundle,
    const AMDAIEDeviceModel &deviceModel) {
  if (deviceModel.device == AMDAIEDevice::xcvc1902)
    return MLIRAIELegacy::VC1902TargetModel::getNumSourceSwitchBoxConnections(
        col, row, bundle);
  if (deviceModel.device == AMDAIEDevice::xcve2802)
    return MLIRAIELegacy::VE2802TargetModel::getNumSourceSwitchBoxConnections(
        col, row, bundle);
  assert(isNPUDevice(deviceModel.device) && "expected NPU device");
  return deviceModel.getNumSourceSwitchBoxConnections(col, row, bundle);
}

uint32_t getNumDestSwitchBoxConnections(int col, int row, StrmSwPortType bundle,
                                        const AMDAIEDeviceModel &deviceModel) {
  if (deviceModel.device == AMDAIEDevice::xcvc1902)
    return MLIRAIELegacy::VC1902TargetModel::getNumDestSwitchBoxConnections(
        col, row, bundle);
  if (deviceModel.device == AMDAIEDevice::xcve2802)
    return MLIRAIELegacy::VE2802TargetModel::getNumDestSwitchBoxConnections(
        col, row, bundle);
  assert(isNPUDevice(deviceModel.device) && "expected NPU device");
  return deviceModel.getNumDestSwitchBoxConnections(col, row, bundle);
}

bool isLegalTileConnection(
    int col, int row, StrmSwPortType srcBundle, int srcChan,
    StrmSwPortType dstBundle, int dstChan,
    const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel) {
  if (deviceModel.device == AMDAIEDevice::xcvc1902)
    return MLIRAIELegacy::VC1902TargetModel::isLegalTileConnection(
        col, row, srcBundle, srcChan, dstBundle, dstChan);
  if (deviceModel.device == AMDAIEDevice::xcve2802)
    return MLIRAIELegacy::VE2802TargetModel::isLegalTileConnection(
        col, row, srcBundle, srcChan, dstBundle, dstChan);
  llvm::report_fatal_error(
      llvm::Twine("isLegalTileConnection unsupported for device: ") +
      stringifyAMDAIEDevice(deviceModel.device));
}

int rows(const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel) {
  if (deviceModel.device == AMDAIEDevice::xcvc1902)
    return MLIRAIELegacy::VC1902TargetModel::rows();
  if (deviceModel.device == AMDAIEDevice::xcve2802)
    return MLIRAIELegacy::VE2802TargetModel::columns();
  llvm::report_fatal_error(llvm::Twine("rows unsupported for device: ") +
                           stringifyAMDAIEDevice(deviceModel.device));
}

int columns(const mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel &deviceModel) {
  if (deviceModel.device == AMDAIEDevice::xcvc1902)
    return MLIRAIELegacy::VC1902TargetModel::columns();
  if (deviceModel.device == AMDAIEDevice::xcve2802)
    return MLIRAIELegacy::VE2802TargetModel::columns();
  llvm::report_fatal_error(llvm::Twine("columns unsupported for device: ") +
                           stringifyAMDAIEDevice(deviceModel.device));
}

}  // namespace MLIRAIELegacy
