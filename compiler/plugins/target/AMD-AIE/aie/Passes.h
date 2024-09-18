// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDAIE_PASSES_H_
#define AMDAIE_PASSES_H_

#include "AIEDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::AMDAIE {

struct AIERoutePathfinderFlowsOptions {
  bool clRouteCircuit = true;
  bool clRoutePacket = true;
};

std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAMDAIEAssignBufferAddressesBasicPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAMDAIEAssignBufferDescriptorIDsPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAMDAIEAssignLockIDsPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAMDAIELocalizeLocksPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAMDAIENormalizeAddressSpacesPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAMDAIEObjectFifoStatefulTransformPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAMDAIEPathfinderPass();
std::unique_ptr<OperationPass<ModuleOp>> createAMDAIECoreToStandardPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAMDAIEDmaToNpuPass();

void registerAMDAIEAssignBufferAddressesBasic();
void registerAMDAIEAssignBufferDescriptorIDs();
void registerAMDAIECoreToStandard();
void registerAMDAIELocalizeLocks();
void registerAMDAIENormalizeAddressSpaces();
void registerAMDAIERoutePathfinderFlows();
void registerAMDAIEDmaToNpu();

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // AMDAIE_PASSES_H_
