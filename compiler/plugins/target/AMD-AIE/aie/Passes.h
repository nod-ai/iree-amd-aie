// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDAIE_AIE_PASSES_H_
#define AMDAIE_AIE_PASSES_H_

#include "AIEDialect.h"
#include "PassDetail.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::AMDAIE {

std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAMDAIEAssignBufferAddressesPass(
    AMDAIEAssignBufferAddressesOptions options = {});
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAMDAIEAssignBufferDescriptorIDsPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAMDAIEAssignLockIDsPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAMDAIELocalizeLocksPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAMDAIENormalizeAddressSpacesPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAMDAIERouteFlowsWithPathfinderPass(
    AMDAIERouteFlowsWithPathfinderOptions options = {});
std::unique_ptr<OperationPass<ModuleOp>> createAMDAIECoreToStandardPass(
    AMDAIECoreToStandardOptions options = {});
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAMDAIEDmaToNpuPass();

void registerAMDAIEAssignBufferDescriptorIDs();
void registerAMDAIELocalizeLocks();
void registerAMDAIENormalizeAddressSpaces();
void registerAMDAIEDmaToNpu();

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // AMDAIE_AIE_PASSES_H_
