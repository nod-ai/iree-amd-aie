// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_ASSIGN_BUFFER_ADDRESS_PASS_BASIC_H_
#define AIE_ASSIGN_BUFFER_ADDRESS_PASS_BASIC_H_

#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {
std::unique_ptr<mlir::OperationPass<xilinx::AIE::DeviceOp>>
createAIEAssignBufferAddressesBasicPass();
void registerAIEAssignBufferAddressesBasic();
}  // namespace xilinx::AIE

#endif  // AIE_ASSIGN_BUFFER_ADDRESS_PASS_BASIC_H_
