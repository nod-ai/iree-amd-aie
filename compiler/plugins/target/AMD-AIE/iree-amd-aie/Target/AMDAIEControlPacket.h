// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TARGET_AMDAIECONTROLPACKET_H_
#define IREE_AMD_AIE_TARGET_AMDAIECONTROLPACKET_H_

#include "aie/AIEDialect.h"

namespace mlir::iree_compiler::AMDAIE {

/// Convert the specified AIE device operation into a sequence of control
/// packets, and output them to a new MLIR file for further processing.
LogicalResult convertAieToControlPacket(ModuleOp moduleOp,
                                        xilinx::AIE::DeviceOp deviceOp,
                                        const std::string &outputMlir,
                                        const std::string &tempDir);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TARGET_AMDAIECONTROLPACKET_H_
