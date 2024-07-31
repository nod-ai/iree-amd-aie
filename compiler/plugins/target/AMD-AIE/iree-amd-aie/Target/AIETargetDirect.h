// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TARGET_DIRECT_AIETARGET_H_
#define IREE_AMD_AIE_TARGET_DIRECT_AIETARGET_H_

#include <string>

#include "iree-amd-aie/Target/AIETarget.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler::AMDAIE {

std::shared_ptr<IREE::HAL::TargetBackend> createBackendDirect(
    const AMDAIEOptions &options);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TARGET_DIRECT_AIETARGET_H_
