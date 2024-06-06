// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TARGET_DIRECT_AIETARGET_H_
#define IREE_AMD_AIE_TARGET_DIRECT_AIETARGET_H_

#include <string>

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler::AMDAIE {

struct AMDAIEDirectOptions {};

std::shared_ptr<IREE::HAL::TargetDevice> createTargetDirect(
    const AMDAIEDirectOptions &options);

std::shared_ptr<IREE::HAL::TargetBackend> createBackendDirect(
    const AMDAIEDirectOptions &options);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TARGET_DIRECT_AIETARGET_H_
