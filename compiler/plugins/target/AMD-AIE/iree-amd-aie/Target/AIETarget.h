// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"

namespace mlir::iree_compiler::AMDAIE {

// Creates the default AIE target.
std::shared_ptr<IREE::HAL::TargetBackend> createTarget();

}  // namespace mlir::iree_compiler::AMDAIE
