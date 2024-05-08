// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_DMAOPINTERFACE_H_
#define IREE_COMPILER_AMDAIE_DMAOPINTERFACE_H_

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::AMDAIE {

class DoublyStridedOpInterface;

namespace detail {

/// Common verifier for doubly-strided operations.
LogicalResult verifyDoublyStridedOp(DoublyStridedOpInterface op);

}  // namespace detail

}  // namespace mlir::iree_compiler::AMDAIE

// clang-format off
#include "iree-amd-aie/IR/AMDAIEDmaOpInterface.h.inc"
// clang-format on

#endif  // IREE_COMPILER_AMDAIE_DMAOPINTERFACE_H_
