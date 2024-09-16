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
#include "mlir/Interfaces/CopyOpInterface.h"

namespace mlir::iree_compiler::AMDAIE {

class DoublyStridedOpInterface;

namespace detail {

/// Return the static base offset on the source side if it can be computed.
/// Otherwise, returns nullopt.
std::optional<int64_t> getSourceStaticBaseOffset(DoublyStridedOpInterface op);

/// Return the static access extent on the source side if it can be computed.
/// Otherwise, returns nullopt.
std::optional<int64_t> getSourceStaticExtent(DoublyStridedOpInterface op);

/// Return the static size of the access on the source side if it can be
/// computed. Otherwise, returns nullopt.
std::optional<int64_t> getSourceStaticSize(DoublyStridedOpInterface op);

/// Return the static base offset on the target side if it can be computed.
/// Otherwise, returns nullopt.
std::optional<int64_t> getTargetStaticBaseOffset(DoublyStridedOpInterface op);

/// Return the static access extent on the target side if it can be computed.
/// Otherwise, returns nullopt.
std::optional<int64_t> getTargetStaticExtent(DoublyStridedOpInterface op);

/// Return the static size of the access on the target side if it can be
/// computed. Otherwise, returns nullopt.
std::optional<int64_t> getTargetStaticSize(DoublyStridedOpInterface op);

/// Common verifier for doubly-strided operations.
LogicalResult verifyDoublyStridedOp(DoublyStridedOpInterface op);

}  // namespace detail

}  // namespace mlir::iree_compiler::AMDAIE

// clang-format off
#include "iree-amd-aie/IR/AMDAIEDmaOpInterface.h.inc"
// clang-format on

#endif  // IREE_COMPILER_AMDAIE_DMAOPINTERFACE_H_
