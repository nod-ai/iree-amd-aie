// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_LOGICALOBJFIFOOPINTERFACE_H_
#define IREE_COMPILER_AMDAIE_LOGICALOBJFIFOOPINTERFACE_H_

#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CopyOpInterface.h"

namespace mlir::iree_compiler::AMDAIE {

class LogicalObjFifoOpInterface;

namespace detail {

/// Return the consumer copy-like operations of the logical objFifo.
SmallVector<mlir::CopyOpInterface> getCopyLikeConsumers(
    LogicalObjFifoOpInterface op);

/// Return the producer copy-like operations of the logical objFifo.
SmallVector<mlir::CopyOpInterface> getCopyLikeProducers(
    LogicalObjFifoOpInterface op);

}  // namespace detail

}  // namespace mlir::iree_compiler::AMDAIE

// clang-format off
#include "iree-amd-aie/IR/AMDAIELogicalObjFifoOpInterface.h.inc"
// clang-format on

#endif  // IREE_COMPILER_AMDAIE_LOGICALOBJFIFOOPINTERFACE_H_
