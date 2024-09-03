// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_TRAITS_H_
#define IREE_COMPILER_AMDAIE_TRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir::OpTrait::iree_compiler::AMDAIE {

template <typename ConcreteType>
class CircularDmaOp : public OpTrait::TraitBase<ConcreteType, CircularDmaOp> {
 public:
  static LogicalResult verifyTrait(Operation *op) { return success(); }
};

}  // namespace mlir::OpTrait::iree_compiler::AMDAIE

#endif  // IREE_COMPILER_AMDAIE_TRAITS_H_
