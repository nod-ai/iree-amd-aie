// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIEOPUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIEOPUTILS_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler::AMDAIE {

/// Return a vector of the parent operations that are of type 'OpTy', including
/// this op if it has type 'OpTy'
template <typename OpTy>
SmallVector<OpTy> getInclusiveParentsOfType(Operation *op) {
  SmallVector<OpTy> res;
  auto *current = op;
  do {
    if (auto typedParent = dyn_cast<OpTy>(current)) {
      res.push_back(typedParent);
    }
  } while ((current = current->getParentOp()));
  return res;
}

}  // namespace mlir::iree_compiler::AMDAIE

#endif
