// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIETRANSFORMS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIETRANSFORMS_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace mlir::iree_compiler::AMDAIE {

/// Hoist an affine apply op on a scf.for op's induction variable
/// TODO(jornt): Can we generalize this to go into upstream?
LogicalResult hoistForAffineApplyOp(RewriterBase &rewriter, affine::AffineApplyOp applyOp);

}  // namespace mlir::iree_compiler::AMDAIE

#endif
