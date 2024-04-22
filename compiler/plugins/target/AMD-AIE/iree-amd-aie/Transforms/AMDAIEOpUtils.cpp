// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIEOpUtils.h"

namespace mlir::iree_compiler::AMDAIE {

SmallVector<scf::ForallOp> getInnermostForallLoops(Operation *op) {
  SmallVector<scf::ForallOp> res;
  op->walk([&](scf::ForallOp forallOp) {
    auto nestedForallOps = forallOp.getOps<scf::ForallOp>();
    if (nestedForallOps.empty()) res.push_back(forallOp);
  });
  return res;
}

}  // namespace mlir::iree_compiler::AMDAIE
