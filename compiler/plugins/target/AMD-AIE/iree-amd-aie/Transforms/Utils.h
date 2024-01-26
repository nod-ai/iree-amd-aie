// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_UTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_UTILS_H_

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

namespace mlir::iree_compiler::AMDAIE {

std::optional<scf::SCFFuseProducerOfSliceResult>
tileAndFuseProducerOfSlice(RewriterBase &rewriter,
                           tensor::ExtractSliceOp candidateSliceOp,
                           scf::ForallOp &loop);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_UTILS_H_
