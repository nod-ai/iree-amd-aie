// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#include "iree_aie_router.h"

#include <set>

#include "d_ary_heap.h"
#include "iree_aie_runtime.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DirectedGraph.h"

#define DEBUG_TYPE "iree-aie-runtime-router"

#define OVER_CAPACITY_COEFF 0.02
#define USED_CAPACITY_COEFF 0.02
#define DEMAND_COEFF 1.1
static constexpr double INF = std::numeric_limits<double>::max();

namespace mlir::iree_compiler::AMDAIE {

}  // namespace mlir::iree_compiler::AMDAIE
