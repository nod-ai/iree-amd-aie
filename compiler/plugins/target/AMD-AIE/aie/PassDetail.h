// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDAIE_AIE_PASSDETAIL_H_
#define AMDAIE_AIE_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::iree_compiler::AMDAIE {

#define GEN_PASS_DECL
#define GEN_PASS_DEF_AMDAIEROUTEFLOWSWITHPATHFINDER
#define GEN_PASS_DEF_AMDAIECORETOSTANDARD

#include "aie/Passes.h.inc"

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // AMDAIE_AIE_PASSDETAIL_H_
