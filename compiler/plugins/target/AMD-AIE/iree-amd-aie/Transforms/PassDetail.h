// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_PASSDETAIL_H_
#define IREE_AMD_AIE_TRANSFORMS_PASSDETAIL_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::iree_compiler::AMDAIE {

#define GEN_PASS_DECL
#define GEN_PASS_DEF_AMDAIEBRIDGETOAIR
#define GEN_PASS_DEF_AMDAIEBUFFERIZETOALLOCATION
#define GEN_PASS_DEF_AMDAIECLEANUP
#define GEN_PASS_DEF_AMDAIEFUSEFILLINTOFORALL
#define GEN_PASS_DEF_AMDAIELOWEREXECUTABLETARGET
#define GEN_PASS_DEF_AMDAIELOWERWORKGROUPCOUNT
#define GEN_PASS_DEF_AMDAIEPACKANDTRANSPOSE
#define GEN_PASS_DEF_AMDAIEPAD
#define GEN_PASS_DEF_AMDAIEPEELFORLOOP
#define GEN_PASS_DEF_AMDAIEPROPAGATEDATALAYOUT
#define GEN_PASS_DEF_AMDAIETILEANDFUSE
#define GEN_PASS_DEF_AMDAIEDECOMPOSELINALGEXTPACKUNPACKTOAIR
#include "iree-amd-aie/Transforms/Passes.h.inc"

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_PASSDETAIL_H_
