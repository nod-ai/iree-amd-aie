// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- AMDAIEEnums.h - AMDAIE enums ---------------------------------------===//
//===----------------------------------------------------------------------===//

#ifndef IREE_AIE_RUNTIME_AMDAIE_ENUMS_H_
#define IREE_AIE_RUNTIME_AMDAIE_ENUMS_H_

#include "mlir/IR/BuiltinAttributes.h"
// clang-format off
#include "iree-amd-aie/aie_runtime/AMDAIEEnums.h.inc"
// clang-format on

namespace mlir::iree_compiler::AMDAIE {

/// Some naming thoughts. Why is the mapping
/// npu1 -> aie2 (Phoenix)
/// npu4 -> aie2p (Strix) ?
/// It seems like npu2 and npu3 are used in the xdna-driver for some custom
/// devices, one might be a 4x4 variation of Strix.

/////////////////////
// AIE2 (Phoenix)  //
/////////////////////
static inline bool isNpu1(AMDAIEDevice d) {
  return d == AMDAIEDevice::npu1 || d == AMDAIEDevice::npu1_1col ||
         d == AMDAIEDevice::npu1_2col || d == AMDAIEDevice::npu1_3col ||
         d == AMDAIEDevice::npu1_4col;
}
static inline bool isAie2(AMDAIEDevice device) { return isNpu1(device); }

////////////////////
// AIE2P (Strix)  //
////////////////////
static inline bool isNpu4(AMDAIEDevice d) { return d == AMDAIEDevice::npu4; }
static inline bool isAie2P(AMDAIEDevice device) { return isNpu4(device); }

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AIE_RUNTIME_AMDAIE_ENUMS_H_
