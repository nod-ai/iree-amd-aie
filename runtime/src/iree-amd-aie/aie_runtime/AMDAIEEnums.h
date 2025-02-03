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

// TODO(someone_who_knows) explain the connection between npu numbering and
// aie numbering, and why npu jumps from '1' to '4'.
namespace mlir::iree_compiler::AMDAIE {
static inline bool isNpu1(AMDAIEDevice device) {
  return device == AMDAIEDevice::npu1 || device == AMDAIEDevice::npu1_1col ||
         device == AMDAIEDevice::npu1_2col ||
         device == AMDAIEDevice::npu1_3col || device == AMDAIEDevice::npu1_4col;
}

static inline bool isAie2(AMDAIEDevice device) { return isNpu1(device); }

static inline bool isNpu4(AMDAIEDevice device) {
  return device == AMDAIEDevice::npu4;
}

static inline bool isAie2P(AMDAIEDevice device) { return isNpu4(device); }

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AIE_RUNTIME_AMDAIE_ENUMS_H_
