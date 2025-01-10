// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_TYPES_H_
#define IREE_COMPILER_AMDAIE_TYPES_H_

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/IR/AMDAIEDmaOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

// clang-format off
#define GET_TYPEDEF_CLASSES
#include "iree-amd-aie/IR/AMDAIETypes.h.inc"
// clang-format on

#endif  // IREE_COMPILER_AMDAIE_TYPES_H_
