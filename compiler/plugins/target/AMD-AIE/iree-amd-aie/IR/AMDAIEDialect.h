// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_DIALECT_IREEAMDAIE_DIALECT_H_
#define IREE_COMPILER_AMDAIE_DIALECT_IREEAMDAIE_DIALECT_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

// clang-format off: must be included after all LLVM/MLIR eaders
#include "iree-amd-aie/IR/AMDAIEDialect.h.inc"  // IWYU pragma: keep
// clang-format on

#endif  // IREE_COMPILER_AMDAIE_DIALECT_IREEAMDAIE_DIALECT_H_
