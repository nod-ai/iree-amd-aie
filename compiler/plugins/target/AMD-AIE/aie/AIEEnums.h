// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_AIE_ENUMS_H
#define MLIR_AIE_ENUMS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

// clang-format off: must include AIEEnums.h.inc after the above includes
#include "aie/AIEEnums.h.inc"
// clang-format on

enum class AllocScheme { Sequential, BankAware, None };

#endif
