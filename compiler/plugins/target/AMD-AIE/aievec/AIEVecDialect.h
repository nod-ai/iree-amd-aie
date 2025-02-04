//===- AIEVecDialect.h - AIE Vector Dialect ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// This file defines the AIE vector dialect.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECDIALECT_H
#define AIE_DIALECT_AIEVEC_IR_AIEVECDIALECT_H

#include "mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "aievec/AIEVecDialect.h.inc"

#endif  // AIE_DIALECT_AIEVEC_IR_AIEVECDIALECT_H
